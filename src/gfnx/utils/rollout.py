import functools
from typing import Any, TypeVar

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from ..base import (
    TAction,
    TBackwardAction,
    TEnvironment,
    TEnvParams,
    TEnvState,
    TObs,
)
from .masking import mask_logits

TPolicyFn = TypeVar("TPolicyFn")
TPolicyParams = TypeVar("TPolicyParams")


# Technical classes for storage of trajectory and transition  data
@chex.dataclass
class TrajectoryData:
    obs: TObs  # [B x ...]
    state: TEnvState  # [B x ...]
    action: TAction | TBackwardAction  # [B]
    log_gfn_reward: Float[Array, " batch_size"]
    done: Bool[Array, " batch_size"]
    pad: Bool[Array, " batch_size"]
    info: dict  # [B x ...]


@chex.dataclass
class TransitionData:
    obs: TObs  # [B x ...]
    state: TEnvState  # [B x ...]
    action: TAction | TBackwardAction  # [B]
    log_gfn_reward: Float[Array, " batch_size"]
    next_obs: TObs  # [B x ...]
    next_state: TEnvState  # [B x ...]
    done: Bool[Array, " batch_size"]
    pad: Bool[Array, " batch_size"]


def forward_rollout(
    rng_key: chex.PRNGKey,
    num_envs: int,
    policy_fn: TPolicyFn,
    policy_params: TPolicyParams,
    env: TEnvironment,
    env_params: TEnvParams,
) -> tuple[TrajectoryData, dict]:
    """Forward rollout in the environment.

    Args:
    - rng_key (chex.PRNGKey): Random number generator key.
    - num_envs (int): Number of parallel environment to perform a rollout.
    - policy_fn (TPolicyFn): Policy function to determine actions.
    This function should have the following signature:
        `
        policy_fn(
            rng_key: chex.PRNGKey, env_obs: TObs, policy_params: TPolicyParams
        ) -> tuple[chex.Array, dict]
        `
        and should return the action logits (not nessesarily masked) as the
        first output and info (forward and backward logits as 'fwd_logits'
        and 'bwd_logits', and additional) as the second one.
    - policy_params (TPolicyParams): Parameters for the policy function.
    - env (TEnvironment): The environment to interact with.
    - env_params (TEnvParams): Parameters for the environment.

    Returns:
        tuple[TrajectoryData, dict]: A tuple containing the trajectory data and
        additional information, such as trajectory-wise entropy and final
        states.
    """
    init_obs, init_state = env.reset(num_envs, env_params)
    return generic_rollout(
        rng_key,
        init_obs,
        init_state,
        policy_fn,
        policy_params,
        env,
        env_params,
        env.step,
        env.get_invalid_mask,
        env.sample_action,
    )


def backward_rollout(
    rng_key: chex.PRNGKey,
    init_state: TEnvState,
    policy_fn: TPolicyFn,
    policy_params: TPolicyParams,
    env: TEnvironment,
    env_params: TEnvParams,
) -> tuple[TrajectoryData, dict]:
    """Forward rollout in the environment.

    Args:
    - rng_key (chex.PRNGKey): Random number generator key.
    - init_state (TEnvState): Initial state of the environment.
    - policy_fn (TPolicyFn): Policy function to determine actions.
    This function should have the following signature:
        `
        policy_fn(
            rng_key: chex.PRNGKey, env_obs: TObs, policy_params: TPolicyParams
        ) -> chex.Array
        `
        and should return the action logits (not nessesarily masked).
    - policy_params (TPolicyParams): Parameters for the policy function.
    - env (TEnvironment): The environment to interact with.
    - env_params (TEnvParams): Parameters for the environment.

    Returns:
        tuple[TrajectoryData, dict]: A tuple containing the trajectory data and
        additional information, such as trajectory-wise entropy and final
        states.
    """
    init_obs = env.get_obs(init_state, env_params)
    return generic_rollout(
        rng_key,
        init_obs,
        init_state,
        policy_fn,
        policy_params,
        env,
        env_params,
        env.backward_step,
        env.get_invalid_backward_mask,
        env.sample_backward_action,
    )


def generic_rollout(
    rng_key: chex.PRNGKey,
    init_obs: TObs,
    init_state: TEnvState,
    policy_fn: TPolicyFn,
    policy_params: TPolicyParams,
    env: TEnvironment,
    env_params: TEnvParams,
    step_fn: callable,
    mask_fn: callable,
    sample_action_fn: callable,
) -> tuple[TrajectoryData, dict]:
    """Generic function for rollouts in the environment.

    Args:
    - rng_key (chex.PRNGKey): Random number generator key.
    - init_obs (TObs): Initial observation from the environment.
    - init_state (TEnvState): Initial state of the environment.
    - policy_fn (TPolicyFn): Policy function to determine actions.
    This function should have the following signature:
        `
        policy_fn(
            rng_key: chex.PRNGKey, env_obs: TObs, policy_params: TPolicyParams
        ) -> tuple[chex.Array, dict]
        `
        and should return the action logits (not nessesarily masked) as the
        first output and info (forward and backward logits as 'fwd_logits'
        and 'bwd_logits', and additional) as the second one.
    - policy_params (TPolicyParams): Parameters for the policy function.
    - env (TEnvironment): The environment to interact with.
    - env_params (TEnvParams): Parameters for the environment.
    - step_fn (callable): Function to perform a step in the environment.
    This function should have the following signature:
        `step_fn(
            env_state: TEnvState,
            action: TAction | TBackwardAction,
            env_params: TEnvParams,
    ) -> Tuple[TObs, TEnvState, TLogReward, TDone, Dict[Any, Any]]`
    - mask_fn (callable): Function to compute masks for invalid actions.
    This function should have the following signature:
        `mask_fn(env_state: TEnvState, env_params: TEnvParams) -> Array`
    - sample_action_fn (callable): Function to sample the corresponding action.
    Signture:
        `sample_action_fn(rng_key: PRNGKey, policy_probs: Array) -> Action`

    Returns:
        tuple[TrajectoryData, dict]: A tuple containing the trajectory data and
        additional information, such as trajectory-wise entropy and final
        states.
    """
    num_envs = jax.tree.leaves(init_state)[0].shape[0]  # Get the batch size

    @chex.dataclass
    class TrajSamplingState:
        env_obs: TObs
        env_state: TEnvState

        rng_key: chex.PRNGKey
        policy_params: Any
        env_params: TEnvParams

    @functools.partial(jax.jit, donate_argnums=(0,))
    def environment_step_fn(
        traj_step_state: TrajSamplingState, _: None
    ) -> tuple[TrajSamplingState, TrajectoryData]:
        # Unpack the sampling state
        # policy = eqx.combine(policy_params, policy_static)
        env_params = traj_step_state.env_params
        env_state = traj_step_state.env_state

        env_obs = traj_step_state.env_obs
        rng_key = traj_step_state.rng_key

        # Split the random key
        rng_key, policy_rng_key, sample_rng_key = jax.random.split(rng_key, 3)

        # Get the invalid mask for the current state
        invalid_mask = mask_fn(env_state, env_params)
        # Call the policy function
        logits, policy_info = policy_fn(policy_rng_key, env_obs, policy_params)
        # Very important part: masking invalid actions
        masked_logits = mask_logits(logits, invalid_mask)
        policy_probs = jax.nn.softmax(masked_logits, axis=-1)
        # Sampling the required action
        action = sample_action_fn(sample_rng_key, policy_probs)
        next_obs, next_env_state, log_gfn_reward, done, step_info = step_fn(
            env_state, action, env_params
        )
        log_probs = jax.nn.log_softmax(masked_logits)
        sampled_log_probs = jnp.take_along_axis(
            log_probs, action[..., None], axis=-1
        ).squeeze(-1)
        info = {
            "entropy": -jnp.sum(policy_probs * log_probs, axis=-1),
            "sampled_log_prob": sampled_log_probs,
            **step_info,
            **policy_info,
        }

        traj_data = TrajectoryData(
            obs=env_obs,
            state=env_state,
            action=action,
            log_gfn_reward=log_gfn_reward,
            done=done,
            pad=next_env_state.is_pad,
            info=info,
        )
        next_traj_state = traj_step_state.replace(
            env_obs=next_obs,
            env_state=next_env_state,
            rng_key=rng_key,
        )

        return next_traj_state, traj_data

    final_traj_stats, traj_data = jax.lax.scan(
        f=environment_step_fn,
        init=TrajSamplingState(
            env_obs=init_obs,
            env_state=init_state,
            rng_key=rng_key,
            policy_params=policy_params,
            env_params=env_params,
        ),
        xs=None,
        # +1 to always have a padding in the end
        length=env.max_steps_in_episode + 1,
    )

    # Now, the shape of traj data is [(T + 1) x B x ...]
    # Need to transpose it to [B x (T + 1) x ...]
    chex.assert_tree_shape_prefix(
        traj_data, (env.max_steps_in_episode + 1, num_envs)
    )
    traj_data = jax.tree.map(
        lambda x: jnp.transpose(x, axes=(1, 0) + tuple(range(2, x.ndim))),
        traj_data,
    )
    chex.assert_tree_shape_prefix(
        traj_data, (num_envs, env.max_steps_in_episode + 1)
    )

    # Logging data
    final_env_state = final_traj_stats.env_state
    traj_entropy = jnp.sum(
        jnp.where(traj_data.pad, 0.0, traj_data.info["entropy"]), axis=1
    )
    return traj_data, {
        "entropy": traj_entropy,
        "final_env_state": final_env_state,
    }


def split_traj_to_transitions(traj_data: TrajectoryData) -> TransitionData:
    """Split a trajectory into transitions.

    This function converts a trajectory (sequence of states, actions, etc.)
    into a sequence of transitions (state-action-next_state tuples) by slicing
    the trajectory data appropriately and reshaping it.

    Args:
        traj_data (TrajectoryData): A trajectory containing observations,
            states, actions, rewards, and other data with shape [B x T x ...]
            where B is batch size and T is trajectory length.

    Returns:
        TransitionData: A dataclass containing transitions with all arrays
            reshaped to [BT x ...] where BT is batch size times trajectory
            length. Contains the following fields:
            - obs: Previous observations
            - state: Previous states
            - action: Actions taken
            - log_gfn_reward: GFlowNet rewards
            - next_obs: Next observations
            - next_state: Next states
            - done: Done flags
            - pad: Padding masks
    """

    def slice_prev(tree: Any) -> Any:
        return jax.tree.map(lambda x: x[:, :-1], tree)

    def slice_next(tree: Any) -> Any:
        return jax.tree.map(lambda x: x[:, 1:], tree)

    base_transition_data = TransitionData(
        obs=slice_prev(traj_data.obs),
        state=slice_prev(traj_data.state),
        action=slice_prev(traj_data.action),
        log_gfn_reward=slice_prev(traj_data.log_gfn_reward),
        next_obs=slice_next(traj_data.obs),
        next_state=slice_next(traj_data.state),
        done=slice_prev(traj_data.done),
        pad=slice_prev(traj_data.pad),
    )
    # Reshape all the arrays to [BT x ...]
    return jax.tree.map(
        lambda x: x.reshape((-1,) + x.shape[2:]), base_transition_data
    )
