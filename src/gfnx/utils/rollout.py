from typing import TypeVar

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
from .masking import compute_action_log_probs, masked_sum

TPolicyFn = TypeVar("TPolicyFn")
TPolicyParams = TypeVar("TPolicyParams")


# Technical classes for storage of trajectory and transition data
@chex.dataclass
class TrajectoryData:
    obs: TObs  # [T+1 x ...]
    state: TEnvState  # [T+1 x ...]
    action: TAction | TBackwardAction  # [T+1]
    done: Bool[Array, " time"]
    pad: Bool[Array, " time"]
    info: dict  # [T+1 x ...]


@chex.dataclass
class TransitionData:
    obs: TObs  # [T x ...]
    state: TEnvState  # [T x ...]
    action: TAction | TBackwardAction  # [T]
    next_obs: TObs  # [T x ...]
    next_state: TEnvState  # [T x ...]
    done: Bool[Array, " transitions"]
    pad: Bool[Array, " transitions"]


def forward_rollout(
    rng_key: chex.PRNGKey,
    policy_fn: TPolicyFn,
    policy_params: TPolicyParams,
    env: TEnvironment,
    env_params: TEnvParams,
) -> tuple[TrajectoryData, TEnvState, dict]:
    """Run a single-environment forward rollout.

    Rolls out the environment from its initial state under the given policy,
    sampling actions until the episode terminates. The trajectory is padded to
    `env.max_steps_in_episode + 1` time steps so shapes are static under
    `jax.jit` / `jax.vmap`.

    Args:
        rng_key: Random key passed to the policy and environment.
        policy_fn: Callable with signature
            `policy_fn(rng_key, env_obs, policy_params) -> tuple[chex.Array, dict]`.
            `env_obs` can be any PyTree matching the environment's observation
            structure. The first output contains (unmasked) action logits of
            shape `[n_actions]`; the info dict may include forward/backward
            logits under the keys `fwd_logits` and `bwd_logits`.
        policy_params: Parameters consumed by `policy_fn`.
        env: Environment instance exposing `reset`, `step`, and `get_invalid_mask`.
        env_params: Environment parameters (typically static).

    Returns:
        A `(TrajectoryData, final_state, info)` tuple. `TrajectoryData` has
        shape `[T+1, ...]`, `final_state` is the state after the last step, and
        `info` contains scalar `entropy` and `trajectory_length`.

    To run multiple environments in parallel::

        rng_keys = jax.random.split(rng_key, num_envs)
        traj_data, final_states, info = jax.vmap(
            lambda rng: forward_rollout(rng, policy_fn, policy_params, env, env_params)
        )(rng_keys)
    """
    init_state = env.reset()
    init_obs = env.get_obs(init_state, env_params)
    return _generic_rollout(
        rng_key,
        init_obs,
        init_state,
        policy_fn,
        policy_params,
        env,
        env_params,
        env.step,
        env.get_invalid_mask,
    )


def backward_rollout(
    rng_key: chex.PRNGKey,
    init_state: TEnvState,
    policy_fn: TPolicyFn,
    policy_params: TPolicyParams,
    env: TEnvironment,
    env_params: TEnvParams,
) -> tuple[TrajectoryData, TEnvState, dict]:
    """Run a single-environment backward rollout starting from a terminal state.

    Walks the environment backwards from `init_state` until the initial state
    is reached, padding the trajectory to `env.max_steps_in_episode + 1` time
    steps so shapes are static under `jax.jit` / `jax.vmap`.

    Args:
        rng_key: Random key passed to the policy and environment.
        init_state: Terminal (or intermediate) state to start from.
        policy_fn: Callable with signature
            `policy_fn(rng_key, env_obs, policy_params) -> tuple[chex.Array, dict]`.
            `env_obs` can be any PyTree matching the environment's observation
            structure. The first output contains (unmasked) backward action
            logits of shape `[n_bwd_actions]`.
        policy_params: Parameters consumed by `policy_fn`.
        env: Environment instance exposing `get_obs`, `backward_step`,
            and `get_invalid_backward_mask`.
        env_params: Environment parameters (typically static).

    Returns:
        A `(TrajectoryData, final_state, info)` tuple mirroring the forward
        rollout contract.
    """
    init_obs = env.get_obs(init_state, env_params)
    return _generic_rollout(
        rng_key,
        init_obs,
        init_state,
        policy_fn,
        policy_params,
        env,
        env_params,
        env.backward_step,
        env.get_invalid_backward_mask,
    )


def _generic_rollout(
    rng_key: chex.PRNGKey,
    init_obs: TObs,
    init_state: TEnvState,
    policy_fn: TPolicyFn,
    policy_params: TPolicyParams,
    env: TEnvironment,
    env_params: TEnvParams,
    step_fn: callable,
    mask_fn: callable,
) -> tuple[TrajectoryData, TEnvState, dict]:
    """Common single-environment rollout implementation shared by forward/backward helpers.

    Args:
        rng_key: Random key passed to the policy and environment.
        init_obs: Observation at which to start the rollout (any PyTree).
        init_state: Environment state matching `init_obs`.
        policy_fn: Callable returning logits (and optionally metadata) given
            `(rng_key, env_obs, policy_params)`.
        policy_params: Parameters consumed by `policy_fn`.
        env: Environment instance used for auxiliary methods such as sampling.
        env_params: Environment parameters (typically static).
        step_fn: Function with signature
            `step_fn(env_state, action, env_params) -> tuple[TObs, TEnvState, Bool, dict]`.
        mask_fn: Function producing invalid-action masks for the current state.

    Returns:
        A `(TrajectoryData, final_state, info)` tuple containing a padded
        trajectory of shape `[T+1, ...]` and rollout-level statistics
        (`entropy`, `trajectory_length`).
    """

    @chex.dataclass
    class TrajSamplingState:
        env_obs: TObs
        env_state: TEnvState
        rng_key: chex.PRNGKey

    def environment_step_fn(
        traj_step_state: TrajSamplingState, _: None
    ) -> tuple[TrajSamplingState, TrajectoryData]:
        env_state = traj_step_state.env_state
        env_obs = traj_step_state.env_obs
        rng_key = traj_step_state.rng_key

        rng_key, policy_rng_key, sample_rng_key = jax.random.split(rng_key, 3)

        invalid_mask = mask_fn(env_state, env_params)
        logits, policy_info = policy_fn(policy_rng_key, env_obs, policy_params)
        policy_probs = jax.nn.softmax(logits, where=jnp.logical_not(invalid_mask), axis=-1)
        policy_log_probs = jax.nn.log_softmax(logits, where=jnp.logical_not(invalid_mask), axis=-1)
        action = jax.random.categorical(sample_rng_key, policy_log_probs, axis=-1)
        next_obs, next_env_state, done, step_info = step_fn(env_state, action, env_params)
        sampled_log_prob = policy_log_probs[action]
        info = {
            "entropy": -masked_sum(
                policy_probs * policy_log_probs, jnp.logical_not(invalid_mask)
            ),
            "sampled_log_prob": sampled_log_prob,
            **step_info,
            **policy_info,
        }

        traj_data = TrajectoryData(
            obs=env_obs,
            state=env_state,
            action=action,
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

    final_traj_state, traj_data = jax.lax.scan(
        f=environment_step_fn,
        init=TrajSamplingState(env_obs=init_obs, env_state=init_state, rng_key=rng_key),
        xs=None,
        # +1 to always have a padding step at the end
        length=env.max_steps_in_episode + 1,
    )

    # traj_data shape: [T+1, ...] — scan is time-major, no batch dim for single env
    chex.assert_tree_shape_prefix(traj_data, (env.max_steps_in_episode + 1,))
    final_state = final_traj_state.env_state
    not_pad = jnp.logical_not(traj_data.pad)
    traj_entropy = masked_sum(traj_data.info["entropy"], not_pad)
    trajectory_length = jnp.sum(not_pad.astype(jnp.int32))

    return (
        traj_data,
        final_state,
        {
            "entropy": traj_entropy,
            "trajectory_length": trajectory_length,
        },
    )


def split_traj_to_transitions(traj_data: TrajectoryData) -> TransitionData:
    """Split a single trajectory into transitions.

    Converts a trajectory (sequence of states, actions, etc.) into a sequence
    of transitions (state-action-next_state tuples) by slicing the trajectory
    data appropriately.

    Args:
        traj_data: A trajectory with shape `[T+1, ...]` where T is the
            trajectory length.

    Returns:
        TransitionData with shape `[T, ...]` containing the following fields:
            - obs: Previous observations.
            - state: Previous states.
            - action: Actions taken.
            - next_obs: Next observations.
            - next_state: Next states.
            - done: Done flags.
            - pad: Padding masks.

        Use `jax.vmap(split_traj_to_transitions)(batched_traj)` for batched
        trajectories.
    """
    return TransitionData(
        obs=jax.tree.map(lambda x: x[:-1], traj_data.obs),
        state=jax.tree.map(lambda x: x[:-1], traj_data.state),
        action=jax.tree.map(lambda x: x[:-1], traj_data.action),
        next_obs=jax.tree.map(lambda x: x[1:], traj_data.obs),
        next_state=jax.tree.map(lambda x: x[1:], traj_data.state),
        done=traj_data.done[:-1],
        pad=traj_data.pad[:-1],
    )


def _compute_trajectory_log_probs(
    env: TEnvironment,
    traj_data: TrajectoryData,
    env_params: TEnvParams,
    is_forward: bool,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute log PF(tau) and log PB(tau) for a single trajectory.

    Args:
        env: Environment instance.
        traj_data: Single trajectory with shape `[T+1, ...]`.
        env_params: Environment parameters.
        is_forward: If True, compute from a forward trajectory; if False, backward.

    Returns:
        Scalar `(log_pf, log_pb)` for the trajectory.
    """
    states = jax.tree.map(lambda x: x[:-1], traj_data.state)  # [T, ...]

    if is_forward:
        next_states = jax.tree.map(lambda x: x[1:], traj_data.state)

        forward_logits = traj_data.info["forward_logits"][:-1]  # [T, n_actions]
        backward_logits = traj_data.info["backward_logits"][1:]  # [T, n_bwd_actions]

        fwd_actions = traj_data.action[:-1]  # [T]
        bwd_actions = env.get_backward_action_batch(states, fwd_actions, next_states, env_params)

        fwd_action_mask = env.get_invalid_mask_batch(states, env_params)
        bwd_action_mask = env.get_invalid_backward_mask_batch(next_states, env_params)
    else:
        prev_states = jax.tree.map(lambda x: x[1:], traj_data.state)

        forward_logits = traj_data.info["forward_logits"][1:]  # [T, n_actions]
        backward_logits = traj_data.info["backward_logits"][:-1]  # [T, n_bwd_actions]

        bwd_actions = traj_data.action[:-1]  # [T]
        fwd_actions = env.get_forward_action_batch(states, bwd_actions, prev_states, env_params)

        bwd_action_mask = env.get_invalid_backward_mask_batch(states, env_params)
        fwd_action_mask = env.get_invalid_mask_batch(prev_states, env_params)

    not_pad = jnp.logical_not(traj_data.pad[:-1])
    sampled_forward_logprobs = compute_action_log_probs(
        forward_logits, fwd_actions, fwd_action_mask
    )
    sampled_backward_logprobs = compute_action_log_probs(
        backward_logits, bwd_actions, bwd_action_mask
    )

    log_pf_traj = masked_sum(sampled_forward_logprobs, not_pad)
    log_pb_traj = masked_sum(sampled_backward_logprobs, not_pad)
    return log_pf_traj, log_pb_traj


def forward_trajectory_log_probs(
    env: TEnvironment,
    fwd_traj_data: TrajectoryData,
    env_params: TEnvParams,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute the log PF(tau) and log PB(tau) of a single forward trajectory.

    Args:
        env: The environment instance.
        fwd_traj_data: A single forward trajectory with shape `[T+1, ...]`
            where T is the trajectory length.
        env_params: Parameters for the environment.

    Returns:
        Scalar `(log_pf, log_pb)` of the forward trajectory.
    """
    return _compute_trajectory_log_probs(env, fwd_traj_data, env_params, is_forward=True)


def backward_trajectory_log_probs(
    env: TEnvironment,
    bwd_traj_data: TrajectoryData,
    env_params: TEnvParams,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute the log PF(tau) and log PB(tau) of a single backward trajectory.

    Args:
        env: The environment instance.
        bwd_traj_data: A single backward trajectory with shape `[T+1, ...]`
            where T is the trajectory length.
        env_params: Parameters for the environment.

    Returns:
        Scalar `(log_pf, log_pb)` of the backward trajectory.
    """
    return _compute_trajectory_log_probs(env, bwd_traj_data, env_params, is_forward=False)
