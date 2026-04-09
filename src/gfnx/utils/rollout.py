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

TPolicyFn = TypeVar("TPolicyFn")
TPolicyParams = TypeVar("TPolicyParams")


# Technical classes for storage of trajectory and transition data
@chex.dataclass
class TrajectoryData:
    obs: TObs  # [T x ...]
    state: TEnvState  # [T x ...]
    action: TAction | TBackwardAction  # [T]
    done: Bool[Array, " time"]
    pad: Bool[Array, " time"]
    info: dict  # [T x ...]


@chex.dataclass
class TransitionData:
    obs: TObs  # [T-1 x ...]
    state: TEnvState  # [T-1 x ...]
    action: TAction | TBackwardAction  # [T-1]
    next_obs: TObs  # [T-1 x ...]
    next_state: TEnvState  # [T-1 x ...]
    done: Bool[Array, " transitions"]
    pad: Bool[Array, " transitions"]


def forward_rollout(
    rng_key: chex.PRNGKey,
    policy_fn: TPolicyFn,
    policy_params: TPolicyParams,
    env: TEnvironment,
    env_params: TEnvParams,
) -> tuple[TrajectoryData, TEnvState, dict]:
    """Single-environment forward rollout.

    Args:
        rng_key: Random key passed to the policy and environment.
        policy_fn: Callable with signature
            `policy_fn(rng_key, obs: [obs_dim], params) -> tuple[logits: [n_actions], dict]`.
        policy_params: Parameters consumed by `policy_fn`.
        env: Environment instance.
        env_params: Environment parameters (typically static).

    Returns:
        A `(TrajectoryData, final_state, info)` tuple. `TrajectoryData` has shape
        `[T+1, ...]`, `final_state` is the state after the last step, and `info`
        contains scalar `entropy` and `trajectory_length`.

    To run multiple environments in parallel::

        rng_keys = jax.random.split(rng_key, num_envs)
        traj_data, final_states, info = jax.vmap(
            lambda rng: forward_rollout(rng, policy_fn, policy_params, env, env_params)
        )(rng_keys)
    """
    init_obs, init_state = env.reset(env_params)
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
        env.sample_action,
    )


def backward_rollout(
    rng_key: chex.PRNGKey,
    init_state: TEnvState,
    policy_fn: TPolicyFn,
    policy_params: TPolicyParams,
    env: TEnvironment,
    env_params: TEnvParams,
) -> tuple[TrajectoryData, TEnvState, dict]:
    """Single-environment backward rollout starting from a terminal state.

    Args:
        rng_key: Random key passed to the policy and environment.
        init_state: Terminal (or intermediate) state to start from.
        policy_fn: Callable with signature
            `policy_fn(rng_key, obs: [obs_dim], params) -> tuple[logits: [n_bwd_actions], dict]`.
        policy_params: Parameters consumed by `policy_fn`.
        env: Environment instance.
        env_params: Environment parameters (typically static).

    Returns:
        A `(TrajectoryData, final_state, info)` tuple mirroring the forward rollout contract.
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
        env.sample_backward_action,
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
    sample_action_fn: callable,
) -> tuple[TrajectoryData, TEnvState, dict]:
    """Common single-environment rollout implementation."""

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
        action = sample_action_fn(sample_rng_key, policy_log_probs)
        next_obs, next_env_state, done, step_info = step_fn(env_state, action, env_params)
        sampled_log_prob = policy_log_probs[action]
        info = {
            "entropy": -jnp.sum(jnp.where(invalid_mask, 0.0, policy_probs * policy_log_probs)),
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
    final_state = final_traj_state.env_state
    traj_entropy = jnp.sum(jnp.where(traj_data.pad, 0.0, traj_data.info["entropy"]))
    trajectory_length = jnp.sum(jnp.where(traj_data.pad, 0, 1))

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

    Args:
        traj_data: A trajectory with shape `[T+1, ...]`.

    Returns:
        TransitionData with shape `[T, ...]` (consecutive state-action-next_state tuples).
        Use `jax.vmap(split_traj_to_transitions)(batched_traj)` for batched trajectories.
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

    forward_logprobs = jax.nn.log_softmax(
        forward_logits, where=jnp.logical_not(fwd_action_mask), axis=-1
    )
    sampled_forward_logprobs = jnp.take_along_axis(
        forward_logprobs, fwd_actions[..., None], axis=-1
    ).squeeze(-1)

    backward_logprobs = jax.nn.log_softmax(
        backward_logits, where=jnp.logical_not(bwd_action_mask), axis=-1
    )
    sampled_backward_logprobs = jnp.take_along_axis(
        backward_logprobs, bwd_actions[..., None], axis=-1
    ).squeeze(-1)

    pad = traj_data.pad[:-1]
    log_pf_traj = jnp.sum(jnp.where(pad, 0.0, sampled_forward_logprobs))
    log_pb_traj = jnp.sum(jnp.where(pad, 0.0, sampled_backward_logprobs))
    return log_pf_traj, log_pb_traj


def forward_trajectory_log_probs(
    env: TEnvironment,
    fwd_traj_data: TrajectoryData,
    env_params: TEnvParams,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute scalar log PF(tau) and log PB(tau) of a single forward trajectory."""
    return _compute_trajectory_log_probs(env, fwd_traj_data, env_params, is_forward=True)


def backward_trajectory_log_probs(
    env: TEnvironment,
    bwd_traj_data: TrajectoryData,
    env_params: TEnvParams,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute scalar log PF(tau) and log PB(tau) of a single backward trajectory."""
    return _compute_trajectory_log_probs(env, bwd_traj_data, env_params, is_forward=False)
