"""Monte Carlo estimation of a GFlowNet's marginal over terminal states."""

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..base import TEnvironment, TEnvParams, TEnvState
from .rollout import TPolicyFn, TPolicyParams, backward_rollout, backward_trajectory_log_probs


def estimate_marginal_log_probs(
    rng_key: chex.PRNGKey,
    terminal_states: TEnvState,
    bwd_policy_fn: TPolicyFn,
    policy_params: TPolicyParams,
    env: TEnvironment,
    env_params: TEnvParams,
    n_rounds: int,
    batch_size: int = 1,
) -> Float[Array, " n_states"]:
    r"""Estimate $\log p_T(x)$ for a batch of terminal states via backward rollouts.

    The marginal probability of the forward policy over terminal states is

    $$
    p_T(x) = \sum_{\tau \rightsquigarrow x} P_F(\tau)
           = \mathbb{E}_{\tau \sim P_B(\cdot \mid x)}
             \left[\frac{P_F(\tau)}{P_B(\tau \mid x)}\right],
    $$

    which is estimated with `n_rounds` backward trajectories per state and averaged
    in probability space (log-sum-exp). When $P_B$ is deterministic (e.g. in
    autoregressive environments) a single round gives the exact value.

    Args:
        rng_key: PRNG key for the backward rollouts.
        terminal_states: Batched terminal states, leading dimension `n_states`.
        bwd_policy_fn: Backward policy whose info dict exposes both
            `forward_logits` and `backward_logits`.
        policy_params: Parameters consumed by `bwd_policy_fn`.
        env: Environment instance.
        env_params: Environment parameters.
        n_rounds: Number of backward trajectories sampled per terminal state.
        batch_size: Number of backward rollouts run in parallel; bounds memory use.
            The states are padded to a multiple of `batch_size` internally.

    Returns:
        Estimated $\log p_T(x)$ for each terminal state, shape `[n_states]`.
    """
    n_terminal_states = terminal_states.is_pad.shape[0]
    # Pad to a multiple of batch_size if needed
    remainder = n_terminal_states % batch_size
    if remainder != 0:
        pad_width = batch_size - remainder
        terminal_states = jax.tree.map(
            lambda x: jnp.pad(x, ((0, pad_width),) + ((0, 0),) * (x.ndim - 1), mode="constant"),
            terminal_states,
        )
    # Use additional batches to avoid OOM
    terminal_states = jax.tree.map(
        lambda x: x.reshape(-1, batch_size, *x.shape[1:]),
        terminal_states,
    )

    def process_batch(rng_key, terminal_states):
        """Process a single batch of terminal states."""
        rng_key, rollout_key = jax.random.split(rng_key)
        rng_keys_bwd = jax.random.split(rollout_key, batch_size)
        bwd_traj_data, _, _ = jax.vmap(
            lambda rng, s: backward_rollout(rng, s, bwd_policy_fn, policy_params, env, env_params),
            in_axes=(0, 0),
        )(rng_keys_bwd, terminal_states)
        log_pf_traj, log_pb_traj = jax.vmap(
            lambda td: backward_trajectory_log_probs(env, td, env_params)
        )(bwd_traj_data)
        return rng_key, log_pf_traj - log_pb_traj

    def process_round(carry: tuple[chex.PRNGKey, TEnvState], xs: None):
        """Process a single round of sampling across all batches."""
        rng_key, terminal_states = carry
        rng_key, log_ratio_traj = jax.lax.scan(process_batch, rng_key, terminal_states)
        chex.assert_shape(log_ratio_traj, terminal_states.is_pad.shape[:2])
        return (rng_key, terminal_states), log_ratio_traj.reshape(-1)

    _, log_ratio_traj = jax.lax.scan(
        process_round,
        (rng_key, terminal_states),
        xs=None,
        length=n_rounds,
    )
    log_ratio_traj = log_ratio_traj[:, :n_terminal_states]
    chex.assert_shape(log_ratio_traj, (n_rounds, n_terminal_states))

    # Average ratios over rounds for each state using log-sum-exp
    return jax.nn.logsumexp(log_ratio_traj, axis=0) - jnp.log(n_rounds)
