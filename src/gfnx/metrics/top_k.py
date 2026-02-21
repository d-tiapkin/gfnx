from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp

from ..base import TEnvironment, TEnvState
from ..utils.rollout import (
    TPolicyFn,
    TPolicyParams,
    forward_rollout,
)
from .base import BaseMetricsModule, BaseProcessArgs, EmptyInitArgs, EmptyUpdateArgs, MetricsState


@chex.dataclass
class TopKMetricsState(MetricsState):
    """State for Top-K reward and diversity metrics.

    This state container stores the computed top-K reward and diversity statistics
    for different values of K. It maintains arrays where each element corresponds
    to a different K value specified during module initialization.

    Attributes:
        top_k_rewards: Array of mean rewards for the top-K samples for each K value.
            Shape: (len(top_k),) where each entry is the mean reward of the top-K samples.
        top_k_diversity: Array of diversity measures for the top-K samples for each K value.
            Shape: (len(top_k),) where each entry is the average pairwise distance
            among top-K samples.
    """

    top_k_rewards: chex.Array
    top_k_diversity: chex.Array


class TopKMetricsModule(BaseMetricsModule):
    """Metric module for computing top-K reward and diversity statistics.

    This module evaluates policy performance by sampling trajectories and computing
    statistics for the top-K highest-reward samples. It measures both the quality
    (reward) and diversity of the best samples, providing insights into the policy's
    ability to find high-reward solutions while maintaining diversity.

    Attributes:
        env: Environment instance for trajectory generation and reward computation
        fwd_policy_fn: Forward policy function for generating trajectories
        num_traj: Total number of trajectories to sample for evaluation
        batch_size: Batch size for processing trajectories (for memory management)
        top_k: List of K values for which to compute top-K statistics
        distance_fn: Function to compute distance between states for diversity measurement
    """

    def __init__(
        self,
        env: TEnvironment,
        fwd_policy_fn: TPolicyFn,
        num_traj: int,
        batch_size: int,
        top_k: list[int] | None = None,
        distance_fn: Callable[[TEnvState, TEnvState], float] | None = None,
    ):
        """Initialize the top-K metrics module.

        Args:
            env: Environment instance for trajectory generation and reward computation
            fwd_policy_fn: Forward policy function for generating trajectory samples
            num_traj: Total number of trajectories to sample during evaluation.
                Must be >= max(top_k) for meaningful statistics.
            batch_size: Batch size for processing trajectories (used for memory management)
            top_k: List of K values for which to compute top-K statistics.
                Default is [10, 50, 100].
            distance_fn: Function that computes distance between two environment states
                for diversity measurement. Must return a scalar distance value.
        """
        if top_k is None:
            top_k = [10, 50, 100]
        self.num_traj = num_traj
        self.batch_size = batch_size
        self.env = env
        self.fwd_policy_fn = fwd_policy_fn
        self.top_k = top_k
        self.distance_fn = distance_fn

    def _get_distance_matrix(self, lhs_states: TEnvState, rhs_states: TEnvState) -> jnp.ndarray:
        """Compute pairwise distance matrix between two sets of states.

        Computes all pairwise distances between states in lhs_states and rhs_states
        using the configured distance function. This is used to measure diversity
        among the top-K samples by computing distances between all pairs.

        Args:
            lhs_states: First set of states, N states
            rhs_states: Second set of states, M states

        Returns:
            jnp.ndarray: Distance matrix of shape (N, M) where entry (i,j) contains
                the distance between lhs_states[i] and rhs_states[j]
        """
        result = jax.vmap(
            lambda lhs_state, rhs_states: jax.vmap(
                lambda rhs_state: self.distance_fn(lhs_state, rhs_state)
            )(rhs_states),
            in_axes=(0, None),
        )(lhs_states, rhs_states)
        chex.assert_shape(result, (lhs_states.is_pad.shape[0], rhs_states.is_pad.shape[0]))
        return result

    InitArgs = EmptyInitArgs

    def init(self, rng_key: chex.PRNGKey, args: InitArgs | None = None) -> TopKMetricsState:
        """Initialize the top-K metrics state.

        Creates initial state with zero-initialized arrays for top-K rewards and
        diversity statistics. The actual values will be computed during the process phase.

        Args:
            rng_key: JAX PRNG key for any random initialization (currently unused)
            args: EmptyInitArgs (no additional initialization parameters needed)

        Returns:
            TopKMetricsState: Initialized state with zero arrays for rewards and diversity
        """
        return TopKMetricsState(
            top_k_rewards=jnp.zeros((len(self.top_k),), dtype=jnp.float32),
            top_k_diversity=jnp.zeros((len(self.top_k),), dtype=jnp.float32),
        )

    UpdateArgs = EmptyUpdateArgs

    def update(
        self,
        metrics_state: TopKMetricsState,
        rng_key: chex.PRNGKey,
        args: UpdateArgs | None = None,
    ) -> TopKMetricsState:
        """Update the metric state with new data (no-op for top-K metrics).

        Top-K metrics are computed entirely during the process phase by sampling
        fresh trajectories, so no incremental updates are needed.

        Args:
            metrics_state: Current metric state (unchanged)
            rng_key: JAX PRNG key for any random operations (currently unused)
            args: EmptyUpdateArgs (no update parameters needed)

        Returns:
            TopKMetricsState: Unchanged metric state
        """
        return metrics_state

    @chex.dataclass
    class ProcessArgs(BaseProcessArgs):
        """Arguments for processing the TopKMetricsModule.

        Attributes:
            policy_params: Current policy parameters used for forward rollouts
                to generate trajectory samples for top-K evaluation.
            env_params: Environment parameters required for trajectory generation
                and reward computation.
        """

        policy_params: TPolicyParams
        env_params: Any

    def process(
        self,
        metrics_state: TopKMetricsState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs,
    ) -> TopKMetricsState:
        """Process the metric state to compute top-K reward and diversity statistics.

        This method performs the core computation for top-K metrics:
        1. Samples trajectories using the current policy
        2. Computes rewards for all terminal states
        3. Identifies the top-K highest-reward samples for each K value
        4. Computes mean rewards and diversity measures for each top-K set

        Args:
            metrics_state: Current metric state (largely ignored, will be replaced)
            rng_key: JAX PRNG key for trajectory sampling
            args: ProcessArgs object containing policy parameters and environment parameters

        Returns:
            TopKMetricsState: Updated state with computed top-K rewards and diversity statistics
        """
        # Sample a batch of trajectory
        _, info = forward_rollout(
            rng_key,
            num_envs=self.num_traj,
            policy_fn=self.fwd_policy_fn,
            policy_params=args.policy_params,
            env=self.env,
            env_params=args.env_params,
        )
        final_env_state = info["final_env_state"]
        rewards = jnp.exp(info["log_gfn_reward"])
        chex.assert_shape(rewards, (self.num_traj,))
        arg_sort_idx = jnp.argsort(rewards)

        # Compute top-k rewards and top-k diversity
        topk_rew = jnp.zeros((len(self.top_k),), dtype=jnp.float32)
        topk_div = jnp.zeros((len(self.top_k),), dtype=jnp.float32)

        for idx, k in enumerate(self.top_k):
            top_idx = arg_sort_idx[-k:]
            topk_rew = topk_rew.at[idx].set(jnp.mean(rewards[top_idx]))
            top_samples = jax.tree.map(lambda x, top_idx=top_idx: x[top_idx], final_env_state)
            num_nonzero_dist = (k - 1) * k
            distance_matrix = self._get_distance_matrix(top_samples, top_samples)
            topk_div = topk_div.at[idx].set(distance_matrix.sum() / num_nonzero_dist)

        return metrics_state.replace(
            top_k_rewards=topk_rew,
            top_k_diversity=topk_div,
        )

    def get(self, metrics_state: TopKMetricsState) -> dict:
        """Get the computed top-K metrics from the current state.

        Extracts the computed top-K reward and diversity statistics and formats them
        into a dictionary with descriptive keys for each K value specified during
        module initialization.

        Args:
            metrics_state: Current metric state containing computed top-K statistics

        Returns:
            Dict[str, float]: Dictionary containing computed top-K metrics with keys:
                - 'top_{k}_reward': Mean reward of the top-K samples for each K
                - 'top_{k}_diversity': Mean pairwise distance among top-K samples for each K

        Example:
            If initialized with top_k=[10, 50], might return:
            {
                "top_10_reward": 0.85,
                "top_10_diversity": 0.42,
                "top_50_reward": 0.78,
                "top_50_diversity": 0.51
            }
        """
        reward_dict = {
            f"top_{k}_reward": metrics_state.top_k_rewards[idx] for idx, k in enumerate(self.top_k)
        }
        diversity_dict = {
            f"top_{k}_diversity": metrics_state.top_k_diversity[idx]
            for idx, k in enumerate(self.top_k)
        }
        return {**reward_dict, **diversity_dict}
