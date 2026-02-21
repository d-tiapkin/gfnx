from typing import Any

import chex
import flashbax as fbx
import jax.numpy as jnp

from ..base import TEnvironment, TEnvParams
from .base import (
    BaseMetricsModule,
    BaseUpdateArgs,
    EmptyInitArgs,
    EmptyProcessArgs,
    MetricsState,
)


@chex.dataclass
class MeanRewardMetricsState(MetricsState):
    """State for accumulating mean reward computation.

    This state container tracks the cumulative sum of rewards and the count
    of samples to compute running mean reward statistics. It enables incremental
    computation of mean reward without storing all individual reward values.

    Attributes:
        sum_reward: Cumulative sum of all rewards (in linear space, not log space)
        num: Total number of reward samples processed
    """

    sum_reward: float
    num: int


class MeanRewardMetricsModule(BaseMetricsModule):
    """Metric module for computing mean reward and its deviation from ground truth.

    This module tracks the empirical mean reward from collected samples and compares
    it against the known ground truth mean reward of the environment. It computes
    both absolute and relative deviations to assess how well the policy's reward
    distribution matches the true environment distribution.

    Attributes:
        env: Environment instance that must support tractable mean reward computation
        gt_mean_reward: Ground truth mean reward from the environment
    """

    def __init__(self, env: TEnvironment, env_params: TEnvParams):
        """Initialize the mean reward metric module.

        Args:
            env: Environment instance that must have tractable mean reward computation
                (env.is_mean_reward_tractable must be True)
            env_params: Environment parameters needed to compute the ground truth mean reward

        Raises:
            ValueError: If the environment does not support tractable mean reward computation
        """
        self.env = env
        if self.env.is_mean_reward_tractable:
            self.gt_mean_reward = self.env.get_mean_reward(env_params)
        else:
            raise ValueError("Ground truth mean reward is not tractable for this environment.")

    InitArgs = EmptyInitArgs

    def init(self, rng_key: chex.PRNGKey, args: InitArgs | None = None) -> MeanRewardMetricsState:
        """Initialize the mean reward metric state.

        Creates initial state with zero cumulative reward and zero sample count.
        The actual mean reward computation will be performed incrementally as
        new reward samples are added via update().

        Args:
            rng_key: JAX PRNG key for any random initialization (currently unused)
            args: EmptyInitArgs (no additional initialization parameters needed)

        Returns:
            MeanRewardMetricsState: Initialized state with zero accumulated reward and count
        """
        return MeanRewardMetricsState(sum_reward=0.0, num=0)

    @chex.dataclass
    class UpdateArgs(BaseUpdateArgs):
        """Arguments for updating the MeanRewardMetricsModule.

        Attributes:
            log_rewards: Array of log-reward values to add to the running statistics.
                These are expected to be in log space and will be converted to
                linear space for mean computation.
        """

        log_rewards: chex.Array

    def update(
        self, metrics_state: MeanRewardMetricsState, rng_key: chex.PRNGKey, args: UpdateArgs
    ) -> MeanRewardMetricsState:
        """Update the metric state with new reward samples.

        Adds new reward samples to the running statistics by converting log-rewards
        to linear space and accumulating them in the sum. Also updates the sample count.

        Args:
            metrics_state: Current metric state with accumulated reward sum and count
            rng_key: JAX PRNG key for any random operations (currently unused)
            args: UpdateArgs object containing the new reward samples

        Returns:
            MeanRewardMetricsState: Updated state with new rewards incorporated
                into the running statistics
        """
        return MeanRewardMetricsState(
            sum_reward=metrics_state.sum_reward + jnp.sum(args.log_rewards),
            num=metrics_state.num + args.log_rewards.shape[0],
        )

    ProcessArgs = EmptyProcessArgs

    def process(
        self,
        metrics_state: MeanRewardMetricsState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs | None = None,
    ) -> MeanRewardMetricsState:
        """Process the metric state for final computation (no-op for mean reward metrics).

        This method performs any final processing needed before metric computation.
        For mean reward metrics, no additional processing is required as the
        statistics are maintained incrementally during updates.

        Args:
            metrics_state: Current metric state with accumulated reward statistics
            rng_key: JAX PRNG key for any random operations (currently unused)
            args: EmptyProcessArgs (no additional processing parameters needed)

        Returns:
            MeanRewardMetricsState: Unchanged metric state ready for get() call
        """
        return metrics_state

    def get(self, metrics_state: MeanRewardMetricsState) -> dict[str, float]:
        """Get the computed mean reward metrics from the current state.

        Computes the empirical mean reward from accumulated samples and calculates
        both absolute and relative deviations from the ground truth mean reward.

        Args:
            metrics_state: Current metric state containing accumulated reward statistics

        Returns:
            Dict[str, float]: Dictionary containing computed reward metrics:
                - 'mean_reward': Empirical mean reward from collected samples
                - 'reward_delta': Absolute difference between empirical and ground truth mean
                - 'rel_reward_delta': Relative difference (normalized by ground truth mean)
        """
        mean_reward = metrics_state.sum_reward / jnp.maximum(metrics_state.num, 1)
        reward_delta = abs(mean_reward - self.gt_mean_reward)
        rel_reward_delta = reward_delta / self.gt_mean_reward
        return {
            "mean_reward": mean_reward,
            "reward_delta": reward_delta,
            "rel_reward_delta": rel_reward_delta,
        }


@chex.dataclass
class SWMeanRewardMetricsState(MetricsState):
    """State for mean reward metrics with sliding window buffer.

    This state container maintains a sliding window buffer of recent rewards
    to compute mean reward statistics over a fixed-size window of the most
    recent samples.

    Attributes:
        reward_buffer: Flashbax buffer storing the most recent reward samples
            in a circular buffer with configurable maximum size
    """

    reward_buffer: Any


class SWMeanRewardSWMetricsModule(BaseMetricsModule):
    """Sliding window mean reward metric module for recent performance tracking.

    This module computes mean reward statistics using a sliding window approach,
    maintaining only the most recent reward samples in a circular buffer.

    Attributes:
        env: Environment instance that must support tractable mean reward computation
        gt_mean_reward: Ground truth mean reward from the environment
        buffer_size: Maximum number of rewards to keep in the sliding window
        buffer_module: Flashbax buffer module for managing the sliding window
    """

    def __init__(self, env: TEnvironment, env_params: TEnvParams, buffer_size: int):
        """Initialize the sliding window mean reward metric module.

        Args:
            env: Environment instance that must have tractable mean reward computation
                (env.is_mean_reward_tractable must be True)
            env_params: Environment parameters needed to compute the ground truth mean reward
            buffer_size: Maximum number of reward samples to keep in the sliding window.
                Must be a positive integer.

        Raises:
            ValueError: If the environment does not support tractable mean reward computation
        """
        self.env = env
        if self.env.is_mean_reward_tractable:
            self.gt_mean_reward = self.env.get_mean_reward(env_params)
        else:
            raise ValueError("Ground truth mean reward is not tractable for this environment.")

        self.buffer_size = buffer_size
        self.buffer_module = fbx.make_item_buffer(
            max_length=buffer_size,
            min_length=1,
            sample_batch_size=1,
            add_batches=True,
        )

    InitArgs = EmptyInitArgs

    def init(
        self, rng_key: chex.PRNGKey, args: InitArgs | None = None
    ) -> SWMeanRewardMetricsState:
        """Initialize the sliding window mean reward metric state.

        Creates initial state with an empty sliding window buffer ready to
        accept reward samples. The buffer will automatically manage the
        sliding window behavior as new samples are added.

        Args:
            rng_key: JAX PRNG key for any random initialization (currently unused)
            args: EmptyInitArgs (no additional initialization parameters needed)

        Returns:
            SWMeanRewardMetricsState: Initialized state with empty sliding window buffer
        """
        buffer_state = self.buffer_module.init(jnp.array(0.0))  # Initialize with a dummy value
        return SWMeanRewardMetricsState(reward_buffer=buffer_state)

    @chex.dataclass
    class UpdateArgs(BaseUpdateArgs):
        """Arguments for updating the SWMeanRewardSWMetricsModule.

        Attributes:
            rewards: Array of reward values to add to the sliding window buffer.
                These rewards will replace the oldest entries when the buffer is full.
        """

        rewards: chex.Array

    def update(
        self, metrics_state: SWMeanRewardMetricsState, rng_key: chex.PRNGKey, args: UpdateArgs
    ) -> SWMeanRewardMetricsState:
        """Update the metric state with new reward samples in the sliding window.

        Adds new reward samples to the sliding window buffer. When the buffer is full,
        the oldest samples are automatically replaced with the new ones, maintaining
        the fixed window size.

        Args:
            metrics_state: Current metric state containing the sliding window buffer
            rng_key: JAX PRNG key for any random operations (currently unused)
            args: UpdateArgs object containing the new reward samples

        Returns:
            SWMeanRewardMetricsState: Updated state with new rewards added to the buffer
        """
        updated_data_buffer = self.buffer_module.add(metrics_state.reward_buffer, args.rewards)
        return metrics_state.replace(reward_buffer=updated_data_buffer)

    ProcessArgs = EmptyProcessArgs

    def process(
        self,
        metrics_state: MeanRewardMetricsState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs | None = None,
    ) -> MeanRewardMetricsState:
        """Process the metric state for final computation (no-op for sliding window metrics).

        This method performs any final processing needed before metric computation.
        For sliding window mean reward metrics, no additional processing is required
        as the statistics are computed directly from the buffer contents.

        Args:
            metrics_state: Current metric state with sliding window buffer
            rng_key: JAX PRNG key for any random operations (currently unused)
            args: EmptyProcessArgs (no additional processing parameters needed)

        Returns:
            MeanRewardMetricsState: Unchanged metric state ready for get() call
        """
        return metrics_state

    def get(self, metrics_state: SWMeanRewardMetricsState) -> dict[str, float]:
        """Get the computed sliding window mean reward metrics from the current state.

        Computes the mean reward from samples in the sliding window buffer and calculates
        both absolute and relative deviations from the ground truth mean reward. Only
        valid (non-empty) buffer entries are included in the computation.

        Args:
            metrics_state: Current metric state containing the sliding window buffer

        Returns:
            Dict[str, float]: Dictionary containing computed reward metrics:
                - 'mean_reward': Mean reward from samples in the sliding window
                - 'reward_delta': Absolute difference between window mean and ground truth
                - 'rel_reward_delta': Relative difference (normalized by ground truth mean)
        """
        buffer_state = metrics_state.reward_buffer
        all_rewards = metrics_state.reward_buffer.experience[0]
        indices = jnp.arange(all_rewards.shape[0])
        valid_mask = jnp.array(
            jnp.logical_or(buffer_state.is_full, indices < buffer_state.current_index),
            dtype=jnp.float32,
        )
        num_valid = jnp.sum(valid_mask)
        mean_reward = jnp.sum(buffer_state.experience * valid_mask) / jnp.maximum(num_valid, 1)
        reward_delta = abs(mean_reward - self.gt_mean_reward)
        rel_reward_delta = reward_delta / self.gt_mean_reward
        return {
            "mean_reward": mean_reward,
            "reward_delta": reward_delta,
            "rel_reward_delta": rel_reward_delta,
        }
