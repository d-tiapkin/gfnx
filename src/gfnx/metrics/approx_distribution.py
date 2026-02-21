from collections.abc import Callable
from typing import Any, ClassVar

import chex
import flashbax as fbx
import jax
import jax.numpy as jnp

from ..base import TEnvironment, TEnvParams, TEnvState
from ..utils.distances import jensen_shannon_divergence, kl_divergence, total_variation_distance
from .base import BaseInitArgs, BaseMetricsModule, BaseProcessArgs, BaseUpdateArgs, MetricsState


@chex.dataclass
class ApproxDistributionMetricsState(MetricsState):
    """State for approximate distribution-based metrics.

    This state container holds the data necessary for computing distribution-based
    metrics such as KL divergence and total variation distance. It stores both the
    true distribution from the environment and the empirical distribution computed
    from collected samples, along with a replay buffer for state storage.

    Attributes:
        true_distribution: The true distribution of the environment (ground truth)
        empirical_distribution: The empirical distribution computed from collected samples
        replay_buffer: Buffer storing environment states for distribution computation
    """

    true_distribution: chex.Array
    empirical_distribution: chex.Array
    replay_buffer: chex.ArrayTree


def marginal_distribution(true_dist: chex.Array, empirical_dist: chex.Array) -> chex.Array:
    """Compute the marginal distribution from the empirical distribution.

    Computes the marginal distribution by summing over all dimensions except the first two,
    then normalizes the result to ensure it represents a valid probability distribution.
    This is useful for visualizing the current convergence of the empirical distribution.

    Args:
        true_dist: True distribution of the environment (unused in current implementation)
        empirical_dist: Empirical distribution tensor from the replay buffer.
                       Expected to have shape (batch_size, ..., additional_dims)

    Returns:
        chex.Array: Normalized marginal distribution after summing over extra dimensions

    Note:
        This function is used as a metric computation function and should follow the
        same interface as other metric functions (tv, kl, jsd) that take true and empirical
        distributions as inputs and return a computed value.
    """
    axis_to_sum = tuple(range(2, len(empirical_dist.shape)))
    empirical_dist = jnp.sum(empirical_dist, axis=axis_to_sum)
    empirical_dist /= jnp.sum(empirical_dist)  # Normalize
    return empirical_dist


class ApproxDistributionMetricsModule(BaseMetricsModule):
    """Distribution-based metrics module for enumerable environments.

    This metric module computes distribution-based metrics by comparing the true
    distribution of an enumerable environment with an empirical distribution
    derived from collected samples. It supports various distance metrics between
    distributions such as KL divergence and total variation distance.

    The module maintains a replay buffer to accumulate environment states and
    computes empirical distributions from these samples. It can only be used
    with enumerable environments that provide access to their true distribution.

    Supported metrics:
        - "tv": Total variation distance between distributions
        - "kl": KL divergence from empirical to true distribution
        - "jsd": Jensen-Shannon divergence from empirical to true distribution
        - "2d_marginal_distribution": Marginal distribution computation

    Attributes:
        metrics: List of metric names to compute, 
            choose from {"tv", "kl", "jsd", "2d_marginal_distribution"}.
        env: Enumerable environment for which to compute metrics.
        buffer_size: Maximum number of states to store in the replay buffer.
    """

    _supported_metrics: ClassVar[dict[str, Callable[..., chex.Array]]] = {
        "tv": total_variation_distance,
        "kl": kl_divergence,
        "jsd": jensen_shannon_divergence,
        "2d_marginal_distribution": marginal_distribution,
    }

    def __init__(
        self,
        metrics: list[str],
        env: TEnvironment,
        buffer_size: int = 1000,
    ):
        """Initialize the distribution metrics module.

        Sets up the metric module with specified metrics, environment, and buffer size.
        Validates that the environment is enumerable and that all requested metrics
        are supported.

        Args:
            metrics: List of metric names to compute. Must be subset of supported metrics.
                    Supported options: ["tv", "kl", "jsd", "marginal_distribution"]
            env: Environment instance for which to compute metrics. Must be enumerable
                (i.e., env.is_enumerable must be True)
            buffer_size: Maximum number of states to store in the replay buffer for
                       empirical distribution computation. Must be positive integer.

        Raises:
            ValueError: If environment is not enumerable, buffer_size is not positive,
                       metrics is not a list of strings, or contains unsupported metrics
        """
        if not env.is_enumerable:
            raise ValueError(f"Environment {env.name} is not enumerable")
        if not isinstance(buffer_size, int) or buffer_size <= 0:
            raise ValueError("buffer_size must be a positive integer")
        if not isinstance(metrics, list) or not all(isinstance(m, str) for m in metrics):
            raise ValueError("metrics must be a list of strings")
        if not all(m in self._supported_metrics for m in metrics):
            raise ValueError(
                f"Unsupported metrics. Supported metrics are: \
                    {self._supported_metrics}"
            )

        self.metrics = metrics
        self.env = env

        self.buffer_module = fbx.make_item_buffer(
            max_length=buffer_size,
            min_length=1,
            sample_batch_size=1,
            add_batches=True,
        )

    @chex.dataclass
    class InitArgs(BaseInitArgs):
        """Arguments for initializing the ApproxDistributionMetricsModule.

        Attributes:
            env_params: Environment parameters needed to obtain the true distribution
                and initialize environment states for the replay buffer.
        """

        env_params: TEnvParams

    def init(self, rng_key: chex.PRNGKey, args: InitArgs) -> ApproxDistributionMetricsState:
        """Initialize the metric state for distribution metrics.

        Creates and initializes all components needed for distribution metric computation:
        the true distribution from the environment, an initial uniform empirical distribution,
        and a replay buffer for storing environment states.

        Args:
            rng_key: JAX PRNG key for any random initialization (currently unused)
            args: InitArgs object containing environment parameters

        Returns:
            ApproxDistributionMetricsState: Initialized state containing:
                - true_distribution: Ground truth distribution from the environment
                - empirical_distribution: Initial uniform distribution
                - replay_buffer: Empty buffer ready to collect states
        """
        _, fake_state = self.env.reset(1, args.env_params)
        fake_single_state = jax.tree.map(lambda x: x[0], fake_state)
        # Replay buffer takes only shapes, but not values
        replay_buffer_state = self.buffer_module.init(fake_single_state)
        true_distribution = self.env.get_true_distribution(args.env_params)
        return ApproxDistributionMetricsState(
            true_distribution=true_distribution,
            empirical_distribution=jnp.ones_like(true_distribution) / true_distribution.size,
            replay_buffer=replay_buffer_state,
        )

    @chex.dataclass
    class UpdateArgs(BaseUpdateArgs):
        """Arguments for updating the ApproxDistributionMetricsModule.

        Attributes:
            states: Environment states to add to the replay buffer for empirical
                distribution computation. These represent states visited during
                training/evaluation episodes.
        """

        states: TEnvState

    def update(
        self,
        metrics_state: ApproxDistributionMetricsState,
        rng_key: chex.PRNGKey,
        args: UpdateArgs,
    ) -> ApproxDistributionMetricsState:
        """Update the metric state with new environment states.

        Adds new environment states to the replay buffer for empirical distribution
        computation.

        Args:
            metrics_state: Current metric state containing the replay buffer
            rng_key: JAX PRNG key for any random operations (currently unused)
            args: UpdateArgs object containing environment states to add

        Returns:
            ApproxDistributionMetricsState: Updated state with new states added to the buffer.
                                         The true_distribution and empirical_distribution
                                         remain unchanged until process() is called.
        """
        updated_buffer = self.buffer_module.add(metrics_state.replay_buffer, args.states)
        return metrics_state.replace(replay_buffer=updated_buffer)

    @chex.dataclass
    class ProcessArgs(BaseProcessArgs):
        """Arguments for processing the ApproxDistributionMetricsModule.

        Attributes:
            env_params: Environment parameters needed for empirical distribution
                computation from the collected states in the replay buffer.
        """

        env_params: TEnvParams

    def process(
        self,
        metrics_state: ApproxDistributionMetricsState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs,
    ) -> ApproxDistributionMetricsState:
        """Process the metric state to compute the empirical distribution.

        This method computes the empirical distribution from the states stored in the
        replay buffer. It calls the environment's `get_empirical_distribution` method
        to convert the collected states into a probability distribution that can be
        compared with the true distribution.

        Args:
            metrics_state: Current metric state containing the replay buffer with collected states
            rng_key: JAX PRNG key for any random operations (currently unused)
            args: ProcessArgs object containing environment parameters

        Returns:
            ApproxDistributionMetricsState: Updated state with the computed empirical distribution.
                                         This state is now ready for metric computation via get().
        """
        # Here you would compute the actual metrics based on the true distribution
        # and the empirical distribution from the replay buffer.
        empirical_distribution = self.env.get_empirical_distribution(
            jax.tree.map(lambda x: x[0], metrics_state.replay_buffer.experience), args.env_params
        )
        return metrics_state.replace(empirical_distribution=empirical_distribution)

    def get(self, metrics_state: ApproxDistributionMetricsState) -> dict[str, Any]:
        """Get the computed distribution metrics from the processed state.

        Computes and returns the requested distribution metrics by comparing the
        true distribution with the empirical distribution. Each metric is computed
        using the corresponding function from the _supported_metrics dictionary.

        Args:
            metrics_state: Processed metric state containing both true and empirical distributions

        Returns:
            Dict[str, Any]: Dictionary containing the computed metrics. Keys are the metric
                          names specified during initialization, and values are the computed
                          metric values (typically float scalars).

        Example:
            If initialized with metrics=["tv", "kl", "jsd"], might return:
            {"tv": 0.123, "kl": 0.456}
        """
        # Here you would return the actual computed metrics
        results = {}
        for metric in self.metrics:
            results[metric] = self._supported_metrics[metric](
                metrics_state.true_distribution, metrics_state.empirical_distribution
            )
        return results
