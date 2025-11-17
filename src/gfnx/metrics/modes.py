from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp


from ..base import TEnvironment, TEnvState
from .base import BaseMetricsModule, MetricsState, BaseInitArgs, BaseUpdateArgs, BaseProcessArgs


@chex.dataclass
class AccumulatedModesMetricsState(MetricsState):
    """State for accumulating mode discovery metrics.

    This state container tracks which modes have been visited during training
    or evaluation by maintaining a record of known modes and a boolean array
    indicating which modes have been encountered.

    Attributes:
        modes: Reference set of mode states to track. These represent the
            target modes that the policy should discover.
        visited_modes_idx: Boolean array indicating which modes have been
            visited at least once. Shape: (num_modes,)
    """

    modes: TEnvState
    visited_modes_idx: chex.Array


class AccumulatedModesMetricsModule(BaseMetricsModule):
    """Metric module for tracking mode discovery in GFlowNet training.

    This module monitors how well a GFlowNet policy discovers different modes
    by tracking which known modes are visited during training or evaluation.
    It maintains a set of reference modes and records which ones have been
    encountered based on a distance threshold.

    Attributes:
        env: Environment instance for mode-related operations
        distance_fn: Function to compute distance between two states
        distance_threshold: Maximum distance for considering a state as visiting a mode
    """

    def __init__(
        self,
        env: TEnvironment,
        distance_fn: Callable[[TEnvState, TEnvState], float],
        distance_threshold: float = 0.1,
    ):
        """Initialize the accumulated modes metric module.

        Args:
            env: Environment instance for mode-related operations
            distance_fn: Function that computes distance between two environment states.
                Must return a scalar distance value.
            distance_threshold: Maximum distance threshold for considering a visited
                state as discovering a mode. States within this distance of a mode
                are considered to have "visited" that mode.
        """
        self.env = env
        self.distance_fn = distance_fn
        self.distance_threshold = distance_threshold

    def _get_distance_matrix(self, lhs_states: TEnvState, rhs_states: TEnvState) -> jnp.ndarray:
        """Compute distance matrix between two sets of states.

        Computes pairwise distances between all states in lhs_states and all states
        in rhs_states using the configured distance function. This is used to determine
        which visited states are close enough to known modes.

        Args:
            lhs_states: First set of states (typically visited states), N states
            rhs_states: Second set of states (typically reference modes), M states

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

    @chex.dataclass
    class InitArgs(BaseInitArgs):
        """Arguments for initializing the AccumulatedModesMetricsModule.

        Attributes:
            modes: Reference set of mode states to track during training/evaluation.
                These represent the target modes that the policy should discover.
        """

        modes: TEnvState

    def init(self, rng_key: chex.PRNGKey, args: InitArgs) -> AccumulatedModesMetricsState:
        """Initialize the accumulated modes metric state.

        Creates initial state with the provided reference modes and initializes
        the visited modes tracking array to all False (no modes visited yet).

        Args:
            rng_key: JAX PRNG key for any random initialization (currently unused)
            args: InitArgs object containing the reference modes to track

        Returns:
            AccumulatedModesMetricsState: Initialized state with reference modes
                and empty visited modes tracking array
        """
        return AccumulatedModesMetricsState(
            modes=args.modes,
            visited_modes_idx=jnp.zeros((args.modes.is_pad.shape[0],), dtype=jnp.bool),
        )

    @chex.dataclass
    class UpdateArgs(BaseUpdateArgs):
        """Arguments for updating the AccumulatedModesMetricsModule.

        Attributes:
            states: Environment states visited during training/evaluation to check
                against the reference modes for mode discovery tracking.
        """

        states: TEnvState

    def update(
        self, metrics_state: AccumulatedModesMetricsState, rng_key: chex.PRNGKey, args: UpdateArgs
    ) -> AccumulatedModesMetricsState:
        """Update the metric state with newly visited states.

        Checks which reference modes have been visited by computing distances between
        the provided states and all reference modes. A mode is considered "visited"
        if any of the provided states is within the distance threshold of that mode.

        Args:
            metrics_state: Current metric state containing modes and visited tracking
            rng_key: JAX PRNG key for any random operations (currently unused)
            args: UpdateArgs object containing the newly visited states

        Returns:
            AccumulatedModesMetricsState: Updated state with updated visited modes tracking.
                The visited_modes_idx array is updated to mark newly discovered modes.
        """
        # Compute distance matrix between current states and modes
        num_modes = metrics_state.visited_modes_idx.shape[0]
        d_matrix = self._get_distance_matrix(args.states, metrics_state.modes)
        mode_passed = jnp.any(d_matrix < self.distance_threshold, axis=0)
        chex.assert_shape(mode_passed, (num_modes,))
        visited_modes_idx = jnp.logical_or(metrics_state.visited_modes_idx, mode_passed)
        return metrics_state.replace(visited_modes_idx=visited_modes_idx)

    @chex.dataclass
    class ProcessArgs(BaseProcessArgs):
        """Arguments for processing the AccumulatedModesMetricsModule.

        Attributes:
            env_params: Environment parameters (currently unused but maintained
                for interface consistency with other metric modules).
        """

        env_params: Any

    def process(
        self,
        metrics_state: AccumulatedModesMetricsState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs | None = None,
    ) -> AccumulatedModesMetricsState:
        """Process the metric state for final computation (no-op for modes metrics).

        This method performs any final processing needed before metric computation.
        For accumulated modes metrics, no additional processing is required as the
        state is maintained incrementally during updates.

        Args:
            metrics_state: Current metric state with accumulated mode visit information
            rng_key: JAX PRNG key for any random operations (currently unused)
            args: ProcessArgs object containing environment parameters (currently unused)

        Returns:
            AccumulatedModesMetricsState: Unchanged metric state ready for get() call
        """
        return metrics_state

    def get(self, metrics_state: AccumulatedModesMetricsState) -> dict:
        """Get the computed mode discovery metrics from the current state.

        Computes and returns metrics that quantify how well the policy has discovered
        the reference modes. Provides both absolute and relative measures of mode coverage.

        Args:
            metrics_state: Current metric state containing mode visit tracking information

        Returns:
            Dict[str, Any]: Dictionary containing computed mode discovery metrics:
                - 'num_modes': Total number of modes discovered (integer count)
                - 'percent_modes': Fraction of modes discovered (float between 0 and 1)
        """
        return {
            "num_modes": metrics_state.visited_modes_idx.sum(),
            "percent_modes": metrics_state.visited_modes_idx.mean(),
        }
