from abc import abstractmethod
from typing import Any, Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from ..base import TEnvironment, TEnvParams, TEnvState
from ..utils.corr import pearson_corr, spearman_corr
from ..utils.rollout import (
    TPolicyFn,
    TPolicyParams,
    backward_rollout,
    backward_trajectory_log_probs,
    forward_rollout,
)
from .base import BaseInitArgs, BaseMetricsModule, BaseProcessArgs, EmptyUpdateArgs, MetricsState


@chex.dataclass
class CorrelationMetricsState(MetricsState):
    """State container for correlation-based metric computation.

    This state class stores the data required for computing correlation metrics
    between transformed distributions of model predictions and true rewards.
    It maintains terminal states, their corresponding transformed log-rewards,
    and computed transformed log-ratios of backward trajectories.

    Attributes:
        test_terminal_states: Terminal states used for correlation evaluation.
            These can be either sampled on-policy or from a fixed test set.
        test_log_rewards_transformed: Transformed log-rewards corresponding to the
            terminal states. Shape: (domain_size,) - obtained by marginalizing
            the distribution over terminal states to a domain of arbitrary size.
        log_ratio_traj_transformed: Transformed log-ratios of backward trajectories from
            terminal states. These ratios represent the model's estimated probability
            of reaching each terminal state, marginalized to domain_size.
            Shape: (domain_size,)
    """

    test_terminal_states: TEnvState
    test_log_rewards_transformed: jnp.ndarray
    log_ratio_traj_transformed: jnp.ndarray


class BaseCorrelationMetricsModule(BaseMetricsModule):
    """Abstract base class for correlation-based GFlowNet evaluation metrics.

    This class provides common functionality for computing correlation metrics between
    transformed model-predicted marginal probabilities and transformed log true rewards.
    It implements the core logic for backward trajectory sampling and log-ratio computation
    that is shared across different correlation metric variants.

    The correlation metrics evaluate how well the learned GFlowNet policy captures
    the true reward distribution by computing correlations between:
    - Transformed log-ratios of backward trajectories (model predictions)
    - Transformed log-rewards of terminal states (ground truth)

    Attributes:
        env: Environment instance for trajectory generation and evaluation
        bwd_policy_fn: Backward policy function for computing trajectory ratios
        n_rounds: Number of sampling rounds for statistical stability
        batch_size: Batch size for processing terminal states during evaluation
        transform_fn: Function to marginalize distributions from terminal states
            to an arbitrary domain size for correlation computation
    """

    def __init__(
        self,
        env: TEnvironment,
        bwd_policy_fn: TPolicyFn,
        n_rounds: int,
        batch_size: int = 1,
        transform_fn: Callable[[TEnvState, jnp.ndarray], jnp.ndarray] = lambda x, y: y,
    ):
        """Initialize the correlation metric module.

        Args:
            env: Environment for trajectory generation and reward computation
            bwd_policy_fn: Backward policy function for trajectory probability estimation
            n_rounds: Number of sampling rounds for averaging log-ratios
            batch_size: Batch size for processing terminal states
            transform_fn: Function to marginalize distributions from terminal states
                to an arbitrary domain size for correlation computation
        """
        self.env = env
        self.bwd_policy_fn = bwd_policy_fn
        self.n_rounds = n_rounds
        self.batch_size = batch_size
        self.transform_fn = transform_fn

    # Ensure the module has a consistent interface
    UpdateArgs = EmptyUpdateArgs

    def update(
        self,
        metrics_state: CorrelationMetricsState,
        rng_key: chex.PRNGKey,
        args: UpdateArgs | None = None,
    ) -> CorrelationMetricsState:
        """Update metric state with new data (no-op for correlation metrics)."""
        return metrics_state

    @chex.dataclass
    class ProcessArgs(BaseProcessArgs):
        """Arguments for processing the correlation metric module.

        Attributes:
            policy_params: Current policy parameters used for trajectory generation,
                backward trajectory sampling, and log-ratio computation.
            env_params: Environment parameters required for trajectory generation,
                reward computation, and rollout operations.
        """

        policy_params: TPolicyParams
        env_params: TEnvParams

    def process(
        self,
        metrics_state: CorrelationMetricsState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs,
    ) -> CorrelationMetricsState:
        rng_key, test_data_key, rollout_key = jax.random.split(rng_key, 3)
        # Compute test terminal states and already transformed log-rewards
        test_terminal_states, test_log_rewards_transformed = self._get_states_and_rewards(
            metrics_state, test_data_key, args
        )
        # Compute log-ratios using backward trajectory sampling
        log_ratio_traj = self._compute_log_ratio(
            rng_key=rollout_key,
            terminal_states=test_terminal_states,
            policy_params=args.policy_params,
            env_params=args.env_params,
        )
        log_ratio_traj_transformed = self.transform_fn(
            test_terminal_states,
            log_ratio_traj,
        )
        chex.assert_equal_shape([
            log_ratio_traj_transformed,
            test_log_rewards_transformed,
        ])
        return metrics_state.replace(
            test_terminal_states=test_terminal_states,
            test_log_rewards_transformed=test_log_rewards_transformed,
            log_ratio_traj_transformed=log_ratio_traj_transformed,
        )

    def get(self, metrics_state: CorrelationMetricsState) -> Dict[str, Any]:
        """Compute and return correlation metrics from the current state.

        Calculates two correlation measures between transformed model predictions and
        transformed true rewards:
        1. Pearson correlation in log-probability space
        2. Spearman rank correlation

        Args:
            metrics_state: Current state containing transformed log-ratios
                and transformed log-rewards

        Returns:
            Dict[str, Any]: Dictionary containing computed correlation metrics:
                - 'pearson': Pearson correlation of log-probabilities
                - 'spearman': Spearman rank correlation of log-probabilities
        """
        chex.assert_equal_shape([
            metrics_state.log_ratio_traj_transformed,
            metrics_state.test_log_rewards_transformed,
        ])
        return {
            "pearson": pearson_corr(
                metrics_state.log_ratio_traj_transformed,
                metrics_state.test_log_rewards_transformed,
            ),
            "spearman": spearman_corr(
                metrics_state.log_ratio_traj_transformed,
                metrics_state.test_log_rewards_transformed,
            ),
        }

    @abstractmethod
    def _get_states_and_rewards(
        self,
        metrics_state: CorrelationMetricsState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs,
    ) -> Tuple[TEnvState, jnp.ndarray]:
        """Get terminal states and transformed log rewards for correlation computation.

        This method should be implemented by subclasses to provide the terminal states
        and transformed log rewards for correlation metric computation.

        Args:
            metrics_state: Current metric state (largely ignored for on-policy)
            rng_key: Random number generator key for sampling
            args: ProcessArgs object containing policy parameters and environment parameters

        Returns:
            Tuple[TEnvState, jnp.ndarray]: A tuple containing:
                - Terminal states used for correlation evaluation
                - Transformed log rewards corresponding to the terminal states
        """
        raise NotImplementedError

    def _compute_log_ratio(
        self,
        rng_key: chex.PRNGKey,
        terminal_states: TEnvState,  # Shape: [T x ...]
        policy_params: TPolicyParams,
        env_params: TEnvParams,
    ) -> jnp.ndarray:
        """Compute log-ratios for terminal states using backward rollouts.

        This method performs multiple rounds of backward rollouts from given terminal
        states and computes the log-ratio of backward trajectories. The log-ratios
        are averaged in a probability space (instead of log space) using log-sum-exp.

        Args:
            rng_key: Random number generator key for sampling
            terminal_states: Terminal states to start backward rollouts from.
                Shape: [T x ...], where T is the total number of terminal states
            policy_params: Parameters for the backward policy function
            env_params: Environment parameters for rollout execution

        Returns:
            jnp.ndarray: Computed log-ratios with shape (n_terminal_states,)
                representing the average log-ratio for each terminal state
                across all sampling rounds

        Note:
            The method uses jax.lax.scan for efficient computation across batches
            and rounds. Log-ratios are computed using backward_trajectory_log_probs
            and averaged using logsumexp for numerical stability.
        """
        n_terminal_states = terminal_states.is_pad.shape[0]
        # Use additional batches to avoid OOM
        terminal_states = jax.tree.map(
            lambda x: x.reshape(-1, self.batch_size, *x.shape[1:]),
            terminal_states,
        )

        def process_batch(rng_key, terminal_states):
            """Process a single batch of terminal states."""
            rng_key, rollout_key = jax.random.split(rng_key)
            bwd_traj_data, _ = backward_rollout(
                rng_key=rollout_key,
                init_state=terminal_states,
                policy_fn=self.bwd_policy_fn,
                policy_params=policy_params,
                env=self.env,
                env_params=env_params,
            )
            log_pf_traj, log_pb_traj = backward_trajectory_log_probs(
                self.env, bwd_traj_data, env_params
            )
            log_ratio_traj = log_pf_traj - log_pb_traj
            return rng_key, log_ratio_traj

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
            length=self.n_rounds,
        )
        chex.assert_shape(log_ratio_traj, (self.n_rounds, n_terminal_states))

        # Average ratios over rounds for each test datum using log-sum-exp
        log_ratio_traj = jax.nn.logsumexp(log_ratio_traj, axis=0)
        log_ratio_traj = log_ratio_traj - jnp.log(self.n_rounds)
        return log_ratio_traj


class OnPolicyCorrelationMetricsModule(BaseCorrelationMetricsModule):
    """On-policy correlation metric module for GFlowNet evaluation.

    This metric module computes correlation metrics between transformed model-predicted
    marginal probabilities and transformed log true rewards. During evaluation,
    it generates fresh terminal states by performing forward rollouts with the current policy,
    then computes backward trajectory log-ratios.

    Attributes:
        n_terminal_states: Number of terminal states to sample for evaluation
        domain_size: Number of points to compute correlations on after marginalization
        fwd_policy_fn: Forward policy function for generating trajectories
    """

    def __init__(
        self,
        n_rounds: int,
        n_terminal_states: int,
        batch_size: int,
        fwd_policy_fn: TPolicyFn,
        bwd_policy_fn: TPolicyFn,
        env: TEnvironment,
        domain_size: int | None = None,
        transform_fn: Callable[[TEnvState, jnp.ndarray], jnp.ndarray] = lambda x, y: y,
    ):
        """Initialize the on-policy correlation metric module.

        Args:
            n_rounds: Number of sampling rounds for statistical stability
            n_terminal_states: Number of terminal states to generate and evaluate
            batch_size: Batch size for efficient processing of terminal states
            fwd_policy_fn: Forward policy function for generating trajectories
            bwd_policy_fn: Backward policy function for computing log-ratios
            env: Environment for trajectory generation and reward computation
            domain_size: Number of points to compute correlations on,
                corresponds to the output size of the transform_fn.
                Defaults to the number of terminal states and corresponds to
                the identity transform.
            transform_fn: Function to marginalize distributions from terminal states
                to an arbitrary domain size for correlation computation
        """
        super().__init__(
            env=env,
            bwd_policy_fn=bwd_policy_fn,
            n_rounds=n_rounds,
            batch_size=batch_size,
            transform_fn=transform_fn,
        )
        self.fwd_policy_fn = fwd_policy_fn
        self.n_terminal_states = n_terminal_states
        self.domain_size = domain_size or n_terminal_states
        assert n_terminal_states % batch_size == 0, (
            f"n_terminal_states ({n_terminal_states}) must be divisible"
            f" by batch_size ({batch_size})"
        )

    @chex.dataclass
    class InitArgs(BaseInitArgs):
        """Arguments for initializing the on-policy correlation metric module.

        Attributes:
            env_params: Environment parameters needed to create dummy terminal states
                during initialization and for environment operations.
        """

        env_params: TEnvParams

    def init(self, rng_key: chex.PRNGKey, args: InitArgs) -> CorrelationMetricsState:
        """Initialize the on-policy correlation metric state.

        Creates an initial state with placeholder data structures. The actual
        terminal states and rewards will be generated during the process phase
        using on-policy sampling.

        The arrays are initialized with domain_size, which represents the size of the
        transformed probability space after marginalization. This allows the correlation
        metrics to be computed on an arbitrary domain rather than just the terminal states.

        Args:
            rng_key: Random number generator key (unused in initialization)
            args: InitArgs object containing environment parameters

        Returns:
            CorrelationMetricsState: Initialized state with dummy terminal states
                and zero-initialized arrays for rewards and log-ratios with shape
                (domain_size,) representing the transformed probability space
        """
        dummy_terminal_states = self.env.reset(self.n_terminal_states, args.env_params)[1]
        return CorrelationMetricsState(
            test_terminal_states=dummy_terminal_states,
            test_log_rewards_transformed=jnp.zeros((self.domain_size)),
            log_ratio_traj_transformed=jnp.zeros((self.domain_size)),
        )

    def _get_states_and_rewards(
        self,
        metrics_state: CorrelationMetricsState,
        rng_key: chex.PRNGKey,
        args: BaseCorrelationMetricsModule.ProcessArgs,
    ) -> Tuple[TEnvState, jnp.ndarray]:
        """Generate fresh data on-policy.

        This method performs the core computation for on-policy correlation evaluation:
        1. Generates fresh terminal states using forward rollouts with current policy
        2. Extracts log-rewards for these terminal states
        3. Transforms the log-rewards using the transform function
        4. Returns the terminal states and transformed log-rewards

        Args:
            metrics_state: Current metric state (largely ignored for on-policy)
            rng_key: Random number generator key for sampling
            args: ProcessArgs object containing policy parameters and environment parameters

        Returns:
            Tuple[TEnvState, jnp.ndarray]: A tuple containing:
                - Terminal states generated via forward rollouts
                - Transformed log-rewards corresponding to the terminal states
        """
        # First, generate fresh terminal states and rewards via forward rollouts
        _, info = forward_rollout(
            rng_key=rng_key,
            num_envs=self.n_terminal_states,
            policy_fn=self.fwd_policy_fn,
            policy_params=args.policy_params,
            env=self.env,
            env_params=args.env_params,
        )
        # Second, extract the terminal states and log-rewards
        terminal_states = info["final_env_state"]
        log_rewards = info["log_gfn_reward"]
        log_rewards_transformed = self.transform_fn(terminal_states, log_rewards)
        # Third, return the terminal states and transformed log-rewards
        return terminal_states, log_rewards_transformed


class TestCorrelationMetricsModule(BaseCorrelationMetricsModule):
    """Fixed test set correlation metric module for GFlowNet evaluation.

    This metric module computes correlation metrics between transformed model predictions
    and transformed true rewards using a fixed set of test terminal states. Unlike the
    on-policy variant, this module evaluates the model's performance on the same set of
    terminal states across different training iterations, providing consistent evaluation
    points for tracking training progress.
    """

    @chex.dataclass
    class InitArgs(BaseInitArgs):
        """Arguments for initializing the test correlation metric module.

        Attributes:
            env_params: Environment parameters needed for computing log-rewards
                from the test set and for environment operations.
            test_set: Fixed set of terminal states that will be used consistently
                for correlation evaluation across training iterations.
        """

        env_params: TEnvParams
        test_set: TEnvState

    def init(self, rng_key: chex.PRNGKey, args: InitArgs) -> CorrelationMetricsState:
        """Initialize metric state with a fixed test set.

        Sets up the metric state using a predefined set of terminal states.
        The log-rewards are computed immediately from the test set, while
        log-ratios are initialized to zero and will be computed during processing.

        Args:
            rng_key: Random number generator key (unused in initialization)
            args: InitArgs object containing environment parameters and test set

        Returns:
            CorrelationMetricsState: Initialized state with fixed test terminal
                states, their computed transformed log-rewards, and zero-initialized
                log-ratios with shape matching the transformed log-rewards
        """
        test_log_rewards = self.env.reward_module.log_reward(args.test_set, args.env_params)
        test_log_rewards_transformed = self.transform_fn(args.test_set, test_log_rewards)
        return CorrelationMetricsState(
            test_terminal_states=args.test_set,
            test_log_rewards_transformed=test_log_rewards_transformed,
            log_ratio_traj_transformed=jnp.zeros_like(test_log_rewards_transformed),
        )

    def _get_states_and_rewards(
        self,
        metrics_state: CorrelationMetricsState,
        rng_key: chex.PRNGKey,
        args: BaseCorrelationMetricsModule.ProcessArgs,
    ) -> Tuple[TEnvState, jnp.ndarray]:
        """Return existing test data from the metrics state.

        This method simply returns the terminal states and transformed log-rewards
        that were computed during initialization. No new data generation is needed
        since this module uses a fixed test set.

        Args:
            metrics_state: Current metric state containing the fixed test set data
            rng_key: Random number generator key (unused for fixed test set)
            args: ProcessArgs object (unused for fixed test set)

        Returns:
            Tuple[TEnvState, jnp.ndarray]: A tuple containing:
                - Terminal states from the fixed test set
                - Transformed log-rewards corresponding to the terminal states
        """
        return metrics_state.test_terminal_states, metrics_state.test_log_rewards_transformed
