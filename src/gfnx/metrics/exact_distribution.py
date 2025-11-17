from typing import Any, Dict

import chex
import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp

from gfnx.utils.distances import kl_divergence, total_variation_distance
from gfnx.utils.rollout import TPolicyFn

from ..base import TEnvironment, TEnvParams, TEnvState
from ..utils.rollout import TPolicyParams
from .base import (
    BaseInitArgs,
    BaseMetricsModule,
    BaseProcessArgs,
    EmptyUpdateArgs,
    MetricsState,
)


@chex.dataclass
class ExactDistributionMetricsState(MetricsState):
    """State for exact distribution metrics.

    Holds the ground-truth terminal distribution from the environment and the
    exact terminal distribution computed under a given forward policy.

    Attributes:
        true_distribution: Ground-truth terminal distribution provided by the environment
        exact_distribution: Exact terminal distribution computed by ``process``
    """

    true_distribution: chex.Array
    exact_distribution: chex.Array


def marginal_distribution(true_dist: chex.Array, exact_dist: chex.Array) -> chex.Array:
    """Compute the marginal distribution from the empirical distribution.

    Computes the marginal distribution by summing over all dimensions except the first two,
    then normalizes the result to ensure it represents a valid probability distribution.
    This is useful for visualizing the current convergence of the empirical distribution.

    Args:
        true_dist: True distribution of the environment (unused in current implementation)
        exact_dist: Exactly calculated distribution produced by the forward policy.

    Returns:
        chex.Array: Normalized marginal distribution after summing over extra dimensions

    Note:
        This function is used as a metric computation function and should follow the
        same interface as other metric functions (tv, kl, jsd) that take true and exact
        distributions as inputs and return a computed value.
    """
    return exact_dist


class ExactDistributionMetricsModule(BaseMetricsModule):
    """Exact distribution metrics for enumerable environments.

    For enumerable environments, this module computes the exact terminal distribution
    induced by simple policy evaluation method.

    Supported metrics:
        - "tv": Total variation distance between distributions
        - "kl": KL divergence between true and exact terminal distributions
        - "2d_marginal_distribution": Marginal distribution computation

    Attributes:
        metrics: List of metric names to compute
        env: The enumerable environment for which to compute metrics
        fwd_policy_fn: Forward policy function producing action logits
        batch_size: Batch size used when evaluating policy over states
        tol_epsilon: Tolerance for convergence in distribution computation
        _supported_metrics: Dictionary mapping metric names to computation functions
    """

    _supported_metrics = {
        "tv": total_variation_distance,
        "kl": kl_divergence,
        "2d_marginal_distribution": marginal_distribution,
    }

    def __init__(
        self,
        metrics: list[str],
        env: TEnvironment,
        fwd_policy_fn: TPolicyFn,
        batch_size: int,
        tol_epsilon: float = 1e-7,
    ):
        """Initialize the exact-distribution metrics module.

        Validates inputs and records configuration. ``batch_size`` controls how
        states are grouped when evaluating the policy over the topological order
        for memory efficiency; it is unrelated to any replay buffer.

        Args:
            metrics: List of metric names to compute. Must be a subset of supported metrics.
            env: Enumerable environment (``env.is_enumerable`` must be True).
            fwd_policy_fn: Function mapping observations to action logits.
            batch_size: Mini-batch size for policy evaluation across states.

        Raises:
            ValueError: If the environment is not enumerable or not topologically sortable,
                or if ``metrics`` is invalid or contains unsupported entries.
        """
        if not env.is_enumerable:
            raise ValueError(f"Environment {env.name} is not enumerable")
        if not isinstance(metrics, list) or not all(isinstance(m, str) for m in metrics):
            raise ValueError("metrics must be a list of strings")
        if not all(m in self._supported_metrics.keys() for m in metrics):
            raise ValueError(
                f"Unsupported metrics. Supported metrics are: \
                    {self._supported_metrics}"
            )

        self.metrics = metrics
        self.env = env
        self.fwd_policy_fn = fwd_policy_fn
        self.batch_size = batch_size
        self.tol_epsilon = tol_epsilon

    @chex.dataclass
    class InitArgs(BaseInitArgs):
        """Arguments for initializing the ExactDistributionMetricsModule.

        Attributes:
            env_params: Environment parameters used to obtain the true distribution
                and query environment functions.
        """

        env_params: TEnvParams

    def init(self, rng_key: chex.PRNGKey, args: InitArgs) -> ExactDistributionMetricsState:
        """Initialize the metric state.

        Obtains the ground-truth terminal distribution from the environment and
        initializes the ``exact_distribution`` with a uniform placeholder of the
        same shape. The exact distribution will be computed in ``process``.

        Args:
            rng_key: JAX PRNG key (currently unused).
            args: Initialization arguments containing environment parameters.

        Returns:
            ExactDistributionMetricsState: Initialized state containing:
                - true_distribution: Ground truth distribution from the environment
                - exact_distribution: Initial uniform distribution
        """
        true_distribution = self.env.get_true_distribution(args.env_params)
        exact_distribution = jnp.ones_like(true_distribution) / true_distribution.size
        return ExactDistributionMetricsState(
            true_distribution=true_distribution,
            exact_distribution=exact_distribution,
        )

    UpdateArgs = EmptyUpdateArgs

    def update(
        self, metrics_state: ExactDistributionMetricsState, rng_key: chex.PRNGKey, args: UpdateArgs
    ) -> ExactDistributionMetricsState:
        return metrics_state

    @chex.dataclass
    class ProcessArgs(BaseProcessArgs):
        """Arguments for processing the ExactDistributionMetricsModule.

        Attributes:
            policy_params: Parameters for the forward policy used during propagation.
            env_params: Environment parameters.
        """

        policy_params: TPolicyParams
        env_params: TEnvParams

    def process(
        self,
        metrics_state: ExactDistributionMetricsState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs,
    ) -> ExactDistributionMetricsState:
        """
        Compute the exact terminal distribution induced by the forward policy.
        Uses a simple power iteration method to propagate the initial state distribution:
            - Initialize the state distribution with all mass on the initial state.
            - Repeatedly apply the transition matrix induced by the forward policy
              until convergence (L1 norm change below ``tol_epsilon``).
            - Extract the terminal distribution from the converged state distribution.

        Overall, we have
            `final_distribution = sum_{t=0}^inf (transition_matrix^t) @ initial_distribution`.

        Args:
            metrics_state: Current metrics state containing the true distribution.
            rng_key: JAX PRNG key for any stochasticity (currently unused).
            args: Processing arguments containing policy and environment parameters.

        Returns:
            ExactDistributionMetricsState: Updated metrics state with the computed
            exact distribution.
        """
        transition = self._preprare_transition_matrix(
            rng_key,
            args.env_params,
            args.policy_params,
        )
        num_states = transition.shape[0] // 2

        initial_vector = jnp.zeros((2 * num_states,))
        initial_state = jax.tree_map(lambda x: x[0], self.env.get_init_state(1))
        initial_idx = self.env.state_to_index(initial_state, args.env_params)
        initial_vector = initial_vector.at[initial_idx].set(1.0)

        def cond_function(carry: tuple):
            last_vec, _, _ = carry
            return jnp.linalg.norm(last_vec, ord=1) > self.tol_epsilon

        def one_step(carry: tuple):
            last_vec, total, transition = carry
            last_vec = transition @ last_vec
            return last_vec, total + last_vec, transition

        _, result, _ = jax.lax.while_loop(
            cond_function, one_step, (initial_vector, initial_vector, transition)
        )

        exact_distribution = result[num_states:].reshape(*metrics_state.true_distribution.shape)
        chex.assert_shape(exact_distribution, metrics_state.true_distribution.shape)
        return metrics_state.replace(exact_distribution=exact_distribution)

    def _preprare_transition_matrix(
        self,
        rng_key: chex.PRNGKey,
        env_params: TEnvParams,
        policy_params: TPolicyParams,
    ) -> chex.Array:
        """
        Returns a transposed sparse transition matrix of size (2 * num_states, 2 * num_states)
        in the BCSR format.
        """
        all_states = self.env.get_all_states(env_params)
        num_states = jax.tree_util.tree_leaves(all_states)[0].shape[0]

        remainder = num_states % self.batch_size
        if remainder != 0:
            pad_width = self.batch_size - remainder
            padded_sorted_states = jax.tree_map(
                lambda x: jnp.pad(
                    x, ((0, pad_width),) + ((0, 0),) * (x.ndim - 1), mode="constant"
                ),
                all_states,
            )
        else:
            padded_sorted_states = all_states

        num_batches = (
            jax.tree_util.tree_leaves(padded_sorted_states)[0].shape[0] // self.batch_size
        )
        batched_sorted_states = jax.tree_util.tree_map(
            lambda x: x.reshape((num_batches, self.batch_size, *x.shape[1:])),
            padded_sorted_states,
        )

        def scan_body(carry, states_batch: TEnvState):
            rng_key = carry
            rng_key, subkey = jax.random.split(rng_key)

            obs_batch = self.env.get_obs(states_batch, env_params)
            invalid_mask_batch = self.env.get_invalid_mask(states_batch, env_params)

            fwd_policy_logits, _ = self.fwd_policy_fn(subkey, obs_batch, policy_params)
            fwd_policy_probs = jax.nn.softmax(
                fwd_policy_logits, axis=-1, where=jnp.logical_not(invalid_mask_batch)
            )
            return rng_key, fwd_policy_probs

        _, batched_fwd_policy_probs = jax.lax.scan(
            scan_body,
            rng_key,
            batched_sorted_states,
        )

        fwd_policy_probs = batched_fwd_policy_probs.reshape(-1, batched_fwd_policy_probs.shape[-1])
        fwd_policy_probs = fwd_policy_probs[:num_states]
        chex.assert_shape(fwd_policy_probs, (num_states, self.env.action_space.n))

        state_idx = jax.vmap(self.env.state_to_index, in_axes=(0, None))(
            all_states, env_params
        )  # [num_states]
        actions = jnp.arange(self.env.action_space.n)  # [num_actions]

        next_state, is_terminal, _ = jax.vmap(
            jax.vmap(self.env._single_transition, in_axes=(None, 0, None)), in_axes=(0, None, None)
        )(all_states, actions, env_params)
        next_state_idx = jax.vmap(
            jax.vmap(self.env.state_to_index, in_axes=(0, None)), in_axes=(0, None)
        )(next_state, env_params)
        next_state_idx = next_state_idx + is_terminal * num_states

        rows = jnp.repeat(state_idx[:, None], self.env.action_space.n, axis=1).reshape(-1)
        cols = next_state_idx.reshape(-1)
        data = fwd_policy_probs.reshape(-1)

        transition_matrix = jsp.BCOO(
            (data, jnp.stack([rows, cols], axis=-1)), shape=(2 * num_states, 2 * num_states)
        )
        return jsp.BCSR.from_bcoo(transition_matrix.T)

    def get(self, metrics_state: ExactDistributionMetricsState) -> Dict[str, Any]:
        return {
            metric: self._supported_metrics[metric](
                metrics_state.true_distribution, metrics_state.exact_distribution
            )
            for metric in self.metrics
        }
