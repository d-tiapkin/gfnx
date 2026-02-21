from typing import Any

import chex
import jax
import jax.numpy as jnp

from ..base import TEnvironment, TEnvParams
from ..utils.rollout import (
    TPolicyFn,
    TPolicyParams,
    backward_rollout,
    backward_trajectory_log_probs,
)
from .base import BaseMetricsModule, BaseProcessArgs, EmptyInitArgs, EmptyUpdateArgs, MetricsState


@chex.dataclass
class EUBOMetricState(MetricsState):
    """State container for the Evidence Upper Bound (EUBO) metric.

    This state container stores the computed EUBO metric.

    Attributes:
        eubo: metric value.
    """

    eubo: jnp.ndarray


class EUBOMetricsModule(BaseMetricsModule):
    """Computes the Evidence Upper Bound (EUBO) for a GFlowNet model.

    This metric evaluates the GFlowNet model by estimating the EUBO.
    The EUBO is computed by sampling trajectories from the backward policy and
    evaluating the log-ratios of the forward and backward probabilities plus the log reward.

    The EUBO is defined as:
    EUBO = {
        if logZ is tractable:
            E_{traj ~ R * Pb} [log Pb(traj | traj_n) + log R(traj_n) - log Pf(traj)]
        else:
            E_{traj ~ R * Pb} [log Pb(traj | traj_n) + log R(traj_n) - log Pf(traj)] - logZ
    },
    where traj is sampled from the trained backward policy.

    Attributes:
        env: Environment instance for trajectory generation and evaluation.
        env_params: Environment parameters.
        bwd_policy_fn: Backward policy function for sampling trajectories
            starting from terminal states and computing log-ratios.
        n_rounds: Number of sampling rounds for statistical stability.
        batch_size: Batch size used when evaluating policy over states.
        rng_key: Key used for pseudo random generation.
    """

    def __init__(
        self,
        env: TEnvironment,
        env_params: TEnvParams,
        bwd_policy_fn: TPolicyFn,
        n_rounds: int,
        batch_size: int,
        rng_key: chex.PRNGKey,
    ):
        """Initializes the EUBO metric module.

        Args:
            env: Environment for trajectory generation and reward computation.
            env_params: Environment parameters.
            bwd_policy_fn: Backward policy function for generating trajectories starting
                from terminal states.
            n_rounds: The number of sampling rounds to perform for estimation.
            batch_size: The number of environments to run in parallel for sampling.
        """
        self.env = env
        rng_key, sample_key = jax.random.split(rng_key)
        self.test_set = env.get_ground_truth_sampling(sample_key, batch_size, env_params)
        if env.is_normalizing_constant_tractable:
            self.logZ = jnp.log(env.get_normalizing_constant(env_params))
        else:
            self.logZ = jnp.array(0.0)
        self.bwd_policy_fn = bwd_policy_fn
        self.n_rounds = n_rounds
        self.batch_size = batch_size

    # Ensure the module has a consistent interface
    InitArgs = EmptyInitArgs

    def init(self, rng_key: chex.PRNGKey, args: InitArgs) -> EUBOMetricState:
        """Initialize the metric state for EUBO metric."""
        return EUBOMetricState(eubo=jnp.array(jnp.inf, dtype=jnp.float32))

    UpdateArgs = EmptyUpdateArgs

    def update(
        self,
        metrics_state: EUBOMetricState,
        rng_key: chex.PRNGKey,
        args: UpdateArgs | None = None,
    ) -> EUBOMetricState:
        """
        Update metric state with new data.
        This is a no-op as the metric is computed on demand.
        """
        return metrics_state

    def get(self, metrics_state: EUBOMetricState) -> dict[str, Any]:
        """Returns the computed EUBO metric from the current state.

        Args:
            metrics_state: The current state containing the computed EUBO.

        Returns:
            A dictionary containing the EUBO value.
        """
        return {"eubo": metrics_state.eubo}

    @chex.dataclass
    class ProcessArgs(BaseProcessArgs):
        """Arguments for processing the EUBO metric module.

        Attributes:
            policy_params: Current policy parameters used for forward and backward rollouts
                to generate terminal states and compute log-ratios.
            env_params: Environment parameters required for trajectory generation
                and reward computation.
        """

        policy_params: TPolicyParams
        env_params: TEnvParams

    def process(
        self,
        metrics_state: EUBOMetricState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs,
    ) -> EUBOMetricState:
        """Computes the EUBO by sampling trajectories from the backward policy.

        This method performs multiple rounds of backward rollouts to sample
        trajectories, and then computes the EUBO for each trajectory. The final
        EUBO is the average over all sampled trajectories across all rounds.

        Args:
            rng_key: Random number generator key for sampling.
            args: Arguments for processing, containing policy and environment parameters.

        Returns:
            An updated metrics state containing the EUBO value, averaged over all
            trajectories and rounds.
        """

        def process_round(carry_rng_key, _):
            """Process a single round of sampling across all batches."""
            rng_key, rollout_key = jax.random.split(carry_rng_key)
            bwd_traj_data, _ = backward_rollout(
                rng_key=rollout_key,
                init_state=self.test_set,
                policy_fn=self.bwd_policy_fn,
                policy_params=args.policy_params,
                env=self.env,
                env_params=args.env_params,
            )
            # EUBO = E_{traj ~ R * Pb} [log Pb(traj | traj_n) + log R(traj_n) - log Pf(traj)]
            # (without normalising constant)
            log_rewards = self.env.reward_module.log_reward(self.test_set, args.env_params)
            log_pf_traj, log_pb_traj = backward_trajectory_log_probs(
                self.env, bwd_traj_data, args.env_params
            )
            eubo = log_pb_traj - log_pf_traj + log_rewards
            chex.assert_shape(eubo, (self.batch_size,))
            return rng_key, eubo

        _, eubo_per_round = jax.lax.scan(
            process_round,
            rng_key,
            None,
            length=self.n_rounds,
        )
        chex.assert_shape(eubo_per_round, (self.n_rounds, self.batch_size))

        # Average over rounds and batch. Normalise using logZ, if it is tractable.
        eubo = jnp.mean(eubo_per_round) - self.logZ
        return metrics_state.replace(eubo=eubo)
