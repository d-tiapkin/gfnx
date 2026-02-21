from typing import Any

import chex
import jax
import jax.numpy as jnp

from ..base import TEnvironment, TEnvParams
from ..utils.rollout import (
    TPolicyFn,
    TPolicyParams,
    forward_rollout,
    forward_trajectory_log_probs,
)
from .base import BaseMetricsModule, BaseProcessArgs, EmptyInitArgs, EmptyUpdateArgs, MetricsState


@chex.dataclass
class ELBOMetricState(MetricsState):
    """State container for the Evidence Lower Bound (ELBO) metric.

    This state container stores the computed ELBO metric.

    Attributes:
        elbo: metric value.
    """

    elbo: jnp.ndarray


class ELBOMetricsModule(BaseMetricsModule):
    """Computes the Evidence Lower Bound (ELBO) for a GFlowNet model.

    This metric evaluates the GFlowNet model by estimating the ELBO.
    The ELBO is computed by sampling trajectories from the forward policy and evaluating
    the log-ratios of the forward and backward probabilities plus the log reward.

    The ELBO is defined as:
    ELBO = {
        if logZ is tractable:
            E_{traj ~ Pf} [log Pb(traj | traj_n) + log R(traj_n) - log Pf(traj)] - logZ
        else:
            E_{traj ~ Pf} [log Pb(traj | traj_n) + log R(traj_n) - log Pf(traj)]
    },
    where traj is sampled from the trained forward policy.

    Attributes:
        env: Environment instance for trajectory generation and evaluation.
        env_params: Environment parameters.
        fwd_policy_fn: Forward policy function producing action logits.
        n_rounds: Number of sampling rounds for statistical stability.
        batch_size: Batch size used when evaluating policy over states.
    """

    def __init__(
        self,
        env: TEnvironment,
        env_params: TEnvParams,
        fwd_policy_fn: TPolicyFn,
        n_rounds: int,
        batch_size: int,
    ):
        """Initializes the ELBO metric module.

        Args:
            env: Environment for trajectory generation and reward computation.
            env_params: Environment parameters used for trajectory generation.
            fwd_policy_fn: Forward policy function for generating trajectories.
            n_rounds: The number of sampling rounds to perform for estimation.
            batch_size: The number of environments to run in parallel for sampling.
        """
        self.env = env
        if env.is_normalizing_constant_tractable:
            self.logZ = jnp.log(env.get_normalizing_constant(env_params))
        else:
            self.logZ = jnp.array(0.0)
        self.fwd_policy_fn = fwd_policy_fn
        self.n_rounds = n_rounds
        self.batch_size = batch_size

    # Ensure the module has a consistent interface
    InitArgs = EmptyInitArgs

    def init(self, rng_key: chex.PRNGKey, args: InitArgs) -> ELBOMetricState:
        """Initialize the metric state for ELBO metric."""
        return ELBOMetricState(elbo=jnp.array(-jnp.inf, dtype=jnp.float32))

    UpdateArgs = EmptyUpdateArgs

    def update(
        self,
        metrics_state: ELBOMetricState,
        rng_key: chex.PRNGKey,
        args: UpdateArgs | None = None,
    ) -> ELBOMetricState:
        """
        Update metric state with new data.
        This is a no-op as the metric is computed on demand.
        """
        return metrics_state

    def get(self, metrics_state: ELBOMetricState) -> dict[str, Any]:
        """Returns the computed ELBO metric from the current state.

        Args:
            metrics_state: The current state containing the computed ELBO.

        Returns:
            A dictionary containing the ELBO value.
        """
        return {"elbo": metrics_state.elbo}

    @chex.dataclass
    class ProcessArgs(BaseProcessArgs):
        """Arguments for processing the ELBO metric module.

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
        metrics_state: ELBOMetricState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs,
    ) -> ELBOMetricState:
        """Computes the ELBO by sampling trajectories from the forward policy.

        This method performs multiple rounds of forward rollouts to sample
        trajectories, and then computes the ELBO for each trajectory. The final
        ELBO is the average over all sampled trajectories across all rounds.

        Args:
            rng_key: Random number generator key for sampling.
            args: Arguments for processing, containing policy and environment parameters.

        Returns:
            An updated metrics state containing the ELBO value, averaged over all
            trajectories and rounds.
        """

        def process_round(carry_rng_key, _):
            """Process a single round of sampling across all batches."""
            rng_key, rollout_key = jax.random.split(carry_rng_key)
            fwd_traj_data, aux_info = forward_rollout(
                rng_key=rollout_key,
                num_envs=self.batch_size,
                policy_fn=self.fwd_policy_fn,
                policy_params=args.policy_params,
                env=self.env,
                env_params=args.env_params,
            )
            # ELBO = E_{traj ~ Pf} [log Pb(traj | traj_n) + log R(traj_n) - log Pf(traj)]
            # (without normalising constant)
            log_pf_traj, log_pb_traj = forward_trajectory_log_probs(
                self.env, fwd_traj_data, args.env_params
            )
            elbo = log_pb_traj - log_pf_traj + aux_info["log_gfn_reward"]
            chex.assert_shape(elbo, (self.batch_size,))
            return rng_key, elbo

        _, elbo_per_round = jax.lax.scan(
            process_round,
            rng_key,
            None,
            length=self.n_rounds,
        )
        chex.assert_shape(elbo_per_round, (self.n_rounds, self.batch_size))

        # Average over rounds and batch. Normalise using logZ, if it is tractable.
        elbo = jnp.mean(elbo_per_round) - self.logZ
        return metrics_state.replace(elbo=elbo)
