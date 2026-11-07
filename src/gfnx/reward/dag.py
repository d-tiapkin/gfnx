"""Posterior reward for DAG structure learning (prior x marginal likelihood).

See [`DAGEnvironment`][gfnx.environment.dag.DAGEnvironment].
"""

import chex
import jax
import jax.numpy as jnp

from ..base import BaseRewardModule, BaseRewardParams, TAction, TLogReward, TReward
from ..environment import DAGEnvParams, DAGEnvState
from .dag_likelihood import BaseDAGLikelihood, BaseDAGLikelihoodParams
from .dag_prior import BaseDAGPrior, BaseDAGPriorParams


@chex.dataclass(frozen=True)
class DAGRewardParams(BaseRewardParams):
    """Parameters of the DAG reward.

    Attributes:
        prior_params: Parameters of the graph prior.
        likelihood_params: Parameters of the marginal likelihood (e.g. the data).
    """

    prior_params: BaseDAGPriorParams
    likelihood_params: BaseDAGLikelihoodParams


@chex.dataclass
class DAGRewardModule(BaseRewardModule[DAGEnvState, DAGEnvParams, DAGRewardParams]):
    r"""Unnormalised posterior over DAGs: likelihood times prior.

    $$
    \log R(G) = \log P(\mathcal{D} \mid G) + \log P(G),
    $$

    so that $R(G)/Z = P(G \mid \mathcal{D})$. Both terms are *modular*:

    $$
    \log R(G) = \sum_{j=1}^{d} \mathrm{LocalScore}\big(X_j \mid \mathrm{Pa}_G(X_j)\big),
    $$

    so adding $X_i \to X_j$ only changes the local score of $X_j$.
    [`delta_score`][gfnx.reward.dag.DAGRewardModule.delta_score] returns that change,
    which Modified Detailed Balance (`baselines/mdb_dag.py`) uses instead of re-scoring
    the whole graph. Construct it as `DAGRewardModule(prior=..., likelihood=...)`.

    Attributes:
        prior: Graph prior, e.g.
            [`UniformDAGPrior`][gfnx.reward.dag_prior.UniformDAGPrior].
        likelihood: Marginal likelihood, e.g.
            [`LinearGaussianScore`][gfnx.reward.dag_likelihood.LinearGaussianScore] or
            [`BGeScore`][gfnx.reward.dag_likelihood.BGeScore].

    References:
        Deleu, T., et al. (2022). Bayesian structure learning with generative flow
        networks. UAI.
    """

    prior: BaseDAGPrior
    likelihood: BaseDAGLikelihood

    def init(self, rng_key: chex.PRNGKey, dummy_state: DAGEnvState) -> DAGRewardParams:
        """Initialises the prior and likelihood with independent keys.

        Args:
            rng_key: Random key.
            dummy_state: Single DAG state, typically `env.reset()`.

        Returns:
            `DAGRewardParams` bundling both parameter sets.
        """
        _, prior_key, likelihood_key = jax.random.split(rng_key, 3)
        return DAGRewardParams(
            prior_params=self.prior.init(prior_key, dummy_state),
            likelihood_params=self.likelihood.init(likelihood_key, dummy_state),
        )

    def reward(self, state: DAGEnvState, reward_params: DAGRewardParams) -> TReward:
        """Computes $R(G)$. Beware of float32 underflow; prefer `log_reward`."""
        return jnp.exp(self.log_reward(state, reward_params))

    def log_reward(self, state: DAGEnvState, reward_params: DAGRewardParams) -> TLogReward:
        r"""Computes $\log P(\mathcal{D} \mid G) + \log P(G)$ for a single graph."""
        return self.likelihood.log_prob(
            state, reward_params.likelihood_params
        ) + self.prior.log_prob(state, reward_params.prior_params)

    def delta_score(
        self,
        state: DAGEnvState,
        action: TAction,
        next_state: DAGEnvState,
        env_params: DAGEnvParams,
        reward_params: DAGRewardParams,
    ) -> TLogReward:
        r"""Computes $\log R(G') - \log R(G)$ for one edge addition $G \to G'$.

        Args:
            state: Graph $G$ before the step.
            action: Edge index `i * d + j` that was added.
            next_state: Graph $G'$ after the step.
            env_params: Environment parameters (supply `num_variables`).
            reward_params: Reward parameters.

        Returns:
            Scalar sum of the prior and likelihood delta-scores.
        """
        return self.prior.delta_score(
            state, action, next_state, env_params, reward_params.prior_params
        ) + self.likelihood.delta_score(
            state, action, next_state, env_params, reward_params.likelihood_params
        )
