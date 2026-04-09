import chex
import jax
import jax.numpy as jnp

from ..base import BaseRewardModule, TAction, TLogReward, TReward, TRewardParams
from ..environment import DAGEnvParams, DAGEnvState
from .dag_likelihood import BaseDAGLikelihood
from .dag_prior import BaseDAGPrior


@chex.dataclass(frozen=True)
class DAGRewardParams:
    prior_params: TRewardParams
    likelihood_params: TRewardParams


@chex.dataclass
class DAGRewardModule(BaseRewardModule[DAGEnvState, DAGEnvParams]):
    """
    Reward module for directed acyclic graph (DAG) structures.
    The reward is defined as the product of a prior over DAGs and a likelihood
    of data given the DAG.
    """

    prior: BaseDAGPrior
    likelihood: BaseDAGLikelihood

    def init(self, rng_key: chex.PRNGKey, dummy_state: DAGEnvState) -> DAGRewardParams:
        _, prior_key, likelihood_key = jax.random.split(rng_key, 3)
        return DAGRewardParams(
            prior_params=self.prior.init(prior_key, dummy_state),
            likelihood_params=self.likelihood.init(likelihood_key, dummy_state),
        )

    def reward(self, state: DAGEnvState, reward_params: DAGRewardParams) -> TReward:
        return jnp.exp(self.log_reward(state, reward_params))

    def log_reward(self, state: DAGEnvState, reward_params: DAGRewardParams) -> TLogReward:
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
        return self.prior.delta_score(
            state, action, next_state, env_params, reward_params.prior_params
        ) + self.likelihood.delta_score(
            state, action, next_state, env_params, reward_params.likelihood_params
        )
