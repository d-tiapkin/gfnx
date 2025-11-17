import chex
import jax.numpy as jnp

from ..base import BaseRewardModule, TAction, TLogReward, TReward
from ..environment import DAGEnvParams, DAGEnvState
from .dag_likelihood import BaseDAGLikelihood
from .dag_prior import BaseDAGPrior


@chex.dataclass
class DAGRewardModule(BaseRewardModule[DAGEnvState, DAGEnvParams]):
    prior: BaseDAGPrior
    likelihood: BaseDAGLikelihood

    def init(self, rng_key: chex.PRNGKey, dummy_state: DAGEnvState) -> None:
        return None

    def reward(self, state: DAGEnvState, env_params: DAGEnvParams) -> TReward:
        return jnp.exp(self.log_reward(state, env_params))

    def log_reward(
        self, state: DAGEnvState, env_params: DAGEnvParams
    ) -> TLogReward:
        return self.likelihood.log_prob(
            state, env_params
        ) + self.prior.log_prob(state, env_params)

    def delta_score(
        self,
        state: DAGEnvState,
        action: TAction,
        next_state: DAGEnvState,
        env_params: DAGEnvParams,
    ) -> TLogReward:
        return self.prior.delta_score(
            state, action, next_state, env_params
        ) + self.likelihood.delta_score(state, action, next_state, env_params)
