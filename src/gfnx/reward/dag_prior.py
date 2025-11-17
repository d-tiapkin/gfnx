import chex
import jax.numpy as jnp

from ..base import TAction, TLogReward
from ..environment import DAGEnvParams, DAGEnvState


@chex.dataclass
class BaseDAGPrior:
    num_variables: int

    def log_prob(
        self, state: DAGEnvState, env_params: DAGEnvParams
    ) -> TLogReward:
        """Computes log P(G).

        Args:
        - state: DAGEnvState, shape [B, ...], batch of states
        - env_params: DAGEnvParams, params of environment,
          always includes reward params

        Returns:
        - TLogReward, shape [B], batch of log P(G)
        """
        raise NotImplementedError

    def delta_score(
        self,
        state: DAGEnvState,
        action: TAction,
        next_state: DAGEnvState,
        env_params: DAGEnvParams,
    ) -> TLogReward:
        """Computes log P(G') - log P(G), where G' is the result of adding
        the edge X_i -> X_j to G.

        Args:
        - state: DAGEnvState, shape [B, ...], batch of states
        - action: DAGEnvAction, shape [B], batch of actions
        - next_state: DAGEnvState, shape [B, ...], batch of next states
        - env_params: DAGEnvParams, params of environment,
          always includes reward params

        Returns:
        - TLogReward, shape [B], batch of log P(G') - log P(G)
        """
        return self.log_prob(next_state, env_params) - self.log_prob(
            state, env_params
        )

    @staticmethod
    def num_parents(state: DAGEnvState) -> chex.Array:
        return jnp.count_nonzero(state.adjacency_matrix, axis=1)


class UniformDAGPrior(BaseDAGPrior):
    def __init__(self, num_variables: int) -> None:
        super().__init__(num_variables=num_variables)
        # We can assign an arbitrary constant here,
        # since we only need an unnormalized score in GFlowNets
        self._log_prior = jnp.zeros(num_variables)

    def log_prob(
        self, state: DAGEnvState, env_params: DAGEnvParams
    ) -> TLogReward:
        num_parents = self.num_parents(state)
        return jnp.sum(self._log_prior[num_parents], axis=1)  # [B]

    def delta_score(
        self,
        state: DAGEnvState,
        action: TAction,
        next_state: DAGEnvState,
        env_params: DAGEnvParams,
    ) -> TLogReward:
        return jnp.zeros(state.time.shape[0])  # [B]
