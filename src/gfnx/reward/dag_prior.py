from typing import Generic, TypeVar

import chex
import jax.numpy as jnp

from ..base import BaseRewardParams, TAction, TLogReward
from ..environment import DAGEnvParams, DAGEnvState


@chex.dataclass(frozen=True)
class BaseDAGPriorParams(BaseRewardParams):
    pass


TDAGPriorParams = TypeVar("TDAGPriorParams", bound=BaseDAGPriorParams)


class BaseDAGPrior(Generic[TDAGPriorParams]):
    def init(self, rng_key: chex.PRNGKey, dummy_state: DAGEnvState) -> TDAGPriorParams:
        """Initialize the prior.

        Args:
        - rng_key: chex.PRNGKey, random key
        - dummy_state: DAGEnvState, single dummy state (no batch dim)
        """
        raise NotImplementedError

    def log_prob(self, state: DAGEnvState, prior_params: TDAGPriorParams) -> TLogReward:
        """Computes log P(G).

        Args:
        - state: DAGEnvState, single state (no batch dim)
        - prior_params: prior-specific parameters produced by ``init``

        Returns:
        - scalar log P(G)
        """
        raise NotImplementedError

    def delta_score(
        self,
        state: DAGEnvState,
        action: TAction,
        next_state: DAGEnvState,
        env_params: DAGEnvParams,
        prior_params: TDAGPriorParams,
    ) -> TLogReward:
        """Computes log P(G') - log P(G), where G' is the result of adding
        the edge X_i -> X_j to G.

        Args:
        - state: DAGEnvState, single state (no batch dim)
        - action: DAGEnvAction, scalar action
        - next_state: DAGEnvState, single next state (no batch dim)
        - env_params: DAGEnvParams, params of environment
        - prior_params: prior-specific parameters produced by ``init``

        Returns:
        - scalar log P(G') - log P(G)
        """
        return self.log_prob(next_state, prior_params) - self.log_prob(state, prior_params)

    @staticmethod
    def num_parents(state: DAGEnvState) -> chex.Array:
        return jnp.count_nonzero(state.adjacency_matrix, axis=1)


class UniformDAGPrior(BaseDAGPrior[BaseDAGPriorParams]):
    def __init__(self, num_variables: int) -> None:
        # We can assign an arbitrary constant here,
        # since we only need an unnormalized score in GFlowNets
        self._log_prior = jnp.zeros(num_variables)

    def init(self, rng_key: chex.PRNGKey, dummy_state: DAGEnvState) -> BaseDAGPriorParams:
        return BaseDAGPriorParams()

    def log_prob(self, state: DAGEnvState, prior_params: BaseDAGPriorParams) -> TLogReward:
        num_parents = self.num_parents(state)
        return jnp.sum(self._log_prior[num_parents])  # scalar

    def delta_score(
        self,
        state: DAGEnvState,
        action: TAction,
        next_state: DAGEnvState,
        env_params: DAGEnvParams,
        prior_params: BaseDAGPriorParams,
    ) -> TLogReward:
        return jnp.zeros(())  # scalar
