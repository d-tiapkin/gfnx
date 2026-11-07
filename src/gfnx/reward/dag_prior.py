"""Priors over DAG structures for Bayesian structure learning.

Combined with a likelihood by [`DAGRewardModule`][gfnx.reward.dag.DAGRewardModule].
"""

from typing import Generic, TypeVar

import chex
import jax.numpy as jnp

from ..base import BaseRewardParams, TAction, TLogReward
from ..environment import DAGEnvParams, DAGEnvState


@chex.dataclass(frozen=True)
class BaseDAGPriorParams(BaseRewardParams):
    """Base (and empty) parameters of a graph prior."""


TDAGPriorParams = TypeVar("TDAGPriorParams", bound=BaseDAGPriorParams)


class BaseDAGPrior(Generic[TDAGPriorParams]):
    r"""Base class for priors $P(G)$ over DAGs.

    Subclasses implement `init` and `log_prob`. `delta_score` defaults to the difference
    of two `log_prob` calls; modular priors should override it with an $O(1)$ local
    update.
    """

    def init(self, rng_key: chex.PRNGKey, dummy_state: DAGEnvState) -> TDAGPriorParams:
        """Initialises the prior.

        Args:
            rng_key: Random key.
            dummy_state: Single DAG state (no batch dimension).

        Returns:
            Prior parameters.
        """
        raise NotImplementedError

    def log_prob(self, state: DAGEnvState, prior_params: TDAGPriorParams) -> TLogReward:
        r"""Computes $\log P(G)$.

        Args:
            state: Single DAG state (no batch dimension).
            prior_params: Prior parameters produced by `init`.

        Returns:
            Scalar $\log P(G)$.
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
        r"""Computes $\log P(G') - \log P(G)$, where $G'$ adds the edge $X_i \to X_j$ to $G$.

        Args:
            state: Graph $G$ (single state, no batch dimension).
            action: Scalar edge index `i * d + j`.
            next_state: Graph $G'$.
            env_params: Environment parameters.
            prior_params: Prior parameters produced by `init`.

        Returns:
            Scalar prior delta-score.
        """
        return self.log_prob(next_state, prior_params) - self.log_prob(state, prior_params)

    @staticmethod
    def num_parents(state: DAGEnvState) -> chex.Array:
        """Returns the in-degree of every node, shape `[d]`."""
        return jnp.count_nonzero(state.adjacency_matrix, axis=0)


class UniformDAGPrior(BaseDAGPrior[BaseDAGPriorParams]):
    r"""Uniform prior over DAGs: $\log P(G) = 0$ and every delta-score is zero.

    Args:
        num_variables: Number of variables $d$.
    """

    def __init__(self, num_variables: int) -> None:
        # We can assign an arbitrary constant here,
        # since we only need an unnormalized score in GFlowNets
        self._log_prior = jnp.zeros(num_variables)

    def init(self, rng_key: chex.PRNGKey, dummy_state: DAGEnvState) -> BaseDAGPriorParams:
        """Returns empty prior parameters."""
        return BaseDAGPriorParams()

    def log_prob(self, state: DAGEnvState, prior_params: BaseDAGPriorParams) -> TLogReward:
        """Returns the (constant) log-prior, `0`."""
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
        """Returns `0`: the uniform prior does not change when an edge is added."""
        return jnp.zeros(())  # scalar
