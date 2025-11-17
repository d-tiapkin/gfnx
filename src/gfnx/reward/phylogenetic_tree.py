import chex
import jax
import jax.numpy as jnp

from gfnx import TLogReward, TReward

from ..base import BaseRewardModule
from ..environment import PhylogeneticTreeEnvParams, PhylogeneticTreeEnvState


class ExponentialRewardParams:
    """Parameters for exponential reward function"""

    scale: float  # Scale parameter for exponential reward
    C: float  # Offset parameter
    offset: float  # Precomputed offset (C / scale) / num_nodes


class PhylogeneticTreeRewardModule(
    BaseRewardModule[PhylogeneticTreeEnvState, PhylogeneticTreeEnvParams]
):
    """
    Reward module for phylogenetic trees using exponential reward function.
    R(x) = exp((C - total_mutations) / scale)
    """

    def __init__(self, num_nodes: int, scale: float = 1.0, C: float = 0.0):
        self.num_nodes = num_nodes
        self.scale = scale
        self.C = C
        self._offset = (C / scale) / num_nodes

    def init(
        self, rng_key: chex.PRNGKey, dummy_state: PhylogeneticTreeEnvState
    ) -> ExponentialRewardParams:
        """Initialize reward parameters"""
        return ExponentialRewardParams(
            scale=self.scale, C=self.C, offset=self._offset
        )

    def get_mutations(
        self,
        state: PhylogeneticTreeEnvState,
        env_params: PhylogeneticTreeEnvParams,
    ) -> chex.Array:
        """Compute total mutations in the tree"""
        # Find internal nodes (type == 2)
        is_internal = state.types == 2

        # For each internal node, compute mutations
        def compute_node_mutations(sequences, types):
            # Only consider sequences at internal nodes
            internal_sequences = jnp.where(types[:, None] == 2, sequences, 0)

            # Count positions where no overlap (mutations occurred)
            mutations = jnp.sum(internal_sequences == 0, axis=(0, 1))
            return mutations

        mutations = jax.vmap(compute_node_mutations)(
            state.sequences, state.types
        )
        return mutations

    def log_reward(
        self,
        state: PhylogeneticTreeEnvState,
        env_params: PhylogeneticTreeEnvParams,
    ) -> TLogReward:
        """Compute log reward: (C - total_mutations) / scale"""
        total_mutations = self.get_mutations(state, env_params)
        return (self.C - total_mutations) / self.scale

    def reward(
        self,
        state: PhylogeneticTreeEnvState,
        env_params: PhylogeneticTreeEnvParams,
    ) -> TReward:
        """Compute reward: exp((C - total_mutations) / scale)"""
        return jnp.exp(self.log_reward(state, env_params))
