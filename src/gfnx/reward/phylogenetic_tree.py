import chex
import jax
import jax.numpy as jnp

from ..base import BaseRewardModule, TLogReward, TReward, TRewardParams
from ..environment import PhyloTreeEnvParams, PhyloTreeEnvState


class PhyloTreeRewardModule(
    BaseRewardModule[PhyloTreeEnvState, PhyloTreeEnvParams]
):
    """
    Reward module for phylogenetic trees using exponential reward function.
    R(x) = exp((offset - total_mutations) / scale)
    """

    def __init__(self, num_nodes: int, scale: float = 1.0, C: float = 0.0):
        self.num_nodes = num_nodes
        self.scale = scale
        self.C = C

        # TODO: check delta score in original paper
        self._offset = (C / scale) / num_nodes

    def init(
        self, rng_key: chex.PRNGKey, dummy_state: PhyloTreeEnvState
    ) -> TRewardParams:
        """Initialize reward parameters"""
        return {}  # No parameters for this reward

    def _get_mutations(
        self,
        state: PhyloTreeEnvState,
    ) -> chex.Array:
        """Compute total mutations in the tree"""

        def compute_mutations(carry, node_idx):
            is_tree = jnp.logical_and(
                state.to_leaf[node_idx] != -1,  # the node is constructed
                state.to_leaf[node_idx] != node_idx,  # the node is not a leaf
            )
            mutations = jnp.where(
                is_tree,
                jnp.sum(
                    state.sequences[state.left_child[node_idx]]
                    & state.sequences[state.right_child[node_idx]]
                    == 0
                ),
                0.0,
            )
            return carry + mutations, None

        return jax.lax.scan(
            compute_mutations, 0.0, jnp.arange(2 * self.num_nodes - 1)
        )[0]

    def delta_score(self, state: PhyloTreeEnvState) -> TReward:
        """Compute delta score"""

        def _single_delta_score(state):
            mutations = jnp.sum(
                state.sequences[state.left_child[state.length - 1]]
                & state.sequences[state.right_child[state.length - 1]]
                == 0
            )
            return (self.C / self.num_nodes - mutations) / self.scale

        return jax.vmap(_single_delta_score)(state)

    def log_reward(
        self,
        state: PhyloTreeEnvState,
        env_params: PhyloTreeEnvParams,
    ) -> TLogReward:
        """Compute log reward: (C - total_mutations) / scale"""

        def _single_log_reward(state):
            total_mutations = self._get_mutations(state)
            return (self.C - total_mutations) / self.scale

        return jax.vmap(_single_log_reward)(state)

    def reward(
        self,
        state: PhyloTreeEnvState,
        env_params: PhyloTreeEnvParams,
    ) -> TReward:
        """Compute reward: exp((C - total_mutations) / scale)"""
        return jnp.exp(self.log_reward(state, env_params))
