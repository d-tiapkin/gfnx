"""Parsimony-based reward for the phylogenetic tree environment.

See [`PhyloTreeEnvironment`][gfnx.environment.phylogenetic_tree.PhyloTreeEnvironment].
"""

import chex
import jax
import jax.numpy as jnp

from ..base import BaseRewardModule, BaseRewardParams, TLogReward, TReward
from ..environment import PhyloTreeEnvParams, PhyloTreeEnvState


@chex.dataclass(frozen=True)
class PhyloTreeRewardParams(BaseRewardParams):
    """Empty reward parameters: the parsimony reward is closed-form."""


class PhyloTreeRewardModule(
    BaseRewardModule[PhyloTreeEnvState, PhyloTreeEnvParams, PhyloTreeRewardParams]
):
    r"""Parsimony reward for phylogenetic trees.

    For a complete tree $T$ with parsimony score $M(T)$ — the minimum number of mutations
    needed to explain the observed sequences, computed with Fitch's algorithm as the
    number of (node, site) pairs whose children's state sets are disjoint —

    $$
    R(T) = \exp\left(\frac{C - M(T)}{\alpha}\right).
    $$

    $\alpha$ (`scale`) is a temperature and $C$ a dataset-specific constant that only
    recentres the log-reward into a numerically friendly range without changing the
    ranking of topologies. Following PhyloGFN, `gfnx.utils.get_phylo_initialization_args`
    sets $\alpha = 4$ and

    | Dataset | DS1 | DS2 | DS3 | DS4 | DS5 | DS6 | DS7 | DS8 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | $C$ | 5800 | 8000 | 8800 | 3500 | 2300 | 2300 | 12500 | 2800 |

    The reward decomposes over merges, which
    [`delta_score`][gfnx.reward.phylogenetic_tree.PhyloTreeRewardModule.delta_score]
    exposes for Forward-Looking Detailed Balance (`baselines/fldb_phylo.py`).

    Args:
        num_nodes: Number of species $n$; must match the environment.
        scale: Temperature $\alpha$.
        C: Recentring constant $C$.

    References:
        Zhou, M. Y., et al. (2024). PhyloGFN: Phylogenetic inference with generative flow
        networks. ICLR.

        Deleu, T., Nouri, P., Malkin, N., Precup, D., Bengio, Y. (2024). Discrete
        probabilistic inference as control in multi-path environments. UAI.
    """

    def __init__(self, num_nodes: int, scale: float = 1.0, C: float = 0.0):
        self.num_nodes = num_nodes
        self.scale = scale
        self.C = C

        # TODO: check delta score in original paper
        self._offset = (C / scale) / num_nodes

    def init(self, rng_key: chex.PRNGKey, dummy_state: PhyloTreeEnvState) -> PhyloTreeRewardParams:
        """Returns empty `PhyloTreeRewardParams`; the reward is closed-form."""
        return PhyloTreeRewardParams()

    def _get_mutations(
        self,
        state: PhyloTreeEnvState,
    ) -> chex.Array:
        """Computes the Fitch parsimony score $M$ summed over all constructed internal nodes."""

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

        return jax.lax.scan(compute_mutations, 0.0, jnp.arange(2 * self.num_nodes - 1))[0]

    def delta_score(self, state: PhyloTreeEnvState) -> TReward:
        r"""Log-reward contribution of the most recent merge.

        Returns $(C / n - m) / \alpha$, where $m$ is the number of sites at which the two
        children of the newest internal node have disjoint state sets. Summed over the
        $n - 1$ merges of a complete tree this gives
        $\big(C (n - 1) / n - M(T)\big) / \alpha$, i.e. $\log R(T)$ up to the constant
        $C / (n \alpha)$.

        Args:
            state: Single state reached by a merge (not the initial state).

        Returns:
            Scalar per-merge log-reward increment.
        """
        mutations = jnp.sum(
            state.sequences[state.left_child[state.length - 1]]
            & state.sequences[state.right_child[state.length - 1]]
            == 0
        )
        return (self.C / self.num_nodes - mutations) / self.scale

    def log_reward(
        self,
        state: PhyloTreeEnvState,
        reward_params: PhyloTreeRewardParams,
    ) -> TLogReward:
        r"""Computes $\log R(T) = (C - M(T)) / \alpha$ for a single (terminal) state."""
        total_mutations = self._get_mutations(state)
        return (self.C - total_mutations) / self.scale

    def reward(
        self,
        state: PhyloTreeEnvState,
        reward_params: PhyloTreeRewardParams,
    ) -> TReward:
        r"""Computes $R(T) = \exp((C - M(T)) / \alpha)$ for a single (terminal) state."""
        return jnp.exp(self.log_reward(state, reward_params))
