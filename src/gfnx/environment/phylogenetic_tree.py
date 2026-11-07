"""Phylogenetic tree environment: build a rooted binary tree by merging species.

Follows the PhyloGFN formulation (Zhou et al., 2024). The parsimony reward lives in
[`gfnx.reward.phylogenetic_tree`][gfnx.reward.phylogenetic_tree]; datasets DS1–DS8 are
loaded with `gfnx.utils.get_phylo_initialization_args`.
"""

from typing import Any

import chex
import jax
import jax.numpy as jnp

from .. import spaces
from ..base import (
    BaseEnvironment,
    BaseEnvParams,
    BaseEnvState,
    TAction,
    TDone,
)


@chex.dataclass(frozen=True)
class EnvState(BaseEnvState):
    """State of a single phylogenetic-tree environment (a forest).

    Nodes are numbered `0 .. 2n - 2`: leaves (species) are `0 .. n - 1`, and internal
    nodes are appended in creation order. Each tree of the forest occupies a *slot*
    `0 .. n - 1`, identified by the smallest leaf it contains.

    Attributes:
        sequences: Fitch state sets of every node, shape `[2n - 1, L]`. Each site is a
            bitmask over `bits_per_seq_elem` characters; unconstructed nodes are zero.
        left_child: Left child of every node, `-1` for leaves / unconstructed nodes.
        right_child: Right child of every node, `-1` for leaves / unconstructed nodes.
        parent: Parent of every node, `-1` for roots / unconstructed nodes.
        to_root: Root node of the tree in each slot, shape `[n]`; `-1` if the slot was
            merged into a smaller one.
        to_leaf: Slot of the subtree rooted at each node, shape `[2n - 1]` (the smallest
            leaf below it); `-1` for unconstructed nodes.
        length: Number of constructed nodes, `n + number_of_merges`.
        is_terminal: `True` once a single tree remains.
        is_initial: `True` for the forest of `n` singleton trees.
        is_pad: `True` for padding steps after termination.
    """

    sequences: chex.Array  # [2 * num_nodes - 1, sequence_length]
    left_child: chex.Array  # [2 * num_nodes - 1]
    right_child: chex.Array  # [2 * num_nodes - 1]
    parent: chex.Array  # [2 * num_nodes - 1]
    to_root: chex.Array  # [num_nodes]
    to_leaf: chex.Array  # [2 * num_nodes - 1]
    length: chex.Array  # scalar

    # Default attributes (scalar, no batch dim)
    is_terminal: chex.Array
    is_initial: chex.Array
    is_pad: chex.Array


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    """Phylogenetic-tree dynamics parameters.

    Attributes:
        num_nodes: Number of species $n$ (leaves).
        sequence_length: Number of aligned sites $L$.
        bits_per_seq_elem: Bits used to encode one site (5 for DNA with gaps).
    """

    num_nodes: int
    sequence_length: int
    bits_per_seq_elem: int = 5


class PhyloTreeEnvironment(BaseEnvironment[EnvState, EnvParams]):
    r"""Bottom-up construction of rooted binary phylogenetic trees over $n$ species.

    The initial state is a forest of $n$ singleton trees, one per species. Each step
    picks two trees of the forest and joins them under a new common ancestor; after
    exactly $n - 1$ merges a single rooted binary tree spans all species and the episode
    terminates. Ancestral sequences are summarised by Fitch state sets: per site, the
    parent's set is the intersection of its children's sets if it is non-empty and their
    union otherwise (each union is one mutation for the parsimony score).

    * **State** — the forest; see [`EnvState`][gfnx.environment.phylogenetic_tree.EnvState].
    * **Forward actions** — `n * (n - 1) / 2` actions, one per unordered pair of slots
      $(i, j)$, $i < j$, in `jnp.triu_indices(n, k=1)` order. The action merges the trees
      in slots $i$ and $j$ into slot $i$; it is valid iff both slots hold a tree.
    * **Backward actions** — `n - 1` actions. Action `b` splits the tree in slot `b` at
      its root; it is valid iff that root is an internal node. Node ids are kept
      contiguous by swapping the removed node with the most recently created one.
    * **Observation** — one-hot Fitch features of each slot's root, `uint8` of shape
      `[n, L, bits_per_seq_elem]` (zeros for empty slots), designed for transformer
      policies.
    * **Episode length** — exactly `n - 1` steps.

    Args:
        sequences: Binary-encoded aligned sequences of shape `[n, L]` (`n > 1`), where
            each entry is a bitmask over `bits_per_seq_elem` characters. Use
            `gfnx.utils.get_phylo_initialization_args` to load DS1–DS8.
        sequence_type: Alphabet name, e.g. `"DNA_WITH_GAP"`.
        bits_per_seq_elem: Bits per site in the encoding (5 for DNA with gaps).

    References:
        Zhou, M. Y., Yan, Z., Layne, E., Malkin, N., Zhang, D., Jain, M., Blanchette, M.,
        Bengio, Y. (2024). PhyloGFN: Phylogenetic inference with generative flow
        networks. ICLR.
    """

    def __init__(
        self,
        sequences: chex.Array,  # [num_nodes, sequence_length]
        sequence_type: str = "DNA_WITH_GAP",
        bits_per_seq_elem: int = 5,
    ):
        self.sequences = sequences
        chex.assert_axis_dimension_gt(sequences, 0, 1)  # num_nodes > 1
        self.num_nodes = sequences.shape[0]
        self.sequence_length = sequences.shape[1]
        self.sequence_type = sequence_type
        self.bits_per_seq_elem = bits_per_seq_elem

        # Pre-compute triu indices for actions
        indices = jnp.triu_indices(self.num_nodes, k=1)
        self.lefts = indices[0]
        self.rights = indices[1]

    @property
    def name(self) -> str:
        """Environment name."""
        return "PhyloTree-v0"

    def reset(self) -> EnvState:
        """Returns the forest of `n` singleton trees (one per species)."""
        sequences = jnp.concatenate(
            [
                self.sequences,
                jnp.zeros((self.num_nodes - 1, self.sequence_length), dtype=jnp.uint8),
            ],
            axis=0,
        )  # [2 * num_nodes - 1, sequence_length]
        to_leaf = jnp.concatenate(
            [jnp.arange(self.num_nodes), jnp.full(self.num_nodes - 1, -1)], axis=0
        )  # [2 * num_nodes - 1]
        return EnvState(
            sequences=sequences,
            left_child=jnp.full((2 * self.num_nodes - 1,), -1, dtype=jnp.int32),
            right_child=jnp.full((2 * self.num_nodes - 1,), -1, dtype=jnp.int32),
            parent=jnp.full((2 * self.num_nodes - 1,), -1, dtype=jnp.int32),
            to_root=jnp.arange(self.num_nodes),
            to_leaf=to_leaf,
            length=jnp.int32(self.num_nodes),
            is_terminal=jnp.bool_(False),
            is_initial=jnp.bool_(True),
            is_pad=jnp.bool_(False),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        """Returns `EnvParams(num_nodes, sequence_length)`. The environment is deterministic."""
        return EnvParams(
            num_nodes=self.num_nodes,
            sequence_length=self.sequence_length,
        )

    def step(
        self, state: EnvState, action: TAction, env_params: EnvParams
    ) -> tuple[chex.Array, EnvState, TDone, dict[str, Any]]:
        """Merges the trees in slots `(i, j) = triu_indices(n, 1)[:, action]` into slot `i`.

        Returns:
            `(obs, next_state, done, info)`; `done` is `True` once one tree remains.
        """
        left = state.to_root[self.lefts[action]]
        right = state.to_root[self.rights[action]]
        overlap = jnp.bitwise_and(state.sequences[left], state.sequences[right])
        union = jnp.bitwise_or(state.sequences[left], state.sequences[right])
        new_sequence = jnp.where(overlap > 0, overlap, union)
        # fmt: off
        next_active = state.replace(
            sequences=state.sequences.at[state.length].set(new_sequence),
            left_child=state.left_child.at[state.length].set(left),
            right_child=state.right_child.at[state.length].set(right),
            parent=state.parent.at[left]
                .set(state.length)
                .at[right]
                .set(state.length),
            to_root=state.to_root.at[self.lefts[action]]
                .set(state.length)
                .at[self.rights[action]]
                .set(-1),
            to_leaf=state.to_leaf.at[state.length].set(self.lefts[action]),
            length=state.length + 1,
            is_initial=False,
        )
        # fmt: on
        next_active = next_active.replace(is_terminal=jnp.all(next_active.to_root[1:] == -1))
        next_pad = state.replace(is_pad=True)
        next_state = jax.tree.map(
            lambda p, a: jnp.where(state.is_terminal, p, a), next_pad, next_active
        )
        return (
            self.get_obs(next_state, env_params),
            next_state,
            jnp.astype(next_state.is_terminal, jnp.bool),
            {},
        )

    def backward_step(
        self, state: EnvState, backward_action: TAction, env_params: EnvParams
    ) -> tuple[chex.Array, EnvState, TDone, dict[str, Any]]:
        """Splits the tree in slot `backward_action` at its root.

        The two subtrees return to their own slots, and the removed root is swapped with
        the most recently created node so that node ids stay contiguous.

        Returns:
            `(obs, prev_state, done, info)`; `done` is `True` once only singletons remain.
        """
        root = state.to_root[backward_action]
        left_child = state.left_child[root]
        right_child = state.right_child[root]
        # fmt: off
        prev_state = state.replace(
            sequences=state.sequences.at[root].set(
                jnp.zeros(self.sequence_length, dtype=jnp.uint8)
            ),
            left_child=state.left_child.at[root].set(-1),
            right_child=state.right_child.at[root].set(-1),
            parent=state.parent.at[left_child]
                .set(-1)
                .at[right_child]
                .set(-1),
            to_root=state.to_root.at[state.to_leaf[left_child]]
                .set(left_child)
                .at[state.to_leaf[right_child]]
                .set(right_child),
            to_leaf=state.to_leaf.at[root].set(-1),
            is_terminal=False,
            is_pad=False,
        )
        # fmt: on

        def swap_root_with_last(prev_state: EnvState) -> EnvState:
            last = prev_state.length - 1
            # fmt: off
            prev_state = prev_state.replace(
                sequences=prev_state.sequences.at[root]
                    .set(prev_state.sequences[last])
                    .at[last]
                    .set(prev_state.sequences[root]),
                left_child=prev_state.left_child.at[root]
                    .set(prev_state.left_child[last])
                    .at[last]
                    .set(prev_state.left_child[root]),
                right_child=prev_state.right_child.at[root]
                    .set(prev_state.right_child[last])
                    .at[last]
                    .set(prev_state.right_child[root]),
                parent=prev_state.parent.at[prev_state.left_child[last]]
                    .set(root)
                    .at[prev_state.right_child[last]]
                    .set(root),
                to_leaf=prev_state.to_leaf.at[root]
                    .set(prev_state.to_leaf[last])
                    .at[last]
                    .set(prev_state.to_leaf[root]),
            )
            # fmt: on

            def swap_internal():
                def swap_internal_left_child():
                    # fmt: off
                    return prev_state.replace(
                        parent=prev_state.parent.at[root]
                            .set(prev_state.parent[last])
                            .at[last]
                            .set(prev_state.parent[root]),
                        left_child=prev_state.left_child.at[
                            prev_state.parent[last]
                        ].set(root),
                    )
                    # fmt: on

                def swap_internal_right_child():
                    # fmt: off
                    return prev_state.replace(
                        parent=prev_state.parent.at[root]
                            .set(prev_state.parent[last])
                            .at[last]
                            .set(prev_state.parent[root]),
                        right_child=prev_state.right_child.at[
                            prev_state.parent[last]
                        ].set(root),
                    )
                    # fmt: on

                return jax.lax.cond(
                    prev_state.left_child[prev_state.parent[last]] == last,
                    swap_internal_left_child,
                    swap_internal_right_child,
                )

            def swap_root():
                return prev_state.replace(
                    to_root=prev_state.to_root.at[prev_state.to_leaf[root]].set(root),
                )

            return jax.lax.cond(
                prev_state.parent[last] == -1,
                swap_root,
                swap_internal,
            )

        last = prev_state.length - 1
        prev_state = jax.lax.cond(
            root == last,
            lambda s: s,
            swap_root_with_last,
            prev_state,
        )
        non_initial = prev_state.replace(
            length=prev_state.length - 1,
            is_initial=jnp.all(prev_state.to_root[: self.num_nodes] == jnp.arange(self.num_nodes)),
        )
        init_pad = state.replace(is_pad=True)
        prev_state = jax.tree.map(
            lambda p, n: jnp.where(state.is_initial, p, n), init_pad, non_initial
        )
        return (
            self.get_obs(prev_state, env_params),
            prev_state,
            jnp.astype(prev_state.is_initial, jnp.bool),
            {},
        )

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.ArrayTree:
        """Returns one-hot Fitch features of each slot's root, shape `[n, L, bits]`."""
        sequences = jnp.where(
            (state.to_root != -1)[:, jnp.newaxis],
            state.sequences[state.to_root],
            jnp.zeros((self.num_nodes, self.sequence_length), dtype=jnp.uint8),
        )  # [num_nodes, sequence_length]
        # Fitch features. One-hot encode each number in the sequence.
        # E.g. for 5-bit encoding for each element in the sequence:
        # [..., 0b00101, ...] <-> [..., [1, 0, 1, 0, 0], ...]
        fitch_features = (
            sequences[..., jnp.newaxis] & (1 << jnp.arange(self.bits_per_seq_elem))
        ) > 0  # [num_nodes, sequence_length, bits_per_seq_elem]
        return jnp.where(fitch_features, 1, 0).astype(jnp.uint8)

    def get_backward_action(
        self,
        _state: EnvState,
        forward_action: TAction,
        _next_state: EnvState,
        _env_params: EnvParams,
    ) -> TAction:
        """Returns slot `i` of the merged pair: the slot that now holds the new tree."""
        return self.lefts[forward_action]

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: TAction,
        _prev_state: EnvState,
        _env_params: EnvParams,
    ) -> TAction:
        """Returns the pair index of the two slots that the split separates."""
        left = state.to_leaf[state.left_child[state.to_root[backward_action]]]
        right = state.to_leaf[state.right_child[state.to_root[backward_action]]]
        return left * (2 * self.num_nodes - 1 - left) // 2 + right - (left + 1)

    def get_action_mask(self, state: EnvState, _env_params: EnvParams) -> chex.Array:
        """Valid forward actions (single state). ``True`` = valid merge."""
        return (state.to_root != -1)[self.lefts] & (state.to_root != -1)[self.rights]

    def get_backward_action_mask(self, state: EnvState, _env_params: EnvParams) -> chex.Array:
        """Valid backward actions (single state). ``True`` = valid undo-merge.

        The last node is never a root, so its slot is always invalid.
        """
        return jnp.logical_and(
            state.to_root[:-1] != -1,
            state.to_root[:-1] != jnp.arange(self.num_nodes - 1),
        )

    @property
    def max_steps_in_episode(self) -> int:
        """Exactly `n - 1` merges."""
        return self.num_nodes - 1

    @property
    def action_space(self) -> spaces.Discrete:
        """`Discrete(n * (n - 1) / 2)`: unordered slot pairs."""
        num_actions = self.num_nodes * (self.num_nodes - 1) // 2
        return spaces.Discrete(num_actions)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """`Discrete(n - 1)`: the slot to split (slot `n - 1` can never hold a merge)."""
        return spaces.Discrete(self.num_nodes - 1)

    @property
    def observation_space(self) -> spaces.Box:
        """`uint8` box of shape `[n, L, bits_per_seq_elem]`."""
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.num_nodes, self.sequence_length, self.bits_per_seq_elem),
            dtype=jnp.uint8,
        )

    @property
    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
            "sequences": spaces.Box(
                low=0,
                high=1,
                shape=(2 * self.num_nodes - 1, self.sequence_length),
                dtype=jnp.uint8,
            ),
            "left_child": spaces.Box(
                low=-1,
                high=2 * self.num_nodes - 2,
                shape=(2 * self.num_nodes - 1,),
                dtype=jnp.int32,
            ),
            "right_child": spaces.Box(
                low=-1,
                high=2 * self.num_nodes - 2,
                shape=(2 * self.num_nodes - 1,),
                dtype=jnp.int32,
            ),
            "parent": spaces.Box(
                low=-1,
                high=2 * self.num_nodes - 2,
                shape=(2 * self.num_nodes - 1,),
                dtype=jnp.int32,
            ),
            "to_root": spaces.Box(
                low=-1,
                high=2 * self.num_nodes - 2,
                shape=(self.num_nodes,),
                dtype=jnp.int32,
            ),
            "to_leaf": spaces.Box(
                low=-1,
                high=self.num_nodes - 1,
                shape=(2 * self.num_nodes - 1,),
                dtype=jnp.int32,
            ),
            "length": spaces.Box(
                low=self.num_nodes,
                high=2 * self.num_nodes - 1,
                shape=(),
                dtype=jnp.int32,
            ),
            "is_initial": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool_),
            "is_terminal": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool_),
            "is_pad": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool_),
        })
