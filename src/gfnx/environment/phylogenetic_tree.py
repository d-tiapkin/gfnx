from typing import Any

import chex
import jax
import jax.numpy as jnp

from .. import spaces
from ..base import (
    BaseEnvironment,
    TAction,
    TDone,
)


@chex.dataclass(frozen=True)
class EnvState:
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
class EnvParams:
    num_nodes: int
    sequence_length: int
    bits_per_seq_elem: int = 5


class PhyloTreeEnvironment(BaseEnvironment[EnvState, EnvParams]):
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
        return "PhyloTree-v0"

    def get_init_state(self) -> EnvState:
        """Returns a single initial state."""
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
        return EnvParams(
            num_nodes=self.num_nodes,
            sequence_length=self.sequence_length,
        )

    def _transition(
        self, state: EnvState, action: TAction, env_params: EnvParams
    ) -> tuple[EnvState, TDone, dict[str, Any]]:
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
        return next_state, next_state.is_terminal, {}

    def _backward_transition(
        self, state: EnvState, backward_action: TAction, env_params: EnvParams
    ) -> tuple[EnvState, chex.Array, dict[str, Any]]:
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
        return prev_state, prev_state.is_initial, {}

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.ArrayTree:
        """Returns observation for a single state."""
        sequences = jnp.where(
            (state.to_root != -1)[:, jnp.newaxis],
            state.sequences[state.to_root],
            jnp.zeros((self.num_nodes, self.sequence_length), dtype=jnp.uint8),
        )  # [num_nodes, sequence_length]
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
        return self.lefts[forward_action]

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: TAction,
        _prev_state: EnvState,
        _env_params: EnvParams,
    ) -> TAction:
        left = state.to_leaf[state.left_child[state.to_root[backward_action]]]
        right = state.to_leaf[state.right_child[state.to_root[backward_action]]]
        return left * (2 * self.num_nodes - 1 - left) // 2 + right - (left + 1)

    def get_invalid_mask(self, state: EnvState, _env_params: EnvParams) -> chex.Array:
        return (state.to_root == -1)[self.lefts] | (state.to_root == -1)[self.rights]

    def get_invalid_backward_mask(self, state: EnvState, _env_params: EnvParams) -> chex.Array:
        return jnp.logical_or(
            state.to_root[:-1] == -1,
            state.to_root[:-1] == jnp.arange(self.num_nodes - 1),
        )

    @property
    def max_steps_in_episode(self) -> int:
        return self.num_nodes - 1

    @property
    def action_space(self) -> spaces.Discrete:
        num_actions = self.num_nodes * (self.num_nodes - 1) // 2
        return spaces.Discrete(num_actions)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.num_nodes - 1)

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.num_nodes, self.sequence_length, self.bits_per_seq_elem),
            dtype=jnp.uint8,
        )

    @property
    def state_space(self) -> spaces.Dict:
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
