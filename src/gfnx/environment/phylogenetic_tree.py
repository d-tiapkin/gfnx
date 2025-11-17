from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from .. import spaces
from ..base import (
    BaseVecEnvironment,
    TAction,
    TDone,
    TRewardModule,
    TRewardParams,
)


@chex.dataclass(frozen=True)
class EnvState:
    sequences: chex.Array  # [B, 2 * num_nodes - 1, sequence_length]
    left_child: chex.Array  # [B, 2 * num_nodes - 1]
    right_child: chex.Array  # [B, 2 * num_nodes - 1]
    parent: chex.Array  # [B, 2 * num_nodes - 1]
    to_root: chex.Array  # [B, num_nodes]
    to_leaf: chex.Array  # [B, 2 * num_nodes - 1]
    length: chex.Array  # [B]

    # Default attributes
    is_terminal: chex.Array  # [B]
    is_initial: chex.Array  # [B]
    is_pad: chex.Array  # [B]
    time: chex.Array  # [B]


@chex.dataclass(frozen=True)
class EnvParams:
    num_nodes: int
    sequence_length: int
    reward_params: TRewardParams = None
    bits_per_seq_elem: int = 5


class PhyloTreeEnvironment(BaseVecEnvironment[EnvState, EnvParams]):
    def __init__(
        self,
        reward_module: TRewardModule,
        sequences: chex.Array,  # [num_nodes, sequence_length]
        sequence_type: str = "DNA_WITH_GAP",
        bits_per_seq_elem: int = 5,
    ):
        super().__init__(reward_module)
        self.sequences = sequences  # each element is already a binary number
        chex.assert_axis_dimension_gt(sequences, 0, 1)  # num_nodes > 1
        self.num_nodes = sequences.shape[0]
        self.sequence_length = sequences.shape[1]
        self.sequence_type = sequence_type
        self.bits_per_seq_elem = bits_per_seq_elem

        # Pre-compute triu indices for actions
        indices = jnp.triu_indices(self.num_nodes, k=1)
        self.lefts = indices[0]
        self.rights = indices[1]

    def get_init_state(self, num_envs: int) -> EnvState:
        """Returns batch of initial states"""
        sequences = jnp.concatenate(
            [
                self.sequences,
                jnp.zeros(
                    (
                        self.num_nodes - 1,
                        self.sequence_length,
                    ),
                    dtype=jnp.uint8,
                ),
            ],
            axis=0,
        )  # [2 * num_nodes - 1, sequence_length]
        to_leaf = jnp.concatenate(
            [
                jnp.arange(self.num_nodes),
                jnp.full(self.num_nodes - 1, -1),
            ],
            axis=0,
        )  # [2 * num_nodes - 1]
        chex.assert_tree_shape_prefix(sequences, (2 * self.num_nodes - 1,))
        return EnvState(
            sequences=jnp.repeat(sequences[jnp.newaxis], num_envs, axis=0),
            left_child=jnp.full(
                (num_envs, 2 * self.num_nodes - 1), -1, dtype=jnp.int32
            ),  # -1 is the padding value
            right_child=jnp.full(
                (num_envs, 2 * self.num_nodes - 1), -1, dtype=jnp.int32
            ),
            parent=jnp.full(
                (num_envs, 2 * self.num_nodes - 1), -1, dtype=jnp.int32
            ),
            to_root=jnp.repeat(
                jnp.arange(self.num_nodes)[jnp.newaxis], num_envs, axis=0
            ),  # every node is a root
            to_leaf=jnp.repeat(to_leaf[jnp.newaxis], num_envs, axis=0),
            length=jnp.full(
                (num_envs,), self.num_nodes, dtype=jnp.int32
            ),  # free slots strat from num_nodes idx
            is_terminal=jnp.zeros((num_envs,), dtype=jnp.bool_),
            is_initial=jnp.ones((num_envs,), dtype=jnp.bool_),
            is_pad=jnp.zeros((num_envs,), dtype=jnp.bool_),
            time=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        """Initialize environment"""
        dummy_state = self.get_init_state(1)
        reward_params = self.reward_module.init(rng_key, dummy_state)
        return EnvParams(
            num_nodes=self.num_nodes,
            sequence_length=self.sequence_length,
            reward_params=reward_params,
        )

    def _single_transition(
        self, state: EnvState, action: TAction, env_params: EnvParams
    ) -> Tuple[EnvState, TDone, Dict[str, Any]]:
        """Single environment step transition"""
        is_terminal = state.is_terminal
        time = state.time

        def get_state_terminal() -> EnvState:
            return state.replace(is_pad=True)

        def get_state_nonterminal() -> EnvState:
            left = state.to_root[self.lefts[action]]
            right = state.to_root[self.rights[action]]
            # If there's overlap (both sequences have 1 in same position), keep it
            # Otherwise, take the union
            overlap = jnp.bitwise_and(
                state.sequences[left], state.sequences[right]
            )
            union = jnp.bitwise_or(
                state.sequences[left], state.sequences[right]
            )
            new_sequence = jnp.where(overlap > 0, overlap, union)
            # fmt: off
            next_state = state.replace(
                sequences=state.sequences.at[state.length].set(new_sequence),
                left_child=state.left_child.at[state.length].set(left),
                right_child=state.right_child.at[state.length].set(right),
                parent=state.parent.at[left]
                    .set(state.length)
                    .at[right]
                    .set(state.length),
                to_root=state.to_root.at[self.lefts[action]]
                    .set(state.length)  # merge to left
                    .at[self.rights[action]]
                    .set(-1),  # remove right
                to_leaf=state.to_leaf.at[state.length].set(self.lefts[action]),
                length=state.length + 1,
                time=time + 1,
                is_initial=False,
            )
            # fmt: on

            return next_state.replace(
                is_terminal=jnp.all(next_state.to_root[1:] == -1)
            )  # all but first node are inner nodes

        next_state = jax.lax.cond(
            is_terminal, get_state_terminal, get_state_nonterminal
        )

        return next_state, next_state.is_terminal, {}

    def _single_backward_transition(
        self, state: EnvState, backward_action: TAction, env_params: EnvParams
    ) -> Tuple[EnvState, chex.Array, Dict[str, Any]]:
        """Single environment step backward transition"""
        is_initial = state.is_initial
        time = state.time

        def get_state_initial() -> EnvState:
            return state.replace(is_pad=True)

        def get_state_non_initial() -> EnvState:
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
                time=time - 1,
                is_terminal=False,
                is_pad=False,
            )
            # fmt: on
            def swap_root_with_last(prev_state: EnvState) -> EnvState:
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
                        to_root=prev_state.to_root.at[
                            prev_state.to_leaf[root]
                        ].set(root),
                    )

                prev_state = jax.lax.cond(
                    prev_state.parent[last] == -1,
                    swap_root,
                    swap_internal,
                )
                return prev_state

            # Swap the last element with the deleted node
            last = prev_state.length - 1
            prev_state = jax.lax.cond(
                root == last,
                lambda prev_state: prev_state,
                swap_root_with_last,
                prev_state,
            )

            return prev_state.replace(
                length=prev_state.length - 1,
                is_initial=jnp.all(
                    prev_state.to_root[: self.num_nodes]
                    == jnp.arange(self.num_nodes)
                ),  # also it is equal to prev_state.length == num_nodes + 1
            )

        prev_state = jax.lax.cond(
            is_initial, get_state_initial, get_state_non_initial
        )
        return prev_state, prev_state.is_initial, {}

    def get_obs(
        self, state: EnvState, env_params: EnvParams
    ) -> chex.ArrayTree:
        """Convert state to observation"""

        def single_get_obs(state: EnvState) -> chex.Array:
            """Take roots from the forest. And one-hot encode each number in the sequence.
            If the node is a root, take its sequence. Otherwise, take zero-array
            """
            sequences = jnp.where(
                (state.to_root != -1)[
                    :, jnp.newaxis
                ],  # Make broadcastable to [num_nodes, sequence_length]
                state.sequences[state.to_root],
                jnp.zeros(
                    (
                        self.num_nodes,
                        self.sequence_length,
                    ),
                    dtype=jnp.uint8,
                ),
            )  # [num_nodes, sequence_length]
            """Fitch features. One-hot encode each number in the sequence.
            E.g. for 5-bit encoding for each element in the sequence:
            [..., 0b00101, ...] <-> [..., [1, 0, 1, 0, 0], ...]
            """
            fitch_features = (
                sequences[..., jnp.newaxis]
                & (1 << jnp.arange(self.bits_per_seq_elem))
            ) > 0  # [num_nodes, sequence_length, bits_per_seq_elem]
            return jnp.where(fitch_features, 1, 0).astype(jnp.uint8)

        return jax.vmap(single_get_obs)(state)

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: TAction,
        next_state: EnvState,
        env_params: EnvParams,
    ) -> TAction:
        """Returns backward action given the forward transition"""
        return self.lefts[forward_action]

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: TAction,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> TAction:
        """Returns forward action given the backward transition"""

        batch_idx = jnp.arange(state.time.shape[0])
        left = state.to_leaf[
            batch_idx,
            state.left_child[
                batch_idx, state.to_root[batch_idx, backward_action]
            ],
        ]
        right = state.to_leaf[
            batch_idx,
            state.right_child[
                batch_idx, state.to_root[batch_idx, backward_action]
            ],
        ]
        forward_action = (
            left * (2 * self.num_nodes - 1 - left) // 2 + right - (left + 1)
        )  # Reverse operation of lefts[forward_action] and rights[forward_action]

        return forward_action

    def get_invalid_mask(
        self, state: EnvState, env_params: EnvParams
    ) -> chex.Array:
        """Returns mask of invalid actions"""

        def single_get_invalid_mask(state: EnvState) -> chex.Array:
            mask = (state.to_root == -1)[self.lefts] | (state.to_root == -1)[
                self.rights
            ]
            return mask

        return jax.vmap(single_get_invalid_mask)(state)

    def get_invalid_backward_mask(
        self, state: EnvState, env_params: EnvParams
    ) -> chex.Array:
        """Returns mask of invalid backward actions"""

        def single_get_invalid_backward_mask(state: EnvState) -> chex.Array:
            return jnp.logical_or(
                state.to_root[:-1] == -1,
                state.to_root[:-1] == jnp.arange(self.num_nodes - 1),
            )  # last node is never a root

        return jax.vmap(single_get_invalid_backward_mask)(state)

    @property
    def max_steps_in_episode(self) -> int:
        """Maximum number of steps in an episode"""
        return self.num_nodes - 1

    @property
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment"""
        num_actions = self.num_nodes * (self.num_nodes - 1) // 2
        return spaces.Discrete(num_actions)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """Backward action space of the environment"""
        return spaces.Discrete(
            self.num_nodes - 1
        )  # Split any node, except the last, as it is never a root

    @property
    def observation_space(self) -> spaces.Box:
        """Observation space of the environment"""
        return spaces.Box(
            low=0,
            high=1,
            shape=(
                self.num_nodes,
                self.sequence_length,
                self.bits_per_seq_elem,
            ),
            dtype=jnp.uint8,
        )

    @property
    def state_space(self) -> spaces.Dict:
        """State space of the environment"""
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
            "is_initial": spaces.Box(
                low=0,
                high=1,
                shape=(),
                dtype=jnp.bool_,
            ),
            "is_terminal": spaces.Box(
                low=0,
                high=1,
                shape=(),
                dtype=jnp.bool_,
            ),
            "is_pad": spaces.Box(
                low=0,
                high=1,
                shape=(),
                dtype=jnp.bool_,
            ),
        })
