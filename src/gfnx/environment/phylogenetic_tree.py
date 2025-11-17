# Code adapted from gfn-maxent-rl by Tristan Deleu, Padideh Nouri, Nikolay Malkin, Doina Precup, Yoshua Bengio
# URL: https://github.com/tristandeleu/gfn-maxent-rl
# License: MIT License

from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

import gfnx.spaces as spaces
from gfnx import TAction, TDone, TRewardModule, TRewardParams

from ..base import BaseEnvParams, BaseEnvState, BaseVecEnvironment

CHARACTERS_MAPS = {
    "DNA": {
        "A": 0b1000,
        "C": 0b0100,
        "G": 0b0010,
        "T": 0b0001,
        "N": 0b1111,
        "?": 0b1111,
    },
    "DNA_WITH_GAP": {
        "A": 0b10000,
        "C": 0b01000,
        "G": 0b00100,
        "T": 0b00010,
        "-": 0b00001,
        "N": 0b11110,
        "?": 0b11110,
    },
}


@chex.dataclass(frozen=True)
class EnvState(BaseEnvState):
    sequences: chex.Array  # [B, num_nodes, sequence_length]  # elems are binary number
    tree: chex.Array  # [B, 2 * num_nodes - 1]
    types: chex.Array  # [B, num_nodes], 0: empty, 1: internal node, 2: merged tree
    merge_order: chex.Array  # [B, num_nodes], tracks the order of merges
    is_done: chex.Array  # [B]
    time: chex.Array  # [B]


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    num_nodes: int
    sequence_length: int
    num_characters: int
    reward_params: TRewardParams


class PhylogeneticTreeEnvironment(BaseVecEnvironment[EnvState, EnvParams]):
    # TODO: fix tree in EnvState
    def __init__(
        self,
        reward_module: TRewardModule,
        sequences: chex.Array,  # [num_nodes, sequence_length]
        sequence_type: str = "DNA_WITH_GAP",
    ):
        super().__init__(reward_module)
        self.sequences = sequences  # each element is already a binary number
        self.num_nodes = sequences.shape[0]
        self.sequence_length = sequences.shape[1]
        self.sequence_type = sequence_type
        self.num_characters = len(CHARACTERS_MAPS[sequence_type])

        # Pre-compute triu indices for actions
        indices = jnp.triu_indices(self.num_nodes, k=1)
        self.lefts = indices[0]
        self.rights = indices[1]

    def get_init_state(self, num_envs: int) -> EnvState:
        """Returns batch of initial states"""
        return EnvState(
            sequences=jnp.repeat(self.sequences[None], num_envs, axis=0),
            tree=jnp.arange((num_envs, 2 * self.num_nodes - 1), dtype=jnp.int32),
            types=jnp.ones((num_envs, self.num_nodes), dtype=jnp.int32),  # All leaves
            merge_order=jnp.zeros((num_envs, self.num_nodes), dtype=jnp.int32),
            is_done=jnp.zeros((num_envs,), dtype=jnp.bool_),
            time=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        dummy_state = self.get_init_state(1)
        reward_params = self.reward_module.init(rng_key, dummy_state)
        return EnvParams(
            num_nodes=self.num_nodes,
            sequence_length=self.sequence_length,
            num_characters=self.num_characters,
            reward_params=reward_params,
        )

    def _single_transition(
        self, state: EnvState, action: TAction, env_params: EnvParams
    ) -> Tuple[EnvState, TDone, Dict[str, Any]]:
        """Single environment step transition"""
        stop_action = self.action_space.n - 1
        is_done = action == stop_action
        action = jnp.where(is_done, 0, action)  # To avoid overflow

        left, right = self.lefts[action], self.rights[action]

        # If there's overlap (both sequences have 1 in same position), keep it
        # Otherwise, take the union
        overlap = state.sequences[left] & state.sequences[right]
        union = state.sequences[left] | state.sequences[right]
        new_sequence = jnp.where(overlap > 0, overlap, union)

        sequences = state.sequences.at[left].set(new_sequence)
        sequences = sequences.at[right].set(0)

        types = state.types.at[left].set(2)  # Rooted tree
        types = types.at[right].set(0)  # Remove right tree

        # FIXME: do we really need it?
        # tree = state.tree.at[left].set(-1)
        # tree = tree.at[right].set(-2)

        time = state.time + 1

        # Check if we're done
        is_done = jnp.logical_or(
            is_done,
            jnp.all(state.types[1:] == 0),  # All but first tree are removed
        )

        # Update merge order
        merge_order = state.merge_order.at[left].set(state.time)

        return (
            EnvState(
                sequences=sequences,
                tree=state.tree,  # We don't update it right now
                types=types,
                merge_order=merge_order,
                is_done=is_done,
                time=time,
            ),
            is_done,
            {},
        )

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.ArrayTree:
        """Convert state to observation"""
        # Convert sequences to one-hot encoding
        sequences = (
            state.sequences[..., None] & (1 << jnp.arange(self.num_characters))
        ) > 0
        return {
            "sequences": sequences.astype(jnp.float32),
            "type": state.types,
            "tree": state.tree,
            "mask": self.get_invalid_mask(state, env_params).astype(jnp.float32),
        }

    def get_invalid_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Returns mask of invalid actions"""
        max_actions = self.num_nodes * (self.num_nodes - 1) // 2
        mask = jnp.ones((state.is_done.shape[0], max_actions), dtype=jnp.bool_)

        # Mask out actions involving removed trees (type == 0)
        mask = jnp.logical_and(mask, state.types[..., self.lefts] != 0)
        mask = jnp.logical_and(mask, state.types[..., self.rights] != 0)

        return mask

    def _single_backward_transition(
        self, state: EnvState, backward_action: TAction, env_params: EnvParams
    ) -> Tuple[EnvState, chex.Array, Dict[Any, Any]]:
        """Environment-specific step backward transition"""
        # TODO: implement
        raise NotImplementedError
        is_done = state.is_done
        time = state.time

        def get_state_done() -> EnvState:
            return state

        def get_state_not_done() -> EnvState:
            # Find the most recently merged tree (type == 2)
            merged_indices = jnp.where(state.types == 2)[0]
            last_merged = merged_indices[-1]

            # The backward action tells us which tree to split off
            split_tree = backward_action

            # Update sequences
            sequences = state.sequences.at[split_tree].set(state.sequences[last_merged])
            sequences = sequences.at[last_merged].set(0)

            # Update types
            types = state.types.at[last_merged].set(1)  # Back to leaf
            types = types.at[split_tree].set(1)  # New leaf

            # Update tree structure
            tree = state.tree.at[last_merged].set(-2)  # No longer internal node
            tree = tree.at[split_tree].set(split_tree)  # Restore leaf node

            # Check if we're at initial state
            is_done = jnp.all(types == 1)  # All nodes are leaves

            return EnvState(
                sequences=sequences,
                tree=tree,
                types=types,
                is_done=is_done,
                time=time + 1,
            )

        prev_state = jax.lax.cond(is_done, get_state_done, get_state_not_done)
        return prev_state, prev_state.is_done, {}

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns backward action given forward transition"""
        # TODO: implement
        raise NotImplementedError
        # For phylogenetic trees, the backward action is the index of the right tree
        # that was merged in the forward action
        return self.rights[forward_action]

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns forward action given backward transition"""
        # Find the action that merges the trees
        merged_indices = jnp.where(state.types == 2)[0]
        last_merged = merged_indices[-1]

        # Find the action index that corresponds to merging these trees
        action_idx = jnp.where(
            (self.lefts == last_merged) & (self.rights == backward_action)
        )[0][0]

        return action_idx

    def get_invalid_backward_mask(
        self, state: EnvState, env_params: EnvParams
    ) -> chex.Array:
        """Returns mask of invalid backward actions"""
        # TODO: implement
        raise NotImplementedError
        mask = jnp.zeros((state.is_done.shape[0], self.num_nodes), dtype=jnp.bool_)

        # Find the most recently merged tree by checking merge_order
        merged_mask = state.types == 2
        last_merged = jnp.where(
            merged_mask, state.merge_order, -jnp.ones_like(state.merge_order)
        ).argmax()

        # Can split into any position that's currently empty (type == 0)
        mask = state.types == 0
        # Except the position of the merged tree itself
        mask = mask.at[last_merged].set(False)

        return mask

    @property
    def max_steps_in_episode(self) -> int:
        """Maximum number of steps in an episode"""
        return self.num_nodes - 1

    @property
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment"""
        num_actions = self.num_nodes * (self.num_nodes - 1) // 2
        return spaces.Discrete(num_actions + 1)  # +1 for the stop action

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """Backward action space of the environment"""
        return spaces.Discrete(self.num_nodes)
