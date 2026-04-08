from itertools import product
from math import prod
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from .. import spaces
from ..base import (
    BaseEnvironment,
    BaseEnvParams,
    BaseEnvState,
    BaseRewardModule,
    TAction,
    TDone,
    TRewardParams,
)


@chex.dataclass(frozen=True)
class EnvState(BaseEnvState):
    state: Int[Array, " dim"]
    is_terminal: Bool[Array, ""]
    is_initial: Bool[Array, ""]
    is_pad: Bool[Array, ""]


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    dim: int = 4
    side: int = 20


class HypergridEnvironment(BaseEnvironment[EnvState, EnvParams]):
    """
    Hypergrid environment
    """

    def __init__(self, dim: int = 4, side: int = 20) -> None:
        self.dim = dim
        self.side = side

        self.stop_action = self.dim  # Stop action id

    def get_init_state(self) -> EnvState:
        return EnvState(
            state=jnp.zeros((self.dim,), dtype=jnp.int32),
            is_terminal=jnp.bool_(False),
            is_initial=jnp.bool_(True),
            is_pad=jnp.bool_(False),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        return EnvParams(dim=self.dim, side=self.side)

    @property
    def is_enumerable(self) -> bool:
        """Whether this environment supports enumerable operations."""
        return True

    @property
    def max_steps_in_episode(self) -> int:
        return self.dim * self.side

    def get_all_states(self, env_params: EnvParams) -> EnvState:
        """Returns all states in the environment in some order."""

        all_states_coords = jnp.array(list(product(range(self.side), repeat=self.dim)))
        num_states = all_states_coords.shape[0]
        is_initial = all_states_coords.sum(axis=1) == 0
        is_terminal = jnp.zeros(num_states, dtype=jnp.bool)
        is_pad = jnp.zeros(num_states, dtype=jnp.bool)

        return EnvState(
            state=all_states_coords,
            is_terminal=is_terminal,
            is_initial=is_initial,
            is_pad=is_pad,
        )

    def state_to_index(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        return jnp.ravel_multi_index(
            state.state.astype(jnp.int32),
            dims=(self.side,) * self.dim,
            mode="clip",
        )

    def _transition(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[EnvState, TDone, dict[Any, Any]]:
        # Compute the "active" next state (always, regardless of is_terminal)
        done = jnp.logical_or(
            action == self.stop_action,
            state.state[action] >= self.side - 1,
        )
        next_inter = state.replace(
            state=state.state.at[action].add(1),
            is_terminal=False,
            is_initial=False,
        )
        next_finished = state.replace(is_terminal=True, is_initial=False)
        next_active = jax.tree.map(lambda f, i: jnp.where(done, f, i), next_finished, next_inter)
        # If already terminal, freeze as pad instead
        next_pad = state.replace(is_pad=True)
        next_state = jax.tree.map(
            lambda p, a: jnp.where(state.is_terminal, p, a), next_pad, next_active
        )
        return next_state, next_state.is_terminal, {}

    def _backward_transition(
        self,
        state: EnvState,
        backward_action: chex.Array,
        env_params: EnvParams,
    ) -> tuple[EnvState, TDone, dict[Any, Any]]:
        # Undo stop: just un-set terminal flag
        undo_stop = EnvState(
            state=state.state,
            is_terminal=False,
            is_initial=jnp.all(state.state == 0),
            is_pad=False,
        )
        # Decrement chosen dimension
        prev_inner = state.state.at[backward_action].add(-1)
        dec_dim = EnvState(
            state=prev_inner,
            is_terminal=False,
            is_initial=jnp.all(prev_inner == 0),
            is_pad=False,
        )
        # Non-initial: pick undo_stop or dec_dim based on is_terminal
        non_initial = jax.tree.map(
            lambda u, d: jnp.where(state.is_terminal, u, d), undo_stop, dec_dim
        )
        # If already initial, freeze as pad
        init_pad = state.replace(is_pad=True)
        prev_state = jax.tree.map(
            lambda p, n: jnp.where(state.is_initial, p, n), init_pad, non_initial
        )
        return prev_state, prev_state.is_initial, {}

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Returns one-hot observation for a single state."""
        state_ohe = jax.nn.one_hot(state.state, self.side, dtype=jnp.float32)
        return jnp.reshape(state_ohe, (self.dim * self.side,))

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        """Returns backward action given the forward transition (single)."""
        return jnp.where(forward_action >= self.backward_action_space.n, 0, forward_action)

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns forward action given the backward transition (single)."""
        return jnp.where(state.is_terminal, self.stop_action, backward_action)

    def get_invalid_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Returns mask of invalid actions for a single state. [dim+1]"""
        augmented_state = jnp.concat([state.state, jnp.zeros((1,))], axis=-1)
        return augmented_state == self.side - 1

    def get_invalid_backward_mask(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Returns mask of invalid backward actions for a single state. [dim]"""
        return jax.lax.cond(
            state.is_terminal,
            lambda x: jnp.ones_like(x, dtype=jnp.bool).at[0].set(False),
            lambda x: x == 0,
            state.state,
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return f"HyperGrid-{self.side}**{self.dim}-v0"

    @property
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.dim + 1)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """Backward action space of the environment."""
        return spaces.Discrete(self.dim)

    @property
    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            low=jnp.zeros(self.dim * self.side),
            high=jnp.ones(self.dim * self.side),
            shape=(self.dim * self.side,),
        )

    @property
    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
            "state": spaces.Box(low=0.0, high=self.side, shape=(self.dim,), dtype=jnp.int32),
            "is_terminal": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
            "is_initial": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
            "is_pad": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
        })

    def _get_states_rewards(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> chex.Array:
        """Returns the reward for all states in the hypergrid."""
        rewards = jnp.zeros((self.side,) * self.dim, dtype=jnp.float32)

        def update_rewards(idx: int, rewards: chex.Array):
            state = jnp.unravel_index(idx, shape=rewards.shape)
            env_state = EnvState(
                state=jnp.array(state),
                is_terminal=jnp.bool_(True),
                is_initial=jnp.bool_(False),
                is_pad=jnp.bool_(False),
            )
            reward = reward_module.reward(env_state, reward_params)
            return rewards.at[state].set(reward)

        return jax.lax.fori_loop(0, self.side**self.dim, update_rewards, rewards)

    def get_true_distribution(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> chex.Array:
        """Returns the true distribution of rewards for all states."""
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return rewards / rewards.sum()

    def get_empirical_distribution(self, states: EnvState, env_params: EnvParams) -> chex.Array:
        """Extracts the empirical distribution from a batch of states."""
        dist_shape = (self.side,) * self.dim
        sample_idx = jax.vmap(lambda x: jnp.ravel_multi_index(x, dims=dist_shape, mode="clip"))(
            states.state
        )

        valid_mask = states.is_terminal.astype(jnp.float32)
        empirical_dist = jax.ops.segment_sum(valid_mask, sample_idx, num_segments=prod(dist_shape))
        empirical_dist = empirical_dist.reshape(dist_shape)
        empirical_dist /= empirical_dist.sum()
        return empirical_dist

    @property
    def is_mean_reward_tractable(self) -> bool:
        return True

    def get_mean_reward(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> float:
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return jnp.pow(rewards, 2).sum() / rewards.sum()

    @property
    def is_normalizing_constant_tractable(self) -> bool:
        return True

    def get_normalizing_constant(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> float:
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return rewards.sum()

    @property
    def is_ground_truth_sampling_tractable(self) -> bool:
        return True

    def get_ground_truth_sampling(
        self,
        rng_key: chex.PRNGKey,
        batch_size: int,
        env_params: EnvParams,
        reward_module: BaseRewardModule,
        reward_params: TRewardParams,
    ) -> EnvState:
        """Returns a batch of states sampled from the ground truth distribution."""
        true_distribution = self.get_true_distribution(env_params, reward_module, reward_params)
        flat_distribution = true_distribution.flatten()

        sampled_indices = jax.random.choice(
            rng_key,
            a=flat_distribution.size,
            shape=(batch_size,),
            p=flat_distribution,
        )

        sampled_coords_unstacked = jnp.unravel_index(
            sampled_indices, shape=true_distribution.shape
        )
        sampled_coords = jnp.stack(sampled_coords_unstacked, axis=1)

        return EnvState(
            state=sampled_coords,
            is_terminal=jnp.ones((batch_size,), dtype=jnp.bool),
            is_initial=jnp.zeros((batch_size,), dtype=jnp.bool),
            is_pad=jnp.zeros((batch_size,), dtype=jnp.bool),
        )
