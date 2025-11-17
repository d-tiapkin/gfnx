from itertools import product
from math import prod
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from .. import spaces
from ..base import (
    BaseEnvParams,
    BaseEnvState,
    BaseVecEnvironment,
    TAction,
    TDone,
    TRewardModule,
)


@chex.dataclass(frozen=True)
class EnvState(BaseEnvState):
    state: Int[Array, " batch_size dim"]
    is_terminal: Bool[Array, " batch_size"]
    is_initial: Bool[Array, " batch_size"]
    is_pad: Bool[Array, " batch_size"]


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    dim: int = 4
    side: int = 20

    reward_params: Any = None


class HypergridEnvironment(BaseVecEnvironment[EnvState, EnvParams]):
    """
    Hypergrid environment
    """

    def __init__(self, reward_module: TRewardModule, dim: int = 4, side: int = 20) -> None:
        super().__init__(reward_module)
        self.dim = dim
        self.side = side

        self.stop_action = self.dim  # Stop action id

    def get_init_state(self, num_envs: int) -> EnvState:
        return EnvState(
            state=jnp.zeros((num_envs, self.dim), dtype=jnp.int32),
            is_terminal=jnp.zeros((num_envs,), dtype=jnp.bool),
            is_initial=jnp.ones((num_envs,), dtype=jnp.bool),
            is_pad=jnp.zeros((num_envs,), dtype=jnp.bool),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        dummy_state = self.get_init_state(1)
        reward_params = self.reward_module.init(rng_key, dummy_state)
        return EnvParams(dim=self.dim, side=self.side, reward_params=reward_params)

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
        # Safe flattening under JIT; avoid raise-mode bounds checks
        return jnp.ravel_multi_index(
            state.state.astype(jnp.int32),
            dims=(self.side,) * self.dim,
            mode="clip",
        )

    def _single_transition(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> Tuple[EnvState, TDone, Dict[Any, Any]]:
        is_terminal = state.is_terminal  # bool

        def get_state_terminal() -> EnvState:
            return state.replace(is_pad=True)

        def get_state_finished() -> EnvState:
            return state.replace(is_terminal=True, is_initial=False)

        def get_state_inter() -> EnvState:
            return state.replace(
                state=state.state.at[action].add(1),
                is_terminal=False,
                is_initial=False,
            )

        def get_state_nonterminal() -> EnvState:
            done = jnp.logical_or(
                action == self.stop_action,
                state.state[action] >= self.side - 1,
            )
            return jax.lax.cond(done, get_state_finished, get_state_inter)

        next_state = jax.lax.cond(is_terminal, get_state_terminal, get_state_nonterminal)

        return next_state, next_state.is_terminal, {}

    def _single_backward_transition(
        self,
        state: EnvState,
        backward_action: chex.Array,
        env_params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        """
        Environment-specific step backward transition. Rewards always zero!
        """
        is_initial = state.is_initial

        def get_state_initial() -> EnvState:
            return state.replace(is_pad=True)

        def undo_stop() -> EnvState:
            # First backward step from a terminal state: just undo the stop action
            return EnvState(
                state=state.state,
                is_terminal=False,
                is_initial=jnp.all(state.state == 0),
                is_pad=False,
            )

        def dec_dim() -> EnvState:
            # Standard backward step on a non-terminal state: decrement the chosen dimension
            prev_inner_state = state.state.at[backward_action].add(-1)
            return EnvState(
                state=prev_inner_state,
                is_terminal=False,
                is_initial=jnp.all(prev_inner_state == 0),
                is_pad=False,
            )

        def get_state_non_initial() -> EnvState:
            return jax.lax.cond(state.is_terminal, undo_stop, dec_dim)

        prev_state = jax.lax.cond(is_initial, get_state_initial, get_state_non_initial)
        return prev_state, prev_state.is_initial, {}

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Applies observation function to state."""

        def single_get_obs(state: EnvState) -> chex.Array:
            state_ohe = jax.nn.one_hot(state.state, self.side, dtype=jnp.float32)
            return jnp.reshape(state_ohe, (self.dim * self.side,))

        return jax.vmap(single_get_obs)(state)

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        """Returns backward action given the forward transition."""
        return jnp.where(forward_action >= self.backward_action_space.n, 0, forward_action)

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns forward action given the backward transition."""
        return jnp.where(state.is_terminal, self.stop_action, backward_action)

    def get_invalid_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Return mask of invalid actions"""

        def single_get_invalid_mask(state: EnvState) -> chex.Array:
            augmeneted_state = jnp.concat([state.state, jnp.zeros((1,))], axis=-1)
            return augmeneted_state == self.side - 1

        return jax.vmap(single_get_invalid_mask)(state)

    def get_invalid_backward_mask(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Returns mask of invalid backward actions."""

        def single_get_invalid_backward_mask(state: EnvState) -> chex.Array:
            return jax.lax.cond(
                state.is_terminal,
                # Set only a fixed zero-action as a valid one
                lambda x: jnp.ones_like(x, dtype=jnp.bool).at[0].set(False),
                lambda x: x == 0,
                state.state,
            )

        return jax.vmap(single_get_invalid_backward_mask)(state)

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

    def _get_states_rewards(self, env_params: EnvParams) -> chex.Array:
        """
        Returns the true distribution of rewards for all states in the hypergrid.
        """
        rewards = jnp.zeros((self.side,) * self.dim, dtype=jnp.float32)

        def update_rewards(idx: int, rewards: chex.Array):
            state = jnp.unravel_index(idx, shape=rewards.shape)  # Unpack index to state
            env_state = EnvState(
                state=jnp.array(state),
                is_terminal=True,
                is_initial=False,
                is_pad=False,
            )
            batched_env_state = jax.tree.map(lambda x: jnp.expand_dims(x, 0), env_state)
            reward = self.reward_module.reward(batched_env_state, env_params)
            return rewards.at[state].set(reward[0])

        rewards = jax.lax.fori_loop(0, self.side**self.dim, update_rewards, rewards)
        return rewards

    def get_true_distribution(self, env_params: EnvParams) -> chex.Array:
        """
        Returns the true distribution of rewards for all states in the hypergrid.
        """
        rewards = self._get_states_rewards(env_params)
        return rewards / rewards.sum()

    def get_empirical_distribution(self, states: EnvState, env_params: EnvParams) -> chex.Array:
        """
        Extracts the empirical distribution from the given states.
        """
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
        """Whether this environment supports mean reward tractability."""
        return True

    def get_mean_reward(self, env_params: EnvParams) -> float:
        """
        Returns the mean reward for the hypergrid environment.
        The mean reward is computed as the sum of rewards divided by the number of states.
        """
        rewards = self._get_states_rewards(env_params)
        return jnp.pow(rewards, 2).sum() / rewards.sum()

    @property
    def is_normalizing_constant_tractable(self) -> bool:
        """Whether this environment supports tractable normalizing constant."""
        return True

    def get_normalizing_constant(self, env_params: EnvParams) -> float:
        """
        Returns the normalizing constant for the hypergrid environment.
        The normalizing constant is computed as the sum of rewards.
        """
        rewards = self._get_states_rewards(env_params)
        return rewards.sum()

    @property
    def is_ground_truth_sampling_tractable(self) -> bool:
        """Whether this environment supports tractable sampling from the GT distribution."""
        return True

    def get_ground_truth_sampling(
        self, rng_key: chex.PRNGKey, batch_size: int, env_params: EnvParams
    ) -> EnvState:
        """
        Returns a batch of states sampled from the ground truth distribution.

        The ground truth distribution is proportional to the rewards of terminal states.

        Args:
            rng_key: JAX random key for sampling.
            batch_size: Number of samples to generate.
            env_params: Environment parameters.

        Returns:
            A batch of ground-truth sampled states.
        """
        true_distribution = self.get_true_distribution(env_params)
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
