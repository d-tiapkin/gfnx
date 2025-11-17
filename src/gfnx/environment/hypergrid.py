from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

import gfnx
import gfnx.spaces as spaces

from ..base import BaseEnvParams, BaseEnvState, BaseVecEnvironment


@chex.dataclass(frozen=True)
class EnvState(BaseEnvState):
    state: chex.Array  # [B, dim]
    time: chex.Array  # [B]
    is_done: chex.Array  # [B]


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    dim: int = 4
    side: int = 20

    reward_params: Any = None


class HypergridEnvironment(BaseVecEnvironment[EnvState, EnvParams]):
    """
    Hypergrid environment
    """

    def __init__(
        self, reward_module: gfnx.TRewardModule, dim: int = 4, side: int = 20
    ) -> None:
        super().__init__(reward_module)
        self.dim = dim
        self.side = side

        self.stop_action = self.dim  # Stop action id

    def get_init_state(self, num_envs: int) -> EnvState:
        return EnvState(
            state=jnp.zeros((num_envs, self.dim), dtype=jnp.int32),
            is_done=jnp.zeros((num_envs,), dtype=jnp.bool),
            time=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        dummy_state = self.get_init_state(1)
        reward_params = self.reward_module.init(rng_key, dummy_state)
        return EnvParams(dim=self.dim, side=self.side, reward_params=reward_params)

    @property
    def max_steps_in_episode(self) -> int:
        return self.dim * self.side

    def _single_transition(
        self,
        state: EnvState,
        action: gfnx.TAction,
        env_params: EnvParams,
    ) -> Tuple[EnvState, gfnx.TDone, Dict[Any, Any]]:
        is_done = state.is_done  # bool
        time = state.time

        def get_state_done() -> EnvState:
            return state

        def get_state_finished() -> EnvState:
            is_done = True
            next_state = EnvState(state=state.state, is_done=is_done, time=time + 1)
            return next_state

        def get_state_inter() -> EnvState:
            is_done = False
            next_state = EnvState(
                state=state.state.at[action].add(1), is_done=is_done, time=time + 1
            )
            return next_state

        def get_state_not_done() -> EnvState:
            is_finished = jnp.logical_or(
                action == self.dim, state.state[action] >= self.side - 1
            )
            return jax.lax.cond(is_finished, get_state_finished, get_state_inter)

        next_state = jax.lax.cond(is_done, get_state_done, get_state_not_done)

        return next_state, next_state.is_done, {}

    def _single_backward_transition(
        self, state: EnvState, backward_action: chex.Array, env_params: EnvParams
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        """
        Environment-specific step backward transition. Rewards always zero!
        """
        is_done = state.is_done
        time = state.time

        def get_state_done() -> EnvState:
            return state

        def get_state_not_done() -> EnvState:
            prev_inner_state = state.state.at[backward_action].add(-1)
            is_done = jnp.all(prev_inner_state == 0)
            return EnvState(state=prev_inner_state, is_done=is_done, time=time + 1)

        prev_state = jax.lax.cond(is_done, get_state_done, get_state_not_done)
        return prev_state, prev_state.is_done, {}

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Applies observation function to state."""

        # One-hot encoding
        def single_get_obs(state: EnvState) -> chex.Array:
            # TODO: improve it with jax one-hot-encoding
            state_ohe = jnp.zeros((self.dim * self.side), dtype=jnp.float32)
            indices = jnp.arange(self.dim) * self.side + state.state
            return state_ohe.at[indices].set(1.0)

        return jax.vmap(single_get_obs)(state)

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        """Returns backward action given the forward transition."""
        return jnp.where(
            forward_action >= self.backward_action_space.n, 0, forward_action
        )

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns forward action given the backward transition."""
        return jnp.where(state.is_done, self.stop_action, backward_action)

    def get_invalid_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Return mask of invalid actions"""

        def single_get_invalid_mask(state: EnvState) -> chex.Array:
            augmeneted_state = jnp.concat([state.state, jnp.zeros((1,))], axis=-1)
            return augmeneted_state == self.side - 1

        return jax.vmap(single_get_invalid_mask)(state)

    def get_invalid_backward_mask(
        self, state: EnvState, params: EnvParams
    ) -> chex.Array:
        """Returns mask of invalid backward actions."""

        def single_get_invalid_backward_mask(state: EnvState) -> chex.Array:
            return jax.lax.cond(
                state.is_done,
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

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            low=jnp.zeros(self.dim * self.side),
            high=jnp.ones(self.dim * self.side),
            shape=(self.dim * self.side,),
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
            "state": spaces.Box(
                low=0.0, high=self.side, shape=(self.dim,), dtype=jnp.int32
            ),
            "is_done": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
        })
