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
    TAction,
    TDone,
)


@chex.dataclass(frozen=True)
class EnvState(BaseEnvState):
    state: Int[Array, " dim"]
    time: Int[Array, ""]
    is_terminal: Bool[Array, ""]
    is_initial: Bool[Array, ""]
    is_pad: Bool[Array, ""]


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    dim: int = 10


class IsingEnvironment(BaseEnvironment[EnvState, EnvParams]):
    """Ising environment for discrete energy-based models.

    This environment is based on the paper https://arxiv.org/pdf/2202.01361.pdf.

    The states are represented as 1d tensors of length `ndim` with values in
    `{-1, 0, 1}`. `s0` is empty (represented as -1), so `s0=[-1, -1, ..., -1]`.
    An action corresponds to replacing a -1 with a 0 or a 1.
    Action `i` in `[0, ndim - 1]` corresponds to replacing `s[i]` with 0.
    Action `i` in `[ndim, 2 * ndim - 1]` corresponds to replacing `s[i - ndim]` with 1.
    NOTE: There is no exit action; the environment terminates when all spins are set.
    """

    def __init__(self, dim: int = 10) -> None:
        self.dim = dim

    def get_init_state(self) -> EnvState:
        return EnvState(
            state=jnp.full((self.dim,), -1, dtype=jnp.int8),
            time=jnp.int32(0),
            is_terminal=jnp.bool_(False),
            is_initial=jnp.bool_(True),
            is_pad=jnp.bool_(False),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        return EnvParams(dim=self.dim)

    @property
    def max_steps_in_episode(self) -> int:
        return self.dim

    def _transition(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[EnvState, TDone, dict[Any, Any]]:
        spin_index = jnp.mod(action, self.dim)
        spin_value = jnp.asarray(action // self.dim, dtype=jnp.int8)
        new_state_arr = state.state.at[spin_index].set(spin_value)
        next_active = state.replace(
            state=new_state_arr,
            time=state.time + 1,
            is_terminal=jnp.all(new_state_arr != -1),
            is_initial=False,
            is_pad=False,
        )
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
        prev_state_arr = state.state.at[backward_action].set(-1)
        non_initial = EnvState(
            state=prev_state_arr,
            time=state.time - 1,
            is_terminal=False,
            is_initial=jnp.all(prev_state_arr == -1),
            is_pad=False,
        )
        init_pad = state.replace(is_pad=True)
        prev_state = jax.tree.map(
            lambda p, n: jnp.where(state.is_initial, p, n), init_pad, non_initial
        )
        return prev_state, prev_state.is_initial, {}

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Returns the lattice partial assignment of spins (single state)."""
        return state.state

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        """Returns backward action given the forward transition (single)."""
        return jnp.mod(forward_action, self.dim)

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns forward action given the backward transition (single)."""
        return backward_action + self.dim * state.state[backward_action].astype(jnp.int32)

    def get_invalid_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Returns mask of invalid forward actions for a single state. [2*dim]"""
        mask = state.state != -1
        return jnp.concatenate([mask, mask], axis=-1)

    def get_invalid_backward_mask(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Returns mask of invalid backward actions for a single state. [dim]"""
        return state.state == -1

    @property
    def name(self) -> str:
        """Environment name."""
        return f"Ising-{self.dim}-v0"

    @property
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2 * self.dim)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """Backward action space of the environment."""
        return spaces.Discrete(self.dim)

    @property
    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            low=jnp.full(self.dim, -1),
            high=jnp.full(self.dim, 1),
            shape=(self.dim,),
        )

    @property
    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
            "state": spaces.Box(
                low=jnp.full(self.dim, -1),
                high=jnp.full(self.dim, 1),
                shape=(self.dim,),
            ),
            "time": spaces.Discrete(self.max_steps_in_episode),
            "is_terminal": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
            "is_initial": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
            "is_pad": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
        })
