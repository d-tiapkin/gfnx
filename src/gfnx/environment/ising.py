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
    time: Int[Array, " batch_size"]
    is_terminal: Bool[Array, " batch_size"]
    is_initial: Bool[Array, " batch_size"]
    is_pad: Bool[Array, " batch_size"]


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    dim: int = 10

    reward_params: Any = None


class IsingEnvironment(BaseVecEnvironment[EnvState, EnvParams]):
    """Ising environment for discrete energy-based models.

    This environment is based on the paper https://arxiv.org/pdf/2202.01361.pdf.

    The states are represented as 1d tensors of length `ndim` with values in
    `{-1, 0, 1}`. `s0` is empty (represented as -1), so `s0=[-1, -1, ..., -1]`.
    An action corresponds to replacing a -1 with a 0 or a 1.
    Action `i` in `[0, ndim - 1]` corresponds to replacing `s[i]` with 0.
    Action `i` in `[ndim, 2 * ndim - 1]` corresponds to replacing `s[i - ndim]` with 1.
    NOTE: There is no exit action; the environment terminates when all spins are set.
    """

    def __init__(self, reward_module: TRewardModule, dim: int = 10) -> None:
        super().__init__(reward_module)
        self.dim = dim

    def get_init_state(self, num_envs: int) -> EnvState:
        return EnvState(
            # TODO: use int4 instead of int8
            state=jnp.full((num_envs, self.dim), -1, dtype=jnp.int8),
            time=jnp.zeros((num_envs,), dtype=jnp.int32),
            is_terminal=jnp.zeros((num_envs,), dtype=jnp.bool),
            is_initial=jnp.ones((num_envs,), dtype=jnp.bool),
            is_pad=jnp.zeros((num_envs,), dtype=jnp.bool),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        dummy_state = self.get_init_state(1)
        reward_params = self.reward_module.init(rng_key, dummy_state)
        return EnvParams(dim=self.dim, reward_params=reward_params)

    @property
    def max_steps_in_episode(self) -> int:
        return self.dim

    def _single_transition(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> Tuple[EnvState, TDone, Dict[Any, Any]]:
        is_terminal = state.is_terminal
        time = state.time

        def get_state_terminal() -> EnvState:
            return state.replace(is_pad=True)

        def get_state_nonterminal() -> EnvState:
            spin_index = jnp.mod(action, self.dim)
            spin_value = jnp.asarray(action // self.dim, dtype=jnp.int8)
            new_state = state.state.at[spin_index].set(spin_value)
            return state.replace(
                state=new_state,
                time=time + 1,
                is_terminal=jnp.all(new_state != -1),
                is_initial=False,
                is_pad=False,
            )

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
        time = state.time

        def get_state_initial() -> EnvState:
            return state.replace(is_pad=True)

        def get_state_non_initial() -> EnvState:
            prev_state = state.state.at[backward_action].set(-1)
            return EnvState(
                state=prev_state,
                time=time - 1,
                is_terminal=False,
                is_initial=jnp.all(prev_state == -1),
                is_pad=False,
            )

        prev_state = jax.lax.cond(is_initial, get_state_initial, get_state_non_initial)
        return prev_state, prev_state.is_initial, {}

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        return state.state

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        return jnp.mod(forward_action, self.dim)

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        batch_size = state.state.shape[0]
        return backward_action + self.dim * state.state[jnp.arange(batch_size), backward_action]

    def get_invalid_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Get mask for a particular state to perform a forward action.

        An action is invalid if there is already a spin (0 or 1) at the index
        of the action.

        Mask is a concatenation of two masks:
        - mask for invalid forward actions for 0-spin
        - mask for invalid forward actions for 1-spin (identical to 0-spin)
        """
        mask = state.state != -1
        return jnp.concatenate([mask, mask], axis=-1)

    def get_invalid_backward_mask(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Get mask for a particular state to perform a backward action.

        An action is invalid if there is no spin at the index of the action.
        """
        mask = state.state == -1
        return mask

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
