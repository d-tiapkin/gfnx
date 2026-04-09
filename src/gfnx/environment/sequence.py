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
    TBackwardAction,
    TDone,
)


@chex.dataclass(frozen=True)
class EnvState(BaseEnvState):
    tokens: Int[Array, " max_length"]
    is_terminal: Bool[Array, ""]
    is_initial: Bool[Array, ""]
    is_pad: Bool[Array, ""]

    @classmethod
    def from_tokens(cls, tokens: Int[Array, " max_length"]) -> "EnvState":
        """Create a single EnvState from tokens."""
        return cls(
            tokens=tokens,
            is_terminal=jnp.bool_(False),
            is_initial=jnp.bool_(False),
            is_pad=jnp.bool_(False),
        )


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    max_length: int
    nchar: int
    ntoken: int

    bos_token: int
    eos_token: int
    pad_token: int


class SequenceEnvironment(BaseEnvironment[EnvState, EnvParams]):
    """
    Class for sequence environments with a fixed length.
    All methods operate on a single state (no batch dim).
    """

    def __init__(
        self,
        max_length: int,
        nchar: int,
        ntoken: int,
        *,
        bos_token: int,
        eos_token: int,
        pad_token: int,
    ) -> None:
        self.max_length = max_length
        self.nchar = nchar
        self.ntoken = ntoken

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

    def get_init_state(self) -> EnvState:
        tokens = jnp.full(
            shape=(self.max_length,),
            fill_value=self.pad_token,
            dtype=jnp.int32,
        )
        return EnvState.from_tokens(tokens).replace(is_initial=jnp.bool_(True))

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        return EnvParams(
            max_length=self.max_length,
            nchar=self.nchar,
            ntoken=self.ntoken,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
        )

    @property
    def max_steps_in_episode(self) -> int:
        return self.max_length

    def _transition(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[EnvState, TDone, dict[Any, Any]]:
        raise NotImplementedError

    def _backward_transition(
        self,
        state: EnvState,
        backward_action: TBackwardAction,
        env_params: EnvParams,
    ) -> tuple[EnvState, TDone, dict[Any, Any]]:
        raise NotImplementedError

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Returns observation: BOS token prepended to tokens (single state)."""
        return jnp.concat(
            [
                jnp.full((1,), fill_value=self.bos_token, dtype=state.tokens.dtype),
                state.tokens,
            ],
            axis=-1,
        )

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        raise NotImplementedError

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        raise NotImplementedError

    def get_invalid_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        raise NotImplementedError

    def get_invalid_backward_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def action_space(self) -> spaces.Discrete:
        raise NotImplementedError

    @property
    def backward_action_space(self) -> spaces.Discrete:
        raise NotImplementedError

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=self.ntoken,
            shape=(self.max_length + 1,),  # +1 for BOS token
            dtype=jnp.int32,
        )

    @property
    def state_space(self) -> spaces.Dict:
        return spaces.Dict({
            "token": spaces.Box(
                low=0,
                high=self.ntoken,
                shape=(self.max_length,),
                dtype=jnp.int32,
            ),
            "is_done": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
        })


class FixedAutoregressiveSequenceEnvironment(SequenceEnvironment):
    """Sequence environment with fixed length and autoregressive generation."""

    def _transition(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[EnvState, TDone, dict[Any, Any]]:
        num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
        pos_to_update = self.max_length - num_pad
        next_tokens = state.tokens.at[pos_to_update].set(action)
        is_done = jnp.all(next_tokens != self.pad_token)
        next_active = EnvState(
            tokens=next_tokens,
            is_terminal=is_done,
            is_initial=False,
            is_pad=False,
        )
        next_pad = state.replace(is_pad=True)
        next_state = jax.tree.map(
            lambda p, a: jnp.where(state.is_terminal, p, a), next_pad, next_active
        )
        return next_state, next_state.is_terminal, {}

    def _backward_transition(
        self, state: EnvState, backward_action: TBackwardAction, env_params: EnvParams
    ) -> tuple[EnvState, TDone, dict[Any, Any]]:
        num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
        last_position = self.max_length - num_pad
        prev_tokens = state.tokens.at[last_position - 1].set(self.pad_token)
        is_initial = jnp.all(prev_tokens == self.pad_token)
        non_initial = state.replace(
            tokens=prev_tokens,
            is_terminal=False,
            is_initial=is_initial,
        )
        init_pad = state.replace(is_pad=True)
        prev_state = jax.tree.map(
            lambda p, n: jnp.where(state.is_initial, p, n), init_pad, non_initial
        )
        return prev_state, prev_state.is_initial, {}

    def get_backward_action(
        self,
        _state: EnvState,
        forward_action: chex.Array,
        _next_state: EnvState,
        _env_params: EnvParams,
    ) -> chex.Array:
        return jnp.zeros((), dtype=forward_action.dtype)

    def get_forward_action(
        self,
        state: EnvState,
        _backward_action: chex.Array,
        _prev_state: EnvState,
        _env_params: EnvParams,
    ) -> chex.Array:
        num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
        last_position = self.max_length - num_pad
        action = state.tokens[last_position - 1]
        return jnp.clip(action, min=0, max=self.action_space.n - 1)

    def get_invalid_mask(self, _state: EnvState, _env_params: EnvParams) -> chex.Array:
        return jnp.zeros((self.nchar,), dtype=jnp.bool)

    def get_invalid_backward_mask(self, _state: EnvState, _env_params: EnvParams) -> chex.Array:
        return jnp.zeros((1,), dtype=jnp.bool)

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.nchar)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(1)


class FixedPrependAppendSequenceEnvironment(SequenceEnvironment):
    """Sequence environment with fixed length and prepend-append generation."""

    def _transition(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[EnvState, TDone, dict[Any, Any]]:
        def get_next_tokens_prepend(state: EnvState, action: TAction) -> chex.Array:
            next_tokens = jax.lax.dynamic_update_slice(state.tokens, state.tokens[:-1], (1,))
            return next_tokens.at[0].set(action)

        def get_next_tokens_append(state: EnvState, action: TAction) -> chex.Array:
            num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
            last_position = self.max_length - num_pad
            return state.tokens.at[last_position].set(action - self.nchar)

        next_tokens = jax.lax.cond(
            action < self.nchar,
            get_next_tokens_prepend,
            get_next_tokens_append,
            state,
            action,
        )
        is_done = jnp.all(next_tokens != self.pad_token)
        next_active = EnvState(
            tokens=next_tokens,
            is_terminal=is_done,
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
        backward_action: TBackwardAction,
        env_params: EnvParams,
    ) -> tuple[EnvState, TDone, dict[Any, Any]]:
        num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
        last_position = self.max_length - num_pad

        def get_prev_tokens_prepend(state: EnvState) -> chex.Array:
            prev_tokens = jax.lax.dynamic_update_slice(state.tokens, state.tokens[1:], (0,))
            return prev_tokens.at[last_position - 1].set(self.pad_token)

        def get_prev_tokens_append(state: EnvState) -> chex.Array:
            return state.tokens.at[last_position - 1].set(self.pad_token)

        prev_tokens = jax.lax.cond(
            backward_action == 0,
            get_prev_tokens_prepend,
            get_prev_tokens_append,
            state,
        )
        is_initial = jnp.all(prev_tokens == self.pad_token)
        non_initial = state.replace(
            tokens=prev_tokens,
            is_terminal=False,
            is_initial=is_initial,
        )
        init_pad = state.replace(is_pad=True)
        prev_state = jax.tree.map(
            lambda p, n: jnp.where(state.is_initial, p, n), init_pad, non_initial
        )
        return prev_state, prev_state.is_initial, {}

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        return jnp.where(forward_action < self.nchar, 0, 1)

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
        last_position = self.max_length - num_pad
        removed_token = state.tokens[last_position - 1]
        action = jnp.where(backward_action == 0, state.tokens[0], self.nchar + removed_token)
        return jnp.clip(action, min=0, max=self.action_space.n - 1)

    def get_invalid_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        return jnp.zeros((2 * self.nchar,), dtype=jnp.bool)

    def get_invalid_backward_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        return jnp.zeros((2,), dtype=jnp.bool)

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(2 * self.nchar)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(2)


class AutoregressiveSequenceEnvironment(SequenceEnvironment):
    """Sequence environment with non-fixed length and autoregressive generation."""

    def __init__(
        self,
        max_length: int,
        nchar: int,
        ntoken: int,
        *,
        bos_token: int,
        eos_token: int,
        pad_token: int,
    ):
        super().__init__(
            max_length,
            nchar,
            ntoken,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
        )
        self.stop_action = nchar

    def _transition(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[EnvState, TDone, dict[Any, Any]]:
        num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
        pos_to_update = self.max_length - num_pad
        action_to_token = jnp.where(action != self.stop_action, action, self.eos_token)
        next_tokens = state.tokens.at[pos_to_update].set(action_to_token)
        is_done = jnp.logical_or(
            jnp.all(next_tokens != self.pad_token),
            action == self.stop_action,
        )
        next_active = EnvState(
            tokens=next_tokens,
            is_terminal=is_done,
            is_initial=False,
            is_pad=False,
        )
        next_pad = state.replace(is_pad=True)
        next_state = jax.tree.map(
            lambda p, a: jnp.where(state.is_terminal, p, a), next_pad, next_active
        )
        return next_state, next_state.is_terminal, {}

    def _backward_transition(
        self, state: EnvState, backward_action: TBackwardAction, env_params: EnvParams
    ) -> tuple[EnvState, TDone, dict[Any, Any]]:
        num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
        last_pos = self.max_length - num_pad
        prev_tokens = state.tokens.at[last_pos - 1].set(self.pad_token)
        is_initial = jnp.all(prev_tokens == self.pad_token)
        non_initial = state.replace(
            tokens=prev_tokens,
            is_terminal=False,
            is_initial=is_initial,
        )
        init_pad = state.replace(is_pad=True)
        prev_state = jax.tree.map(
            lambda p, n: jnp.where(state.is_initial, p, n), init_pad, non_initial
        )
        return prev_state, prev_state.is_initial, {}

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        return jnp.zeros((), dtype=forward_action.dtype)

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
        last_pos = self.max_length - num_pad
        all_actions = jnp.where(state.tokens != self.eos_token, state.tokens, self.stop_action)
        action = all_actions[last_pos - 1]
        return jnp.clip(action, min=0, max=self.action_space.n - 1)

    def get_invalid_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        return jnp.zeros((self.nchar + 1,), dtype=jnp.bool)

    def get_invalid_backward_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        return jnp.zeros((1,), dtype=jnp.bool)

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.nchar + 1)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(1)


class NonAutoregressiveSequenceEnvironment(SequenceEnvironment):
    """Sequence environment with fixed length and non-autoregressive generation."""

    def _transition(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[EnvState, TDone, dict[Any, Any]]:
        pos, word = jnp.unravel_index(action, (self.max_length, self.nchar))
        next_tokens = state.tokens.at[pos].set(word)
        is_done = jnp.all(next_tokens != self.pad_token)
        next_active = EnvState(
            tokens=next_tokens,
            is_terminal=is_done,
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
        backward_action: TBackwardAction,
        env_params: EnvParams,
    ) -> tuple[EnvState, TDone, dict[Any, Any]]:
        prev_tokens = state.tokens.at[backward_action].set(self.pad_token)
        is_initial = jnp.all(prev_tokens == self.pad_token)
        non_initial = state.replace(
            tokens=prev_tokens,
            is_terminal=False,
            is_initial=is_initial,
        )
        init_pad = state.replace(is_pad=True)
        prev_state = jax.tree.map(
            lambda p, n: jnp.where(state.is_initial, p, n), init_pad, non_initial
        )
        return prev_state, prev_state.is_initial, {}

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        pos, _ = jnp.unravel_index(forward_action, (self.max_length, self.nchar))
        return pos

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        word = state.tokens[backward_action]
        return jnp.ravel_multi_index(
            (backward_action, word), (self.max_length, self.nchar), mode="clip"
        )

    def get_invalid_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        pos_mask = state.tokens != self.pad_token  # [max_length]
        invalid_mask_2d = jnp.repeat(jnp.expand_dims(pos_mask, axis=1), repeats=self.nchar, axis=1)
        invalid_mask_flat = invalid_mask_2d.reshape(-1)
        all_filled = jnp.all(pos_mask)
        return jnp.where(all_filled, jnp.zeros_like(invalid_mask_flat), invalid_mask_flat)

    def get_invalid_backward_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        pos_mask = state.tokens == self.pad_token  # [max_length]
        all_filled = jnp.all(pos_mask)
        return jnp.where(all_filled, jnp.zeros_like(pos_mask), pos_mask)

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.max_length * self.nchar)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.max_length)
