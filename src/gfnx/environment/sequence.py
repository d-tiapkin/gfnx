"""Generic sequence-generation environments.

Four generation regimes share one state layout (a padded token buffer):

| Class | Regime | Forward actions | Backward actions |
| --- | --- | --- | --- |
| [`FixedAutoregressiveSequenceEnvironment`][gfnx.environment.sequence.FixedAutoregressiveSequenceEnvironment] | append, fixed length | `nchar` | 1 |
| [`AutoregressiveSequenceEnvironment`][gfnx.environment.sequence.AutoregressiveSequenceEnvironment] | append, variable length (stop action) | `nchar + 1` | 1 |
| [`FixedPrependAppendSequenceEnvironment`][gfnx.environment.sequence.FixedPrependAppendSequenceEnvironment] | prepend or append, fixed length | `2 * nchar` | 2 |
| [`NonAutoregressiveSequenceEnvironment`][gfnx.environment.sequence.NonAutoregressiveSequenceEnvironment] | fill any empty position, fixed length | `max_length * nchar` | `max_length` |

Concrete tasks subclass one of them: [`BitseqEnvironment`][gfnx.environment.bitseq.BitseqEnvironment]
(non-autoregressive), [`TFBind8Environment`][gfnx.environment.tfbind.TFBind8Environment] and
[`GFPEnvironment`][gfnx.environment.gfp.GFPEnvironment] (fixed autoregressive),
[`AMPEnvironment`][gfnx.environment.amp.AMPEnvironment] (variable-length autoregressive) and
[`QM9SmallEnvironment`][gfnx.environment.qm9_small.QM9SmallEnvironment] (prepend/append).
"""  # noqa: E501

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
    """State of a single sequence environment.

    Attributes:
        tokens: Token buffer of shape `[max_length]`. Filled positions hold character
            ids in `[0, nchar)` (or `eos_token` after a stop action); empty positions hold
            `pad_token`.
        is_terminal: `True` once the sequence is complete.
        is_initial: `True` for the empty sequence.
        is_pad: `True` for padding steps after termination.
    """

    tokens: Int[Array, " max_length"]
    is_terminal: Bool[Array, ""]
    is_initial: Bool[Array, ""]
    is_pad: Bool[Array, ""]

    @classmethod
    def from_tokens(cls, tokens: Int[Array, " max_length"]) -> "EnvState":
        """Creates a non-initial, non-terminal single state holding `tokens`."""
        return cls(
            tokens=tokens,
            is_terminal=jnp.bool_(False),
            is_initial=jnp.bool_(False),
            is_pad=jnp.bool_(False),
        )


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    """Sequence dynamics parameters (mirrors the constructor arguments).

    Attributes:
        max_length: Length of the token buffer.
        nchar: Number of regular characters (the vocabulary size $m$).
        ntoken: Total number of token ids, including special tokens.
        bos_token: Beginning-of-sequence token id, prepended to observations.
        eos_token: End-of-sequence token id, written by a stop action.
        pad_token: Id marking empty positions.
    """

    max_length: int
    nchar: int
    ntoken: int

    bos_token: int
    eos_token: int
    pad_token: int


class SequenceEnvironment(BaseEnvironment[EnvState, EnvParams]):
    """Abstract base for environments that generate token sequences.

    The state is a token buffer of length `max_length` initialised with `pad_token`;
    subclasses define how characters are written into it (see the module overview).
    The observation is the buffer with `bos_token` prepended, i.e. an integer vector of
    shape `[max_length + 1]`, suitable for embedding by a transformer policy.
    Trajectories take at most `max_length` steps.

    Args:
        max_length: Length $n$ of the sequences / token buffer.
        nchar: Vocabulary size $m$ (number of regular characters, ids `0 .. nchar - 1`).
        ntoken: Total number of token ids, including the special tokens below.
        bos_token: Beginning-of-sequence token id.
        eos_token: End-of-sequence token id.
        pad_token: Padding token id, marking empty positions.
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

    def reset(self) -> EnvState:
        """Returns the empty sequence (a buffer of `pad_token`)."""
        tokens = jnp.full(
            shape=(self.max_length,),
            fill_value=self.pad_token,
            dtype=jnp.int32,
        )
        return EnvState.from_tokens(tokens).replace(is_initial=jnp.bool_(True))

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        """Returns `EnvParams` mirroring the constructor arguments (deterministic)."""
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
        """At most `max_length` steps: one token is written per step."""
        return self.max_length

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Returns `[bos_token, *tokens]`, an integer vector of shape `[max_length + 1]`."""
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

    def get_action_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        raise NotImplementedError

    def get_backward_action_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
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
    """Fixed-length, left-to-right (autoregressive) sequence generation.

    * **Forward actions** — `nchar` actions; action `c` appends character `c` at the
      first empty position. All actions are always valid.
    * **Backward actions** — a single action that removes the last character. The
      generation DAG is a tree, so $P_B$ is degenerate (always 1).
    * **Termination** — after exactly `max_length` steps, when the buffer is full.

    Each object has exactly one trajectory, so the number of terminal states is
    `nchar ** max_length`.
    """

    def step(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
        """Appends character `action` at the first empty position.

        Returns:
            `(obs, next_state, done, info)`; `done` is `True` once the buffer is full.
        """
        num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
        pos_to_update = self.max_length - num_pad
        next_tokens = state.tokens.at[pos_to_update].set(action)
        is_done = jnp.all(next_tokens != self.pad_token)
        next_active = state.replace(
            tokens=next_tokens,
            is_terminal=is_done,
            is_initial=False,
            is_pad=False,
        )
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
        self, state: EnvState, backward_action: TBackwardAction, env_params: EnvParams
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
        """Removes the last character (`backward_action` is ignored; it is always `0`).

        Returns:
            `(obs, prev_state, done, info)`; `done` is `True` once the buffer is empty.
        """
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
        return (
            self.get_obs(prev_state, env_params),
            prev_state,
            jnp.astype(prev_state.is_initial, jnp.bool),
            {},
        )

    def get_backward_action(
        self,
        _state: EnvState,
        forward_action: chex.Array,
        _next_state: EnvState,
        _env_params: EnvParams,
    ) -> chex.Array:
        """Always `0`: there is a single backward action."""
        return jnp.zeros((), dtype=forward_action.dtype)

    def get_forward_action(
        self,
        state: EnvState,
        _backward_action: chex.Array,
        _prev_state: EnvState,
        _env_params: EnvParams,
    ) -> chex.Array:
        """Returns the last character of `state`, i.e. the token the backward step removes."""
        num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
        last_position = self.max_length - num_pad
        action = state.tokens[last_position - 1]
        return jnp.clip(action, min=0, max=self.action_space.n - 1)

    def get_action_mask(self, _state: EnvState, _env_params: EnvParams) -> chex.Array:
        """All `nchar` actions are always valid. Shape `[nchar]`."""
        return jnp.ones((self.nchar,), dtype=jnp.bool)

    def get_backward_action_mask(self, _state: EnvState, _env_params: EnvParams) -> chex.Array:
        """The single backward action is always valid. Shape `[1]`."""
        return jnp.ones((1,), dtype=jnp.bool)

    @property
    def action_space(self) -> spaces.Discrete:
        """`Discrete(nchar)`."""
        return spaces.Discrete(self.nchar)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """`Discrete(1)`."""
        return spaces.Discrete(1)


class FixedPrependAppendSequenceEnvironment(SequenceEnvironment):
    r"""Fixed-length generation by prepending or appending characters.

    * **Forward actions** — `2 * nchar` actions. Action `c < nchar` prepends character
      `c` (shifting the sequence right); action `nchar + c` appends character `c`. All
      actions are always valid.
    * **Backward actions** — 2 actions: `0` removes the first character, `1` removes the
      last one. Both are always valid, so a sequence of length $\ell > 1$ has two
      parents, which coincide only when all of its characters are equal.
    * **Termination** — after exactly `max_length` steps, when the buffer is full.
    """

    def step(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
        """Prepends (`action < nchar`) or appends (`action >= nchar`) a character.

        Returns:
            `(obs, next_state, done, info)`; `done` is `True` once the buffer is full.
        """

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
        next_active = state.replace(
            tokens=next_tokens,
            is_terminal=is_done,
            is_initial=False,
            is_pad=False,
        )
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
        self,
        state: EnvState,
        backward_action: TBackwardAction,
        env_params: EnvParams,
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
        """Removes the first (`backward_action == 0`) or last (`1`) character.

        Returns:
            `(obs, prev_state, done, info)`; `done` is `True` once the buffer is empty.
        """
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
        return (
            self.get_obs(prev_state, env_params),
            prev_state,
            jnp.astype(prev_state.is_initial, jnp.bool),
            {},
        )

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Maps a prepend to backward action `0` and an append to `1`."""
        return jnp.where(forward_action < self.nchar, 0, 1)

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns the prepend (`first char`) or append (`nchar + last char`) action."""
        num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
        last_position = self.max_length - num_pad
        removed_token = state.tokens[last_position - 1]
        action = jnp.where(backward_action == 0, state.tokens[0], self.nchar + removed_token)
        return jnp.clip(action, min=0, max=self.action_space.n - 1)

    def get_action_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """All `2 * nchar` actions are always valid. Shape `[2 * nchar]`."""
        return jnp.ones((2 * self.nchar,), dtype=jnp.bool)

    def get_backward_action_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Both backward actions are always valid. Shape `[2]`."""
        return jnp.ones((2,), dtype=jnp.bool)

    @property
    def action_space(self) -> spaces.Discrete:
        """`Discrete(2 * nchar)`: prepend actions first, then append actions."""
        return spaces.Discrete(2 * self.nchar)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """`Discrete(2)`: remove first, remove last."""
        return spaces.Discrete(2)


class AutoregressiveSequenceEnvironment(SequenceEnvironment):
    """Variable-length, left-to-right (autoregressive) generation with a stop action.

    * **Forward actions** — `nchar + 1` actions. Action `c < nchar` appends character
      `c`; action `nchar` (`stop_action`) writes `eos_token` and terminates. All actions
      are always valid.
    * **Backward actions** — a single action that removes the last token (character or
      `eos_token`). $P_B$ is degenerate.
    * **Termination** — on the stop action, or when the buffer is full (`max_length`
      characters).

    Args:
        max_length: Maximum sequence length.
        nchar: Vocabulary size.
        ntoken: Total number of token ids.
        bos_token: Beginning-of-sequence token id.
        eos_token: End-of-sequence token id, written by the stop action.
        pad_token: Padding token id.

    Attributes:
        stop_action: Index of the stop action (equal to `nchar`).
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

    def step(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
        """Appends character `action`, or `eos_token` if `action == stop_action`.

        Returns:
            `(obs, next_state, done, info)`; `done` is `True` after the stop action or
            once the buffer is full.
        """
        num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
        pos_to_update = self.max_length - num_pad
        action_to_token = jnp.where(action != self.stop_action, action, self.eos_token)
        next_tokens = state.tokens.at[pos_to_update].set(action_to_token)
        is_done = jnp.logical_or(
            jnp.all(next_tokens != self.pad_token),  # all pad tokens are replaced by characters
            action == self.stop_action,  # EOS token is generated
        )
        next_active = state.replace(
            tokens=next_tokens,
            is_terminal=is_done,
            is_initial=False,
            is_pad=False,
        )
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
        self, state: EnvState, backward_action: TBackwardAction, env_params: EnvParams
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
        """Removes the last token (`backward_action` is ignored; it is always `0`).

        Returns:
            `(obs, prev_state, done, info)`; `done` is `True` once the buffer is empty.
        """
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
        return (
            self.get_obs(prev_state, env_params),
            prev_state,
            jnp.astype(prev_state.is_initial, jnp.bool),
            {},
        )

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Always `0`: there is a single backward action."""
        return jnp.zeros((), dtype=forward_action.dtype)

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns the last character of `state`, or `stop_action` if it is `eos_token`."""
        num_pad = jnp.sum(state.tokens == self.pad_token, axis=-1)
        last_pos = self.max_length - num_pad
        all_actions = jnp.where(state.tokens != self.eos_token, state.tokens, self.stop_action)
        action = all_actions[last_pos - 1]
        return jnp.clip(action, min=0, max=self.action_space.n - 1)

    def get_action_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """All `nchar + 1` actions (including stop) are always valid. Shape `[nchar + 1]`."""
        return jnp.ones((self.nchar + 1,), dtype=jnp.bool)

    def get_backward_action_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """The single backward action is always valid. Shape `[1]`."""
        return jnp.ones((1,), dtype=jnp.bool)

    @property
    def action_space(self) -> spaces.Discrete:
        """`Discrete(nchar + 1)`: characters, then the stop action."""
        return spaces.Discrete(self.nchar + 1)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """`Discrete(1)`."""
        return spaces.Discrete(1)


class NonAutoregressiveSequenceEnvironment(SequenceEnvironment):
    r"""Fixed-length generation by filling empty positions in any order.

    * **Forward actions** — `max_length * nchar` actions. Action `a` writes character
      `a % nchar` at position `a // nchar` (i.e. `(pos, char) = unravel(a)`); it is valid
      iff that position is still empty.
    * **Backward actions** — `max_length` actions. Action `i` clears position `i`; valid
      iff it is filled.
    * **Termination** — after exactly `max_length` steps, when every position is filled.

    A sequence with $\ell$ filled positions has $\ell$ parents, and every complete
    sequence is reached by `max_length!` distinct trajectories.
    """

    def step(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
        """Writes character `action % nchar` at the empty position `action // nchar`.

        Returns:
            `(obs, next_state, done, info)`; `done` is `True` once every position is filled.
        """
        pos, word = jnp.unravel_index(action, (self.max_length, self.nchar))
        next_tokens = state.tokens.at[pos].set(word)
        is_done = jnp.all(next_tokens != self.pad_token)
        next_active = state.replace(
            tokens=next_tokens,
            is_terminal=is_done,
            is_initial=False,
            is_pad=False,
        )
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
        self,
        state: EnvState,
        backward_action: TBackwardAction,
        env_params: EnvParams,
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
        """Clears position `backward_action`.

        Returns:
            `(obs, prev_state, done, info)`; `done` is `True` once the buffer is empty.
        """
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
        return (
            self.get_obs(prev_state, env_params),
            prev_state,
            jnp.astype(prev_state.is_initial, jnp.bool),
            {},
        )

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns the position `forward_action // nchar` that was filled."""
        pos, _ = jnp.unravel_index(forward_action, (self.max_length, self.nchar))
        return pos

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns `pos * nchar + char`, re-writing the character cleared at `pos`."""
        word = state.tokens[backward_action]
        return jnp.ravel_multi_index(
            (backward_action, word), (self.max_length, self.nchar), mode="clip"
        )

    def get_action_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Return the mask of valid actions (single state). [max_length * nchar].

        An action ``(pos, char)`` is valid iff ``pos`` is not yet filled.
        Falls back to all-True at terminal states so softmax remains finite.
        """
        valid_pos_mask = state.tokens == self.pad_token  # [max_length]
        chex.assert_shape(valid_pos_mask, (self.max_length,))
        action_mask_2d = jnp.repeat(
            jnp.expand_dims(valid_pos_mask, axis=1), repeats=self.nchar, axis=1
        )
        chex.assert_shape(action_mask_2d, (self.max_length, self.nchar))
        action_mask_flat = action_mask_2d.reshape(-1)
        # Terminal: all positions filled → no action is valid; fall back to all-True.
        all_filled = jnp.all(state.tokens != self.pad_token)
        return jnp.where(all_filled, jnp.ones_like(action_mask_flat), action_mask_flat)

    def get_backward_action_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Return the mask of valid backward actions (single state). [max_length].

        A backward action (remove the token at ``pos``) is valid iff ``pos`` is
        currently filled. Falls back to all-True at the initial state.
        """
        valid_pos_mask = state.tokens != self.pad_token  # [max_length]
        all_empty = jnp.all(state.tokens == self.pad_token)
        return jnp.where(all_empty, jnp.ones_like(valid_pos_mask), valid_pos_mask)

    @property
    def action_space(self) -> spaces.Discrete:
        """`Discrete(max_length * nchar)`, indexed as `pos * nchar + char`."""
        return spaces.Discrete(self.max_length * self.nchar)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """`Discrete(max_length)`: the position to clear."""
        return spaces.Discrete(self.max_length)
