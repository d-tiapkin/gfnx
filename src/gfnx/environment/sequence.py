from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

import gfnx
import gfnx.spaces as spaces

from ..base import BaseEnvParams, BaseEnvState, BaseVecEnvironment


@chex.dataclass(frozen=True)
class EnvState(BaseEnvState):
    tokens: chex.Array  # [B, max_length]
    is_done: chex.Array  # [B]
    time: chex.Array  # [B]


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    max_length: int
    nchar: int
    ntoken: int

    bos_token: int
    eos_token: int
    pad_token: int

    reward_params: gfnx.TRewardParams


class SequenceEnvironment(BaseVecEnvironment[EnvState, EnvParams]):
    """
    Class for sequence environments with a fixed length.
    """

    def __init__(
        self,
        reward_module: gfnx.TRewardModule,
        max_length: int,  # Maximal length of the sequence
        nchar: int,  # Number of active characters in the vocabulary
        ntoken: int,  # Size of the vocabulary including special tokens
        *,
        bos_token: int,  # id of beginning of sentence token
        eos_token: int,  # id of end of sentence token
        pad_token: int,  # id of padding token
    ) -> None:
        super().__init__(reward_module)
        self.max_length = max_length
        self.nchar = nchar
        self.ntoken = ntoken

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

    def get_init_state(self, num_envs: int) -> EnvState:
        # Fill empty tokens with [PAD] token
        return EnvState(
            tokens=jnp.full(
                shape=(num_envs, self.max_length),
                fill_value=self.pad_token,  # [PAD] token
                dtype=jnp.int32,
            ),
            is_done=jnp.zeros((num_envs,), dtype=jnp.bool),
            time=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        dummy_state = self.get_init_state(1)
        reward_params = self.reward_module.init(rng_key, dummy_state)
        return EnvParams(
            reward_params=reward_params,
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

    def _single_transition(
        self,
        state: EnvState,
        action: gfnx.TAction,
        env_params: EnvParams,
    ) -> Tuple[EnvState, gfnx.TDone, Dict[Any, Any]]:
        raise NotImplementedError

    def _single_backward_transition(
        self,
        state: EnvState,
        backward_action: gfnx.TBackwardAction,
        env_params: EnvParams,
    ) -> Tuple[EnvState, gfnx.TDone, Dict[Any, Any]]:
        """
        Environment-specific step backward transition. Rewards always zero!
        """
        raise NotImplementedError

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Applies observation function to state."""
        # Add BOS token to the beginning of the sentence
        num_envs = state.time.shape[0]
        obs = jnp.concat(
            [
                jnp.full(
                    shape=(num_envs, 1),
                    fill_value=self.bos_token,
                    dtype=state.tokens.dtype,
                ),
                state.tokens,
            ],
            axis=-1,
        )
        return obs

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
        """Return mask of invalid actions"""
        raise NotImplementedError

    def get_invalid_backward_mask(
        self, state: EnvState, env_params: EnvParams
    ) -> chex.Array:
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Environment name."""
        raise NotImplementedError

    @property
    def action_space(self) -> spaces.Discrete:
        raise NotImplementedError

    @property
    def backward_action_space(self) -> spaces.Discrete:
        raise NotImplementedError

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            low=0,
            high=params.ntoken,  # Includes all special tokens
            shape=(self.max_length + 1,),  # +1 because of BOS token
            dtype=jnp.int32,
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
            "token": spaces.Box(
                low=0,
                high=params.ntoken,  # Includes special tokens
                # (e.g. PAD and EOS)
                shape=(self.max_length,),
                dtype=jnp.int32,
            ),
            "is_done": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
        })


'''
FIX ME
class AutoregressiveSequenceEnvironment(SequenceEnvironment):
    """
    Class for sequence environments with a fixed length and 
    non-autoregressive generation.
    """

    def _single_transition(
        self,
        state: EnvState,
        action: base.TAction,
        env_params: EnvParams,
    ) -> Tuple[EnvState, base.TDone, Dict[Any, Any]]:
        is_done = state.is_done
        time = state.time

        def get_next_state_done(
            state: EnvState, action: base.TAction
        ) -> EnvState:
            return state

        def get_next_state_not_done(
            state: EnvState, action: base.TAction
        ) -> EnvState:
            # action is a raveled multi-index of a pair (pos, char)
            pos_to_update = time # time is exactly the position to update
            print(pos_to_update)
            next_tokens = state.tokens.at[pos_to_update].set(action)

            is_done = jnp.logical_or(
                jnp.all(next_tokens != self.pad_token), 
                # All pad tokens are replaced by characters
                action == self.eos_token  # EOS token is generated
            )
            next_state = EnvState(
                tokens=next_tokens, 
                is_done=is_done, 
                time=time + 1
            )
            return next_state

        next_state : EnvState = jax.lax.cond(
            is_done, 
            get_next_state_done, get_next_state_not_done,
            state, action
        )

        return next_state, next_state.is_done, {}

    def _single_backward_transition(
        self, 
        state: EnvState, 
        backward_action: base.TBackwardAction, 
        env_params: EnvParams
    ) -> Tuple[EnvState, base.TDone, Dict[Any, Any]]:
        """
        Environment-specific step backward transition. Rewards always zero!
        """
        is_done = state.is_done
        time = state.time

        def get_prev_state_done(
            state: EnvState, backward_action: base.TAction
        ) -> EnvState:
            return state

        def get_prev_state_not_done(
            state: EnvState, backward_action: base.TAction
        ) -> EnvState:
            prev_tokens = state.tokens.at[state.time-1].set(self.pad_token)
            is_done = jnp.all(prev_tokens == self.pad_token)
            return EnvState(tokens=prev_tokens, is_done=is_done, time=time - 1)

        prev_state : EnvState = jax.lax.cond(
            is_done, 
            get_prev_state_done, get_prev_state_not_done,
            state, backward_action
        )
        return  prev_state, prev_state.is_done, {}

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns backward action given the complete characterization 
        of the forward transition."""
        num_envs = state.time.shape[0]
        return jnp.zeros((num_envs,), dtype=forward_action.dtype)

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns forward action given the complete characterization 
        of the backward transition."""
        return state.tokens[state.time-1]

    def get_invalid_mask(
        self, state: EnvState, env_params: EnvParams
    ) -> chex.Array:
        """Return mask of invalid actions"""
        num_envs = state.time.shape[0]
        return jnp.zeros((num_envs, self.nchar + 1), dtype=jnp.bool)

    def get_invalid_backward_mask(
        self, state: EnvState, env_params: EnvParams
    ) -> chex.Array:
        """Returns mask of invalid backward actions."""
        num_envs = state.time.shape[0]
        return jnp.zeros((num_envs, 1), dtype=jnp.bool)

    @property
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment, consists of characters 
        and EOS token"""
        return spaces.Discrete(self.nchar + 1)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """Backward action space of the environment, 
        only about removing the last character."""
        return spaces.Discrete(1)
'''


class NonAutoregressiveSequenceEnvironment(SequenceEnvironment):
    """
    Class for sequence environments with a fixed length and
    non-autoregressive generation.
    """

    def _single_transition(
        self,
        state: EnvState,
        action: gfnx.TAction,
        env_params: EnvParams,
    ) -> Tuple[EnvState, gfnx.TDone, Dict[Any, Any]]:
        is_done = state.is_done
        time = state.time

        def get_next_state_done(state: EnvState, action: gfnx.TAction) -> EnvState:
            return state

        def get_next_state_not_done(state: EnvState, action: gfnx.TAction) -> EnvState:
            # action is a raveled multi-index of a pair (pos, char)
            pos, word = jnp.unravel_index(action, (self.max_length, self.nchar))
            next_tokens = state.tokens.at[pos].set(word)
            is_done = jnp.all(next_tokens != self.pad_token)
            next_state = EnvState(tokens=next_tokens, is_done=is_done, time=time + 1)
            return next_state

        next_state: EnvState = jax.lax.cond(
            is_done, get_next_state_done, get_next_state_not_done, state, action
        )

        return next_state, next_state.is_done, {}

    def _single_backward_transition(
        self,
        state: EnvState,
        backward_action: gfnx.TBackwardAction,
        env_params: EnvParams,
    ) -> Tuple[EnvState, gfnx.TDone, Dict[Any, Any]]:
        """
        Environment-specific step backward transition. Rewards always zero!
        """
        is_done = state.is_done
        time = state.time

        def get_prev_state_done(
            state: EnvState, backward_action: gfnx.TAction
        ) -> EnvState:
            return state

        def get_prev_state_not_done(
            state: EnvState, backward_action: gfnx.TAction
        ) -> EnvState:
            prev_tokens = state.tokens.at[backward_action].set(self.pad_token)
            is_done = jnp.all(prev_tokens == self.pad_token)
            return EnvState(tokens=prev_tokens, is_done=is_done, time=time - 1)

        prev_state: EnvState = jax.lax.cond(
            is_done,
            get_prev_state_done,
            get_prev_state_not_done,
            state,
            backward_action,
        )
        return prev_state, prev_state.is_done, {}

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns backward action given the forward transition."""
        pos, _ = jnp.unravel_index(forward_action, (self.max_length, self.nchar))
        return pos

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns forward action given the backward transition."""
        word = jnp.take_along_axis(
            state.tokens, jnp.expand_dims(backward_action, axis=1), axis=1
        ).squeeze()
        return jnp.ravel_multi_index(
            (backward_action, word), (self.max_length, self.nchar)
        )

    def get_invalid_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Return mask of invalid actions"""
        pos_mask = state.tokens != self.pad_token  # [B, token_len]
        chex.assert_shape(pos_mask, (state.tokens.shape[0], self.max_length))
        invalid_mask_2d = jnp.repeat(
            jnp.expand_dims(pos_mask, axis=2), repeats=self.nchar, axis=2
        )
        chex.assert_shape(
            invalid_mask_2d, (state.tokens.shape[0], self.max_length, self.nchar)
        )
        return invalid_mask_2d.reshape(state.tokens.shape[0], -1)

    def get_invalid_backward_mask(
        self, state: EnvState, env_params: EnvParams
    ) -> chex.Array:
        """Returns mask of invalid backward actions."""
        return state.tokens == self.pad_token

    @property
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment, consists of pairs
        (position, word)"""
        return spaces.Discrete(self.max_length * self.nchar)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """Backward action space of the environment, consists of position"""
        return spaces.Discrete(self.max_length)


class PrependAppendSequenceEnvironment(SequenceEnvironment):
    """
    Class for sequence environments with a fixed length and
    non-autoregressive generation.
    """

    def _single_transition(
        self,
        state: EnvState,
        action: gfnx.TAction,
        env_params: EnvParams,
    ) -> Tuple[EnvState, gfnx.TDone, Dict[Any, Any]]:
        is_done = state.is_done
        time = state.time

        def get_next_state_done(state: EnvState, action: gfnx.TAction) -> EnvState:
            return state

        def get_next_state_not_done(state: EnvState, action: gfnx.TAction) -> EnvState:
            def get_next_tokens_prepend(
                state: EnvState, action: gfnx.TAction
            ) -> chex.Array:
                next_tokens = jax.lax.dynamic_update_slice(
                    state.tokens, state.tokens[:-1], (1,)
                )
                return next_tokens.at[0].set(action)

            def get_next_tokens_append(
                state: EnvState, action: gfnx.TAction
            ) -> chex.Array:
                return state.tokens.at[state.time].set(
                    jnp.where(
                        action == 2 * self.nchar, self.eos_token, action - self.nchar
                    )
                )

            next_tokens = jax.lax.cond(
                action < self.nchar,
                get_next_tokens_prepend,
                get_next_tokens_append,
                state,
                action,
            )
            is_done = jnp.logical_or(
                jnp.all(next_tokens != self.pad_token), action == 2 * self.nchar
            )
            next_state = EnvState(tokens=next_tokens, is_done=is_done, time=time + 1)
            return next_state

        next_state: EnvState = jax.lax.cond(
            is_done, get_next_state_done, get_next_state_not_done, state, action
        )

        return next_state, next_state.is_done, {}

    def _single_backward_transition(
        self,
        state: EnvState,
        backward_action: gfnx.TBackwardAction,
        env_params: EnvParams,
    ) -> Tuple[EnvState, gfnx.TDone, Dict[Any, Any]]:
        """
        Environment-specific step backward transition. Rewards always zero!
        """
        is_done = state.is_done
        time = state.time

        def get_prev_state_done(
            state: EnvState, backward_action: gfnx.TAction
        ) -> EnvState:
            return state

        def get_prev_state_not_done(
            state: EnvState, backward_action: gfnx.TAction
        ) -> EnvState:
            def get_prev_tokens_prepend(state: EnvState) -> chex.Array:
                prev_tokens = jax.lax.dynamic_update_slice(
                    state.tokens, state.tokens[1:], (0,)
                )
                return prev_tokens.at[state.time - 1].set(self.pad_token)

            def get_prev_tokens_append(state: EnvState) -> chex.Array:
                return state.tokens.at[state.time - 1].set(self.pad_token)

            prev_tokens = jax.lax.cond(
                backward_action == 0,
                get_prev_tokens_prepend,
                get_prev_tokens_append,
                state,
            )

            is_done = jnp.all(prev_tokens == self.pad_token)
            return EnvState(tokens=prev_tokens, is_done=is_done, time=time - 1)

        prev_state: EnvState = jax.lax.cond(
            is_done,
            get_prev_state_done,
            get_prev_state_not_done,
            state,
            backward_action,
        )
        return prev_state, prev_state.is_done, {}

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns backward action given the forward transition."""
        return jnp.where(forward_action < self.nchar, 0, 1)

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns forward action given thebackward transition."""
        num_envs = state.time.shape[0]
        last_tokens_actions = jnp.where(
            state.tokens[jnp.arange(num_envs), state.time - 1] == self.eos_token,
            2 * self.nchar,
            self.nchar + state.tokens[jnp.arange(num_envs), state.time - 1],
        )

        return jnp.where(backward_action == 0, state.tokens[:, 0], last_tokens_actions)

    def get_invalid_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Return mask of invalid actions"""
        num_envs = state.time.shape[0]
        return jnp.zeros((num_envs, 2 * self.nchar + 1), dtype=jnp.bool)

    def get_invalid_backward_mask(
        self, state: EnvState, env_params: EnvParams
    ) -> chex.Array:
        """Returns mask of invalid backward actions."""
        num_envs = state.time.shape[0]
        return jnp.zeros((num_envs, 2), dtype=jnp.bool)

    @property
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment, consists of characters
        and EOS token."""
        return spaces.Discrete(2 * self.nchar + 1)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """Backward action space of the environment,
        only about removing the last character."""
        return spaces.Discrete(2)
        return spaces.Discrete(2)
