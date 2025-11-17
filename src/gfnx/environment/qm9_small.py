from math import prod

import chex
import jax
import jax.numpy as jnp

from ..base import TRewardModule
from ..utils import QM9_SMALL_BLOCKS, QM9_SMALL_FULL_ALPHABET
from .sequence import (
    EnvParams,  # noqa: F401
    EnvState,  # noqa: F401
    FixedPrependAppendSequenceEnvironment,
)


class QM9SmallEnvironment(FixedPrependAppendSequenceEnvironment):
    def __init__(self, reward_module: TRewardModule) -> None:
        self.char_to_id = {char: i for i, char in enumerate(QM9_SMALL_FULL_ALPHABET)}

        super().__init__(
            reward_module,
            max_length=5,
            nchar=len(QM9_SMALL_BLOCKS),
            ntoken=len(QM9_SMALL_FULL_ALPHABET),
            bos_token=self.char_to_id["[BOS]"],
            eos_token=self.char_to_id["[EOS]"],
            pad_token=self.char_to_id["[PAD]"],
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "QM9Small-v0"

    @property
    def is_enumerable(self) -> bool:
        """Whether the environment is enumerable."""
        return True

    def _get_states_rewards(self, env_params: EnvParams) -> chex.Array:
        """
        Returns the true distribution of rewards for all states in the hypergrid.
        """
        rewards = jnp.zeros((self.nchar,) * self.max_length, dtype=jnp.float32)

        def update_rewards(idx: int, rewards: chex.Array):
            state = jnp.unravel_index(idx, shape=rewards.shape)  # Unpack index to state
            env_state = EnvState(
                tokens=jnp.array(state),
                is_terminal=True,
                is_initial=False,
                is_pad=False,
            )
            batched_env_state = jax.tree.map(lambda x: jnp.expand_dims(x, 0), env_state)
            reward = self.reward_module.reward(batched_env_state, env_params)
            return rewards.at[state].set(reward[0])

        rewards = jax.lax.fori_loop(0, self.nchar**self.max_length, update_rewards, rewards)
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
        dist_shape = (self.nchar,) * self.max_length
        sample_idx = jax.vmap(lambda x: jnp.ravel_multi_index(x, dims=dist_shape, mode="clip"))(
            states.tokens
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
