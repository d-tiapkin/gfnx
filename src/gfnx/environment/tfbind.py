from math import prod

import chex
import jax
import jax.numpy as jnp

from ..base import TRewardModule
from ..utils import NUCLEOTIDES, NUCLEOTIDES_FULL_ALPHABET
from .sequence import (
    EnvParams,
    EnvState,
    FixedAutoregressiveSequenceEnvironment,
)


class TFBind8Environment(FixedAutoregressiveSequenceEnvironment):
    def __init__(self, reward_module: TRewardModule) -> None:
        self.char_to_id = {char: i for i, char in enumerate(NUCLEOTIDES_FULL_ALPHABET)}

        super().__init__(
            reward_module,
            max_length=8,
            nchar=len(NUCLEOTIDES),
            ntoken=len(NUCLEOTIDES_FULL_ALPHABET),
            bos_token=self.char_to_id["[BOS]"],
            eos_token=self.char_to_id["[EOS]"],
            pad_token=self.char_to_id["[PAD]"],
        )

    @property
    def is_enumerable(self) -> bool:
        """Whether the environment is enumerable."""
        return True

    @property
    def name(self) -> str:
        """Environment name."""
        return "TFBind8-v0"

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

        return jax.lax.fori_loop(0, self.nchar**self.max_length, update_rewards, rewards)

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

    @property
    def is_ground_truth_sampling_tractable(self) -> bool:
        """Whether this environment supports tractable sampling from the GT distribution."""
        return True

    def get_ground_truth_sampling(
        self, rng_key: chex.PRNGKey, batch_size: int, env_params: EnvParams
    ) -> EnvState:
        """
        Returns a batch of terminal states sampled from the ground-truth distribution
        proportional to rewards over all sequences of length `max_length`.

        Args:
            rng_key: JAX random key for sampling.
            batch_size: Number of samples to generate.
            env_params: Environment parameters.

        Returns:
            EnvState with shape [batch_size, max_length].
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
        sampled_tokens = jnp.stack(sampled_coords_unstacked, axis=1)

        return EnvState(
            tokens=sampled_tokens.astype(jnp.int32),
            is_terminal=jnp.ones((batch_size,), dtype=jnp.bool_),
            is_initial=jnp.zeros((batch_size,), dtype=jnp.bool_),
            is_pad=jnp.zeros((batch_size,), dtype=jnp.bool_),
        )
