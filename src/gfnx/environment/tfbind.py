"""TFBind-8 environment: DNA sequences of length 8, generated left to right.

The lookup-table reward lives in [`gfnx.reward.tfbind`][gfnx.reward.tfbind].
"""

from math import prod

import chex
import jax
import jax.numpy as jnp

from ..base import BaseRewardModule, TRewardParams
from ..utils import NUCLEOTIDES, NUCLEOTIDES_FULL_ALPHABET
from .sequence import (
    EnvParams,
    EnvState,
    FixedAutoregressiveSequenceEnvironment,
)


class TFBind8Environment(FixedAutoregressiveSequenceEnvironment):
    """Autoregressive generation of DNA 8-mers over the nucleotides A, C, G, T.

    The reward is the wet-lab measured binding activity of the 8-mer to the human
    transcription factor SIX6. This is a
    [`FixedAutoregressiveSequenceEnvironment`][gfnx.environment.sequence.FixedAutoregressiveSequenceEnvironment]
    with `max_length = 8` and `nchar = 4`:

    * **State** — `tokens` of shape `[8]` over `gfnx.utils.NUCLEOTIDES_FULL_ALPHABET`
      (`A, C, G, T = 0..3`, then `[BOS]`, `[EOS]`, `[PAD]`).
    * **Forward actions** — 4 actions, appending a nucleotide.
    * **Backward actions** — a single action removing the last nucleotide.
    * **Observation** — `[bos_token, *tokens]`, shape `[9]`.
    * **Episode length** — exactly 8 steps.

    The environment is enumerable over its $4^8 = 65{,}536$ terminal states: it exposes
    the exact target distribution, normalising constant, expected reward and exact
    sampling. Distributions are arrays of shape `(4,) * 8` indexed by the tokens.

    The environment has no constructor arguments.

    References:
        Shen, M. W., et al. (2023). Towards understanding and improving GFlowNet
        training. ICML.

        Barrera, L. A., et al. (2016). Survey of variation in human transcription factors
        reveals prevalent DNA binding changes. Science, 351(6280).
    """

    def __init__(self) -> None:
        self.char_to_id = {char: i for i, char in enumerate(NUCLEOTIDES_FULL_ALPHABET)}

        super().__init__(
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

    def _get_states_rewards(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> chex.Array:
        """Evaluates the reward on every 8-mer; returns an array of shape `(4,) * 8`."""
        rewards = jnp.zeros((self.nchar,) * self.max_length, dtype=jnp.float32)

        def update_rewards(idx: int, rewards: chex.Array):
            state = jnp.unravel_index(idx, shape=rewards.shape)
            env_state = EnvState(
                tokens=jnp.array(state),
                is_terminal=jnp.bool_(True),
                is_initial=jnp.bool_(False),
                is_pad=jnp.bool_(False),
            )
            reward = reward_module.reward(env_state, reward_params)
            return rewards.at[state].set(reward)

        return jax.lax.fori_loop(0, self.nchar**self.max_length, update_rewards, rewards)

    def get_true_distribution(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> chex.Array:
        """Returns $R(x)/Z$ for every 8-mer, as an array of shape `(4,) * 8`."""
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return rewards / rewards.sum()

    def get_empirical_distribution(self, states: EnvState, env_params: EnvParams) -> chex.Array:
        """Histogram of a batch of terminal states, as an array of shape `(4,) * 8`."""
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
    def is_expected_reward_tractable(self) -> bool:
        """Whether this environment supports tractable expected reward."""
        return True

    def get_expected_reward(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> float:
        r"""Returns $\mathbb{E}_{x \sim R/Z}[R(x)] = \sum_x R(x)^2 / \sum_x R(x)$."""
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return jnp.pow(rewards, 2).sum() / rewards.sum()

    @property
    def is_normalizing_constant_tractable(self) -> bool:
        """Whether this environment supports tractable normalizing constant."""
        return True

    def get_normalizing_constant(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> float:
        r"""Returns $Z = \sum_x R(x)$ over all $4^8$ sequences."""
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return rewards.sum()

    @property
    def is_ground_truth_sampling_tractable(self) -> bool:
        """Whether this environment supports tractable sampling from the GT distribution."""
        return True

    def get_ground_truth_sampling(
        self,
        rng_key: chex.PRNGKey,
        batch_size: int,
        env_params: EnvParams,
        reward_module: BaseRewardModule,
        reward_params: TRewardParams,
    ) -> EnvState:
        """Draws `batch_size` terminal sequences exactly from $R(x)/Z$.

        Args:
            rng_key: JAX random key for sampling.
            batch_size: Number of samples to generate.
            env_params: Environment parameters.
            reward_module: TFBind-8 reward module.
            reward_params: Its parameters (the reward lookup table).

        Returns:
            Batched terminal `EnvState` with `tokens` of shape `[batch_size, 8]`.
        """
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
        sampled_tokens = jnp.stack(sampled_coords_unstacked, axis=1)

        return EnvState(
            tokens=sampled_tokens.astype(jnp.int32),
            is_terminal=jnp.ones((batch_size,), dtype=jnp.bool_),
            is_initial=jnp.zeros((batch_size,), dtype=jnp.bool_),
            is_pad=jnp.zeros((batch_size,), dtype=jnp.bool_),
        )
