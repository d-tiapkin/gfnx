"""QM9 (small) environment: molecules assembled from 2-stem building blocks.

The lookup-table reward lives in [`gfnx.reward.qm9_small`][gfnx.reward.qm9_small].
"""

from math import prod

import chex
import jax
import jax.numpy as jnp

from ..base import BaseRewardModule, TRewardParams
from ..utils import QM9_SMALL_BLOCKS, QM9_SMALL_FULL_ALPHABET
from .sequence import (
    EnvParams,
    EnvState,
    FixedPrependAppendSequenceEnvironment,
)


class QM9SmallEnvironment(FixedPrependAppendSequenceEnvironment):
    """Small-molecule generation from 11 two-stem building blocks, 5 blocks per molecule.

    Every block has exactly two attachment points (stems), so a molecule is a linear
    chain of blocks and the task reduces to a prepend/append sequence MDP. This is a
    [`FixedPrependAppendSequenceEnvironment`][gfnx.environment.sequence.FixedPrependAppendSequenceEnvironment]
    with `max_length = 5` and `nchar = 11`:

    * **State** — `tokens` of shape `[5]` over `gfnx.utils.QM9_SMALL_FULL_ALPHABET`
      (blocks `0..10`, then `[BOS]`, `[EOS]`, `[PAD]`).
    * **Forward actions** — 22 actions: prepend block `c` (`c < 11`) or append block `c`
      (`11 + c`).
    * **Backward actions** — 2 actions: remove the first block (`0`) or the last (`1`).
    * **Observation** — `[bos_token, *tokens]`, shape `[6]`.
    * **Episode length** — exactly 5 steps.

    The reward is a proxy prediction of the HOMO-LUMO gap. The environment is enumerable
    over its $11^5 = 161{,}051$ terminal states (exact target distribution, normalising
    constant and expected reward; arrays of shape `(11,) * 5`).

    The environment has no constructor arguments.

    References:
        Shen, M. W., et al. (2023). Towards understanding and improving GFlowNet
        training. ICML.
    """

    def __init__(self) -> None:
        self.char_to_id = {char: i for i, char in enumerate(QM9_SMALL_FULL_ALPHABET)}

        super().__init__(
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

    def _get_states_rewards(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> chex.Array:
        """Evaluates the reward on every molecule; returns an array of shape `(11,) * 5`."""
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
        """Returns $R(x)/Z$ for every molecule, as an array of shape `(11,) * 5`."""
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return rewards / rewards.sum()

    def get_empirical_distribution(self, states: EnvState, env_params: EnvParams) -> chex.Array:
        """Histogram of a batch of terminal states, as an array of shape `(11,) * 5`."""
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
        r"""Returns $Z = \sum_x R(x)$ over all $11^5$ molecules."""
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return rewards.sum()
