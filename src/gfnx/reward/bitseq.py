"""Mode-distance reward for the bit-sequence environment.

See [`BitseqEnvironment`][gfnx.environment.bitseq.BitseqEnvironment].
"""

import chex
import jax
import jax.numpy as jnp

from gfnx.utils.distances import hamming_distance

from ..base import BaseRewardModule, BaseRewardParams, TLogReward, TReward
from ..environment import BitseqEnvParams, BitseqEnvState
from ..utils.bitseq import construct_mode_set, detokenize


@chex.dataclass(frozen=True)
class BitseqRewardParams(BaseRewardParams):
    """Parameters of the bit-sequence reward.

    Attributes:
        mode_set: Boolean array of shape `[mode_set_size, sentence_len]` holding the
            modes $M$. The GFlowNet only observes rewards, never $M$ itself.
    """

    mode_set: chex.Array


class BitseqRewardModule(BaseRewardModule[BitseqEnvState, BitseqEnvParams, BitseqRewardParams]):
    r"""Reward based on the Hamming distance to the closest of a hidden set of modes.

    For a binary string $x$ of length $n$,

    $$
    R(x) = \exp\left(-\beta \cdot \frac{\min_{x' \in M} d(x, x')}{n}\right),
    $$

    where $d$ is the Hamming distance, $M$ is the mode set and $\beta$ is the reward
    exponent (inverse temperature); larger $\beta$ gives a peakier target.

    The mode set is sampled in [`init`][gfnx.reward.bitseq.BitseqRewardModule.init]:
    each of the `mode_set_size` modes is a concatenation of `sentence_len / 8` blocks
    drawn uniformly from `{00000000, 11111111, 11110000, 00001111, 00111100}`.

    Args:
        sentence_len: Length $n$ of the binary strings; must match the environment's `n`
            and be divisible by 8.
        k: Bits per token; must match the environment's `k` (used to detokenize states).
        mode_set_size: Number of modes $|M|$.
        reward_exponent: Inverse temperature $\beta$.

    References:
        Malkin, N., et al. (2022). Trajectory balance: Improved credit assignment in
        GFlowNets. NeurIPS.

        Tiapkin, D., et al. (2024). Generative flow networks as entropy-regularized RL.
        AISTATS.
    """

    def __init__(
        self,
        sentence_len: int = 120,
        k: int = 8,
        mode_set_size: int = 60,
        reward_exponent: float = 1.0,
    ):
        self.block_len = 8
        self.block_set = jnp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 0, 0],
            ],
            dtype=jnp.bool,
        )
        self.sentence_len = sentence_len
        self.k = k
        assert sentence_len % self.block_len == 0
        self.mode_set_size = mode_set_size
        self.reward_exponent = reward_exponent

    def init(self, rng_key: chex.PRNGKey, dummy_state: BitseqEnvState) -> BitseqRewardParams:
        """Samples the mode set.

        Args:
            rng_key: Random key used to sample the modes.
            dummy_state: Unused.

        Returns:
            `BitseqRewardParams` holding the sampled `mode_set`.
        """
        return BitseqRewardParams(
            mode_set=construct_mode_set(
                self.sentence_len,
                self.block_len,
                self.block_set,
                self.mode_set_size,
                rng_key,
            )
        )

    def _mode_set_distance(self, s: chex.Array, mode_set: chex.Array):
        distances = jax.vmap(lambda ms: hamming_distance(s, ms))(mode_set)
        return jnp.min(distances)

    def log_reward(self, state: BitseqEnvState, reward_params: BitseqRewardParams) -> TLogReward:
        r"""Computes $-\beta \min_{x' \in M} d(x, x') / n$ for a single state.

        Args:
            state: Terminal bit-sequence state; its tokens are detokenized to bits.
            reward_params: Parameters holding the mode set.

        Returns:
            Scalar log-reward in $[-\beta, 0]$.
        """
        bitseq = detokenize(state.tokens, self.k)
        mode_dist = self._mode_set_distance(bitseq, reward_params.mode_set)
        return -self.reward_exponent * mode_dist.astype(jnp.float32) / bitseq.shape[0]

    def reward(self, state: BitseqEnvState, reward_params: BitseqRewardParams) -> TReward:
        """Computes $R(x)$, the exponential of `log_reward`."""
        return jnp.exp(self.log_reward(state, reward_params))
