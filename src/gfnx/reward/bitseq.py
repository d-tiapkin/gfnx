"""Reward functions used for hypergrid environment"""

import chex
import jax
import jax.numpy as jnp

from gfnx.utils.distances import hamming_distance

from ..base import BaseRewardModule, TLogReward, TReward, TRewardParams
from ..environment import BitseqEnvParams, BitseqEnvState
from ..utils.bitseq import construct_mode_set, detokenize


class BitseqRewardModule(BaseRewardModule[BitseqEnvState, BitseqEnvParams]):
    def __init__(
        self,
        sentence_len: int = 120,
        k: int = 8,
        mode_set_size: int = 60,
        reward_exponent: float = 1.0,
    ):
        """
        General reward function for bitseqs
        """
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

    def init(self, rng_key: chex.PRNGKey, dummy_state: BitseqEnvState) -> TRewardParams:
        return {
            "mode_set": construct_mode_set(
                self.sentence_len,
                self.block_len,
                self.block_set,
                self.mode_set_size,
                rng_key,
            )
        }

    def _mode_set_distance(self, s: chex.Array, mode_set: chex.Array):
        distances = jax.vmap(lambda ms: hamming_distance(s, ms))(mode_set)
        return jnp.min(distances)

    def log_reward(self, state: BitseqEnvState, reward_params: TRewardParams) -> TLogReward:
        bitseq = detokenize(state.tokens, self.k)
        mode_dist = self._mode_set_distance(bitseq, reward_params["mode_set"])
        return -self.reward_exponent * mode_dist.astype(jnp.float32) / bitseq.shape[0]

    def reward(self, state: BitseqEnvState, reward_params: TRewardParams) -> TReward:
        return jnp.exp(self.log_reward(state, reward_params))
