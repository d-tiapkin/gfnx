"""Reward functions used for hypergrid environment"""


import chex
import jax
import jax.numpy as jnp

import gfnx
from gfnx import BitseqEnvParams, BitseqEnvState
from ..base import BaseRewardModule
from ..utils import (
    construct_mode_set,
    detokenize,
    mode_set_distance,
)


class BitseqRewardModule(
    BaseRewardModule[BitseqEnvState, BitseqEnvParams]
):
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

    def init(
        self, rng_key: chex.PRNGKey, dummy_state: BitseqEnvState
    ) -> gfnx.TRewardParams:
        return {
            "mode_set": construct_mode_set(
                self.sentence_len,
                self.block_len,
                self.block_set,
                self.mode_set_size,
                rng_key,
            )
        }

    def log_reward(
        self, state: BitseqEnvState, env_params: BitseqEnvParams
    ) -> gfnx.TLogReward:
        def single_log_reward(tokens, reward_params):
            bitseq = detokenize(tokens, self.k)
            mode_dist = mode_set_distance(bitseq, reward_params["mode_set"])
            return -self.reward_exponent * mode_dist.astype(jnp.float32)
        return -jax.vmap(
            single_log_reward, in_axes=(0, None)
        )(state.tokens, env_params.reward_params)
    
    def reward(
        self, state: BitseqEnvState, env_params: BitseqEnvParams
    ) -> gfnx.TReward:
        return jnp.exp(self.log_reward(state, env_params))
