"""Reward functions used for QM9Small environment."""

import pickle

import chex
import jax.numpy as jnp

from ..base import BaseRewardModule, TLogReward, TReward
from ..environment import (
    QM9SmallEnvParams,
    QM9SmallEnvState,
)


class QM9SmallRewardModule(BaseRewardModule[QM9SmallEnvState, QM9SmallEnvParams]):
    def __init__(
        self,
        nchar: int = 11,
        max_length: int = 5,
        min_reward: float = 1e-3,
        reward_exponent: float = 5.0,
        reward_scale: float = 100.0,
    ):
        """
        TODO: Add description
        """
        self.nchar = nchar
        self.max_length = max_length
        self.min_reward = min_reward
        self.reward_exponent = reward_exponent
        self.reward_scale = reward_scale

    def init(self, rng_key: chex.PRNGKey, dummy_state: QM9SmallEnvState) -> None:
        # Source: https://github.com/maxwshen/gflownet/blob/main/datasets/qm9str/block_qm9str_v1_s5.pkl
        with open("proxy/weights/qm9_small/block_qm9str_v1_s5.pkl", "rb") as f:
            oracle_d = pickle.load(f)
        oracle = {tuple(x): float(y) for x, y in oracle_d.items()}

        values_raw = jnp.array(list(oracle.values()))

        # Normalization as in https://github.com/maxwshen/gflownet/blob/main/exps/qm9str/qm9str.py
        values = jnp.clip(values_raw, min=self.min_reward)
        values = jnp.pow(values, self.reward_exponent)
        values = values * self.reward_scale / values.max()
        return {"rewards": values}

    def reward(self, state: QM9SmallEnvState, reward_params) -> TReward:
        powers_array = jnp.array([
            self.nchar ** (self.max_length - i - 1) for i in range(self.max_length)
        ])
        index = jnp.sum(state.tokens * powers_array)
        return reward_params["rewards"][index]

    def log_reward(self, state: QM9SmallEnvState, reward_params) -> TLogReward:
        return jnp.log(self.reward(state, reward_params))
