"""Reward functions used for TFBind-8 environment.

pip install git+https://github.com/brandontrabucco/design-bench.git@chris/fixes-v2
"""

import itertools
import pickle

import chex
import jax.numpy as jnp
import numpy as np

from ..base import BaseRewardModule, TLogReward, TReward
from ..environment import (
    TFBind8EnvParams,
    TFBind8EnvState,
)


class TFBind8RewardModule(
    BaseRewardModule[TFBind8EnvState, TFBind8EnvParams]
):
    def __init__(
        self,
        nchar: int = 4,
        max_length: int = 8,
        min_reward: float = 1e-3,
        reward_exponent: float = 3.0,
        reward_scale: float = 10.0,
    ):
        """
        TODO: Add description
        """
        self.nchar = nchar
        self.max_length = max_length
        self.min_reward = min_reward
        self.reward_exponent = reward_exponent
        self.reward_scale = reward_scale

    def init(
        self, rng_key: chex.PRNGKey, dummy_state: TFBind8EnvState
    ) -> None:
        # Make a full loop to get the values for all possible states
        
        # Generate all possible values of characters
        values = list(range(self.nchar))
        # Generate all possible arrays
        all_states = np.array(
            list(itertools.product(values, repeat=self.max_length))
        )

        # Source: https://github.com/maxwshen/gflownet/blob/main/datasets/tfbind8/tfbind8-exact-v0-all.pkl
        with open('proxy/weights/tfbind/tfbind8-exact-v0-all.pkl', 'rb') as f:
            oracle_d = pickle.load(f)
        oracle = {tuple(x): float(y[0]) for x, y in zip(oracle_d['x'], oracle_d['y'])}

        values_raw = jnp.array([oracle[tuple(state)] for state in all_states])
        
        # Normalization as in https://github.com/maxwshen/gflownet/blob/main/exps/tfbind8/tfbind8_oracle.py
        values = jnp.pow(values_raw, self.reward_exponent)
        values = (values * self.reward_scale / values.max())
        values = jnp.clip(values, min=self.min_reward)
        return {"rewards": values}  # Dict with all possible values

    def reward(
        self, state: TFBind8EnvState, env_params: TFBind8EnvParams
    ) -> TReward:
        tokens = state.tokens
        powers_array = jnp.array([
            self.nchar ** (self.max_length - i - 1)
            for i in range(self.max_length)
        ])
        indices = jnp.sum(tokens * powers_array, axis=-1)
        return jnp.take_along_axis(
            env_params.reward_params["rewards"],
            indices,
            axis=0,
            mode="fill",
            fill_value=self.min_reward,
        )

    def log_reward(
        self, state: TFBind8EnvState, env_params: TFBind8EnvParams
    ) -> TLogReward:
        return jnp.log(self.reward(state, env_params))
