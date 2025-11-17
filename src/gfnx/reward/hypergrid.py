"""Reward functions used for hypergrid environment"""
import chex
import jax.numpy as jnp

import gfnx
from gfnx import (
    HypergridEnvParams,
    HypergridEnvState,
)

from ..base import BaseRewardModule


class GeneralHypergridRewardModule(
    BaseRewardModule[HypergridEnvState, HypergridEnvParams]
):
    def __init__(self, R0: float = 1e-3, R1: float = 0.5, R2: float = 2.0):
        """
        General reward function for hypegrids, defined as
        $$
            R(s) = R0 + R1 \cdot \prod_{d=1}^D \ind{| s^d/(H-1) - 0.5| \in (0.25, 0.5)}
            + R2 \cdot \prod_{d=1}^D \ind\{ | s^d/(H-1) - 0.5| \in (0.3, 0.4) \}
        $$

        Source: Madan, Kanika, et al. "Learning gflownets from partial episodes 
        for improved convergence and stability." 
        International Conference on Machine Learning. PMLR, 2023.
        """  # noqa: E501
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.min_reward = 1e-6  # TODO: Make this a parameter

    def init(
        self, rng_key: chex.PRNGKey, dummy_state: HypergridEnvState
    ) -> None:
        return None  # No parameters needed to jit

    def reward(
        self, state: HypergridEnvState, env_params: HypergridEnvParams
    ) -> gfnx.TReward:
        state = state.state
        ax = jnp.abs(state / (env_params.side - 1) - 0.5)
        reward = (
            self.R0
            + jnp.prod((ax > 0.25), axis=-1) * self.R1
            + jnp.prod((ax < 0.4) * (ax > 0.3), axis=-1) * self.R2
        )
        chex.assert_shape(reward, (state.shape[0],))    # [B]
        return jnp.clip(reward, a_min=self.min_reward)
    
    def log_reward(
        self, state: HypergridEnvState, env_params: HypergridEnvParams
    ) -> gfnx.TLogReward:
        return jnp.log(self.reward(state, env_params))


# Two specific use cases
class EasyHypergridRewardModule(GeneralHypergridRewardModule):
    def __init__(self) -> None:
        super().__init__(R0=1e-3, R1=0.5, R2=2.0)


class HardHypergridRewardModule(GeneralHypergridRewardModule):
    def __init__(self) -> None:
        super().__init__(R0=1e-4, R1=1.0, R2=3.0)
