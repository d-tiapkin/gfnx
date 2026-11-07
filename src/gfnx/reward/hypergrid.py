"""Reward modules for the hypergrid environment.

All hypergrid rewards are analytical functions of the terminal cell, so their
`reward_params` are empty. See
[`HypergridEnvironment`][gfnx.environment.hypergrid.HypergridEnvironment].
"""

import chex
import jax.numpy as jnp

from ..base import BaseRewardModule, BaseRewardParams, TLogReward, TReward
from ..environment import HypergridEnvParams, HypergridEnvState


@chex.dataclass(frozen=True)
class HypergridRewardParams(BaseRewardParams):
    """Empty reward parameters: hypergrid rewards are closed-form."""


class GeneralHypergridRewardModule(
    BaseRewardModule[HypergridEnvState, HypergridEnvParams, HypergridRewardParams]
):
    r"""Standard multimodal hypergrid reward.

    For a terminal cell $s = (s^1, \ldots, s^D)$ on a grid with side $H$,

    $$
    R(s) = R_0
    + R_1 \prod_{i=1}^D \mathbb{I}\left[0.25 < \left|\tfrac{s^i}{H-1} - 0.5\right|\right]
    + R_2 \prod_{i=1}^D \mathbb{I}\left[0.3 < \left|\tfrac{s^i}{H-1} - 0.5\right| < 0.4\right].
    $$

    The indicator products carve out $2^D$ symmetric modes near the corners of the grid:
    instead of peaking at the centre, the reward places mass on bands away from the
    middle, which makes exploration highly multimodal. $R_0$ sets the background reward
    (lower is harder), $R_1$ the plateau around each corner and $R_2$ the peak inside
    it. The reward is clipped from below at `min_reward`.

    Args:
        side: Side length $H$ of the grid. Must equal the environment's `side`; it has no
            default so that a mismatch cannot happen silently.
        R0: Background reward $R_0$.
        R1: Height of the outer plateau $R_1$.
        R2: Height of the inner peak $R_2$.
        min_reward: Lower clip applied to the reward (keeps $\log R$ finite).

    References:
        Bengio, E., et al. (2021). Flow network based generative models for non-iterative
        diverse candidate generation. NeurIPS.

        Madan, K., et al. (2023). Learning GFlowNets from partial episodes for improved
        convergence and stability. ICML.
    """

    def __init__(
        self,
        *,
        side: int,
        R0: float = 1e-3,
        R1: float = 0.5,
        R2: float = 2.0,
        min_reward: float = 1e-6,
    ) -> None:
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.side = side
        self.min_reward = min_reward

    def init(self, rng_key: chex.PRNGKey, dummy_state: HypergridEnvState) -> HypergridRewardParams:
        """Returns empty `HypergridRewardParams`; the reward is closed-form."""
        return HypergridRewardParams()

    def reward(self, state: HypergridEnvState, reward_params: HypergridRewardParams) -> TReward:
        """Computes $R(s)$ for a single cell.

        Args:
            state: Hypergrid state; only `state.state` (the coordinates) is used.
            reward_params: Unused.

        Returns:
            Scalar reward, at least `min_reward`.
        """
        ax = jnp.abs(state.state / (self.side - 1) - 0.5)
        reward = (
            self.R0 + jnp.prod(ax > 0.25) * self.R1 + jnp.prod((ax < 0.4) * (ax > 0.3)) * self.R2
        )
        return jnp.clip(reward, min=self.min_reward)

    def log_reward(
        self, state: HypergridEnvState, reward_params: HypergridRewardParams
    ) -> TLogReward:
        r"""Computes $\log R(s)$ for a single cell (see `reward`)."""
        return jnp.log(self.reward(state, reward_params))


# Two specific use cases
class EasyHypergridRewardModule(GeneralHypergridRewardModule):
    """Hypergrid reward with $(R_0, R_1, R_2) = (10^{-3}, 0.5, 2.0)$.

    Args:
        side: Side length $H$ of the grid; must equal the environment's `side`.
    """

    def __init__(self, *, side: int) -> None:
        super().__init__(side=side, R0=1e-3, R1=0.5, R2=2.0)


class HardHypergridRewardModule(GeneralHypergridRewardModule):
    """Hypergrid reward with $(R_0, R_1, R_2) = (10^{-4}, 1.0, 3.0)$.

    The lower background reward $R_0$ makes the modes harder to discover.

    Args:
        side: Side length $H$ of the grid; must equal the environment's `side`.
    """

    def __init__(self, *, side: int) -> None:
        super().__init__(side=side, R0=1e-4, R1=1.0, R2=3.0)
