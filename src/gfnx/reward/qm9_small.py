"""Lookup-table reward for the QM9 (small) environment.

See [`QM9SmallEnvironment`][gfnx.environment.qm9_small.QM9SmallEnvironment].
"""

import pickle

import chex
import jax.numpy as jnp

from ..base import BaseRewardModule, BaseRewardParams, TLogReward, TReward
from ..environment import (
    QM9SmallEnvParams,
    QM9SmallEnvState,
)


@chex.dataclass(frozen=True)
class QM9SmallRewardParams(BaseRewardParams):
    """Parameters of the QM9 (small) reward.

    Attributes:
        rewards: Normalised rewards of all $11^5$ block sequences, indexed by the base-11
            number spelled by the tokens (first token most significant).
    """

    rewards: chex.Array


class QM9SmallRewardModule(
    BaseRewardModule[QM9SmallEnvState, QM9SmallEnvParams, QM9SmallRewardParams]
):
    r"""Proxy HOMO-LUMO gap of small molecules, precomputed for every block sequence.

    The proxy predictions $y(x)$ (a model trained on QM9, as released by Shen et al.,
    2023) are normalised once at initialisation:

    $$
    R(x) = \text{scale} \cdot \frac{\max(y(x), R_{\min})^{\gamma}}
    {\max_{x'} \max(y(x'), R_{\min})^{\gamma}},
    $$

    with $\gamma$ = `reward_exponent`, scale = `reward_scale` and $R_{\min}$ =
    `min_reward`. The resulting table is stored in `reward_params`, so evaluating the
    reward is a single lookup.

    Args:
        nchar: Number of building blocks (11).
        max_length: Blocks per molecule (5).
        min_reward: Lower clip $R_{\min}$ applied to the raw predictions.
        reward_exponent: Exponent $\gamma$.
        reward_scale: Value of the best molecule after normalisation.

    References:
        Shen, M. W., et al. (2023). Towards understanding and improving GFlowNet
        training. ICML.
    """

    def __init__(
        self,
        nchar: int = 11,
        max_length: int = 5,
        min_reward: float = 1e-3,
        reward_exponent: float = 5.0,
        reward_scale: float = 100.0,
    ):
        self.nchar = nchar
        self.max_length = max_length
        self.min_reward = min_reward
        self.reward_exponent = reward_exponent
        self.reward_scale = reward_scale

    def init(self, rng_key: chex.PRNGKey, dummy_state: QM9SmallEnvState) -> QM9SmallRewardParams:
        """Loads and normalises the precomputed proxy predictions.

        Reads `proxy/weights/qm9_small/block_qm9str_v1_s5.pkl` relative to the current
        working directory, so run from the repository root.

        Args:
            rng_key: Unused.
            dummy_state: Unused.

        Returns:
            `QM9SmallRewardParams` with the normalised reward table.
        """
        # Source: https://github.com/maxwshen/gflownet/blob/main/datasets/qm9str/block_qm9str_v1_s5.pkl
        with open("proxy/weights/qm9_small/block_qm9str_v1_s5.pkl", "rb") as f:
            oracle_d = pickle.load(f)
        oracle = {tuple(x): float(y) for x, y in oracle_d.items()}

        values_raw = jnp.array(list(oracle.values()))

        # Normalization as in https://github.com/maxwshen/gflownet/blob/main/exps/qm9str/qm9str.py
        values = jnp.clip(values_raw, min=self.min_reward)
        values = jnp.pow(values, self.reward_exponent)
        values = values * self.reward_scale / values.max()
        return QM9SmallRewardParams(rewards=values)

    def reward(self, state: QM9SmallEnvState, reward_params: QM9SmallRewardParams) -> TReward:
        """Looks up $R(x)$ for a single molecule.

        Args:
            state: Terminal QM9 state.
            reward_params: Parameters holding the reward table.

        Returns:
            Scalar reward. Only meaningful for terminal states: a non-terminal state
            containing `[PAD]` tokens can map to an unrelated entry of the table.
            Indices beyond the table return `min_reward`.
        """
        powers_array = jnp.array([
            self.nchar ** (self.max_length - i - 1) for i in range(self.max_length)
        ])
        index = jnp.sum(state.tokens * powers_array)
        # Out-of-range indices (e.g. from PAD tokens) get min_reward, not a clamped entry
        return reward_params.rewards.at[index].get(mode="fill", fill_value=self.min_reward)

    def log_reward(
        self, state: QM9SmallEnvState, reward_params: QM9SmallRewardParams
    ) -> TLogReward:
        r"""Computes $\log R(x)$ (log of `reward`)."""
        return jnp.log(self.reward(state, reward_params))
