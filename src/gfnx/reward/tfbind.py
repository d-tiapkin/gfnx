"""Lookup-table reward for the TFBind-8 environment.

See [`TFBind8Environment`][gfnx.environment.tfbind.TFBind8Environment].
"""

import itertools
import pickle

import chex
import jax.numpy as jnp
import numpy as np

from ..base import BaseRewardModule, BaseRewardParams, TLogReward, TReward
from ..environment import (
    TFBind8EnvParams,
    TFBind8EnvState,
)


@chex.dataclass(frozen=True)
class TFBind8RewardParams(BaseRewardParams):
    """Parameters of the TFBind-8 reward.

    Attributes:
        rewards: Normalised rewards of all $4^8$ sequences, shape `[65536]`, indexed by
            the base-4 number spelled by the tokens (first token most significant).
    """

    rewards: chex.Array


class TFBind8RewardModule(
    BaseRewardModule[TFBind8EnvState, TFBind8EnvParams, TFBind8RewardParams]
):
    r"""Wet-lab binding activity of DNA 8-mers to the transcription factor SIX6.

    The raw measurements $y(x)$ for all $4^8$ sequences (Barrera et al., 2016, as packaged
    by Shen et al., 2023) are normalised once at initialisation:

    $$
    R(x) = \max\left(\text{scale} \cdot \frac{y(x)^{\gamma}}{\max_{x'} y(x')^{\gamma}},\;
    R_{\min}\right),
    $$

    with $\gamma$ = `reward_exponent`, scale = `reward_scale` and $R_{\min}$ =
    `min_reward`, following Shen et al. (2023). The resulting table is stored in
    `reward_params`, so evaluating the reward is a single lookup.

    Args:
        nchar: Vocabulary size (4 nucleotides).
        max_length: Sequence length (8).
        min_reward: Lower clip $R_{\min}$.
        reward_exponent: Exponent $\gamma$ sharpening the measured activities.
        reward_scale: Value of the best sequence after normalisation.

    References:
        Shen, M. W., et al. (2023). Towards understanding and improving GFlowNet
        training. ICML.

        Barrera, L. A., et al. (2016). Survey of variation in human transcription factors
        reveals prevalent DNA binding changes. Science, 351(6280).
    """

    def __init__(
        self,
        nchar: int = 4,
        max_length: int = 8,
        min_reward: float = 1e-3,
        reward_exponent: float = 3.0,
        reward_scale: float = 10.0,
    ):
        self.nchar = nchar
        self.max_length = max_length
        self.min_reward = min_reward
        self.reward_exponent = reward_exponent
        self.reward_scale = reward_scale

    def init(self, rng_key: chex.PRNGKey, dummy_state: TFBind8EnvState) -> TFBind8RewardParams:
        """Loads and normalises the exhaustive TFBind-8 measurements.

        Reads `proxy/weights/tfbind/tfbind8-exact-v0-all.pkl` relative to the current
        working directory, so run from the repository root.

        Args:
            rng_key: Unused.
            dummy_state: Unused.

        Returns:
            `TFBind8RewardParams` with the normalised reward table.
        """
        # Make a full loop to get the values for all possible states

        # Generate all possible values of characters
        values = list(range(self.nchar))
        # Generate all possible arrays
        all_states = np.array(list(itertools.product(values, repeat=self.max_length)))

        # Source: https://github.com/maxwshen/gflownet/blob/main/datasets/tfbind8/tfbind8-exact-v0-all.pkl
        with open("proxy/weights/tfbind/tfbind8-exact-v0-all.pkl", "rb") as f:
            oracle_d = pickle.load(f)
        oracle = {tuple(x): float(y[0]) for x, y in zip(oracle_d["x"], oracle_d["y"])}

        values_raw = jnp.array([oracle[tuple(state)] for state in all_states])

        # Normalization as in https://github.com/maxwshen/gflownet/blob/main/exps/tfbind8/tfbind8_oracle.py
        values = jnp.pow(values_raw, self.reward_exponent)
        values = values * self.reward_scale / values.max()
        values = jnp.clip(values, min=self.min_reward)
        return TFBind8RewardParams(rewards=values)

    def reward(self, state: TFBind8EnvState, reward_params: TFBind8RewardParams) -> TReward:
        """Looks up $R(x)$ for a single sequence.

        Args:
            state: Terminal TFBind-8 state.
            reward_params: Parameters holding the reward table.

        Returns:
            Scalar reward; out-of-range indices (e.g. from `[PAD]` tokens in a
            non-terminal state) get `min_reward`.
        """
        powers_array = jnp.array([
            self.nchar ** (self.max_length - i - 1) for i in range(self.max_length)
        ])
        index = jnp.sum(state.tokens * powers_array)
        # Out-of-range indices (e.g. from PAD tokens) get min_reward, not a clamped entry
        return reward_params.rewards.at[index].get(mode="fill", fill_value=self.min_reward)

    def log_reward(self, state: TFBind8EnvState, reward_params: TFBind8RewardParams) -> TLogReward:
        r"""Computes $\log R(x)$ (log of `reward`)."""
        return jnp.log(self.reward(state, reward_params))
