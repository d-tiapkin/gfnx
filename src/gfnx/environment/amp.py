import chex
import jax.numpy as jnp

from ..base import TRewardModule
from ..utils import AMINO_ACIDS, PROTEINS_FULL_ALPHABET
from .sequence import (
    AutoregressiveSequenceEnvironment,
    EnvParams,
    EnvState,
)


class AMPEnvironment(AutoregressiveSequenceEnvironment):
    def __init__(self, reward_module: TRewardModule) -> None:
        self.char_to_id = {char: i for i, char in enumerate(PROTEINS_FULL_ALPHABET)}

        super().__init__(
            reward_module,
            max_length=60,
            nchar=len(AMINO_ACIDS),
            ntoken=len(PROTEINS_FULL_ALPHABET),
            bos_token=self.char_to_id["[BOS]"],
            eos_token=self.char_to_id["[EOS]"],
            pad_token=self.char_to_id["[PAD]"],
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "AMP-v0"

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Applies observation function to state."""

        # Use PAD if the last token is already PAD or EOS, otherwise use EOS
        last_token = state.tokens[:, -1]
        to_append = jnp.where(
            jnp.logical_or(last_token == self.pad_token, last_token == self.eos_token),
            self.pad_token,
            self.eos_token
        )
        to_append = to_append[:, None]  # Add dimension to match concatenation

        return jnp.concat(
            [
                state.tokens,
                to_append,
            ],
            axis=-1,
        )
