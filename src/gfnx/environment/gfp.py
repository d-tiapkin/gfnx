from ..base import TRewardModule
from ..utils import AMINO_ACIDS, PROTEINS_FULL_ALPHABET
from .sequence import (
    EnvParams,  # noqa: F401
    EnvState,  # noqa: F401
    FixedAutoregressiveSequenceEnvironment,
)


class GFPEnvironment(FixedAutoregressiveSequenceEnvironment):
    def __init__(self, reward_module: TRewardModule) -> None:
        self.char_to_id = {
            char: i for i, char in enumerate(PROTEINS_FULL_ALPHABET)
        }

        super().__init__(
            reward_module,
            max_length=237,
            nchar=len(AMINO_ACIDS),
            ntoken=len(PROTEINS_FULL_ALPHABET),
            bos_token=self.char_to_id["[BOS]"],
            eos_token=self.char_to_id["[EOS]"],
            pad_token=self.char_to_id["[PAD]"],
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "GFP-v0"
