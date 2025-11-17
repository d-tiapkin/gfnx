import gfnx

from .sequence import (
    NonAutoregressiveSequenceEnvironment, 
    EnvState,  # noqa: F401
    EnvParams  # noqa: F401
)
from gfnx.utils import AMINO_ACIDS, PROTEINS_FULL_ALPHABET


class GFPEnvironment(NonAutoregressiveSequenceEnvironment):
    def __init__(
        self,
        reward_module: gfnx.TRewardModule
    ) -> None:
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
            pad_token=self.char_to_id["[PAD]"]
        )
