from ..base import TRewardModule
from ..utils import NUCLEOTIDES, NUCLEOTIDES_FULL_ALPHABET
from .sequence import (
    EnvParams,  # noqa: F401
    EnvState,  # noqa: F401
    NonAutoregressiveSequenceEnvironment,
    FixedAutoregressiveSequenceEnvironment,
)


class TFBind8Environment(FixedAutoregressiveSequenceEnvironment):
    def __init__(self, reward_module: TRewardModule) -> None:
        self.char_to_id = {
            char: i for i, char in enumerate(NUCLEOTIDES_FULL_ALPHABET)
        }

        super().__init__(
            reward_module,
            max_length=8,
            nchar=len(NUCLEOTIDES),
            ntoken=len(NUCLEOTIDES_FULL_ALPHABET),
            bos_token=self.char_to_id["[BOS]"],
            eos_token=self.char_to_id["[EOS]"],
            pad_token=self.char_to_id["[PAD]"],
        )
