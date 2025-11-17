import gfnx
from gfnx.utils import PROTEINS_FULL_ALPHABET, AMINO_ACIDS

from .sequence import (
    EnvParams,  # noqa: F401
    EnvState,  # noqa: F401
    PrependAppendSequenceEnvironment,
)


class AMPEnvironment(PrependAppendSequenceEnvironment):
    def __init__(
        self,
        reward_module: gfnx.TRewardModule
    ) -> None:
        self.char_to_id = {
            char: i for i, char in enumerate(PROTEINS_FULL_ALPHABET)
        }

        super().__init__(
            reward_module, 
            max_length=60,
            nchar=len(AMINO_ACIDS),
            ntoken=len(PROTEINS_FULL_ALPHABET),
            bos_token=self.char_to_id["[BOS]"],
            eos_token=self.char_to_id["[EOS]"],
            pad_token=self.char_to_id["[PAD]"]
        )
