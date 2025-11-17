import gfnx

from .sequence import (
    EnvParams,  # noqa: F401
    EnvState,  # noqa: F401
    NonAutoregressiveSequenceEnvironment,
)


class BitseqEnvironment(NonAutoregressiveSequenceEnvironment):
    def __init__(
        self,
        reward_module: gfnx.TRewardModule,
        n: int = 120,
        k: int = 8
    ) -> None:
        self.n = n
        self.k = k
        assert n % k == 0, "n should be divisible by k"

        super().__init__(
            reward_module, 
            max_length=n // k,
            nchar=2**k,
            ntoken=2**k + 3,
            bos_token=2**k + 1,
            eos_token=2**k + 2,
            pad_token=2**k
        )
