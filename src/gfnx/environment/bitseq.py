from .sequence import (
    EnvParams,  # noqa: F401
    EnvState,  # noqa: F401
    NonAutoregressiveSequenceEnvironment,
)


class BitseqEnvironment(NonAutoregressiveSequenceEnvironment):
    def __init__(self, n: int = 120, k: int = 8) -> None:
        self.n = n
        self.k = k
        assert n % k == 0, "n should be divisible by k"

        super().__init__(
            max_length=n // k,
            nchar=2**k,
            ntoken=2**k + 3,
            bos_token=2**k + 1,
            eos_token=2**k + 2,
            pad_token=2**k,
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return f"Bitseq-{self.n}-{self.k}-v0"
