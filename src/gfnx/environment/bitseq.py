"""Bit-sequence environment: fill an $n$-bit string with $k$-bit words in any order.

Rewards live in [`gfnx.reward.bitseq`][gfnx.reward.bitseq].
"""

from .sequence import (
    EnvParams,  # noqa: F401
    EnvState,  # noqa: F401
    NonAutoregressiveSequenceEnvironment,
)


class BitseqEnvironment(NonAutoregressiveSequenceEnvironment):
    r"""Non-autoregressive generation of binary strings of length $n$.

    The string is split into $n/k$ words of $k$ bits. A state is a sequence of $n/k$
    words, each either empty ($\oslash$) or one of the $2^k$ $k$-bit words, and
    $s_0 = (\oslash, \ldots, \oslash)$. Each step replaces one empty word with a
    non-empty one, in any order; terminal states are the sequences without empty words,
    i.e. the $2^n$ binary strings of length $n$.

    This is a [`NonAutoregressiveSequenceEnvironment`][gfnx.environment.sequence.NonAutoregressiveSequenceEnvironment]
    with `max_length = n // k` and vocabulary `nchar = 2 ** k`:

    * **State** — `tokens` of shape `[n // k]`; each entry is a word id in `[0, 2^k)`, or
      `pad_token = 2^k` for $\oslash$ (`bos_token = 2^k + 1`, `eos_token = 2^k + 2`).
    * **Forward actions** — `(n // k) * 2^k` actions; action `a` writes word
      `a % 2^k` at position `a // 2^k` (valid iff the position is empty).
    * **Backward actions** — `n // k` actions; action `i` empties position `i`.
    * **Observation** — `[bos_token, *tokens]`, shape `[n // k + 1]`.
    * **Episode length** — exactly `n // k` steps.

    Use `gfnx.utils.bitseq.detokenize(tokens, k)` to recover the bit string from
    `tokens`.

    Args:
        n: Total number of bits $n$.
        k: Bits per word $k$; must divide `n`. Larger `k` gives shorter trajectories
            and a larger action space.

    References:
        Malkin, N., et al. (2022). Trajectory balance: Improved credit assignment in
        GFlowNets. NeurIPS.

        Tiapkin, D., et al. (2024). Generative flow networks as entropy-regularized RL.
        AISTATS.
    """  # noqa: E501

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
