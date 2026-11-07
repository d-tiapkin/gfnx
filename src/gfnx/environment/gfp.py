"""Green fluorescent protein (GFP) environment (experimental).

The proxy reward lives in [`gfnx.reward.gfp`][gfnx.reward.gfp].
"""

from ..utils import AMINO_ACIDS, PROTEINS_FULL_ALPHABET
from .sequence import (
    EnvParams,  # noqa: F401
    EnvState,  # noqa: F401
    FixedAutoregressiveSequenceEnvironment,
)


class GFPEnvironment(FixedAutoregressiveSequenceEnvironment):
    """Autoregressive generation of protein sequences of fixed length 237 (experimental).

    This is a
    [`FixedAutoregressiveSequenceEnvironment`][gfnx.environment.sequence.FixedAutoregressiveSequenceEnvironment]
    with `max_length = 237` and `nchar = 20` (the standard amino acids):

    * **State** — `tokens` of shape `[237]` over `gfnx.utils.PROTEINS_FULL_ALPHABET`.
    * **Forward actions** — 20 actions, appending an amino acid.
    * **Backward actions** — a single action removing the last amino acid.
    * **Observation** — `[bos_token, *tokens]`, shape `[238]`.
    * **Episode length** — exactly 237 steps.

    The environment has no constructor arguments. No pretrained proxy is shipped yet;
    see [`EqxProxyGFPRewardModule`][gfnx.reward.gfp.EqxProxyGFPRewardModule].

    References:
        Madan, K., et al. (2023). Learning GFlowNets from partial episodes for improved
        convergence and stability. ICML.

        Sarkisyan, K., et al. (2016). Local fitness landscape of the green fluorescent
        protein. Nature, 533.
    """

    def __init__(self) -> None:
        self.char_to_id = {char: i for i, char in enumerate(PROTEINS_FULL_ALPHABET)}

        super().__init__(
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
