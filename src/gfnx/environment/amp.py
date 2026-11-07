"""Antimicrobial peptide (AMP) environment: variable-length protein sequences.

The proxy reward lives in [`gfnx.reward.amp`][gfnx.reward.amp].
"""

import chex
import jax.numpy as jnp

from ..utils import AMINO_ACIDS, PROTEINS_FULL_ALPHABET
from .sequence import (
    AutoregressiveSequenceEnvironment,
    EnvParams,
    EnvState,
)


class AMPEnvironment(AutoregressiveSequenceEnvironment):
    """Autoregressive generation of peptides of length at most 60.

    Peptides are written left to right over the 20 standard amino acids, and the agent
    decides when to stop. This is an
    [`AutoregressiveSequenceEnvironment`][gfnx.environment.sequence.AutoregressiveSequenceEnvironment]
    with `max_length = 60` and `nchar = 20`:

    * **State** — `tokens` of shape `[60]` over `gfnx.utils.PROTEINS_FULL_ALPHABET`
      (amino acids `0..19`, then `[BOS]`, `[EOS]`, `[PAD]`).
    * **Forward actions** — 21 actions: append amino acid `c < 20`, or stop (`20`), which
      writes `[EOS]`.
    * **Backward actions** — a single action removing the last token ($P_B$ degenerate).
    * **Observation** — `[*tokens, last]`, shape `[61]`, where `last` is `[EOS]` if the
      buffer is full without a stop token, and `[PAD]` otherwise. Every complete
      peptide, including one that fills the buffer without a stop action, therefore
      ends with an end-of-sequence marker in the policy observation. Unlike the other
      sequence environments, no `[BOS]` is prepended.
    * **Episode length** — at most 60 steps.

    The environment has no constructor arguments.

    References:
        Malkin, N., et al. (2022). Trajectory balance: Improved credit assignment in
        GFlowNets. NeurIPS.

        Pirtskhalava, M., et al. (2021). DBAASP v3: database of antimicrobial/cytotoxic
        activity and structure of peptides as a resource for development of new
        therapeutics. Nucleic Acids Research, 49.
    """

    def __init__(self) -> None:
        self.char_to_id = {char: i for i, char in enumerate(PROTEINS_FULL_ALPHABET)}

        super().__init__(
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
        """Returns `tokens` followed by one extra token, shape `[max_length + 1]`.

        The extra token is `eos_token` when the buffer is full and its last token is a
        character (the sequence hit `max_length` without a stop action), and `pad_token`
        otherwise.
        """
        last_token = state.tokens[-1]
        to_append = jnp.where(
            jnp.logical_or(last_token == self.pad_token, last_token == self.eos_token),
            self.pad_token,
            self.eos_token,
        )
        return jnp.concat([state.tokens, to_append[None]], axis=-1)
