"""
To use this code, install `clamp_common_eval` library to load a dataset: https://github.com/MJ10/clamp-gen-data/tree/master
Important: you need the main branch
Also, install `biopython`, `polyleven`, `pandas`, `torch`, and `transformers`
before running this code: `pip install biopython polyleven`
Make sure that git-lfs is installed in your system before cloning the
repository, as the data files are stored using git-lfs.
To install git-lfs, follow the instructions here: https://git-lfs.github.com/
"""

from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
from datasets.base import RewardProxyDataset
from sklearn.model_selection import GroupKFold


class AMPRewardProxyDataset(RewardProxyDataset):
    """
    This class is a dataset for the AMP reward proxy task, needed to train a
    reward proxy model
    Credits: https://github.com/MJ10/BioSeq-GFN-AL/blob/faster-impl/lib/dataset/amp.py
    """

    def __init__(self, split="D", nfold=5) -> None:
        rng = np.random.RandomState(142857)
        # Get the data
        from clamp_common_eval.defaults import get_default_data_splits  # pyright: ignore[reportMissingImports]

        source = get_default_data_splits(setting="Target")
        data = source.sample(split, -1)
        # Get groups and split the data
        groups = np.concatenate((
            np.array(source.d1_pos.group),
            np.array(source.d2_pos.group),
        ))
        n_pos, n_neg = len(data["AMP"]), len(data["nonAMP"])
        pos_train, pos_valid = next(
            GroupKFold(nfold).split(np.arange(n_pos), groups=groups)
        )
        neg_train, neg_valid = next(
            GroupKFold(nfold).split(
                np.arange(n_neg), groups=rng.randint(0, nfold, n_neg)
            )
        )

        # Collect splitted data
        pos_train = [data["AMP"][i] for i in pos_train]
        neg_train = [data["nonAMP"][i] for i in neg_train]
        pos_valid = [data["AMP"][i] for i in pos_valid]
        neg_valid = [data["nonAMP"][i] for i in neg_valid]
        # Next, tokenize and pad the sequences
        self.train_data = self._tokenize_and_pad(pos_train + neg_train)
        self.test_data = self._tokenize_and_pad(pos_valid + neg_valid)

        self.train_score = jnp.concatenate([
            jnp.ones(len(pos_train)),
            jnp.zeros(len(neg_train)),
        ])
        self.test_score = jnp.concatenate([
            jnp.ones(len(pos_valid)),
            jnp.zeros(len(neg_valid)),
        ])

    def _tokenize_and_pad(self, seqs) -> chex.Array:
        eos_token = self.char_to_id["[EOS]"]
        pad_token = self.char_to_id["[PAD]"]
        processed_seqs = []
        for seq in seqs:
            seq = [
                self.char_to_id[char] for char in seq
            ]  # All characters should be correct
            if len(seq) < self.max_len:
                seq += [eos_token] + [pad_token] * (
                    self.max_len - len(seq) - 1
                )
            processed_seqs.append(seq[: self.max_len])
        return jnp.array(processed_seqs)

    def train_set(self) -> Tuple[chex.Array, chex.Array]:
        return self.train_data, self.train_score

    def test_set(self) -> Tuple[chex.Array, chex.Array]:
        return self.test_data, self.test_score

    @property
    def max_len(self) -> int:
        return 60
