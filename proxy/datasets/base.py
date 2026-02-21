from typing import ClassVar

import chex

from gfnx.utils import PROTEINS_FULL_ALPHABET


class RewardProxyDataset:
    """Base class for reward proxy datasets for protein design tasks."""

    char_to_id: ClassVar[dict[str, int]] = {
        char: i for i, char in enumerate(PROTEINS_FULL_ALPHABET)
    }

    def train_set(self) -> tuple[chex.Array, chex.Array]:
        raise NotImplementedError

    def test_set(self) -> tuple[chex.Array, chex.Array]:
        raise NotImplementedError

    @property
    def max_len(self) -> int:
        raise NotImplementedError

    @property
    def offset(self) -> float:
        return 0.0
