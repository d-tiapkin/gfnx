from typing import Tuple

import chex

class RewardProxyDataset:
    def train_set(self) -> Tuple[chex.Array, chex.Array]:
        raise NotImplementedError

    def test_set(self) -> Tuple[chex.Array, chex.Array]:
        raise NotImplementedError

    @property
    def max_len(self) -> int:
        raise NotImplementedError