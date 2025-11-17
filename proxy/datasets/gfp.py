'''
To use this code, you need to install the library `design-bench`: https://github.com/brandontrabucco/design-bench/tree/chris/fixes-v2
IMPORTANT: install exactly this branch, not the main branch, as the main branch has some issues with the tokenizers.
'''
from typing import Tuple

import chex

from design_bench.datasets.discrete.gfp_dataset import GFPDataset
import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split

from .base import RewardProxyDataset

class GFPRewardProxyDataset(RewardProxyDataset):
    '''
    This class is a dataset for the GFP reward proxy task, needed to train a reward proxy model
    Credits: https://github.com/MJ10/BioSeq-GFN-AL/blob/faster-impl/lib/dataset/gfp.py
    '''
    def __init__(self) -> None:
        split_rng_key = np.random.RandomState(0)
        dataset = GFPDataset()
        dataset.map_normalize_y()
        # Do test-valid-train splitting
        x, y = dataset.x, dataset.y
        self.offset = jnp.min(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=split_rng_key)
        # Convert to jax arrays
        self.train_data, self.train_score = jnp.array(x_train), jnp.array(y_train)
        self.test_data, self.test_score = jnp.array(x_test), jnp.array(y_test)
        # TODO: they are doing some offsetting here, but I don't know why
    
    def train_set(self) -> Tuple[chex.Array, chex.Array]:
        return self.train_data, self.train_score
    
    def test_set(self) -> Tuple[chex.Array, chex.Array]:
        return self.test_data, self.test_score

    @property
    def max_len(self) -> int:
        return 237
    
# if __name__ == '__main__':
#     dataset = GFPRewardProxyDataset()
#     print('Offset:', dataset.offset)