import chex
import jax
import jax.numpy as jnp
from scipy import stats


def spearmanr(a: chex.Array, b: chex.Array):
    """Computation of Spearman rank correlation.

    Assumes that all components of a and b are different
    """
    chex.assert_equal_shape([a, b])

    def callback(a, b):
        res = stats.spearmanr(a, b)
        return jnp.float32(res.statistic)

    return jax.pure_callback(callback, jnp.float32(0), a, b)
