import chex
import jax.numpy as jnp


def mask_logits(logits: chex.Array, invalid_actions_mask: chex.Array) -> chex.Array:
    chex.assert_equal_shape([logits, invalid_actions_mask])
    min_logit = jnp.finfo(logits.dtype).min
    return jnp.where(invalid_actions_mask, min_logit, logits)
