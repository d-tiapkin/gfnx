import chex
import jax
import jax.numpy as jnp


def masked_sum(
    x: chex.Array, mask: chex.Array, axis: int | tuple[int, ...] | None = None
) -> chex.Array:
    """Sum of `x` over `axis`, counting only entries where `mask` is True."""
    return jnp.sum(jnp.where(mask, x, jnp.zeros_like(x)), axis=axis)


def masked_mean(
    x: chex.Array, mask: chex.Array, axis: int | tuple[int, ...] | None = None
) -> chex.Array:
    """Mean of `x` over `axis`, counting only entries where `mask` is True."""
    total = masked_sum(x, mask, axis=axis)
    count = jnp.sum(mask.astype(x.dtype), axis=axis)
    return total / jnp.maximum(count, 1)


def compute_action_log_probs(
    logits: chex.Array,
    actions: chex.Array,
    invalid_mask: chex.Array,
    pad_mask: chex.Array | None = None,
) -> chex.Array:
    """Compute log-probabilities of selected actions under a masked softmax policy.

    Args:
        logits: Raw logits, shape [..., n_actions].
        actions: Selected action indices, shape [...].
        invalid_mask: Boolean mask where True = invalid action, shape [..., n_actions].
        pad_mask: Optional boolean mask where True = padding step (zero out), shape [...].

    Returns:
        Log-probabilities of selected actions, shape [...]. Padding steps are 0.0.
    """
    log_probs = jax.nn.log_softmax(logits, where=jnp.logical_not(invalid_mask), axis=-1)
    selected = jnp.take_along_axis(log_probs, jnp.expand_dims(actions, -1), axis=-1).squeeze(-1)
    if pad_mask is not None:
        selected = jnp.where(pad_mask, 0.0, selected)
    return selected
