import chex
import jax
import jax.numpy as jnp


def mask_logits(logits: chex.Array, invalid_actions_mask: chex.Array) -> chex.Array:
    chex.assert_equal_shape([logits, invalid_actions_mask])
    min_logit = jnp.finfo(logits.dtype).min
    return jnp.where(invalid_actions_mask, min_logit, logits)


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
    log_probs = jax.nn.log_softmax(mask_logits(logits, invalid_mask), axis=-1)
    selected = jnp.take_along_axis(log_probs, jnp.expand_dims(actions, -1), axis=-1).squeeze(-1)
    if pad_mask is not None:
        selected = jnp.where(pad_mask, 0.0, selected)
    return selected
