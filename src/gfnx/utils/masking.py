"""Masking utilities for GFlowNet computations.

Mask convention used throughout this module (and the rest of the library):

    Every ``*_mask`` argument follows ``True = valid / include``.

That is, a ``True`` entry means "count this element / consider this action /
keep this step", while ``False`` means "ignore / mask out". This matches the
semantics of ``jax.nn.{soft,log_soft}max(..., where=mask, axis=-1)`` and the
``mask=`` argument accepted by :func:`masked_sum` / :func:`masked_mean` below.
"""

import chex
import jax
import jax.numpy as jnp


def masked_sum(
    x: chex.Array, mask: chex.Array, axis: int | tuple[int, ...] | None = None
) -> chex.Array:
    """Sum of ``x`` over ``axis``, counting only entries where ``mask`` is True.

    Convention: ``True`` = include, ``False`` = ignore.
    """
    return jnp.sum(jnp.where(mask, x, jnp.zeros_like(x)), axis=axis)


def masked_mean(
    x: chex.Array, mask: chex.Array, axis: int | tuple[int, ...] | None = None
) -> chex.Array:
    """Mean of ``x`` over ``axis``, counting only entries where ``mask`` is True.

    Convention: ``True`` = include, ``False`` = ignore. Returns 0 when no
    entries are included along ``axis`` (rather than NaN).
    """
    total = masked_sum(x, mask, axis=axis)
    count = jnp.sum(mask.astype(x.dtype), axis=axis)
    return total / jnp.maximum(count, 1)


def compute_action_log_probs(
    logits: chex.Array,
    actions: chex.Array,
    action_mask: chex.Array,
    step_mask: chex.Array | None = None,
) -> chex.Array:
    """Compute log-probabilities of selected actions under a masked softmax policy.

    Args:
        logits: Raw logits, shape ``[..., n_actions]``.
        actions: Selected action indices, shape ``[...]``.
        action_mask: Boolean mask, shape ``[..., n_actions]``. ``True`` = valid
            action (included in the softmax), ``False`` = invalid (excluded).
        step_mask: Optional boolean mask, shape ``[...]``. ``True`` = real step
            (keep its log-prob), ``False`` = padding step (zero out the
            log-prob in the output). When omitted, no per-step zeroing is
            applied.

    Returns:
        Log-probabilities of the selected actions, shape ``[...]``. Padding
        steps (``step_mask == False``) are returned as ``0.0`` when
        ``step_mask`` is provided.
    """
    log_probs = jax.nn.log_softmax(logits, where=action_mask, axis=-1)
    selected = jnp.take_along_axis(log_probs, jnp.expand_dims(actions, -1), axis=-1).squeeze(-1)
    if step_mask is not None:
        selected = jnp.where(step_mask, selected, 0.0)
    return selected
