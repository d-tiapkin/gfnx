from dataclasses import dataclass
from typing import Callable, Literal

import chex
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class ExplorationState:
    """State of the exploration process."""

    schedule: Callable[[chex.Array], chex.Array]
    step: chex.Array


ExplorationScheduleType = Literal["linear", "exponential", "constant"]


def create_exploration_schedule(
    start_eps: float,
    end_eps: float = 0.0,
    exploration_steps: int = 10000,
    schedule_type: ExplorationScheduleType = "constant",
    decay_rate: float = 0.995,  # for exponential decay
) -> Callable[[chex.Array], chex.Array]:
    """Creates an exploration schedule function.

    Args:
        schedule_type: Type of exploration schedule ("linear", "exponential", "constant")
        start_eps: Initial exploration value
        end_eps: Final exploration value (for linear decay)
        exploration_steps: Number of steps for decay (for linear decay)
        decay_rate: Decay rate for exponential schedule

    Returns:
        A function that takes step index and returns current epsilon value
    """

    def linear_schedule(idx: chex.Array) -> chex.Array:
        """Linear decay from start_eps to end_eps over exploration_steps."""
        progress = jnp.minimum(idx / exploration_steps, 1.0)
        return start_eps + (end_eps - start_eps) * progress

    def exponential_schedule(idx: chex.Array) -> chex.Array:
        """Exponential decay with given rate."""
        return start_eps * (decay_rate**idx)

    def constant_schedule(idx: chex.Array) -> chex.Array:
        """Constant exploration rate."""
        return jnp.array(start_eps)

    schedules = {
        "linear": linear_schedule,
        "exponential": exponential_schedule,
        "constant": constant_schedule,
    }

    return schedules[schedule_type]


def apply_epsilon_greedy(
    rng_key: chex.PRNGKey,
    logits: chex.Array,
    epsilon: chex.Array,
) -> chex.Array:
    """Applies epsilon-greedy exploration to logits.

    Args:
        rng_key: JAX random key
        logits: Original logits from policy
        epsilon: Current exploration rate

    Returns:
        Modified logits with exploration
    """
    random_key, choice_key = jax.random.split(rng_key)

    # Generate random logits
    random_logits = jax.random.uniform(
        random_key,
        shape=logits.shape,
        minval=-1.0,
        maxval=1.0,
    )

    # Choose between random and policy logits
    explore = jax.random.uniform(choice_key) < epsilon
    return jnp.where(explore, random_logits, logits)


def apply_epsilon_greedy_vmap(
    rng_key: chex.PRNGKey,
    logits: chex.Array,
    epsilon: chex.Array,
) -> chex.Array:
    rng_keys = jax.random.split(rng_key, logits.shape[0])
    return jax.vmap(apply_epsilon_greedy, in_axes=(0, 0, None))(
        rng_keys, logits, epsilon
    )
