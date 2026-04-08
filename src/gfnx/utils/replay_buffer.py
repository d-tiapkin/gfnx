from collections.abc import Callable
from typing import Generic, TypeVar

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flashbax import make_item_buffer
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState

Experience = TypeVar("Experience", bound=chex.ArrayTree)
ReplayBufferState = TrajectoryBufferState


@chex.dataclass(frozen=True)
class ReplayBuffer(Generic[Experience]):
    """Pure functions defining a replay buffer.

    Mostly follows flashbax's item buffer API.
    """

    init: Callable[[Experience], ReplayBufferState[Experience]]
    add: Callable[[ReplayBufferState[Experience], Experience], ReplayBufferState[Experience]]
    sample: Callable[[ReplayBufferState[Experience], chex.PRNGKey], Experience]
    can_sample: Callable[[ReplayBufferState[Experience]], chex.Array]


def make_replay_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
) -> ReplayBuffer:
    """Creates a replay buffer that filters out padded transitions before adding.
    """

    buffer = make_item_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_sequences=False,
        add_batches=True,
    )

    def init_fn(single_experience: Experience) -> ReplayBufferState[Experience]:
        """Initializes the replay buffer with a single experience.
        """
        return buffer.init(single_experience)

    def add_fn(
        state: ReplayBufferState[Experience], transitions: Experience
    ) -> ReplayBufferState[Experience]:
        """Adds a batch of transitions to the replay buffer.

        Given a mixed batch of padded and unpadded transitions (distinguished by a
        boolean ``.pad`` field), only the unpadded ones are inserted. The number of
        unpadded items is decomposed into powers of two so that each ``buffer.add``
        call receives a static-shaped slice.
        """
        # Sort transitions by padding: [unpadded, padded]
        sort_idx = jnp.argsort(transitions.pad, stable=True)
        transitions = jax.tree.map(lambda x: x[sort_idx], transitions)

        n = transitions.pad.shape[0]
        num_unpadded = jnp.sum(~transitions.pad)

        # Compute the length of binary decomposition
        with jax.ensure_compile_time_eval():
            max_exp = int(np.floor(np.log2(n))) + 1
            sizes = (2 ** np.arange(max_exp))[::-1]

        processed_idx = jnp.int32(0)
        for bs in sizes:
            to_add = processed_idx + bs <= num_unpadded
            state = jax.lax.cond(
                to_add,
                lambda state, processed_idx, bs=bs: buffer.add(
                    state,
                    jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, processed_idx, bs),
                                 transitions),
                ),
                lambda state, processed_idx, bs=bs: state,
                state, processed_idx
            )
            processed_idx = processed_idx + jnp.where(to_add, bs, 0)

        return state

    def sample_fn(state: ReplayBufferState[Experience], rng_key: chex.PRNGKey) -> Experience:
        """Samples a batch of transitions from the replay buffer.
        """
        return buffer.sample(state, rng_key).experience

    def can_sample_fn(state: ReplayBufferState[Experience]) -> chex.Array:
        """Checks if the replay buffer can sample.
        """
        return buffer.can_sample(state)

    return ReplayBuffer(
        init=init_fn,
        add=add_fn,
        sample=sample_fn,
        can_sample=can_sample_fn,
    )