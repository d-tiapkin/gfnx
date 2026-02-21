import os
from typing import Any

import equinox as eqx
import orbax.checkpoint as ocp


def save_checkpoint(path: str, state: Any):
    """Save state to checkpoint, filtering out non-array fields.

    Args:
        path: Directory path for the checkpoint
        state: State (typically Equinox model)
    """

    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    params = eqx.filter(state, eqx.is_array)
    ckptr.save(
        os.path.abspath(path),  # Checkpoint path should be absolute.
        args=ocp.args.StandardSave(params)
    )
    ckptr.wait_until_finished()


def load_checkpoint(path: str, state: Any) -> Any:
    """Load state from checkpoint, preserving static components.

    Args:
        path: Directory path of the checkpoint
        state: Template state with same structure as saved checkpoint

    Returns:
        Restored state with loaded parameters and original static components

    Example:
        >>> model = eqx.nn.Linear(10, 5, key=jax.random.PRNGKey(0))
        >>> model = load_checkpoint("./checkpoints/model_state", model)
    """

    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    state_params, state_static = eqx.partition(state, eqx.is_array)
    loaded_params = ckptr.restore(
        os.path.abspath(path),  # Checkpoint path should be absolute.
        state_params
    )
    return eqx.combine(loaded_params, state_static)

