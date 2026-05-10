import functools

import equinox as eqx
import jax
from jax_tqdm import loop_tqdm


def run_training_loop(train_step_fn, init_state, num_steps: int, tqdm_print_rate: int = 20):
    """Run a training loop using jax.lax.fori_loop with eqx.partition handling.

    Args:
        train_step_fn: Function (idx: int, state) -> state called at each step.
        init_state: Initial training state (any equinox pytree).
        num_steps: Number of training steps.
        tqdm_print_rate: How often to update the tqdm progress bar.

    Returns:
        Final training state (same type as init_state).
    """
    params, static = eqx.partition(init_state, eqx.is_array)

    @functools.partial(jax.jit, donate_argnums=(1,))
    @loop_tqdm(num_steps, print_rate=tqdm_print_rate)
    def wrapper(idx, params):
        state = eqx.combine(params, static)
        new_state = train_step_fn(idx, state)
        new_params, _ = eqx.partition(new_state, eqx.is_array)
        return new_params

    final_params = jax.lax.fori_loop(0, num_steps, wrapper, params)
    jax.block_until_ready(final_params)
    return eqx.combine(final_params, static)
