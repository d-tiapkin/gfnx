import chex
import jax.numpy as jnp


def torus_adjacency(N: int, dtype=jnp.uint8) -> chex.Array:
    """
    Adjacency for an N x N torus grid (von Neumann neighborhood).
    Returns an (N^2, N^2) matrix A with A[u, v] = 1 iff u and v are neighbors.
    """
    idx = jnp.arange(N * N).reshape(N, N)

    up = jnp.roll(idx, -1, axis=0)
    down = jnp.roll(idx, 1, axis=0)
    left = jnp.roll(idx, -1, axis=1)
    right = jnp.roll(idx, 1, axis=1)

    rows = jnp.tile(idx.ravel(), 4)
    cols = jnp.concatenate([up.ravel(), down.ravel(), left.ravel(), right.ravel()])

    A = jnp.zeros((N * N, N * N), dtype=dtype)
    A = A.at[rows, cols].set(1)

    A = A.at[jnp.arange(N * N), jnp.arange(N * N)].set(0)
    return A


def get_true_ising_J(N: int, sigma: float) -> chex.Array:
    """
    Get the true Ising model couplings J for an N x N torus grid.
    """
    A = torus_adjacency(N)
    return sigma * A
