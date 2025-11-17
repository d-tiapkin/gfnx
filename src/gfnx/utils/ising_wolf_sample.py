"""JAX implementation of Wolff sampling for J = sigma * A_N on an N x N torus.
Energy convention: P(x) ∝ exp{-alpha * E(x)},  E(x) = - x^T J x,  J = sigma * A_N.
With this convention, the Wolff/Swendsen-Wang bond probability is:
    p_bond = 1 - exp(-4 * alpha * |sigma|).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax_tqdm import scan_tqdm


def _checkerboard_like(spins: jnp.ndarray) -> jnp.ndarray:
    """Return η[i,j] = (-1)^(i+j) with same shape/dtype as spins (±1 in int8)."""
    N = spins.shape[0]
    ii = jnp.arange(N)[:, None]
    jj = jnp.arange(N)[None, :]
    eta = 1 - 2 * ((ii + jj) & 1)  # +1 on even parity, -1 on odd
    return eta.astype(spins.dtype)


def _bond_prob(alpha: float, sigma: float) -> jnp.ndarray:
    """p_bond = 1 - exp(-4 * alpha * |sigma|) from the given energy convention."""
    return 1.0 - jnp.exp(-4.0 * alpha * jnp.abs(sigma))


def _wolff_step_ferro(
    key: jax.Array, spins: jnp.ndarray, p_bond: jax.Array
) -> tuple[jnp.ndarray, jax.Array]:
    """
    One Wolff cluster flip in a ferromagnetic frame (spins ∈ {±1}, shape (N,N)).

    Implementation notes:
    - Uniformly picks a seed site.
    - Grows a single same-spin cluster using bond prob p_bond on equal-spin edges.
    - Uses periodic BCs via jnp.roll.
    - No Python loops: BFS is a lax.while_loop; RNG handled per-iteration.
    """
    N = spins.shape[0]

    # Seed site (uniform over lattice)
    key, k_seed = jax.random.split(key)
    flat = jax.random.randint(k_seed, (), 0, N * N)
    i0 = flat // N
    j0 = flat % N

    # Seed mask and seed spin
    seed_mask = jnp.zeros((N, N), dtype=bool).at[i0, j0].set(True)
    seed_spin = spins[i0, j0]
    same_as_seed = spins == seed_spin

    # Precompute equal-spin adjacency (aligned at "source" sites)
    eq_r = spins == jnp.roll(spins, -1, axis=1)
    eq_l = spins == jnp.roll(spins, +1, axis=1)
    eq_d = spins == jnp.roll(spins, -1, axis=0)
    eq_u = spins == jnp.roll(spins, +1, axis=0)

    # While there is a frontier, try to add neighbors via Bernoulli(p_bond)
    def cond_fun(state):
        _cluster, frontier, _key = state
        return jnp.any(frontier)

    def body_fun(state):
        cluster, frontier, key = state

        # Neighbor not yet in cluster (aligned at source sites)
        nb_out_r = ~jnp.roll(cluster, -1, axis=1)
        nb_out_l = ~jnp.roll(cluster, +1, axis=1)
        nb_out_d = ~jnp.roll(cluster, -1, axis=0)
        nb_out_u = ~jnp.roll(cluster, +1, axis=0)

        # Candidate edges from current frontier to neighbors
        cand_r = frontier & eq_r & nb_out_r & same_as_seed
        cand_l = frontier & eq_l & nb_out_l & same_as_seed
        cand_d = frontier & eq_d & nb_out_d & same_as_seed
        cand_u = frontier & eq_u & nb_out_u & same_as_seed

        # One RNG draw for all four directions this iteration
        key, k_iter = jax.random.split(key)
        bern = jax.random.uniform(k_iter, (4, N, N)) < p_bond

        add_r_sites = bern[0] & cand_r
        add_l_sites = bern[1] & cand_l
        add_d_sites = bern[2] & cand_d
        add_u_sites = bern[3] & cand_u

        # Shift site decisions onto neighbor positions to form next frontier
        add_r = jnp.roll(add_r_sites, +1, axis=1)  # goes to (i, j+1)
        add_l = jnp.roll(add_l_sites, -1, axis=1)  # goes to (i, j-1)
        add_d = jnp.roll(add_d_sites, +1, axis=0)  # goes to (i+1, j)
        add_u = jnp.roll(add_u_sites, -1, axis=0)  # goes to (i-1, j)

        new_nodes = (add_r | add_l | add_d | add_u) & (~cluster)
        new_cluster = cluster | new_nodes
        new_front = new_nodes
        return (new_cluster, new_front, key)

    cluster0 = seed_mask
    frontier0 = seed_mask
    cluster, _, key = lax.while_loop(cond_fun, body_fun, (cluster0, frontier0, key))

    # Flip the cluster (rejection-free)
    new_spins = jnp.where(cluster, -spins, spins)
    return new_spins, key


def _wolff_step_any_sign(
    key: jax.Array, spins: jnp.ndarray, sigma: float, alpha: float
) -> tuple[jnp.ndarray, jax.Array]:
    """
    One Wolff step for J = sigma * A_N under the given energy convention.
    For sigma < 0, applies the checkerboard gauge (valid for bipartite cases).
    """
    p = _bond_prob(alpha, sigma)
    pred = jnp.greater_equal(sigma, 0.0)  # tracer-safe predicate

    def ferro(args):
        k, s = args
        s, k = _wolff_step_ferro(k, s, p)
        return (s, k)

    def anti(args):
        k, s = args
        eta = _checkerboard_like(s)
        s_f = eta * s
        s_f, k = _wolff_step_ferro(k, s_f, p)
        return (eta * s_f, k)

    spins, key = lax.cond(pred, ferro, anti, operand=(key, spins))
    return spins, key


def _burn_in_scan(carry, _):
    key, spins, sigma, alpha = carry
    spins, key = _wolff_step_any_sign(key, spins, sigma, alpha)
    return (key, spins, sigma, alpha), None


def _sweep_scan(carry, _):
    key, spins, sigma, alpha = carry
    spins, key = _wolff_step_any_sign(key, spins, sigma, alpha)
    return (key, spins, sigma, alpha), None


def _sample_outer_scan_factory(sweeps_per_sample: int):
    """Close over a static sweeps_per_sample and return an outer-scan body."""

    def _body(carry, _):
        key, spins, sigma, alpha = carry
        (key, spins, sigma, alpha), _ = lax.scan(
            _sweep_scan,
            (key, spins, sigma, alpha),
            xs=None,
            length=sweeps_per_sample,
        )
        return (key, spins, sigma, alpha), spins

    return _body


@partial(
    jax.jit,
    static_argnames=("N", "num_samples", "burn_in", "sweeps_per_sample"),
)
def wolff_sampler(
    key: jax.Array,
    N: int,
    sigma: float,
    alpha: float,
    num_samples: int,
    burn_in: int = 200,
    sweeps_per_sample: int = 1,
    init_spins: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    """
    Generate Wolff samples for the N x N torus with J = sigma * A_N.

    Args (static where noted):
      key : PRNGKey
      N   : int [static]  - lattice side (shape is (N,N))
      sigma : float       - coupling scale (can be ±; when <0 we gauge)
      alpha : float       - inverse-noise scale in P(x) ∝ exp{-alpha * E(x)}
      num_samples : int [static] - number of configurations to return
      burn_in     : int [static] - Wolff steps to discard up front
      sweeps_per_sample : int [static] - Wolff steps between saved samples
      init_spins : optional (N,N) array of ±1 (int8/ int32 / float) initial state

    Returns:
      samples     : (num_samples, N, N) int8 array of spins in {+1,-1}
      final_state : (N, N) int8 array (last state)
      final_key   : PRNGKey

    Notes:
      • Uses the correct bond prob for E = - x^T J x:  p = 1 - exp(-4 * alpha * |sigma|).
      • For sigma < 0, applies the checkerboard gauge internally (valid for bipartite cases).
      • No Python loops: burn-in and sampling use lax.scan; cluster growth uses while_loop.
      • If you change N/loop lengths/sweeps, JAX recompiles (once) for the new static values.
    """
    # Initialize spins (±1 in int8)
    if init_spins is None:
        key, k0 = jax.random.split(key)
        spins = 2 * jax.random.randint(k0, (N, N), 0, 2, dtype=jnp.int8) - 1
    else:
        spins = jnp.sign(init_spins).astype(jnp.int8)
        spins = jnp.where(spins == 0, jnp.int8(1), spins)  # avoid zeros

    # Burn-in
    (key, spins, _, _), _ = lax.scan(
        _burn_in_scan, (key, spins, sigma, alpha), xs=None, length=burn_in
    )

    # Sampling
    outer_body = _sample_outer_scan_factory(sweeps_per_sample)
    wrapped_outer_body = scan_tqdm(num_samples)(outer_body)
    (key, spins, _, _), samples = lax.scan(
        wrapped_outer_body,
        (key, spins, sigma, alpha),
        xs=jnp.arange(num_samples),  # for tqdm
        length=num_samples,
    )

    samples = samples.astype(jnp.int8)
    spins = spins.astype(jnp.int8)
    return samples, spins, key
