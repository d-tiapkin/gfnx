"""
Parallel Tempering (Replica Exchange) + checkerboard Heat-Bath for
J = sigma * A_N on an N x N torus under P(x) ∝ exp{-alpha * E(x)}, E(x) = - x^T (sigma A_N) x.

This sampler is always valid (handles frustration: e.g., odd N, sigma < 0).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax_tqdm import scan_tqdm


def _checkerboard_masks(N: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Boolean masks (N,N): black = (i+j)%2==0, white = ~black."""
    ii = jnp.arange(N)[:, None]
    jj = jnp.arange(N)[None, :]
    black = ((ii + jj) & 1) == 0
    white = ~black
    return black, white


def _neighbor_sum(spins: jnp.ndarray) -> jnp.ndarray:
    """
    Sum of 4 nearest neighbors with periodic BCs.
    spins: (..., N, N) in {±1}. Returns same shape as spins.
    """
    s = spins
    up = jnp.roll(s, +1, axis=-2)
    down = jnp.roll(s, -1, axis=-2)
    left = jnp.roll(s, +1, axis=-1)
    right = jnp.roll(s, -1, axis=-1)
    return up + down + left + right


def _energy_sigma_A_batch(spins: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """
    E(x) = - x^T (sigma A_N) x = -2*sigma * sum_{<i,j>} s_i s_j
    spins: (R,N,N) in {±1}. Returns (R,) energies (float32).
    """
    s = spins.astype(jnp.int32)
    right = jnp.roll(s, -1, axis=-1)
    down = jnp.roll(s, -1, axis=-2)
    edge_sum = (s * right + s * down).sum(axis=(-2, -1), dtype=jnp.int64)  # (R,)
    return (-2.0 * sigma) * edge_sum.astype(jnp.float32)


def _make_alpha_ladder(
    alpha_target: float, num_replicas: int, alpha_min_fraction: float
) -> jnp.ndarray:
    """
    Geometric ladder α_0 < ... < α_{R-1} = alpha_target  (float32).
    All math is JAX-friendly; no Python float() on tracers.
    """
    alpha_target = jnp.asarray(alpha_target, jnp.float32)
    frac = jnp.asarray(alpha_min_fraction, jnp.float32)
    # TODO: replace 1e-6 with a provided value in args
    alpha_min = jnp.maximum(jnp.asarray(1e-6, jnp.float32), alpha_target * frac)
    if num_replicas == 1:
        return jnp.array([alpha_target], dtype=jnp.float32)
    t = jnp.linspace(0.0, 1.0, num_replicas, dtype=jnp.float32)
    ladder = alpha_min * jnp.power(alpha_target / alpha_min, t)
    ladder = ladder.at[-1].set(alpha_target)  # exact target at the top
    return ladder


def _heat_bath_checkerboard(
    key: jax.Array,
    spins: jnp.ndarray,
    alphas: jnp.ndarray,
    sigma: float,
    black_mask: jnp.ndarray,
    white_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jax.Array]:
    """
    One full heat-bath sweep over all replicas using checkerboard updates.
    spins : (R, N, N) in {±1}, alphas : (R,)
    """
    R, N, _ = spins.shape
    bmask = jnp.broadcast_to(black_mask, (R, N, N))
    wmask = jnp.broadcast_to(white_mask, (R, N, N))

    # Update BLACK sites
    nbr = _neighbor_sum(spins)
    # With E = -2*sigma * sum_<i,j> s_i s_j, the local conditional uses field 4*alpha*sigma*m_i.
    # We'll compute p_plus via sigmoid(2H) with H = 2*alpha*sigma*m_i.
    H = 2.0 * alphas[:, None, None] * sigma * nbr
    p_plus = jax.nn.sigmoid(2.0 * H)
    key, kB = jax.random.split(key)
    uB = jax.random.uniform(kB, (R, N, N))
    new_black = jnp.where(uB < p_plus, jnp.int8(1), jnp.int8(-1))
    spins = jnp.where(bmask, new_black, spins)

    # Update WHITE sites (recompute neighbor sums)
    nbr = _neighbor_sum(spins)
    H = 2.0 * alphas[:, None, None] * sigma * nbr
    p_plus = jax.nn.sigmoid(2.0 * H)
    key, kW = jax.random.split(key)
    uW = jax.random.uniform(kW, (R, N, N))
    new_white = jnp.where(uW < p_plus, jnp.int8(1), jnp.int8(-1))
    spins = jnp.where(wmask, new_white, spins)

    return spins, key


def _replica_exchange(
    key: jax.Array,
    spins: jnp.ndarray,  # (R, N, N), int8 in {±1}
    energies: jnp.ndarray,  # (R,), float32
    alphas: jnp.ndarray,  # (R,), float32
    parity: jnp.ndarray,  # scalar int32, 0 or 1
) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    """
    Parallel Tempering neighbor-exchange step (static shapes, vectorized).

    Active pairs depend on `parity`:
      parity = 0  -> edges (0,1), (2,3), (4,5), ...
      parity = 1  -> edges (1,2), (3,4), (5,6), ...

    Acceptance per adjacent edge e=(e, e+1):
      A = min(1, exp( (alpha[e] - alpha[e+1]) * (E[e+1] - E[e]) ))

    Returns updated (spins, energies, key). Safe when R<2 (no-op).
    """
    R = spins.shape[0]

    # All adjacent edges have a static shape (R-1,), independent of parity.
    edges = jnp.arange(R - 1, dtype=jnp.int32)  # 0..R-2
    left = edges  # left endpoint index
    right = edges + 1  # right endpoint index

    # Active mask selects parity-aligned edges; others are inactive (masked out).
    active = (edges & 1) == parity  # (R-1,) bool

    # Vectorized Metropolis acceptance for EVERY adjacent edge, then mask.
    Ei, Ej = energies[left], energies[right]  # (R-1,), (R-1,)
    ai, aj = alphas[left], alphas[right]  # (R-1,), (R-1,)
    logA = (ai - aj) * (Ej - Ei)  # (R-1,)

    key, kU = jax.random.split(key)
    u = jax.random.uniform(kU, shape=logA.shape)
    accept_pairs = active & (jnp.log(u) < logA)  # (R-1,) bool

    # Build a PERMUTATION of length R:
    #  - identity everywhere,
    #  - transposition (left<->right) where accept_pairs == True,
    #  - identity on inactive or rejected edges.
    partner = jnp.arange(R, dtype=jnp.int32)
    new_left_partner = jnp.where(accept_pairs, right, left)  # (R-1,)
    new_right_partner = jnp.where(accept_pairs, left, right)  # (R-1,)
    partner = partner.at[left].set(new_left_partner)
    partner = partner.at[right].set(new_right_partner)
    # partner now encodes disjoint swaps (a valid permutation).

    # Apply the permutation to spins and energies (gathers, no in-place hazards).
    spins = spins[partner]  # (R, N, N)
    energies = energies[partner]  # (R,)

    return spins, energies, key


def _local_sweeps_scan_factory(
    sweeps_per_exchange: int, black_mask: jnp.ndarray, white_mask: jnp.ndarray
):
    """Close over static sweep count; return body to apply that many HB sweeps."""

    def _one_sweep(carry, _):
        key, spins, alphas, sigma = carry
        spins, key = _heat_bath_checkerboard(key, spins, alphas, sigma, black_mask, white_mask)
        return (key, spins, alphas, sigma), None

    def _many_sweeps(carry, _):
        (key, spins, alphas, sigma), _ = lax.scan(
            _one_sweep, carry, xs=None, length=sweeps_per_exchange
        )
        return (key, spins, alphas, sigma), None

    return _many_sweeps


def _outer_step_factory(
    sweeps_per_exchange: int, black_mask: jnp.ndarray, white_mask: jnp.ndarray
):
    """
    One PT iteration:
      1) sweeps_per_exchange HB sweeps at each replica,
      2) compute energies,
      3) attempt neighbor exchanges with alternating parity,
      4) toggle parity.
    """
    local_sweeps = _local_sweeps_scan_factory(sweeps_per_exchange, black_mask, white_mask)

    def _outer(carry, _):
        key, spins, energies, alphas, sigma, parity = carry
        # Local HB sweeps
        (key, spins, alphas, sigma), _ = lax.scan(
            local_sweeps, (key, spins, alphas, sigma), xs=None, length=1
        )
        # Update energies after local moves
        energies = _energy_sigma_A_batch(spins, sigma)
        # Try swaps at current parity
        spins, energies, key = _replica_exchange(key, spins, energies, alphas, parity)
        # Toggle parity (0 -> 1 -> 0 ...)
        parity = jnp.int32(1) - parity
        return (key, spins, energies, alphas, sigma, parity), None

    return _outer


def _sample_outer_scan_factory(
    exchanges_per_sample: int,
    sweeps_per_exchange: int,
    black_mask: jnp.ndarray,
    white_mask: jnp.ndarray,
):
    """
    Close over static counts; return body that:
      - runs exchanges_per_sample outer PT steps,
      - records the TARGET replica (last index) as the sample.
    """
    outer_step = _outer_step_factory(sweeps_per_exchange, black_mask, white_mask)

    def _body(carry, _):
        key, spins, energies, alphas, sigma, parity = carry
        (key, spins, energies, alphas, sigma, parity), _ = lax.scan(
            outer_step,
            (key, spins, energies, alphas, sigma, parity),
            xs=None,
            length=exchanges_per_sample,
        )
        sample = spins[-1]  # target replica at α_target is last
        return (key, spins, energies, alphas, sigma, parity), sample

    return _body


@partial(
    jax.jit,
    static_argnames=(
        "N",
        "num_samples",
        "burn_in",
        "num_replicas",
        "sweeps_per_exchange",
        "exchanges_per_sample",
        "alpha_min_fraction",
    ),
)
def pt_sampler(
    key: jax.Array,
    N: int,
    sigma: float,
    alpha: float,
    num_samples: int,
    *,
    burn_in: int = 200,
    num_replicas: int = 16,
    sweeps_per_exchange: int = 1,
    exchanges_per_sample: int = 1,
    alpha_min_fraction: float = 0.3,
    init_spins: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    """
    Parallel Tempering + Heat-Bath sampler for N x N torus with J = sigma * A_N.

    Args (static where noted):
      key : PRNGKey
      N   : int [static]  - lattice side; spins are (N,N)
      sigma : float       - coupling scale (±)
      alpha : float       - scale in P(x) ∝ exp{-alpha * E(x)}
      num_samples : int [static] - #configs to return (from TARGET replica at α)
      burn_in     : int [static] - #outer PT iterations before saving
      num_replicas: int [static] - size of temperature/alpha ladder (≥1)
      sweeps_per_exchange : int [static] - HB sweeps between exchanges
      exchanges_per_sample: int [static] - PT outer steps between saved samples
      alpha_min_fraction : float [static] - α_min = alpha * alpha_min_fraction
      init_spins : optional (N,N) array of ±1 to initialize all replicas

    Returns:
      samples     : (num_samples, N, N) int8 spins in {+1,-1} (from target replica)
      final_state : (N, N) int8 (last state of TARGET replica)
      final_key   : PRNGKey
    """
    # Build alpha ladder (ascending to alpha at the last index)
    alphas = _make_alpha_ladder(alpha, num_replicas, alpha_min_fraction)  # (R,)
    R = num_replicas

    # Initialize replica states (R,N,N), either random or from provided init
    if init_spins is None:
        key, k0 = jax.random.split(key)
        spins = 2 * jax.random.randint(k0, (R, N, N), 0, 2, dtype=jnp.int8) - 1
    else:
        s0 = jnp.sign(init_spins).astype(jnp.int8)
        s0 = jnp.where(s0 == 0, jnp.int8(1), s0)
        spins = jnp.broadcast_to(s0, (R, N, N)).copy()

    # Initial energies (for exchanges)
    energies = _energy_sigma_A_batch(spins, sigma)

    # Checkerboard masks
    black_mask, white_mask = _checkerboard_masks(N)

    # Burn-in: run 'burn_in' outer PT steps
    outer_step = _outer_step_factory(sweeps_per_exchange, black_mask, white_mask)
    parity0 = jnp.array(0, dtype=jnp.int32)
    (key, spins, energies, alphas, sigma, _), _ = lax.scan(
        outer_step,
        (key, spins, energies, alphas, sigma, parity0),
        xs=None,
        length=burn_in,
    )

    # Sampling: repeat PT steps and record target replica
    sample_body = _sample_outer_scan_factory(
        exchanges_per_sample, sweeps_per_exchange, black_mask, white_mask
    )
    wrapped_sample_body = scan_tqdm(n=num_samples, print_rate=10)(sample_body)
    (key, spins, energies, alphas, sigma, _), samples = lax.scan(
        wrapped_sample_body,
        (key, spins, energies, alphas, sigma, jnp.array(0, jnp.int32)),
        xs=jnp.arange(num_samples),  # for tqdm
        length=num_samples,
    )

    samples = samples.astype(jnp.int8)
    final_state = spins[-1].astype(jnp.int8)
    return samples, final_state, key
