from functools import partial

import chex
import jax
import jax.numpy as jnp

##### Distribution distances


def total_variation_distance(p: chex.Array, q: chex.Array) -> chex.Array:
    """
    Compute the Total Variation distance between two probability distributions.

    Args:
        p: First probability distribution (1D array).
        q: Second probability distribution (1D array).

    Returns:
        Total Variation distance as a scalar.
    """
    chex.assert_equal_shape([p, q])
    return jnp.sum(jnp.abs(p - q)) / 2.0


def kl_divergence(p: chex.Array, q: chex.Array, epsilon: float = 1e-9) -> chex.Array:
    """
    Compute the Kullback-Leibler divergence between two probability distributions.

    Args:
        p: First probability distribution (1D array).
        q: Second probability distribution (1D array).
        epsilon: Small value to avoid division by zero.

    Returns:
        Kullback-Leibler divergence as a scalar.
    """
    chex.assert_equal_shape([p, q])
    # Handle zero entries in p: 0 * log(anything) = 0
    # Only compute log(p/q) where p > 0, otherwise the term is 0
    return jnp.sum(jnp.where(p > 0, p * jnp.log(p / (q + epsilon)), 0.0))


def jensen_shannon_divergence(p: chex.Array, q: chex.Array, epsilon: float = 1e-9) -> chex.Array:
    """
    Compute the Jensen-Shannon divergence between two probability distributions.

    Args:
        p: First probability distribution (1D array).
        q: Second probability distribution (1D array).
        epsilon: Small value to avoid division by zero.

    Returns:
        Jensen-Shannon divergence as a scalar.
    """
    chex.assert_equal_shape([p, q])
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m, epsilon) + kl_divergence(q, m, epsilon))


##### String distances


def hamming_distance(s1: chex.Array, s2: chex.Array) -> chex.Array:
    """
    Compute the Hamming distance between two arrays.

    Args:
        s1: First array.
        s2: Second array.

    Returns:
        Hamming distance as a scalar.
    """
    chex.assert_equal_shape([s1, s2])
    return jnp.sum(s1 != s2, dtype=jnp.float32)


@partial(jax.jit, static_argnames=("max_len", "eos_id", "pad_id", "dtype"))
def _levenshtein_core_padded(
    s1_padded: chex.Array,
    s2_padded: chex.Array,
    s1_len: int,
    s2_len: int,
    max_len: int,
    eos_id: int,
    pad_id: int,
    dtype=jnp.int32,
):
    """
    Core JAX implementation of Levenshtein distance using lax.scan
    on padded arrays of fixed size max_len.

    Args:
        s1_padded (chex.Array): First sequence padded to max_len.
        s2_padded (chex.Array): Second sequence padded to max_len.
        s1_len (int): Original length of s1.
        s2_len (int): Original length of s2.
        max_len (int): The static maximum length used for padding.
        eos_id (int): End-of-sequence token ID.
        pad_id (int): Padding token ID.
        dtype (jnp.dtype): Data type of the output.

    Returns:
        The Levenshtein distance for the original s1, s2.
    """
    # Initialize the first row (dp[0, :]) - padded to max_len + 1
    # Represents distance from empty string to prefixes of s2_padded.
    # Correct values up to s2_len, others don't matter initially but will be computed.
    initial_row = jnp.arange(max_len + 1, dtype=dtype)

    # Outer scan: Iterate max_len times (for padded s1)
    # The 'carry' is the previous DP row.
    # 'xs' contains the character from s1_padded for the current iteration 'i'.
    def compute_row(prev_row: chex.Array, s1_char_i_tuple: tuple) -> tuple:
        s1_char, i = s1_char_i_tuple  # i is the effective row index (0 to max_len-1)
        # prev_row corresponds to dp[i, :] conceptually

        # First element of the current row: dp[i+1, 0] = i + 1
        current_row_0 = i + 1  # Note: index i corresponds to s1[i], row i+1 in DP table

        # Inner scan: Iterate max_len times (for padded s2)
        # Computes elements dp[i+1, 1] to dp[i+1, max_len]
        def compute_element(carry: tuple, s2_packed_j_tuple: tuple) -> tuple:
            # carry = (current_val_minus_1, prev_row_diag)
            #       = (dp[i+1, j], dp[i, j])
            current_val_minus_1, prev_row_diag = carry

            # s2_packed_j_tuple = (s2_char, prev_row_j, j)
            s2_char, prev_row_j, j = s2_packed_j_tuple  # j is effective col index (0 to max_len-1)
            # s2_char is s2_padded[j]
            # prev_row_j is dp[i, j+1]

            # Cost: 0 if match, 1 otherwise. Padding value never matches.
            cost = jnp.where(
                (s1_char == s2_char) & (s1_char != pad_id) & (s1_char != eos_id), 0, 1
            )

            substitution_cost = prev_row_diag + cost
            deletion_cost = prev_row_j + 1  # Deletion from s1 (comes from dp[i, j+1])
            insertion_cost = current_val_minus_1 + 1  # Insertion into s1 (comes from dp[i+1, j])

            current_val = jnp.minimum(
                jnp.minimum(substitution_cost, deletion_cost), insertion_cost
            )

            # Update carry for the next inner step: (new_left_val, new_diag_val)
            # new_left_val is current_val (dp[i+1, j+1])
            # new_diag_val for next step (j+1) is dp[i, j+1], which is prev_row_j
            next_carry = (current_val, prev_row_j)

            return next_carry, current_val  # Return computed dp[i+1, j+1]

        # Prepare inputs for the inner scan
        s2_chars = s2_padded  # Padded s2 characters
        prev_row_js = prev_row[1:]  # dp[i, 1] to dp[i, max_len]
        indices_j = jnp.arange(max_len, dtype=dtype)  # Column indices 0 to max_len-1

        # Initial carry for the inner scan: (dp[i+1, 0], dp[i, 0])
        initial_inner_carry = (current_row_0, prev_row[0])

        # Pack the inputs for the inner scan
        inner_scan_inputs = (s2_chars, prev_row_js, indices_j)

        # Execute the inner scan
        _, current_row_elems = jax.lax.scan(
            compute_element, initial_inner_carry, inner_scan_inputs, length=max_len
        )

        # Combine the first element (dp[i+1, 0]) with the rest of the row
        current_row = jnp.concatenate([jnp.array([current_row_0], dtype=dtype), current_row_elems])

        # Return the computed row (dp[i+1, :])
        # This becomes the 'carry' for the next outer step
        # And is also collected as the 'y' output of this step
        return current_row, current_row

    # Prepare inputs for the outer scan
    indices_i = jnp.arange(max_len, dtype=dtype)  # Row indices 0 to max_len-1
    outer_scan_inputs = (s1_padded, indices_i)

    # Execute the outer scan
    # final_row is dp[max_len, :]
    # all_rows contains dp[1, :], dp[2, :], ..., dp[max_len, :]
    final_row, all_rows = jax.lax.scan(
        compute_row,
        initial_row,  # Initial carry is dp[0, :]
        outer_scan_inputs,  # Iterate through padded s1 and indices
        length=max_len,
    )

    # We need the value dp[s1_len, s2_len].
    # 'all_rows' has shape (max_len, max_len + 1).
    # all_rows[k] corresponds to dp[k+1, :].
    # So, dp[s1_len, :] corresponds to all_rows[s1_len - 1].
    # The desired value is all_rows[s1_len - 1, s2_len].
    # If s1_len=0, use initial_row. If s1_len>0, use all_rows[s1_len-1]
    return jnp.where(s1_len == 0, initial_row[s2_len], all_rows[s1_len - 1, s2_len])


def levenshtein_distance(
    s1: chex.Array, s2: chex.Array, eos_id: int = 0, pad_id: int = -1
) -> float:
    """
    Computes the Levenshtein distance between two padded JAX arrays.

    Args:
        s1 (chex.Array): The first padded array of integers.
        s2 (chex.Array): The second padded array of integers.
        EOS_ID (int): The integer representing the End-of-String token.
        PAD_ID (int): The integer representing the padding token.

    Returns:
        int: The Levenshtein distance between the two strings.
    """
    chex.assert_equal_shape([s1, s2])
    # Find the actual lengths of the strings by finding the first EOS or PAD token
    len_s1 = jnp.argmax(jnp.logical_or(s1 == eos_id, s1 == pad_id))
    len_s2 = jnp.argmax(jnp.logical_or(s2 == eos_id, s2 == pad_id))

    # If the first element is EOS or PAD, the length is 0
    len_s1 = jnp.where(jnp.logical_or(s1[0] == eos_id, s1[0] == pad_id), 0, len_s1)
    len_s2 = jnp.where(jnp.logical_or(s2[0] == eos_id, s2[0] == pad_id), 0, len_s2)

    # Add 1 to length if no EOS/PAD is found, as argmax returns 0 for all same elements
    len_s1 = jnp.where(jnp.all(s1 != eos_id) & jnp.all(s1 != pad_id), s1.shape[0], len_s1)
    len_s2 = jnp.where(jnp.all(s2 != eos_id) & jnp.all(s2 != pad_id), s2.shape[0], len_s2)

    return _levenshtein_core_padded(
        s1, s2, len_s1, len_s2, max_len=s1.shape[0], eos_id=eos_id, pad_id=pad_id
    )
