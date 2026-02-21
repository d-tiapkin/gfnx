import functools

import chex
import jax
import jax.numpy as jnp

from .distances import hamming_distance


@functools.partial(jax.jit, static_argnums=(0, 1, 3))
def construct_mode_set(
    sentence_len: int,
    block_len: int,
    block_set: chex.Array,
    mode_set_size: int,
    rng_key: chex.PRNGKey,
):
    n_choices = sentence_len // block_len
    choices = jax.random.choice(rng_key, block_set, shape=(mode_set_size, n_choices))
    mode_set = choices.reshape(mode_set_size, -1)
    chex.assert_shape(mode_set, (mode_set_size, sentence_len))
    return mode_set


@functools.partial(jax.jit, static_argnums=(1,))
def tokenize(bitseq: chex.Array, k: int):
    bitseq = jnp.reshape(bitseq, (-1, k))

    def tokenize_one_word(word):
        result = 0

        def loop_fn(i, loop_carry):
            word, result = loop_carry
            result = result + 2 ** (k - i - 1) * word[i]
            return word, result

        _, result = jax.lax.fori_loop(0, k, loop_fn, (word, result))
        return result

    return jax.vmap(tokenize_one_word)(bitseq)


@functools.partial(jax.jit, static_argnums=(1,))
def detokenize(tokens: chex.Array, k: int):
    @jax.jit
    def detokenize_one_word(word):
        # Decompose a number into a binary
        result = jnp.zeros(k, dtype=jnp.bool)

        def loop_fn(i, loop_carry):
            word, result = loop_carry
            result = result.at[k - i - 1].set(jnp.astype(word % 2, jnp.bool))
            word = word // 2
            return word, result

        word, result = jax.lax.fori_loop(0, k, loop_fn, (word, result))
        return result

    return jnp.reshape(jax.vmap(detokenize_one_word)(tokens), -1)


def construct_binary_test_set(rng_key: chex.PRNGKey, mode_set: chex.Array):
    test_set = []
    len_mode = mode_set.shape[1]
    # Modify cnt random bits randomly
    for mode in mode_set:
        test_set.append(mode)
        for cnt in range(1, len_mode):
            rng_key, choice_key = jax.random.split(rng_key)
            subset = jax.random.choice(choice_key, len_mode, shape=(cnt,), replace=False)
            change_mask = jnp.zeros(len_mode, dtype=jnp.bool)
            change_mask = change_mask.at[subset].set(True)
            test_set.append(jnp.logical_xor(mode, change_mask))
            assert len(test_set[-1]) == len_mode
            assert hamming_distance(test_set[-1], mode) == cnt
    return jnp.array(test_set)
