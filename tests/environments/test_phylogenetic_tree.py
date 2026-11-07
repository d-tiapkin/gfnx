import jax
import jax.numpy as jnp
import pytest

from gfnx.environment.phylogenetic_tree import PhyloTreeEnvironment
from gfnx.utils.phylogenetic_tree import CHARACTERS_MAPS


@pytest.mark.parametrize("sequence_type,bits_per_seq_elem", [("DNA", 4), ("DNA_WITH_GAP", 5)])
def test_init_passes_bits_per_seq_elem(sequence_type, bits_per_seq_elem):
    """`env_params.bits_per_seq_elem` must follow the constructor, not the dataclass default."""
    char_map = CHARACTERS_MAPS[sequence_type]
    sequences = jnp.array(
        [[char_map[c] for c in seq] for seq in ("ACGT", "AACC", "GTTA")], dtype=jnp.uint8
    )
    env = PhyloTreeEnvironment(
        sequences, sequence_type=sequence_type, bits_per_seq_elem=bits_per_seq_elem
    )
    env_params = env.init(jax.random.PRNGKey(0))

    assert env_params.bits_per_seq_elem == bits_per_seq_elem
    # fldb_phylo.py sizes its input layer from env_params, so it must match the observation.
    obs = env.get_obs(env.reset(), env_params)
    assert obs.shape == (3, 4, env_params.bits_per_seq_elem)
    assert obs.shape == env.observation_space.shape
