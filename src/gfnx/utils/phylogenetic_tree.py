import json
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp

CHARACTERS_MAPS = {
    "DNA": {
        "A": 0b1000,
        "C": 0b0100,
        "G": 0b0010,
        "T": 0b0001,
        "N": 0b1111,
        "?": 0b1111,
    },
    "RNA": {
        "A": 0b1000,
        "C": 0b0100,
        "G": 0b0010,
        "U": 0b0001,
        "N": 0b1111,
        "?": 0b1111,
    },
    "DNA_WITH_GAP": {
        "A": 0b10000,
        "C": 0b01000,
        "G": 0b00100,
        "T": 0b00010,
        "-": 0b00001,
        "N": 0b11110,
        "?": 0b11110,
    },
    "RNA_WITH_GAP": {
        "A": 0b10000,
        "C": 0b01000,
        "G": 0b00100,
        "U": 0b00010,
        "-": 0b00001,
        "N": 0b11110,
        "?": 0b11110,
    },
}

CONFIGS = {
    "DS1": ("DNA_WITH_GAP", 5800.0, 4.0, 5),
    "DS2": ("DNA_WITH_GAP", 8000.0, 4.0, 5),
    "DS3": ("DNA_WITH_GAP", 8800.0, 4.0, 5),
    "DS4": ("DNA_WITH_GAP", 3500.0, 4.0, 5),
    "DS5": ("DNA_WITH_GAP", 2300.0, 4.0, 5),
    "DS6": ("DNA_WITH_GAP", 2300.0, 4.0, 5),
    "DS7": ("DNA_WITH_GAP", 12500.0, 4.0, 5),
    "DS8": ("DNA_WITH_GAP", 2800.0, 4.0, 5),
}


def get_phylo_initialization_args(
    dataset_name: str, data_folder: Path
) -> Tuple[dict, dict]:
    """
    Prepares the arguments required to initialize the Phylogenetic Tree Environment
    and its associated reward module from the provided dataset.

    Args:
        dataset_name (str): The name of the dataset (e.g., "DS1", "DS2", ...)
        data_folder (Path): Path to the folder containing the dataset JSON files.

    Returns:
        env_kwargs (dict):
            Additional arguments for environment initialization (excluding the reward module).
            Contains:
                - sequences (chex.Array): Binary-encoded sequences.
                - sequence_type (str): The type of sequences (e.g., "DNA_WITH_GAP").
                - bits_per_seq_elem (int): Number of bits used to encode a sequence element.
        reward_kwargs (dict):
            Parameters for constructing the reward module.
            Contains:
                - num_nodes (int): Number of sequences in the dataset.
                - C (float): Reward parameter C.
                - scale (float): Reward scale.
    """
    if dataset_name not in CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_folder = Path(data_folder)

    # Load the raw sequences from the JSON file.
    data_file = data_folder / f"{dataset_name}.json"
    with open(data_file, "r") as f:
        sequences_dict = json.load(f)

    # Retrieve the configuration tuple for the given dataset.
    config = CONFIGS[dataset_name]
    sequence_type, C, scale, bits_per_seq_elem = config

    # Map each character to its binary encoding.
    char_dict = CHARACTERS_MAPS[sequence_type]
    sequences = jnp.array(
        [
            [char_dict[c] for c in sequence]
            for sequence in sequences_dict.values()
        ],
        dtype=jnp.uint8,
    )

    # Prepare arguments for the environment initialization.
    env_kwargs = {
        "sequences": sequences,
        "sequence_type": sequence_type,
        "bits_per_seq_elem": bits_per_seq_elem,
    }

    # Prepare parameters for constructing the reward module.
    reward_kwargs = {
        "num_nodes": len(sequences_dict),
        "C": C,
        "scale": scale,
    }

    return env_kwargs, reward_kwargs
