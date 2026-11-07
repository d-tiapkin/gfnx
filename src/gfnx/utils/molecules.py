"""
Technical details for molecule generation tasks
"""

QM9_SMALL_BLOCKS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
]

SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[PAD]"]

QM9_SMALL_FULL_ALPHABET = QM9_SMALL_BLOCKS + SPECIAL_TOKENS
