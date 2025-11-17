"""
Technical details for protein generation tasks
"""

AMINO_ACIDS = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

NUCLEOTIDES = ["A", "C", "G", "T"]

SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[PAD]"]

NUCLEOTIDES_FULL_ALPHABET = NUCLEOTIDES + SPECIAL_TOKENS
PROTEINS_FULL_ALPHABET = AMINO_ACIDS + SPECIAL_TOKENS
