'''
Technical details for protein generation tasks
'''

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

PROTEINS_SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[PAD]"]

PROTEINS_FULL_ALPHABET = AMINO_ACIDS + PROTEINS_SPECIAL_TOKENS
