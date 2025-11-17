"""Neural networks for different environments"""

import math

from dataclasses import dataclass
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class TransformerEncoderConfig:
    max_len: int = 2048
    num_heads: int = 8
    num_layers: int = 3

    mlp_dim: int = 64
    qkv_dim: int = 64
    emb_dim: int = 64

    dtype: jnp.dtype = jnp.float32

    dropout_rate: float = 0.1
    vocab_size: int = 256


class PositionalEncoding(nn.Module):
    config: TransformerEncoderConfig

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = np.zeros((self.config.max_len, self.config.emb_dim))
        position = np.arange(0, self.config.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, self.config.emb_dim, 2)
            * (-math.log(10000.0) / self.config.emb_dim)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x) -> chex.Array:
        x = x + self.pe[:, : x.shape[1]]
        return x


class TransformerMLPBlock(nn.Module):
    """Transformer MLP / feed-forward block.
    Simplified version of the original code from Flax:
    https://github.com/google/flax/blob/main/examples/wmt/models.py

    Attributes:
        config: TransformerConfig dataclass containing hyperparameters.
        out_dim: optionally specify out dimension.
    """

    config: TransformerEncoderConfig

    @nn.compact
    def __call__(self, inputs: chex.Array, training: bool) -> chex.Array:
        """Applies Transformer MlpBlock module."""
        config = self.config
        actual_out_dim = inputs.shape[-1]
        x = nn.Dense(features=config.mlp_dim, dtype=jnp.float32)(inputs)
        x = nn.relu(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=not training)
        output = nn.Dense(features=actual_out_dim, dtype=jnp.float32)(x)
        output = nn.Dropout(rate=config.dropout_rate)(
            output, deterministic=not training
        )
        return output


class EncoderBlock(nn.Module):
    """Transformer encoder layer.

    Simplified version of the original code from Flax:
    https://github.com/google/flax/blob/main/examples/wmt/models.py

    Attributes:
        config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerEncoderConfig

    @nn.compact
    def __call__(self, inputs : chex.Array, training: bool, encoder_mask=None) -> chex.Array:
        config = self.config

        assert inputs.ndim == 3
        x = nn.MultiHeadAttention(
            num_heads=config.num_heads,
            dtype=jnp.float32,
            qkv_features=config.qkv_dim,
            use_bias=True,
            broadcast_dropout=False,
            dropout_rate=0.0,  # It is dropout inside of attention mask, turned off by default
            deterministic=not training,
            precision=jax.lax.Precision("float32"),
        )(inputs, mask=encoder_mask)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=not training)
        x = x + inputs
        x = nn.LayerNorm(dtype=jnp.float32)(x)
        y = TransformerMLPBlock(config=config)(x, training)
        return nn.LayerNorm(dtype=jnp.float32)(x + y)


class TransformerEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
        config: TransformerConfig dataclass containing hyperparameters.
        shared_embedding: a shared embedding layer to use.
    """

    config: TransformerEncoderConfig

    @nn.compact
    def __call__(self, inputs, training: bool, encoder_mask=None):
        """Applies Transformer model on the inputs.

        Args:
        inputs: input data
        inputs_positions: input subsequence positions for packed examples.
        encoder_mask: encoder self-attention mask.

        Returns:
        output of a transformer encoder.
        """
        config = self.config
        assert inputs.ndim == 2  # (batch, len)

        x = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.emb_dim,
            embedding_init=nn.initializers.normal(stddev=1.0),
            dtype=jnp.float32,
        )(inputs)
        x = PositionalEncoding(config=config)(x)
        x = nn.Dropout(rate=config.dropout_rate, deterministic=not training)(x)
        x = x.astype(jnp.float32)
        # Input Encoder
        for lyr in range(config.num_layers):
            x = EncoderBlock(config=config, name=f"encoderblock_{lyr+1}")(
                x, training, encoder_mask
            )
        return x
