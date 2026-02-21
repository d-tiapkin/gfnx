"""A generic transformer network for policies and proxies.

This module implements a standard transformer encoder architecture using Equinox.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class PositionalEncoding(eqx.Module):
    """
    A fixed sinusoidal positional encoding layer for a SINGLE example.

    This module is designed to be vectorized with `jax.vmap` to handle batches.
    """

    pe: jax.Array
    dropout: eqx.nn.Dropout

    def __init__(self, d_model: int, dropout_p: float, max_len: int = 5000):
        """
        Args:
            d_model (int): The dimensionality of the embeddings.
            dropout_p (float): The dropout probability.
            max_len (int): The maximum possible sequence length.
        """
        super().__init__()
        self.dropout = eqx.nn.Dropout(p=dropout_p)

        # Create the positional encoding matrix for a single sequence
        position = jnp.arange(max_len).reshape(-1, 1)
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))

        pe = jnp.zeros((max_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        # Final shape is (max_len, d_model)
        self.pe = pe

    def __call__(
        self,
        x: jax.Array,
        position_ids: jax.Array,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> jax.Array:
        """
        Applies positional encoding to a single input tensor.

        Args:
            x (jax.Array): The input tensor of shape (d_model,).
            position_ids (jax.Array): The positions of a token in the sequence.
            key (jax.random.PRNGKey): A JAX random key for the dropout layer.

        Returns:
            jax.Array: The output tensor with positional information.
        """
        # Add the sliced positional encodings
        x = x + self.pe[position_ids, :]
        return self.dropout(x, inference=not enable_dropout, key=key)


class EmbedderBlock(eqx.Module):
    """Transformer embedder."""

    token_embedder: eqx.nn.Embedding
    position_embedder: PositionalEncoding

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embedding_size: int,
        dropout_rate: float,
        *,
        key: chex.PRNGKey,
    ):
        token_key = jax.random.split(key)[0]

        self.token_embedder = eqx.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=embedding_size,
            key=token_key,
        )
        self.position_embedder = PositionalEncoding(
            d_model=embedding_size,
            dropout_p=dropout_rate,
            max_len=max_length,
        )

    def __call__(
        self,
        token_ids: Int[Array, " seq_len"],
        position_ids: Int[Array, " seq_len"],
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> Float[Array, "seq_len embedding_size"]:
        token_emb = jax.vmap(self.token_embedder)(token_ids)
        keys = None if key is None else jax.random.split(key, num=token_ids.shape[0])
        return jax.vmap(
            lambda x, y, z: self.position_embedder(x, y, enable_dropout=enable_dropout, key=z)
        )(token_emb, position_ids, keys)


class FeedForwardBlock(eqx.Module):
    """A single transformer feed forward block."""

    linear: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm1: eqx.nn.LayerNorm
    layernorm2: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_rate: float,
        *,
        key: chex.PRNGKey,
    ):
        linear_key, output_key = jax.random.split(key)
        self.linear = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=intermediate_size,
            key=linear_key,
        )
        self.output = eqx.nn.Linear(
            in_features=intermediate_size,
            out_features=hidden_size,
            key=output_key,
        )

        self.layernorm1 = eqx.nn.LayerNorm(shape=hidden_size)
        self.layernorm2 = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, " hidden_size"],
        enable_dropout: bool = True,
        key: chex.PRNGKey | None = None,
    ) -> Float[Array, " hidden_size"]:
        # Pre-layernorm
        key1, key2 = (None, None) if key is None else jax.random.split(key)
        inputs = self.layernorm1(inputs)
        # Feed-forward
        hidden = self.linear(inputs)
        hidden = jax.nn.relu(hidden)
        hidden = self.dropout(hidden, inference=not enable_dropout, key=key1)
        # Project back to input size.
        output = self.output(hidden)
        output = self.dropout(output, inference=not enable_dropout, key=key2)
        # Residual and layer norm
        output += inputs
        return self.layernorm2(output)


class AttentionBlock(eqx.Module):
    """A single transformer attention block."""

    attention: eqx.nn.MultiheadAttention
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        *,
        key: chex.PRNGKey,
    ):
        self.num_heads = num_heads
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=attention_dropout_rate,
            key=key,
        )
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, "seq_len hidden_size"],
        mask: Int[Array, " seq_len"] | None,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        if mask is not None:
            mask = self.make_self_attention_mask(mask)
        attention_key, dropout_key = (None, None) if key is None else jax.random.split(key)

        attention_output = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            mask=mask,
            inference=not enable_dropout,
            key=attention_key,
        )

        result = attention_output
        return self.dropout(result, inference=not enable_dropout, key=dropout_key)

    def make_self_attention_mask(
        self, mask: Int[Array, " seq_len"]
    ) -> Float[Array, "num_heads seq_len seq_len"]:
        """Create self-attention mask from sequence-level mask."""
        mask = jnp.multiply(jnp.expand_dims(mask, axis=-1), jnp.expand_dims(mask, axis=-2))
        mask = jnp.expand_dims(mask, axis=-3)
        mask = jnp.repeat(mask, repeats=self.num_heads, axis=-3)
        return mask.astype(jnp.float32)


class TransformerLayer(eqx.Module):
    """A single transformer layer."""

    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        *,
        key: chex.PRNGKey,
    ):
        attention_key, ff_key = jax.random.split(key)

        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=attention_key,
        )
        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=ff_key,
        )

    def __call__(
        self,
        inputs: Float[Array, "seq_len hidden_size"],
        mask: Int[Array, " seq_len"] | None = None,
        *,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        seq_len = inputs.shape[0]
        attn_key, ff_key = (None, None) if key is None else jax.random.split(key)
        attention_output = self.attention_block(
            inputs, mask, enable_dropout=enable_dropout, key=attn_key
        )
        attention_output = attention_output + inputs  # Residual connection
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)
        return jax.vmap(self.ff_block, in_axes=(0, None, 0))(
            attention_output, enable_dropout, ff_keys
        )


class Encoder(eqx.Module):
    """Full transformer encoder."""

    embedder_block: EmbedderBlock
    layers: list[TransformerLayer]
    pad_id: int

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embedding_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float = 0.0,
        pad_id: int = 0,
        *,
        key: chex.PRNGKey,
    ):
        self.pad_id = pad_id  # Padding token to identify masks
        embedder_key, layer_key = jax.random.split(key, num=2)
        self.embedder_block = EmbedderBlock(
            vocab_size=vocab_size,
            max_length=max_length,
            embedding_size=embedding_size,
            dropout_rate=dropout_rate,
            key=embedder_key,
        )

        layer_keys = jax.random.split(layer_key, num=num_layers)
        self.layers = []
        for layer_key in layer_keys:
            self.layers.append(
                TransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    key=layer_key,
                )
            )

    def __call__(
        self,
        token_ids: Int[Array, " seq_len"],
        position_ids: Int[Array, " seq_len"],
        *,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> dict[str, Array]:
        emb_key, l_key = (None, None) if key is None else jax.random.split(key)

        embeddings = self.embedder_block(
            token_ids=token_ids,
            position_ids=position_ids,
            enable_dropout=enable_dropout,
            key=emb_key,
        )

        # We assume that all pad_id values should be masked out.
        mask = jnp.asarray(token_ids != self.pad_id, dtype=jnp.int32)

        x = embeddings
        layer_outputs = []
        for layer in self.layers:
            cl_key, l_key = (None, None) if l_key is None else jax.random.split(l_key)
            x = layer(x, mask, enable_dropout=enable_dropout, key=cl_key)
            layer_outputs.append(x)

        return {"embeddings": embeddings, "layers_out": layer_outputs}
