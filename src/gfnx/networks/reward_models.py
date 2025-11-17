import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .transformer import Encoder


class EqxTransformerRewardModel(eqx.Module):
    encoder: Encoder
    pooler: eqx.nn.Linear
    offset: float = 0.0

    def __init__(
        self,
        encoder_params: dict,
        output_dim: int,
        offset: float = 0.0,
        *,
        key: chex.PRNGKey,
    ):
        encoder_key, pooler_key = jax.random.split(key)
        self.encoder = Encoder(**encoder_params, key=encoder_key)
        self.pooler = eqx.nn.Linear(
            in_features=encoder_params["hidden_size"],
            out_features=output_dim,
            key=pooler_key,
        )
        self.offset = offset

    def __call__(
        self,
        inputs: Int[Array, " seq_len"],
        *,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> Float[Array, " output_dim"]:
        pos_ids = jnp.arange(inputs.shape[0])
        x = self.encoder(
            inputs, pos_ids, enable_dropout=enable_dropout, key=key
        )["layers_out"][-1]  # [seq_len, hidden_size]
        x = x.mean(axis=0)  # Average pooling
        # MLP layers for a final prediction
        x = self.pooler(x)
        return x
