import os

import chex
import jax
import jax.numpy as jnp

from ..base import BaseRewardModule, TLogReward, TReward, TRewardParams
from ..environment import AMPEnvParams, AMPEnvState
from ..utils.proteins import PROTEINS_FULL_ALPHABET


class EqxProxyAMPRewardModule(BaseRewardModule[AMPEnvState, AMPEnvParams]):
    def __init__(
        self,
        proxy_config_path: str,
        pretrained_proxy_path: str,
        reward_exponent: float = 1.0,
        min_reward: float = 1e-6,
    ):
        """
        Proxy reward model for amp
        """
        # Load config to a proxy model
        import omegaconf

        cfg = omegaconf.OmegaConf.load(proxy_config_path)
        self.network_params = omegaconf.OmegaConf.to_container(cfg.network)
        self.pretrained_proxy_path = pretrained_proxy_path
        if not os.path.isabs(self.pretrained_proxy_path):
            # Assume that the path is relative to the root of the project
            module_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            )
            self.pretrained_proxy_path = os.path.join(
                module_path, self.pretrained_proxy_path
            )

        self.reward_exponent = reward_exponent
        self.min_reward = min_reward

    def init(
        self, rng_key: chex.PRNGKey, dummy_state: AMPEnvState
    ) -> TRewardParams:
        # Lazy imports to avoid importing equinox in the main module
        import equinox as eqx
        import orbax.checkpoint as ocp

        from ..networks.reward_models import EqxTransformerRewardModel

        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        model = EqxTransformerRewardModel(
            encoder_params={
                "pad_id": len(PROTEINS_FULL_ALPHABET) - 1,
                **self.network_params["encoder_params"],
            },
            output_dim=1,
            key=rng_key,
        )

        abstract_model = jax.tree_util.tree_map(
            ocp.utils.to_shape_dtype_struct, model
        )
        model = ckptr.restore(self.pretrained_proxy_path, abstract_model)
        model_params, model_static = eqx.partition(model, eqx.is_array)
        self.model_static = model_static
        self.offset = model.offset

        return {"model_params": model_params}

    def log_reward(
        self, state: AMPEnvState, env_params: AMPEnvParams
    ) -> TLogReward:
        return jnp.log(self.reward(state, env_params))

    def reward(self, state: AMPEnvState, env_params: AMPEnvParams) -> TReward:
        # Lazy imports to avoid importing equinox in the main module
        import equinox as eqx

        model = eqx.combine(
            env_params.reward_params["model_params"], self.model_static
        )
        reward = jax.vmap(lambda x: model(x, enable_dropout=False, key=None))(
            state.tokens
        )
        reward = jnp.clip(
            jnp.pow(jax.nn.sigmoid(reward), self.reward_exponent),
            min=self.min_reward,
        ).squeeze(axis=-1)
        chex.assert_shape(reward, (state.tokens.shape[0],))  # [B]
        return reward
