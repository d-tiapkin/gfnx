
import chex
import jax.numpy as jnp
import omegaconf
import orbax.checkpoint as ocp
from hydra.utils import instantiate

import gfnx
from gfnx import GFPEnvParams, GFPEnvState
from ..base import BaseRewardModule


class ProxyGFPRewardModule(BaseRewardModule[GFPEnvState, GFPEnvParams]):
    def __init__(
        self,
        proxy_config_path: str,
        pretrained_proxy_path: str,
        offset: float = 0.0,
        reward_exponent: float = 1.0,
        min_reward: float = 1e-6
    ):
        """
        Proxy reward model for gfp
        """
        # Load config to a proxy model
        self.cfg = omegaconf.OmegaConf.load(proxy_config_path)
        self.pretrained_proxy_path = pretrained_proxy_path

        self.offset = offset
        self.reward_exponent = reward_exponent
        self.min_reward = min_reward

    def init(
        self, rng_key: chex.PRNGKey, dummy_state: GFPEnvState
    ) -> gfnx.TRewardParams:
        #print(f'Config for reward model: {self.cfg}')
        self.model = instantiate(self.cfg.network)
        variables = self.model.init(
            rng_key, dummy_state.tokens, training=False
        )
        #params = variables["params"]

        orbax_checkpointer = ocp.StandardCheckpointer()
        variables = orbax_checkpointer.restore(
            self.pretrained_proxy_path
        )
        return {
            "network_params": variables["params"]
        }

    def log_reward(
        self, state: GFPEnvState, env_params: GFPEnvParams
    ) -> gfnx.TLogReward:
        return jnp.log(self.reward(state, env_params))
    
    def reward(
        self, state: GFPEnvState, env_params: GFPEnvParams
    ) -> gfnx.TReward:
        reward = self.model.apply(
            {'params': env_params.reward_params["network_params"]},
            state.tokens,
            training=False
        )
        reward = jnp.clip(reward + self.offset, a_min=self.min_reward)
        reward = reward.squeeze(axis=-1)
        chex.assert_shape(reward, (state.tokens.shape[0],))    # [B]
        return reward
