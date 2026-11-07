"""Proxy-model reward for the green fluorescent protein (GFP) environment.

See [`GFPEnvironment`][gfnx.environment.gfp.GFPEnvironment].
"""

import os
from typing import Any

import chex
import jax
import jax.numpy as jnp

from ..base import BaseRewardModule, BaseRewardParams, TLogReward, TReward
from ..environment import GFPEnvParams, GFPEnvState
from ..utils.proteins import PROTEINS_FULL_ALPHABET


@chex.dataclass(frozen=True)
class GFPRewardParams(BaseRewardParams):
    """Parameters of the GFP reward.

    Attributes:
        model_params: Array leaves of the pretrained Equinox proxy regressor.
    """

    model_params: Any


class EqxProxyGFPRewardModule(BaseRewardModule[GFPEnvState, GFPEnvParams, GFPRewardParams]):
    r"""Fluorescence predicted by a pretrained transformer regressor (experimental).

    $$
    R(x) = \max\big(f_\phi(x) + c,\; R_{\min}\big),
    $$

    where $f_\phi$ is a regressor trained on the GFP fluorescence dataset of Sarkisyan
    et al. (2016) and $c$ is the offset stored with the model. No pretrained GFP proxy is
    shipped yet: `proxy/weights/dummy_gfp` holds placeholder weights, and a real proxy is
    trained with `python proxy/train_proxy.py --config-name gfp`.

    Args:
        proxy_config_path: Path to the Hydra/OmegaConf config used to train the proxy
            (its `network` section defines the architecture), e.g.
            `"proxy/configs/gfp.yaml"`. Resolved relative to the working directory.
        pretrained_proxy_path: Orbax checkpoint directory, e.g.
            `"proxy/weights/gfp/model"`. Relative paths are resolved against the
            repository root, so use a repository checkout (editable install) or pass an
            absolute path.
        reward_exponent: Stored for API symmetry with the AMP reward; currently unused.
        min_reward: Lower clip $R_{\min}$.

    References:
        Madan, K., et al. (2023). Learning GFlowNets from partial episodes for improved
        convergence and stability. ICML.

        Sarkisyan, K., et al. (2016). Local fitness landscape of the green fluorescent
        protein. Nature, 533.
    """

    def __init__(
        self,
        proxy_config_path: str,
        pretrained_proxy_path: str,
        reward_exponent: float = 1.0,
        min_reward: float = 1e-6,
    ):
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
            self.pretrained_proxy_path = os.path.join(module_path, self.pretrained_proxy_path)

        self.reward_exponent = reward_exponent
        self.min_reward = min_reward

    def init(self, rng_key: chex.PRNGKey, dummy_state: GFPEnvState) -> GFPRewardParams:
        """Restores the pretrained proxy network.

        Builds an `EqxTransformerRewardModel` from the `network` section of the config,
        restores its weights from `pretrained_proxy_path` with Orbax, and splits it into
        arrays (returned as `reward_params`) and static structure (stored on the module
        as `model_static`). `reward` therefore only works on a module whose `init` has
        been called. Requires `equinox` and `orbax-checkpoint` (the `baselines` extra).

        Args:
            rng_key: Key used to build the network skeleton before restoring weights.
            dummy_state: Unused.

        Returns:
            Reward parameters holding the proxy weights.
        """
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

        abstract_model = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, model)
        model = ckptr.restore(self.pretrained_proxy_path, abstract_model)
        model_params, model_static = eqx.partition(model, eqx.is_array)
        self.model_static = model_static
        self.offset = model.offset

        return GFPRewardParams(model_params=model_params)

    def log_reward(self, state: GFPEnvState, reward_params: GFPRewardParams) -> TLogReward:
        r"""Computes $\log R(x)$ (log of `reward`)."""
        return jnp.log(self.reward(state, reward_params))

    def reward(self, state: GFPEnvState, reward_params: GFPRewardParams) -> TReward:
        """Runs the proxy on `state.tokens` (dropout off) and returns the clipped reward.

        Args:
            state: Terminal GFP state.
            reward_params: Proxy weights returned by `init`.

        Returns:
            Scalar reward, at least `min_reward`.
        """
        # Lazy imports to avoid importing equinox in the main module
        import equinox as eqx

        model = eqx.combine(reward_params.model_params, self.model_static)
        reward = model(state.tokens, enable_dropout=False, key=None)
        reward = jnp.clip(reward + self.offset, min=self.min_reward)
        return reward.squeeze()
