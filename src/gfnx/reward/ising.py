import chex
import jax.numpy as jnp

from ..base import BaseRewardModule, TLogReward, TReward, TRewardParams
from ..environment import IsingEnvParams, IsingEnvState


@chex.dataclass(frozen=True)
class IsingRewardParams:
    J: chex.Array


class IsingRewardModule(BaseRewardModule[IsingEnvState, IsingEnvParams]):
    def init(
        self,
        rng_key: chex.PRNGKey,
        dummy_state: IsingEnvState,
    ) -> TRewardParams:
        dim = dummy_state.state.shape[-1]
        J = jnp.zeros((dim, dim))
        return IsingRewardParams(J=J)

    def reward(self, state: IsingEnvState, env_params: IsingEnvParams) -> TReward:
        return jnp.exp(self.log_reward(state, env_params))

    def log_reward(self, state: IsingEnvState, env_params: IsingEnvParams) -> TLogReward:
        """Compute log reward for Ising model states.

        Args:
        - state: IsingEnvState with state field of shape [B, dim] containing values in {0, 1}
        - env_params: Environment parameters containing reward_params with alpha and J

        Returns:
        - Log reward tensor of shape [B] for each state in the batch

        The Ising model energy is computed as:
            E = -alpha * sum_{i,j} J_{ij} * s_i * s_j
        where s_i are the spin values in {-1, 1} (transformed from {0, 1} input).

        The log reward is simply -E, so higher energy states have lower reward.
        """
        canonical = 2 * state.state - 1
        J = env_params.reward_params.J
        log_reward = jnp.einsum("bi,ij,bj->b", canonical, J, canonical)
        return log_reward
