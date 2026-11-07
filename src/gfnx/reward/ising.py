"""Energy-based reward for the Ising environment.

See [`IsingEnvironment`][gfnx.environment.ising.IsingEnvironment].
"""

import chex
import jax.numpy as jnp

from ..base import BaseRewardModule, BaseRewardParams, TLogReward, TReward
from ..environment import IsingEnvParams, IsingEnvState


@chex.dataclass(frozen=True)
class IsingRewardParams(BaseRewardParams):
    """Parameters of the Ising reward.

    Attributes:
        J: Interaction matrix of shape `[dim, dim]`. It is a *trainable* reward parameter:
            EB-GFN updates it jointly with the GFlowNet policy.
    """

    J: chex.Array


class IsingRewardModule(BaseRewardModule[IsingEnvState, IsingEnvParams, IsingRewardParams]):
    r"""Gibbs reward of an Ising model with interaction matrix $J$.

    For a full configuration $\mathbf{x} \in \{-1, +1\}^D$,

    $$
    \mathcal{E}_J(\mathbf{x}) = -\mathbf{x}^\top J \mathbf{x},
    \qquad
    R(\mathbf{x}) = \exp\big(-\mathcal{E}_J(\mathbf{x})\big),
    \qquad
    \log R(\mathbf{x}) = \mathbf{x}^\top J \mathbf{x}.
    $$

    Positive $J_{ij}$ favour aligned spins and negative $J_{ij}$ anti-aligned ones. The
    environment stores spins in `{0, 1}`; they are mapped to $\{-1, +1\}$ by
    $\mathbf{x} = 2s - 1$.

    $J$ lives in [`IsingRewardParams`][gfnx.reward.ising.IsingRewardParams], not on the
    module, so it can be learned. [`init`][gfnx.reward.ising.IsingRewardModule.init]
    returns $J = 0$ (a uniform target). To use a fixed ground-truth model, replace it,
    e.g. with the toroidal lattice `gfnx.utils.get_true_ising_J(N, sigma)`:

    ```python
    reward_params = reward_module.init(key, env.reset()).replace(
        J=gfnx.utils.get_true_ising_J(N, sigma)
    )
    ```

    References:
        Zhang, D., et al. (2022). Generative flow networks for discrete probabilistic
        modeling. ICML.

        Ising, E. (1925). Beitrag zur Theorie des Ferromagnetismus. Zeitschrift für
        Physik, 31(1):253–258.
    """

    def init(
        self,
        rng_key: chex.PRNGKey,
        dummy_state: IsingEnvState,
    ) -> IsingRewardParams:
        """Returns `IsingRewardParams` with a zero interaction matrix.

        Args:
            rng_key: Unused.
            dummy_state: Single Ising state; its `state` length sets the size of $J$.

        Returns:
            `IsingRewardParams(J=zeros([dim, dim]))`.
        """
        dim = dummy_state.state.shape[-1]
        J = jnp.zeros((dim, dim))
        return IsingRewardParams(J=J)

    def reward(self, state: IsingEnvState, reward_params: IsingRewardParams) -> TReward:
        r"""Computes $R(\mathbf{x}) = \exp(\mathbf{x}^\top J \mathbf{x})$ for a single state."""
        return jnp.exp(self.log_reward(state, reward_params))

    def log_reward(self, state: IsingEnvState, reward_params: IsingRewardParams) -> TLogReward:
        r"""Computes $\log R(\mathbf{x}) = \mathbf{x}^\top J \mathbf{x}$ for a single state.

        Args:
            state: Ising state whose `state` field has shape `[dim]` with values in
                `{0, 1}` (a terminal configuration).
            reward_params: Parameters holding the interaction matrix `J`.

        Returns:
            Scalar log-reward, i.e. the negative energy.
        """
        canonical = 2 * state.state - 1
        J = reward_params.J
        return jnp.einsum("i,ij,j->", canonical, J, canonical)
