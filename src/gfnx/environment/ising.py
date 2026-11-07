"""Ising model environment: assign the spins of a lattice one at a time.

Implements the sequential construction of Ising configurations used by EB-GFN
(Zhang et al., 2022). The energy-based reward lives in
[`gfnx.reward.ising`][gfnx.reward.ising]; its coupling matrix can be trained jointly with
the GFlowNet (see `baselines/tb_ising.py`).
"""

from typing import Any

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from .. import spaces
from ..base import (
    BaseEnvironment,
    BaseEnvParams,
    BaseEnvState,
    TAction,
    TDone,
)


@chex.dataclass(frozen=True)
class EnvState(BaseEnvState):
    """State of a single Ising environment.

    Attributes:
        state: Partial spin assignment of shape `[dim]` with entries in `{-1, 0, 1}`:
            `-1` is an unassigned site, `0` is spin $-1$ and `1` is spin $+1$.
        time: Number of spins assigned so far.
        is_terminal: `True` once every site is assigned.
        is_initial: `True` for the empty configuration.
        is_pad: `True` for padding steps after termination.
    """

    state: Int[Array, " dim"]
    time: Int[Array, ""]
    is_terminal: Bool[Array, ""]
    is_initial: Bool[Array, ""]
    is_pad: Bool[Array, ""]


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    r"""Ising dynamics parameters.

    Attributes:
        dim: Number of lattice sites $D$ (e.g. `N * N` for an $N \times N$ lattice).
    """

    dim: int = 10


class IsingEnvironment(BaseEnvironment[EnvState, EnvParams]):
    r"""Ising environment: sequential spin assignment on $D$ lattice sites.

    An object is a full spin configuration $\mathbf{x} \in \{-1, +1\}^D$. It is built from
    the empty lattice by assigning one unassigned site per step, in any order, so every
    trajectory has exactly $D$ steps:
    $s_0 = (\varnothing, \ldots, \varnothing) \to s_1 \to \cdots \to s_D = \mathbf{x}$.

    * **State** — `state` of shape `[dim]` with entries in `{-1, 0, 1}`: `-1` encodes an
      unassigned site $\varnothing$, `0` encodes spin $-1$ and `1` encodes spin $+1$
      (the canonical spin is `2 * state - 1`). $s_0$ is all `-1`.
    * **Forward actions** — `2 * dim` actions. Action `i < dim` sets site `i` to `0`
      (spin $-1$); action `dim + i` sets site `i` to `1` (spin $+1$). Only unassigned
      sites can be set. There is no stop action: the episode terminates automatically
      when all sites are assigned.
    * **Backward actions** — `dim` actions. Action `i` clears site `i` back to `-1`
      (valid iff it is assigned).
    * **Observation** — the raw `state` vector in `{-1, 0, 1}^dim`.
    * **Episode length** — exactly `dim` steps.

    Args:
        dim: Number of lattice sites $D$. For an $N \times N$ lattice pass `N * N`.

    References:
        Zhang, D., Malkin, N., Liu, Z., Volokhova, A., Courville, A., Bengio, Y. (2022).
        Generative flow networks for discrete probabilistic modeling. ICML.
    """

    def __init__(self, dim: int = 10) -> None:
        self.dim = dim

    def reset(self) -> EnvState:
        """Returns the empty lattice (all sites `-1`)."""
        return EnvState(
            state=jnp.full((self.dim,), -1, dtype=jnp.int8),
            time=jnp.int32(0),
            is_terminal=jnp.bool_(False),
            is_initial=jnp.bool_(True),
            is_pad=jnp.bool_(False),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        """Returns `EnvParams(dim)`. The environment is deterministic."""
        return EnvParams(dim=self.dim)

    @property
    def max_steps_in_episode(self) -> int:
        """Every trajectory assigns all `dim` sites."""
        return self.dim

    def step(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
        """Assigns one spin.

        Args:
            state: Current state.
            action: Action in `[0, 2 * dim)`; site `action % dim` receives value
                `action // dim` (`0` = spin $-1$, `1` = spin $+1$).
            env_params: Environment parameters.

        Returns:
            `(obs, next_state, done, info)`; `done` is `True` once all sites are assigned.
        """
        spin_index = jnp.mod(action, self.dim)
        spin_value = jnp.asarray(action // self.dim, dtype=jnp.int8)
        new_state_arr = state.state.at[spin_index].set(spin_value)
        next_active = state.replace(
            state=new_state_arr,
            time=state.time + 1,
            is_terminal=jnp.all(new_state_arr != -1),
            is_initial=False,
            is_pad=False,
        )
        next_pad = state.replace(is_pad=True)
        next_state = jax.tree.map(
            lambda p, a: jnp.where(state.is_terminal, p, a), next_pad, next_active
        )
        return (
            self.get_obs(next_state, env_params),
            next_state,
            jnp.astype(next_state.is_terminal, jnp.bool),
            {},
        )

    def backward_step(
        self,
        state: EnvState,
        backward_action: chex.Array,
        env_params: EnvParams,
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
        """Clears one assigned spin.

        Args:
            state: Current state.
            backward_action: Site index in `[0, dim)` to reset to `-1`.
            env_params: Environment parameters.

        Returns:
            `(obs, prev_state, done, info)`; `done` is `True` once the lattice is empty.
        """
        prev_state_arr = state.state.at[backward_action].set(-1)
        non_initial = EnvState(
            state=prev_state_arr,
            time=state.time - 1,
            is_terminal=False,
            is_initial=jnp.all(prev_state_arr == -1),
            is_pad=False,
        )
        init_pad = state.replace(is_pad=True)
        prev_state = jax.tree.map(
            lambda p, n: jnp.where(state.is_initial, p, n), init_pad, non_initial
        )
        return (
            self.get_obs(prev_state, env_params),
            prev_state,
            jnp.astype(prev_state.is_initial, jnp.bool),
            {},
        )

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Returns the partial spin assignment `state.state` (values in `{-1, 0, 1}`)."""
        return state.state

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        """Returns the site index `forward_action % dim` that was assigned."""
        return jnp.mod(forward_action, self.dim)

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns `site + dim * value`, the forward action that re-assigns the cleared site."""
        return backward_action + self.dim * state.state[backward_action].astype(jnp.int32)

    def get_action_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Returns the mask of valid forward actions for a single state. [2*dim].

        ``True`` = valid action (cell is empty), ``False`` = invalid (cell
        already has a spin). The mask is the concatenation of two identical
        sub-masks (one per spin value).
        """
        mask = state.state == -1
        return jnp.concatenate([mask, mask], axis=-1)

    def get_backward_action_mask(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Returns the mask of valid backward actions for a single state. [dim].

        ``True`` = valid backward action (a spin is set at this index),
        ``False`` = invalid (cell is empty).
        """
        return state.state != -1

    @property
    def name(self) -> str:
        """Environment name."""
        return f"Ising-{self.dim}-v0"

    @property
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2 * self.dim)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """Backward action space of the environment."""
        return spaces.Discrete(self.dim)

    @property
    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            low=jnp.full(self.dim, -1),
            high=jnp.full(self.dim, 1),
            shape=(self.dim,),
        )

    @property
    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
            "state": spaces.Box(
                low=jnp.full(self.dim, -1),
                high=jnp.full(self.dim, 1),
                shape=(self.dim,),
            ),
            "time": spaces.Discrete(self.max_steps_in_episode),
            "is_terminal": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
            "is_initial": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
            "is_pad": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
        })
