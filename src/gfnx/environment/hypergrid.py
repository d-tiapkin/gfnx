"""Hypergrid environment: walk on a $D$-dimensional grid and decide when to stop.

The hypergrid is the classic GFlowNet benchmark of Bengio et al. (2021). It is small,
fully enumerable and highly multimodal under the standard rewards, which makes it the
default environment for debugging objectives and policies. Rewards live in
[`gfnx.reward.hypergrid`][gfnx.reward.hypergrid].
"""

from itertools import product
from math import prod
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
    BaseRewardModule,
    TAction,
    TDone,
    TRewardParams,
)


@chex.dataclass(frozen=True)
class EnvState(BaseEnvState):
    r"""State of a single hypergrid environment.

    Attributes:
        state: Integer coordinates $(s^1, \ldots, s^D)$, each in `[0, side - 1]`.
        is_terminal: `True` once the stop action has been taken.
        is_initial: `True` at the origin before any step.
        is_pad: `True` for padding steps after termination.
    """

    state: Int[Array, " dim"]
    is_terminal: Bool[Array, ""]
    is_initial: Bool[Array, ""]
    is_pad: Bool[Array, ""]


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    """Hypergrid dynamics parameters.

    Attributes:
        dim: Number of dimensions $D$.
        side: Number of cells $H$ per dimension.
    """

    dim: int = 4
    side: int = 20


class HypergridEnvironment(BaseEnvironment[EnvState, EnvParams]):
    r"""Hypergrid environment of size $H^D$.

    An object is a cell of the grid $\{0, \ldots, H-1\}^D$. It is built by starting at
    the origin and repeatedly incrementing one coordinate by one, until the agent takes
    the stop action. The terminal state is the cell where the agent stopped, so every
    cell is a possible terminal object and $|\mathcal{X}| = H^D$.

    * **State** — the coordinate vector `state` of shape `[dim]`; $s_0 = (0, \ldots, 0)$.
    * **Forward actions** — `dim + 1` actions. Action `i < dim` increments coordinate
      `i`; action `dim` (`env.stop_action`) terminates the trajectory. Incrementing a
      coordinate that is already at `side - 1` is masked out.
    * **Backward actions** — `dim` actions. Action `i` decrements coordinate `i`
      (valid iff it is positive). From a terminal state the only valid backward action
      is `0`, which undoes the stop action.
    * **Observation** — concatenated one-hot encodings of the coordinates, a float vector
      of shape `[dim * side]`.
    * **Episode length** — at most `dim * (side - 1) + 1` steps, including the stop
      action; `max_steps_in_episode` returns the looser bound `dim * side`.

    The environment is enumerable: it exposes the exact target distribution, the
    normalising constant, the expected reward and exact sampling for any hypergrid
    reward module.

    Args:
        dim: Number of dimensions $D$. The state space grows as $H^D$.
        side: Number of cells $H$ per dimension. Must match the `side` of the reward
            module.

    Attributes:
        stop_action: Index of the stop action (equal to `dim`).

    References:
        Bengio, E., Jain, M., Korablyov, M., Precup, D., Bengio, Y. (2021). Flow network
        based generative models for non-iterative diverse candidate generation. NeurIPS.
    """

    def __init__(self, dim: int = 4, side: int = 20) -> None:
        self.dim = dim
        self.side = side

        self.stop_action = self.dim  # Stop action id

    def reset(self) -> EnvState:
        """Returns the initial state: the origin of the grid."""
        return EnvState(
            state=jnp.zeros((self.dim,), dtype=jnp.int32),
            is_terminal=jnp.bool_(False),
            is_initial=jnp.bool_(True),
            is_pad=jnp.bool_(False),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        """Returns `EnvParams(dim, side)`. The environment is deterministic."""
        return EnvParams(dim=self.dim, side=self.side)

    @property
    def is_enumerable(self) -> bool:
        """Whether this environment supports enumerable operations."""
        return True

    @property
    def max_steps_in_episode(self) -> int:
        """Upper bound `dim * side` on the trajectory length."""
        return self.dim * self.side

    def get_all_states(self, env_params: EnvParams) -> EnvState:
        """Returns every cell of the grid as a batched `EnvState` of size `side ** dim`.

        Cells are listed in row-major (`itertools.product`) order, which matches
        `state_to_index`.
        """

        all_states_coords = jnp.array(list(product(range(self.side), repeat=self.dim)))
        num_states = all_states_coords.shape[0]
        is_initial = all_states_coords.sum(axis=1) == 0
        is_terminal = jnp.zeros(num_states, dtype=jnp.bool)
        is_pad = jnp.zeros(num_states, dtype=jnp.bool)

        return EnvState(
            state=all_states_coords,
            is_terminal=is_terminal,
            is_initial=is_initial,
            is_pad=is_pad,
        )

    def state_to_index(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Returns the row-major index of a single cell in `[0, side ** dim)`."""
        return jnp.ravel_multi_index(
            state.state.astype(jnp.int32),
            dims=(self.side,) * self.dim,
            mode="clip",
        )

    def step(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
        """Increments a coordinate or stops.

        Args:
            state: Current state.
            action: Coordinate index in `[0, dim)` to increment, or `stop_action`.
            env_params: Environment parameters.

        Returns:
            `(obs, next_state, done, info)`; `done` is `True` after the stop action.
        """
        # Compute the "active" next state (always, regardless of is_terminal)
        done = jnp.logical_or(
            action == self.stop_action,
            state.state[action] >= self.side - 1,
        )
        next_inter = state.replace(
            state=state.state.at[action].add(1),
            is_terminal=False,
            is_initial=False,
        )
        next_finished = state.replace(is_terminal=True, is_initial=False)
        next_active = jax.tree.map(lambda f, i: jnp.where(done, f, i), next_finished, next_inter)
        # If already terminal, freeze as pad instead
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
        """Decrements a coordinate, or undoes the stop action at a terminal state.

        Args:
            state: Current state.
            backward_action: Coordinate index in `[0, dim)` to decrement. Ignored at a
                terminal state, where the only backward move is to un-terminate.
            env_params: Environment parameters.

        Returns:
            `(obs, prev_state, done, info)`; `done` is `True` once the origin is reached.
        """
        # Undo stop: just un-set terminal flag
        undo_stop = EnvState(
            state=state.state,
            is_terminal=False,
            is_initial=jnp.all(state.state == 0),
            is_pad=False,
        )
        # Decrement chosen dimension
        prev_inner = state.state.at[backward_action].add(-1)
        dec_dim = EnvState(
            state=prev_inner,
            is_terminal=False,
            is_initial=jnp.all(prev_inner == 0),
            is_pad=False,
        )
        # Non-initial: pick undo_stop or dec_dim based on is_terminal
        non_initial = jax.tree.map(
            lambda u, d: jnp.where(state.is_terminal, u, d), undo_stop, dec_dim
        )
        # If already initial, freeze as pad
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
        """Returns the flattened one-hot encoding of the coordinates, shape `[dim * side]`."""
        state_ohe = jax.nn.one_hot(state.state, self.side, dtype=jnp.float32)
        return jnp.reshape(state_ohe, (self.dim * self.side,))

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        """Maps an increment of coordinate `i` to backward action `i`, and stop to `0`."""
        return jnp.where(forward_action >= self.backward_action_space.n, 0, forward_action)

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        env_params: EnvParams,
    ) -> chex.Array:
        """Returns `stop_action` if `state` is terminal, otherwise `backward_action`."""
        return jnp.where(state.is_terminal, self.stop_action, backward_action)

    def get_action_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Returns the mask of valid forward actions for a single state. [dim+1]

        ``True`` = valid action, ``False`` = invalid (cell already at the
        upper boundary along that axis).
        """
        augmented_state = jnp.concat([state.state, jnp.zeros((1,))], axis=-1)
        return augmented_state != self.side - 1

    def get_backward_action_mask(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Returns the mask of valid backward actions for a single state. [dim]

        ``True`` = valid backward move along that axis, ``False`` = invalid.
        """
        return jax.lax.cond(
            state.is_terminal,
            lambda x: jnp.zeros_like(x, dtype=jnp.bool).at[0].set(True),
            lambda x: x != 0,
            state.state,
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return f"HyperGrid-{self.side}**{self.dim}-v0"

    @property
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.dim + 1)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """Backward action space of the environment."""
        return spaces.Discrete(self.dim)

    @property
    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            low=jnp.zeros(self.dim * self.side),
            high=jnp.ones(self.dim * self.side),
            shape=(self.dim * self.side,),
        )

    @property
    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
            "state": spaces.Box(low=0.0, high=self.side, shape=(self.dim,), dtype=jnp.int32),
            "is_terminal": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
            "is_initial": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
            "is_pad": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
        })

    def _get_states_rewards(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> chex.Array:
        """Evaluates the reward on every cell; returns an array of shape `(side,) * dim`."""
        reward_side = getattr(reward_module, "side", self.side)
        if reward_side != self.side:
            raise ValueError(f"reward_module.side={reward_side} != environment side={self.side}")
        rewards = jnp.zeros((self.side,) * self.dim, dtype=jnp.float32)

        def update_rewards(idx: int, rewards: chex.Array):
            state = jnp.unravel_index(idx, shape=rewards.shape)
            env_state = EnvState(
                state=jnp.array(state),
                is_terminal=jnp.bool_(True),
                is_initial=jnp.bool_(False),
                is_pad=jnp.bool_(False),
            )
            reward = reward_module.reward(env_state, reward_params)
            return rewards.at[state].set(reward)

        return jax.lax.fori_loop(0, self.side**self.dim, update_rewards, rewards)

    def get_true_distribution(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> chex.Array:
        """Returns $R(x)/Z$ for every cell, as an array of shape `(side,) * dim`."""
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return rewards / rewards.sum()

    def get_empirical_distribution(self, states: EnvState, env_params: EnvParams) -> chex.Array:
        """Histogram of a batch of terminal states, as an array of shape `(side,) * dim`."""
        dist_shape = (self.side,) * self.dim
        sample_idx = jax.vmap(lambda x: jnp.ravel_multi_index(x, dims=dist_shape, mode="clip"))(
            states.state
        )

        valid_mask = states.is_terminal.astype(jnp.float32)
        empirical_dist = jax.ops.segment_sum(valid_mask, sample_idx, num_segments=prod(dist_shape))
        empirical_dist = empirical_dist.reshape(dist_shape)
        empirical_dist /= empirical_dist.sum()
        return empirical_dist

    @property
    def is_expected_reward_tractable(self) -> bool:
        """`True`: the expected reward is computed by enumeration."""
        return True

    def get_expected_reward(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> float:
        r"""Returns $\mathbb{E}_{x \sim R/Z}[R(x)] = \sum_x R(x)^2 / \sum_x R(x)$."""
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return jnp.pow(rewards, 2).sum() / rewards.sum()

    @property
    def is_normalizing_constant_tractable(self) -> bool:
        """`True`: $Z$ is computed by enumeration."""
        return True

    def get_normalizing_constant(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> float:
        r"""Returns $Z = \sum_x R(x)$ over all `side ** dim` cells."""
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return rewards.sum()

    @property
    def is_ground_truth_sampling_tractable(self) -> bool:
        """`True`: exact samples are drawn from the enumerated distribution."""
        return True

    def get_ground_truth_sampling(
        self,
        rng_key: chex.PRNGKey,
        batch_size: int,
        env_params: EnvParams,
        reward_module: BaseRewardModule,
        reward_params: TRewardParams,
    ) -> EnvState:
        """Draws `batch_size` terminal cells exactly from $R(x)/Z$.

        Args:
            rng_key: Random key.
            batch_size: Number of samples.
            env_params: Environment parameters.
            reward_module: Hypergrid reward module.
            reward_params: Its parameters.

        Returns:
            Batched terminal `EnvState` with `state` of shape `[batch_size, dim]`.
        """
        true_distribution = self.get_true_distribution(env_params, reward_module, reward_params)
        flat_distribution = true_distribution.flatten()

        sampled_indices = jax.random.choice(
            rng_key,
            a=flat_distribution.size,
            shape=(batch_size,),
            p=flat_distribution,
        )

        sampled_coords_unstacked = jnp.unravel_index(
            sampled_indices, shape=true_distribution.shape
        )
        sampled_coords = jnp.stack(sampled_coords_unstacked, axis=1)

        return EnvState(
            state=sampled_coords,
            is_terminal=jnp.ones((batch_size,), dtype=jnp.bool),
            is_initial=jnp.zeros((batch_size,), dtype=jnp.bool),
            is_pad=jnp.zeros((batch_size,), dtype=jnp.bool),
        )
