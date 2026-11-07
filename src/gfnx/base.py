"""Abstract base classes shared by every gfnx environment and reward module.

gfnx separates a GFlowNet task into two independent objects:

* an **environment** ([`BaseEnvironment`][gfnx.base.BaseEnvironment]) that only describes
  the dynamics of the generation DAG — forward/backward transitions, action masks and
  observations;
* a **reward module** ([`BaseRewardModule`][gfnx.base.BaseRewardModule]) that maps a
  terminal state to a (log-)reward.

Neither object holds training state: everything that changes during training lives in
JAX pytrees: the environment state (`BaseEnvState`), the environment parameters
(`BaseEnvParams`) and the reward parameters (`BaseRewardParams`). All methods operate on a
**single** state without a batch dimension; use `jax.vmap` to batch them.

Typical initialisation:

```python
env = gfnx.HypergridEnvironment(dim=4, side=20)
env_params = env.init(rng_key)  # dynamics only
reward_module = gfnx.EasyHypergridRewardModule(side=20)
reward_params = reward_module.init(reward_key, env.reset())  # single dummy state
```
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

TEnvironment = TypeVar("TEnvironment", bound="BaseEnvironment")
TEnvParams = TypeVar("TEnvParams", bound="BaseEnvParams")

TObs = chex.ArrayTree
TEnvState = TypeVar("TEnvState", bound="BaseEnvState")
TAction = chex.Array
TBackwardAction = chex.Array

TRewardModule = TypeVar("TRewardModule", bound="BaseRewardModule")
TRewardParams = TypeVar("TRewardParams", bound="BaseRewardParams")
TLogReward = chex.Array
TReward = chex.Array
TDone = chex.Array


@chex.dataclass(frozen=True)
class BaseEnvState:
    r"""Base class for the state of a single environment instance.

    Concrete environments subclass it and add their domain fields (coordinates, tokens,
    adjacency matrices, ...). States are frozen `chex` dataclasses, i.e. JAX pytrees, so
    they can be stacked along a leading batch/time axis by `jax.vmap` / `jax.lax.scan`.

    Attributes:
        is_terminal: `True` if the state is a terminal state $x \in \mathcal{X}$.
        is_initial: `True` if the state is the initial state $s_0$.
        is_pad: `True` if the state is a padding step. A forward step from a terminal
            state (or a backward step from the initial state) returns the same state
            with `is_pad=True`, so that fixed-length rollouts can be produced with
            `jax.lax.scan`.
    """

    is_terminal: Bool[Array, ""]
    is_initial: Bool[Array, ""]
    is_pad: Bool[Array, ""]

    @property
    def is_valid(self) -> Bool[Array, ""]:
        """Whether the state corresponds to a real (non-padding) step.

        Convention: ``True`` for a real step, ``False`` for padding (the
        complement of `is_pad`, exposed for the ``True = include``
        masking convention used throughout the library). Pass directly to
        ``masked_sum`` / ``masked_mean`` or as ``where=`` to JAX reductions;
        no negation needed.
        """
        return jnp.logical_not(self.is_pad)


@chex.dataclass(frozen=True)
class BaseEnvParams:
    """Base class for environment parameters.

    Holds only the parameters of the *dynamics* (sizes, vocabularies, ...), as returned
    by [`BaseEnvironment.init`][gfnx.base.BaseEnvironment.init]. Reward parameters are
    never stored here; see [`BaseRewardParams`][gfnx.base.BaseRewardParams].
    """


@chex.dataclass(frozen=True)
class BaseRewardParams:
    """Base class for reward parameters.

    Holds every array a reward module needs at call time (lookup tables, proxy-model
    weights, mode sets, interaction matrices, ...), as returned by
    [`BaseRewardModule.init`][gfnx.base.BaseRewardModule.init]. Keeping them out of the
    module object lets them be swapped or trained (e.g. the Ising coupling matrix) without
    recompiling the environment logic.
    """


class BaseRewardModule(ABC, Generic[TEnvState, TEnvParams, TRewardParams]):
    r"""Base class for reward functions.

    A reward module maps a single **terminal** state to a scalar (log-)reward. It is
    decoupled from the environment: `reward` and `log_reward` never see the environment
    object or its parameters, only the terminal state and the module's own
    `reward_params`.

    Subclasses must implement:

    * [`init`][gfnx.base.BaseRewardModule.init] — build the reward parameters;
    * [`log_reward`][gfnx.base.BaseRewardModule.log_reward] — scalar $\log R(x)$;
    * [`reward`][gfnx.base.BaseRewardModule.reward] — scalar $R(x)$.

    Batched evaluation is done with `jax.vmap`:

    ```python
    log_rewards = jax.vmap(reward_module.log_reward, in_axes=(0, None))(
        final_states, reward_params
    )  # [B]
    ```
    """

    @abstractmethod
    def init(self, rng_key: chex.PRNGKey, dummy_state: TEnvState) -> TRewardParams:
        """Builds the reward parameters.

        Args:
            rng_key: Random key (used e.g. to sample a mode set or initialise a proxy).
            dummy_state: A single environment state without batch dimension, typically
                `env.reset()`. Used to infer shapes.

        Returns:
            The reward parameters pytree passed to every subsequent call.
        """
        raise NotImplementedError

    @abstractmethod
    def log_reward(self, state: TEnvState, reward_params: TRewardParams) -> Float[Array, ""]:
        r"""Computes the log-reward of a single state.

        Args:
            state: A single terminal state (no batch dimension).
            reward_params: Reward parameters returned by `init`.

        Returns:
            Scalar $\log R(x)$.
        """
        raise NotImplementedError

    @abstractmethod
    def reward(self, state: TEnvState, reward_params: TRewardParams) -> Float[Array, ""]:
        r"""Computes the reward of a single state.

        Args:
            state: A single terminal state (no batch dimension).
            reward_params: Reward parameters returned by `init`.

        Returns:
            Scalar $R(x) \geq 0$.
        """
        raise NotImplementedError


class BaseEnvironment(ABC, Generic[TEnvState, TEnvParams]):
    r"""Abstract base class for all gfnx environments.

    An environment describes the generation DAG $\mathcal{G} = (\mathcal{S}, \mathcal{E})$
    of a GFlowNet: how states are built forward from the initial state $s_0$, how they
    are taken apart backward, which actions are valid, and what the policy observes. It
    is **stateless** (all mutable data lives in the `EnvState` pytree) and knows nothing
    about rewards.

    All methods operate on a **single** state (no batch dimension). Use `jax.vmap`
    externally to run many environments in parallel:

    ```python
    rng_keys = jax.random.split(rng_key, num_envs)
    traj_data, final_states, info = jax.vmap(
        lambda rng: gfnx.utils.forward_rollout(rng, policy_fn, policy_params, env, env_params)
    )(rng_keys)
    ```

    **Backward actions.** Backward actions encode *structural* choices (e.g. "remove the
    token at position $i$") rather than the inverse of each specific forward edge.
    [`get_backward_action`][gfnx.base.BaseEnvironment.get_backward_action] and
    [`get_forward_action`][gfnx.base.BaseEnvironment.get_forward_action] convert between
    the two encodings for a given transition.

    **Padding.** Stepping forward from a terminal state (or backward from the initial
    state) is a no-op that returns the same state with `is_pad=True`, which lets
    rollouts use a fixed number of `jax.lax.scan` steps.

    **Masks.** Every `*_mask` uses the convention `True` = valid / include.

    **Enumerable environments** may additionally implement `get_all_states`,
    `state_to_index`, `get_true_distribution`, `get_empirical_distribution`,
    `get_expected_reward`, `get_normalizing_constant` and `get_ground_truth_sampling`.
    Which helpers exist depends on the environment: check `is_enumerable` and the
    `is_*_tractable` properties. The helpers take `reward_module` and `reward_params`
    explicitly because the environment does not own a reward.
    """

    @abstractmethod
    def init(self, rng_key: chex.PRNGKey) -> TEnvParams:
        """Builds the environment (dynamics) parameters.

        Args:
            rng_key: Random key. Unused by deterministic environments.

        Returns:
            Environment parameters passed to every other method.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def max_steps_in_episode(self) -> int:
        """Upper bound on the number of forward steps from $s_0$ to any terminal state.

        Counts the stop action, if any. `forward_rollout` allocates
        `max_steps_in_episode + 1` time steps: a value below the true maximum truncates
        trajectories, while a loose bound only adds padding.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, state: TEnvState, action: TAction, env_params: TEnvParams
    ) -> tuple[TObs, TEnvState, TDone, dict[Any, Any]]:
        """Applies one forward action to a single state.

        Args:
            state: Current state (no batch dimension).
            action: Scalar forward action, valid under `get_action_mask(state, ...)`.
            env_params: Environment parameters.

        Returns:
            A tuple `(obs, next_state, done, info)`: the observation of `next_state`, the
            next state, a boolean flag that is `True` iff `next_state` is terminal, and an
            (empty) info dict. If `state` is already terminal, `next_state` is `state`
            marked as padding.
        """
        raise NotImplementedError

    @abstractmethod
    def backward_step(
        self,
        state: TEnvState,
        backward_action: TBackwardAction,
        env_params: TEnvParams,
    ) -> tuple[TObs, TEnvState, TDone, dict[Any, Any]]:
        """Applies one backward action to a single state.

        Args:
            state: Current state (no batch dimension).
            backward_action: Scalar backward action, valid under
                `get_backward_action_mask(state, ...)`.
            env_params: Environment parameters.

        Returns:
            A tuple `(obs, prev_state, done, info)`, where `done` is `True` iff `prev_state`
            is the initial state. If `state` is already initial, `prev_state` is `state`
            marked as padding.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> TEnvState:
        """Returns the initial state $s_0$ of a single environment (no batch dimension)."""
        raise NotImplementedError

    @abstractmethod
    def get_obs(self, state: TEnvState, env_params: TEnvParams) -> chex.ArrayTree:
        """Returns the policy observation of a single state (no batch dimension)."""
        raise NotImplementedError

    @abstractmethod
    def get_backward_action(
        self,
        state: TEnvState,
        forward_action: TAction,
        next_state: TEnvState,
        env_params: TEnvParams,
    ) -> chex.Array:
        """Returns the backward action that undoes a forward transition.

        Args:
            state: State before the forward step.
            forward_action: Forward action taken in `state`.
            next_state: State after the forward step.
            env_params: Environment parameters.

        Returns:
            Scalar backward action `b` such that `backward_step(next_state, b)` returns
            `state`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_forward_action(
        self,
        state: TEnvState,
        backward_action: TAction,
        prev_state: TEnvState,
        env_params: TEnvParams,
    ) -> chex.Array:
        """Returns the forward action that redoes a backward transition.

        Args:
            state: State before the backward step.
            backward_action: Backward action taken in `state`.
            prev_state: State after the backward step (a parent of `state`).
            env_params: Environment parameters.

        Returns:
            Scalar forward action `a` such that `step(prev_state, a)` returns `state`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_action_mask(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Bool[Array, " n_actions"]:
        """Returns the mask of currently valid forward actions for a single state.

        Convention: ``True`` at index ``a`` means action ``a`` is **valid** (allowed)
        in this state; ``False`` means **invalid** (forbidden). Use directly with
        ``jax.nn.log_softmax(logits, where=action_mask, axis=-1)`` and with
        ``masked_sum`` / ``masked_mean`` (all of which treat ``True`` as include).

        Not batched — apply ``jax.vmap`` for batches, or use ``get_action_mask_batch``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_backward_action_mask(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Bool[Array, " n_bwd_actions"]:
        """Returns the mask of currently valid backward actions for a single state.

        Convention: ``True`` = valid backward action, ``False`` = invalid. See
        [`get_action_mask`][gfnx.base.BaseEnvironment.get_action_mask] for the rationale and
        intended usage.

        Not batched — use `get_backward_action_mask_batch` for batches.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience batched methods (defined once here, not overridden).
    # Use these in loss functions where states have a leading batch dim.
    # ------------------------------------------------------------------

    def get_action_mask_batch(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Bool[Array, "batch_size n_actions"]:
        """Batched `get_action_mask` for use in loss functions. state: [B, ...]."""
        return jax.vmap(self.get_action_mask, in_axes=(0, None))(state, env_params)

    def get_backward_action_mask_batch(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Bool[Array, "batch_size n_bwd_actions"]:
        """Batched `get_backward_action_mask` for use in loss functions. state: [B, ...]."""
        return jax.vmap(self.get_backward_action_mask, in_axes=(0, None))(state, env_params)

    def get_backward_action_batch(
        self,
        state: TEnvState,
        forward_action: TAction,
        next_state: TEnvState,
        env_params: TEnvParams,
    ) -> chex.Array:
        """Batched get_backward_action for use in loss functions. All inputs: [B, ...]"""
        return jax.vmap(self.get_backward_action, in_axes=(0, 0, 0, None))(
            state, forward_action, next_state, env_params
        )

    def get_forward_action_batch(
        self,
        state: TEnvState,
        backward_action: TAction,
        prev_state: TEnvState,
        env_params: TEnvParams,
    ) -> chex.Array:
        """Batched get_forward_action for use in loss functions. All inputs: [B, ...]"""
        return jax.vmap(self.get_forward_action, in_axes=(0, 0, 0, None))(
            state, backward_action, prev_state, env_params
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    @abstractmethod
    def action_space(self):
        """Action space of the environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def backward_action_space(self):
        """Backward action space of the environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self):
        """Observation space of the environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def state_space(self):
        """State space of the environment."""
        raise NotImplementedError

    @property
    def is_enumerable(self) -> bool:
        """Whether this environment supports enumerable operations."""
        return False

    def get_true_distribution(
        self, env_params: TEnvParams, reward_module: TRewardModule, reward_params: TRewardParams
    ) -> chex.Array:
        """Returns the target distribution $R(x)/Z$ over all terminal states.

        Only available when `is_enumerable` is `True`.

        Args:
            env_params: Environment parameters.
            reward_module: Reward module defining $R$.
            reward_params: Parameters of `reward_module`.

        Returns:
            Array of probabilities, indexed consistently with `get_empirical_distribution`.
        """
        if not self.is_enumerable:
            raise ValueError(f"Environment {self.name} is not enumerable")
        raise NotImplementedError

    def get_empirical_distribution(self, states: TEnvState, env_params: TEnvParams) -> chex.Array:
        """Computes the empirical distribution of a batch of terminal states.

        Unlike most methods, this one is naturally batched.

        Args:
            states: Batched states with leading dimension `[B, ...]`. Non-terminal entries
                are ignored.
            env_params: Environment parameters.

        Returns:
            Normalised histogram with the same layout as `get_true_distribution`.
        """
        if not self.is_enumerable:
            raise ValueError(f"Environment {self.name} is not enumerable")
        raise NotImplementedError

    def get_all_states(self, env_params: TEnvParams) -> chex.Array:
        """Returns all states if enumerable."""
        if not self.is_enumerable:
            raise ValueError(f"Environment {self.name} does not support getting all states")
        raise NotImplementedError

    def state_to_index(self, state: TEnvState, env_params: TEnvParams) -> chex.Array:
        """Converts a single state to its index in get_all_states."""
        if not self.is_enumerable:
            raise ValueError(f"Environment {self.name} does not support getting all states")
        raise NotImplementedError

    @property
    def is_expected_reward_tractable(self) -> bool:
        """Whether this environment supports tractable expected reward."""
        return False

    def get_expected_reward(
        self, env_params: TEnvParams, reward_module: TRewardModule, reward_params: TRewardParams
    ) -> float:
        """Returns the expected reward under the reward-proportional distribution."""
        if not self.is_expected_reward_tractable:
            raise ValueError(f"Expected reward for environment {self.name} is not tractable")
        raise NotImplementedError

    @property
    def is_normalizing_constant_tractable(self) -> bool:
        """Whether this environment supports tractable normalizing constant."""
        return False

    def get_normalizing_constant(
        self, env_params: TEnvParams, reward_module: TRewardModule, reward_params: TRewardParams
    ) -> float:
        """Returns the normalizing constant (sum of rewards over all states)."""
        if not self.is_normalizing_constant_tractable:
            raise ValueError(f"Normalizing constant for environment {self.name} is not tractable")
        raise NotImplementedError

    def get_log_normalizing_constant(
        self, env_params: TEnvParams, reward_module: TRewardModule, reward_params: TRewardParams
    ) -> float:
        """Returns log Z, the log of the sum of rewards over all terminal states.

        Defaults to ``log(get_normalizing_constant(...))``. Environments whose
        rewards can underflow in float32 should override it with a log-space
        computation.
        """
        return jnp.log(self.get_normalizing_constant(env_params, reward_module, reward_params))

    @property
    def is_ground_truth_sampling_tractable(self) -> bool:
        """Whether this environment supports tractable GT distribution sampling."""
        return False

    def get_ground_truth_sampling(
        self,
        rng_key: chex.PRNGKey,
        batch_size: int,
        env_params: TEnvParams,
        reward_module: TRewardModule,
        reward_params: TRewardParams,
    ) -> TEnvState:
        """Samples terminal states exactly from the target distribution $R(x)/Z$.

        Args:
            rng_key: Random key.
            batch_size: Number of samples.
            env_params: Environment parameters.
            reward_module: Reward module defining $R$.
            reward_params: Parameters of `reward_module`.

        Returns:
            Terminal states with a leading batch dimension `[batch_size, ...]`.
        """
        if not self.is_ground_truth_sampling_tractable:
            raise ValueError(f"GT sampling for environment {self.name} is not tractable")
        raise NotImplementedError


class BaseRenderer(ABC, Generic[TEnvState]):
    """Base class for rendering environments."""

    @abstractmethod
    def init_state(self, state: TEnvState):
        """Initialize visual representation of the given state."""
        raise NotImplementedError

    @abstractmethod
    def transition(self, state: TEnvState, next_state: TEnvState, action: TAction):
        """Update visualization for state transition."""
        raise NotImplementedError

    @property
    @abstractmethod
    def figure(self):
        """Return the current figure for rendering."""
        raise NotImplementedError
