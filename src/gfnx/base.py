"""Abstract base class for all gfnx Environments"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

TEnvironment = TypeVar("TEnvironment", bound="BaseEnvironment")
TEnvParams = TypeVar("TEnvParams", bound="BaseEnvParams")

TObs = chex.ArrayTree
TEnvState = TypeVar("TEnvState", bound="BaseEnvState")
TAction = chex.Array
TBackwardAction = chex.Array

TRewardModule = TypeVar("TRewardModule", bound="BaseRewardModule")
TRewardParams = TypeVar("TRewardParams")
TLogReward = chex.Array
TReward = chex.Array
TDone = chex.Array


@chex.dataclass(frozen=True)
class BaseEnvState:
    is_terminal: Bool[Array, ""]
    is_initial: Bool[Array, ""]
    is_pad: Bool[Array, ""]


@chex.dataclass(frozen=True)
class BaseEnvParams:
    pass


class BaseRewardModule(ABC, Generic[TEnvState, TEnvParams]):
    """
    Base class for reward and log reward implementations.

    Subclasses must implement:
        - init: Initialize and return reward parameters.
        - log_reward: Compute log reward for a single state.
        - reward: Compute reward for a single state.

    Use jax.vmap to apply over batches of states.
    """

    @abstractmethod
    def init(self, rng_key: chex.PRNGKey, dummy_state: TEnvState) -> TRewardParams:
        """
        Initialize reward module, returns TRewardParams.
        Args:
        - rng_key: chex.PRNGKey, random key
        - dummy_state: TEnvState, a single dummy state (no batch dim)
        """
        raise NotImplementedError

    @abstractmethod
    def log_reward(self, state: TEnvState, reward_params: TRewardParams) -> Float[Array, ""]:
        """
        Compute the log reward for a single state.
        Args:
        - state: TEnvState, a single state (no batch dim)
        - reward_params: TRewardParams, reward parameters
        Returns:
        - scalar log reward
        """
        raise NotImplementedError

    @abstractmethod
    def reward(self, state: TEnvState, reward_params: TRewardParams) -> Float[Array, ""]:
        """
        Compute the reward for a single state.
        Args:
        - state: TEnvState, a single state (no batch dim)
        - reward_params: TRewardParams, reward parameters
        Returns:
        - scalar reward
        """
        raise NotImplementedError


class BaseEnvironment(ABC, Generic[TEnvState, TEnvParams]):
    """
    Abstract base class for all gfnx environments.

    All methods operate on a **single** state (no batch dimension).
    Use jax.vmap externally to run multiple environments in parallel:

        rng_keys = jax.random.split(rng_key, num_envs)
        traj_data, final_states, info = jax.vmap(
            lambda rng: forward_rollout(rng, policy_fn, policy_params, env, env_params)
        )(rng_keys)
    """

    @abstractmethod
    def get_init_state(self) -> TEnvState:
        """Returns a single initial state of the environment (no batch dim)."""
        raise NotImplementedError

    @abstractmethod
    def init(self, rng_key: chex.PRNGKey) -> TEnvParams:
        """Init and return environment parameters."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_steps_in_episode(self) -> int:
        raise NotImplementedError

    def step(
        self, state: TEnvState, action: TAction, env_params: TEnvParams
    ) -> tuple[TObs, TEnvState, TDone, dict[Any, Any]]:
        """Performs a single-environment step transition."""
        next_state, done, info = self._transition(state, action, env_params)
        done = jnp.astype(done, jnp.bool)
        return self.get_obs(next_state, env_params), next_state, done, info

    def backward_step(
        self,
        state: TEnvState,
        backward_action: TBackwardAction,
        env_params: TEnvParams,
    ) -> tuple[TObs, TEnvState, TDone, dict[Any, Any]]:
        """
        Performs a single-environment backward step transition.
        `done` is true when the state reaches the initial state.
        """
        prev_state, done, info = self._backward_transition(state, backward_action, env_params)
        done = jnp.astype(done, jnp.bool)
        return self.get_obs(prev_state, env_params), prev_state, done, info

    def reset(self, env_params: TEnvParams) -> tuple[TObs, TEnvState]:
        """Returns observation and initial state for a single environment."""
        state = self.get_init_state()
        return self.get_obs(state, env_params), state

    @abstractmethod
    def _transition(
        self, state: TEnvState, action: TAction, env_params: TEnvParams
    ) -> tuple[TEnvState, TDone, dict[Any, Any]]:
        """Single-environment forward transition. Not batched."""
        raise NotImplementedError

    @abstractmethod
    def _backward_transition(
        self,
        state: TEnvState,
        backward_action: TAction,
        env_params: TEnvParams,
    ) -> tuple[TEnvState, TDone, dict[Any, Any]]:
        """Single-environment backward transition. Not batched."""
        raise NotImplementedError

    @abstractmethod
    def get_obs(self, state: TEnvState, env_params: TEnvParams) -> chex.ArrayTree:
        """Returns observation for a single state (no batch dim)."""
        raise NotImplementedError

    @abstractmethod
    def get_backward_action(
        self,
        state: TEnvState,
        forward_action: TAction,
        next_state: TEnvState,
        env_params: TEnvParams,
    ) -> chex.Array:
        """Returns backward action for a single forward transition (not batched)."""
        raise NotImplementedError

    @abstractmethod
    def get_forward_action(
        self,
        state: TEnvState,
        backward_action: TAction,
        prev_state: TEnvState,
        env_params: TEnvParams,
    ) -> chex.Array:
        """Returns forward action for a single backward transition (not batched)."""
        raise NotImplementedError

    @abstractmethod
    def get_invalid_mask(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Bool[Array, " n_actions"]:
        """Returns mask of invalid forward actions for a single state. Not batched."""
        raise NotImplementedError

    @abstractmethod
    def get_invalid_backward_mask(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Bool[Array, " n_bwd_actions"]:
        """Returns mask of invalid backward actions for a single state. Not batched."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience batched methods (defined once here, not overridden).
    # Use these in loss functions where states have a leading batch dim.
    # ------------------------------------------------------------------

    def get_invalid_mask_batch(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Bool[Array, "batch_size n_actions"]:
        """Batched get_invalid_mask for use in loss functions. state: [B, ...]"""
        return jax.vmap(self.get_invalid_mask, in_axes=(0, None))(state, env_params)

    def get_invalid_backward_mask_batch(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Bool[Array, "batch_size n_bwd_actions"]:
        """Batched get_invalid_backward_mask for use in loss functions. state: [B, ...]"""
        return jax.vmap(self.get_invalid_backward_mask, in_axes=(0, None))(state, env_params)

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

    def sample_action(self, rng_key: chex.PRNGKey, policy_logprobs: chex.Array) -> Int[Array, ""]:
        """Sample a single action from policy logprobs. logprobs: [n_actions]"""
        return jax.random.categorical(rng_key, policy_logprobs, axis=-1)

    def sample_backward_action(
        self,
        rng_key: chex.PRNGKey,
        policy_logprobs: chex.Array,
    ) -> Int[Array, ""]:
        """Sample a single backward action from policy logprobs. logprobs: [n_bwd_actions]"""
        return jax.random.categorical(rng_key, policy_logprobs, axis=-1)

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
        self, env_params: TEnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> chex.Array:
        """Returns the true reward distribution over all states (enumerable envs only)."""
        if not self.is_enumerable:
            raise ValueError(f"Environment {self.name} is not enumerable")
        raise NotImplementedError

    def get_empirical_distribution(self, states: TEnvState, env_params: TEnvParams) -> chex.Array:
        """
        Extracts the empirical distribution from a batch of states.
        states: [B, ...] — batched states (this method is naturally batched over states).
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
    def is_mean_reward_tractable(self) -> bool:
        """Whether this environment supports mean reward tractability."""
        return False

    def get_mean_reward(
        self, env_params: TEnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> float:
        """Returns the mean reward over the true distribution."""
        if not self.is_mean_reward_tractable:
            raise ValueError(f"Mean reward for environment {self.name} is not tractable")
        raise NotImplementedError

    @property
    def is_normalizing_constant_tractable(self) -> bool:
        """Whether this environment supports tractable normalizing constant."""
        return False

    def get_normalizing_constant(
        self, env_params: TEnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> float:
        """Returns the normalizing constant (sum of rewards over all states)."""
        if not self.is_normalizing_constant_tractable:
            raise ValueError(f"Normalizing constant for environment {self.name} is not tractable")
        raise NotImplementedError

    @property
    def is_ground_truth_sampling_tractable(self) -> bool:
        """Whether this environment supports tractable GT distribution sampling."""
        return False

    def get_ground_truth_sampling(
        self,
        rng_key: chex.PRNGKey,
        batch_size: int,
        env_params: TEnvParams,
        reward_module: "BaseRewardModule",
        reward_params: TRewardParams,
    ) -> TEnvState:
        """
        Returns a batch of states sampled from the ground truth distribution.
        Returns: TEnvState with leading batch dim [batch_size, ...]
        """
        if not self.is_ground_truth_sampling_tractable:
            raise ValueError(f"GT sampling for environment {self.name} is not tractable")
        raise NotImplementedError


# Backward-compatibility alias
BaseVecEnvironment = BaseEnvironment


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
