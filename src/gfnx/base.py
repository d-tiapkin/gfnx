"""Abstract base class for all gfnx Environments"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

TEnvironment = TypeVar("TEnvironment", bound="BaseVecEnvironment")
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
    is_terminal: Bool[Array, " batch_size"]
    is_initial: Bool[Array, " batch_size"]
    is_pad: Bool[Array, " batch_size"]


@chex.dataclass(frozen=True)
class BaseEnvParams:
    reward_params: TRewardParams


class BaseRewardModule(ABC, Generic[TEnvState, TEnvParams]):
    """
    Base class for reward and log reward implementations.

    This class defines the interface for reward modules, which are
    responsible for computing rewards and log rewards given the state of
    the environment and its parameters.
    Subclasses should implement the following methods:
        - init: Initialize the reward module and return its parameters.
        - log_reward: Compute the log reward given the state and environment
          parameters.
        - reward: Compute the reward given the state and environment
          parameters.
    """

    @abstractmethod
    def init(self, rng_key: chex.PRNGKey, dummy_state: TEnvState) -> TRewardParams:
        """
        Initialize reward module, returns TRewardParams.
        Args:
        - rng_key: chex.PRNGKey, random key
        - dummy_state: TEnvState, shape [B, ...], batch of dummy states
        """
        raise NotImplementedError

    @abstractmethod
    def log_reward(self, state: TEnvState, env_params: TEnvParams) -> Float[Array, " batch_size"]:
        """
        Compute the log reward given the state and environment parameters.
        Args:
        - state: TEnvState, shape [B, ...], batch of states
        - env_params: TEnvParams, params of environment,
          always includes reward params
        Returns:
        - TLogReward, shape [B, ...], batch of log rewards
        """
        raise NotImplementedError

    @abstractmethod
    def reward(self, state: TEnvState, env_params: TEnvParams) -> Float[Array, " batch_size"]:
        """
        Log reward function, returns TReward
        Args:
        - state: TEnvState, shape [B, ...], batch of states
        - env_params: TEnvParams, params of environment,
          always includes reward params
        Returns:
        - TReward, shape [B, ...], batch of rewards
        """
        raise NotImplementedError


class BaseVecEnvironment(ABC, Generic[TEnvState, TEnvParams]):
    """
    Jittable abstract base class for all gfnx Environments.
    Note: all environments are vectorized by default.

    Args:
    - reward_module: TRewardModule, reward module
    """

    def __init__(self, reward_module: TRewardModule):
        self.reward_module = reward_module

    @abstractmethod
    def get_init_state(self, num_envs: int) -> TEnvState:
        """Returns batch of initial states of the environment."""
        raise NotImplementedError

    @abstractmethod
    def init(self, rng_key: chex.PRNGKey) -> TEnvParams:
        """
        Init params of the environment and reward module.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def max_steps_in_episode(self) -> int:
        raise NotImplementedError

    def step(
        self, state: TEnvState, action: TAction, env_params: TEnvParams
    ) -> Tuple[TObs, TEnvState, TLogReward, TDone, Dict[Any, Any]]:
        """Performs batched step transitions in the environment."""
        next_state, done, info = self.transition(state, action, env_params)
        done = jnp.astype(done, jnp.bool)  # Ensure that done is boolean
        # Compute reward only for a states that became terminal on this step
        new_dones = jnp.logical_and(done, jnp.logical_not(state.is_terminal))

        # Since computation of log rewards is expensive, we do it only if at
        # least one of the environments is done
        log_reward = jax.lax.cond(
            jnp.any(new_dones),
            self.reward_module.log_reward,
            lambda state, _: jnp.zeros_like(state.is_pad, dtype=jnp.float32),
            next_state,  # Args for log_reward
            env_params,  # Args for log_reward
        )
        log_reward = jnp.where(new_dones, log_reward, jnp.zeros_like(log_reward))
        return (
            self.get_obs(next_state, env_params),
            next_state,
            log_reward,
            done,
            info,
        )

    def backward_step(
        self,
        state: TEnvState,
        backward_action: TBackwardAction,
        env_params: TEnvParams,
    ) -> Tuple[TObs, TEnvState, TLogReward, TDone, Dict[Any, Any]]:
        """
        Performs batched backward step transitions in the environment.
        Important: `done` is true if the state is the initial one.
        """
        state, done, info = self.backward_transition(state, backward_action, env_params)
        done = jnp.astype(done, jnp.bool)  # Ensure that done is boolean
        # log reward is always zero for backward steps
        log_rewards = jnp.zeros(state.is_pad.shape, dtype=jnp.float32)
        return self.get_obs(state, env_params), state, log_rewards, done, info

    def reset(self, num_envs: int, env_params: TEnvParams) -> Tuple[TObs, TEnvState]:
        """Performs batched resetting of environment."""
        state = self.get_init_state(num_envs)
        return self.get_obs(state, env_params), state

    def transition(
        self, state: TEnvState, action: TAction, env_params: TEnvParams
    ) -> Tuple[TEnvState, TDone, Dict[Any, Any]]:
        """Environment-specific step transition."""
        next_state, done, info = jax.vmap(self._single_transition, in_axes=(0, 0, None))(
            state, action, env_params
        )
        return next_state, done, info

    def backward_transition(
        self,
        state: TEnvState,
        backward_action: TAction,
        env_params: TEnvParams,
    ) -> Tuple[TEnvState, TDone, Dict[Any, Any]]:
        """Environment-specific step backward transition."""
        prev_state, done, info = jax.vmap(self._single_backward_transition, in_axes=(0, 0, None))(
            state, backward_action, env_params
        )
        return prev_state, done, info

    @abstractmethod
    def _single_transition(
        self, state: TEnvState, action: TAction, env_params: TEnvParams
    ) -> Tuple[TEnvState, TDone, Dict[Any, Any]]:
        """Environment-specific step transition. NOTE: this is not batched!"""
        raise NotImplementedError

    @abstractmethod
    def _single_backward_transition(
        self,
        state: TEnvState,
        backward_action: TAction,
        env_params: TEnvParams,
    ) -> Tuple[TEnvState, TDone, Dict[Any, Any]]:
        """
        Environment-specific step backward transition.
        NOTE: this is not batched!
        """
        raise NotImplementedError

    @abstractmethod
    def get_obs(self, state: TEnvState, env_params: TEnvParams) -> chex.ArrayTree:
        """Applies observation function to state. Should be batched."""
        raise NotImplementedError

    @abstractmethod
    def get_backward_action(
        self,
        state: TEnvState,
        forward_action: TAction,
        next_state: TEnvState,
        env_params: TEnvParams,
    ) -> chex.Array:
        """
        Returns backward action given the complete characterization of the
        forward transition. Should be batched.
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
        """
        Returns forward action given the complete characterization of the
        backward transition. Should be batched.
        """
        raise NotImplementedError

    @abstractmethod
    def get_invalid_mask(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Bool[Array, " batch_size"]:
        """Returns mask of invalid actions. Should be batched"""
        raise NotImplementedError

    @abstractmethod
    def get_invalid_backward_mask(
        self, state: TEnvState, env_params: TEnvParams
    ) -> Bool[Array, " batch_size"]:
        """Returns mask of invalid backward actions. Should be batched."""
        raise NotImplementedError

    def sample_action(
        self, rng_key: chex.PRNGKey, policy_logprobs: chex.Array
    ) -> Int[Array, " batch_size"]:
        """
        Helping function for sampling actions from policy.
        """
        batch_size = policy_logprobs.shape[0]
        action = jax.random.categorical(rng_key, policy_logprobs, axis=-1)
        chex.assert_shape(action, (batch_size,))
        return action

    def sample_backward_action(
        self,
        rng_key: chex.PRNGKey,
        policy_logprobs: chex.Array,
    ) -> Int[Array, " batch_size"]:
        """
        Helping function for sampling actions from policy.
        """
        batch_size = policy_logprobs.shape[0]
        action = jax.random.categorical(rng_key, policy_logprobs, axis=-1)
        chex.assert_shape(action, (batch_size,))
        return action

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
        """Action space of the environment."""
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

    # Additional methods for enumerable environments
    def get_true_distribution(self, env_params: TEnvParams) -> chex.Array:
        """
        Returns the true distribution of rewards for all states if the
        environment is enumerable.
        Args:
            env_params: TEnvParams, params of environment
        Returns:
            chex.Array, true distribution of rewards for all states
        """
        if not self.is_enumerable:
            raise ValueError(f"Environment {self.name} is not enumerable")
        raise NotImplementedError

    def get_empirical_distribution(self, states: TEnvState, env_params: TEnvParams) -> chex.Array:
        """
        Extracts the empirical distribution from the given states if the
        environment is enumerable.
        Args:
            states: TEnvState, shape [B, ...], batch of states
            env_params: TEnvParams, params of environment
        Returns:
            chex.Array, empirical distribution of rewards for all states
        """
        if not self.is_enumerable:
            raise ValueError(f"Environment {self.name} is not enumerable")
        raise NotImplementedError

    def get_all_states(self, env_params: TEnvParams) -> chex.Array:
        """Returns a list of all states if this functionality is supported."""
        if not self.is_enumerable:
            raise ValueError(f"Environment {self.name} does not support getting all states")
        raise NotImplementedError

    def state_to_index(self, state: TEnvState, env_params: TEnvParams) -> chex.Array:
        """
        Converts a state to its corresponding index returned by `get_all_states` function
        if this functionality is supported.
        """
        if not self.is_enumerable:
            raise ValueError(f"Environment {self.name} does not support getting all states")
        raise NotImplementedError

    @property
    def is_mean_reward_tractable(self) -> bool:
        """Whether this environment supports mean reward tractability."""
        return False

    def get_mean_reward(self, env_params: TEnvParams) -> float:
        """
        Returns the mean reward over the true distribution.
        Args:
            env_params: TEnvParams, params of environment
        Returns:
           float, mean reward over the true distribution
        """
        if not self.is_mean_reward_tractable:
            raise ValueError(f"Mean reward for environment {self.name} is not tractable")
        raise NotImplementedError

    @property
    def is_normalizing_constant_tractable(self) -> bool:
        """Whether this environment supports tractable normalizing constant."""
        return False

    def get_normalizing_constant(self, env_params: TEnvParams) -> float:
        """
        Returns the normalizing constant for the hypergrid environment.
        The normalizing constant is computed as the sum of rewards.
        """
        if not self.is_normalizing_constant_tractable:
            raise ValueError(f"Normalizing constant for environment {self.name} is not tractable")
        raise NotImplementedError

    @property
    def is_ground_truth_sampling_tractable(self) -> bool:
        """Whether this environment supports tractable sampling from the GT distribution."""
        return False

    def get_ground_truth_sampling(
        self, rng_key: chex.PRNGKey, batch_size: int, env_params: TEnvParams
    ) -> TEnvState:
        """
        Returns the ground truth sampling for the hypergrid environment.
        The ground truth sampling is computed as the sum of rewards.
        Args:
            batch_size: int, number of samples to generate
            env_params: TEnvParams, params of environment
        Returns:
            TEnvState, shape [batch_size, ...], batch of ground truth sampled states
        """
        if not self.is_ground_truth_sampling_tractable:
            raise ValueError(f"GT sampling for environment {self.name} is not tractable")
        raise NotImplementedError


class BaseRenderer(ABC, Generic[TEnvState]):
    """
    Base class for rendering environments.
    """

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
