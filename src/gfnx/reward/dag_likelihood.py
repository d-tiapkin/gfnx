import math
from typing import Optional

import chex
import jax.numpy as jnp
from scipy.special import gammaln

from ..base import TAction, TLogReward, TRewardParams
from ..environment import DAGEnvParams, DAGEnvState
from ..utils.dag import load_dag_samples


class BaseDAGLikelihood:
    def init(self, rng_key: chex.PRNGKey, dummy_state: DAGEnvState) -> TRewardParams:
        """Initialize the likelihood. Default implementation returns None.

        Args:
        - rng_key: chex.PRNGKey, random key
        - dummy_state: DAGEnvState, shape [1, ...], a dummy state

        """
        return None

    def log_prob(self, state: DAGEnvState, env_params: DAGEnvParams) -> TLogReward:
        """Computes the log-likelihood of the data given the state - graph G:

            log P(D | G) = sum_j LocalScore(X_j | Pa_G(X_j))

        Args:
        - state: DAGEnvState, shape [B, ...], batch of states
        - env_params: DAGEnvParams, params of environment,
          always includes reward params

        Returns:
        - TLogReward, shape [B], batch of log-likelihoods
        """
        num_graphs, num_variables = state.adjacency_matrix.shape[:2]
        adjacency_matrix = state.adjacency_matrix.transpose(0, 2, 1)
        parents = adjacency_matrix.reshape(-1, num_variables)
        variables = jnp.tile(jnp.arange(num_variables), num_graphs)
        likelihood_params = env_params.reward_params.likelihood_params

        local_scores = self._local_score(variables, parents, likelihood_params)
        local_scores = local_scores.reshape(num_graphs, num_variables)
        return jnp.sum(local_scores, axis=1)  # [B]

    def delta_score(
        self,
        state: DAGEnvState,
        action: TAction,
        next_state: DAGEnvState,
        env_params: DAGEnvParams,
    ) -> TLogReward:
        """Computes the delta-score for adding an edge X_i -> X_j to some grpah
        G, for a specific choice of local score. The delta-score is given by:

            LocalScore(X_j | Pa_G(X_j) U X_i) - LocalScore(X_j | Pa_G(X_j))

        Args:
        - state: DAGEnvState, shape [B, ...], batch of states
        - action: DAGEnvAction, shape [B], batch of actions
        - next_state: DAGEnvState, shape [B, ...], batch of next states
        - env_params: DAGEnvParams, params of environment,
            always includes reward params

        Returns:
        - TLogReward, shape [B], batch of delta-scores
        """
        arange = jnp.arange(state.is_pad.shape[0])  # [B]
        source, target = jnp.divmod(action, env_params.num_variables)
        parents = state.adjacency_matrix[arange, :, target]  # [B, num_variables]
        next_parents = next_state.adjacency_matrix[arange, :, target]  # [B, num_variables]
        likelihood_params = env_params.reward_params.likelihood_params
        return self._local_score(target, next_parents, likelihood_params) - self._local_score(
            target, parents, likelihood_params
        )

    def _local_score(
        self,
        variables: chex.Array,
        parents: chex.Array,
        likelihood_params: TRewardParams,
    ) -> TLogReward:
        """Computes the local score LocalScore(X_j | Pa_G(X_j)).

        Args:
        - variables: chex.Array, shape [B], batch of variables
        - parents: chex.Array, shape [B, num_variables], batched mask of parents
        - likelihood_params: TRewardParams, params of likelihood

        Returns:
        - TLogReward, shape [B], batch of local scores
        """
        raise NotImplementedError


class ZeroScore(BaseDAGLikelihood):
    def delta_score(
        self,
        state: DAGEnvState,
        action: TAction,
        next_state: DAGEnvState,
        env_params: DAGEnvParams,
    ) -> TLogReward:
        return jnp.zeros(state.is_pad.shape[0])  # [B]

    def _local_score(
        self, variables: chex.Array, parents: chex.Array, likelihood_params: TRewardParams
    ) -> TLogReward:
        return jnp.zeros(variables.shape[0])  # [B]


class LinearGaussianScore(BaseDAGLikelihood):
    def __init__(
        self,
        data_path: str,
        prior_mean: float = 0.0,
        prior_scale: float = 1.0,
        obs_scale: float = math.sqrt(0.1),
    ):
        self.data_path = data_path
        self.prior_mean = prior_mean
        self.prior_scale = prior_scale
        self.obs_scale = obs_scale

    def init(self, rng_key: chex.PRNGKey, dummy_state: DAGEnvState) -> TRewardParams:
        """
        Initialize the likelihood.
        Args:
        - rng_key: chex.PRNGKey, random key
        - dummy_state: DAGEnvState, shape [1, ...], a dummy state

         Returns:
        - Loaded samples as a dictionary with key "data"
        """
        data = load_dag_samples(self.data_path)
        return {"data": data}

    def _local_score(
        self, variables: chex.Array, parents: chex.Array, likelihood_params: TRewardParams
    ) -> TLogReward:
        data = likelihood_params["data"]
        num_samples, num_variables = data.shape
        masked_data = data * parents[:, jnp.newaxis]

        means = self.prior_mean * jnp.sum(masked_data, axis=2)
        diffs = (data[:, variables].T - means) / self.obs_scale
        y_matrix = self.prior_scale * jnp.matmul(diffs[:, None], masked_data)
        y = jnp.squeeze(y_matrix, axis=1)
        sigma_matrix = (self.obs_scale**2) * jnp.eye(num_variables) + (
            self.prior_scale**2
        ) * jnp.matmul(masked_data.transpose(0, 2, 1), masked_data)
        term1 = jnp.sum(diffs**2, axis=1)
        term2 = -jnp.sum(y * jnp.linalg.solve(sigma_matrix, y[..., None])[..., 0], axis=1)
        _, term3 = jnp.linalg.slogdet(sigma_matrix)
        term4 = 2 * (num_samples - num_variables) * jnp.log(self.obs_scale)
        term5 = num_samples * jnp.log(2 * jnp.pi)

        return -0.5 * (term1 + term2 - term3 - term4 - term5)


class BGeScore(BaseDAGLikelihood):
    def __init__(
        self,
        data: chex.Array,
        mean_obs: Optional[chex.Array] = None,
        alpha_mu: float = 1.0,
        alpha_w: Optional[float] = None,
    ):
        self.num_samples, self.num_variables = data.shape
        if mean_obs is None:
            mean_obs = jnp.zeros((self.num_variables,))
        if alpha_w is None:
            alpha_w = self.num_variables + 2.0

        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w

        self.t = (self.alpha_mu * (self.alpha_w - self.num_variables - 1)) / (self.alpha_mu + 1)

        t_matrix = self.t * jnp.eye(self.num_variables)
        data_mean = jnp.mean(data, axis=0, keepdims=True)
        data_centered = data - data_mean

        self.r_matrix = (
            t_matrix
            + jnp.matmul(data_centered.T, data_centered)
            + ((self.num_samples * self.alpha_mu) / (self.num_samples + self.alpha_mu))
            * jnp.dot((data_mean - self.mean_obs).T, data_mean - self.mean_obs)
        )
        all_parents = jnp.arange(self.num_variables)
        self.log_gamma_term = (
            0.5 * (jnp.log(self.alpha_mu) - jnp.log(self.num_samples + self.alpha_mu))
            + gammaln(
                0.5 * (self.num_samples + self.alpha_w - self.num_variables + all_parents + 1)
            )
            - gammaln(0.5 * (self.alpha_w - self.num_variables + all_parents + 1))
            - 0.5 * self.num_samples * jnp.log(jnp.pi)
            + 0.5 * (self.alpha_w - self.num_variables + 2 * all_parents + 1) * jnp.log(self.t)
        )

    def _local_score(
        self, variables: chex.Array, parents: chex.Array, likelihood_params: TRewardParams
    ) -> TLogReward:
        def _logdet(array: chex.Array, mask: chex.Array) -> chex.Array:
            mask = mask[:, None, :] * mask[:, :, None]
            array = mask * array + (1.0 - mask) * jnp.eye(self.num_variables)
            _, logdet = jnp.linalg.slogdet(array)
            return logdet

        num_parents = jnp.sum(parents, axis=1)  # (num_graphs,)
        arange = jnp.arange(parents.shape[0])
        parents_and_variable = parents.at[arange, variables].set(True)

        factor = self.num_samples + self.alpha_w - self.num_variables + num_parents

        log_term_r = 0.5 * factor * _logdet(self.r_matrix, parents) - 0.5 * (factor + 1) * _logdet(
            self.r_matrix, parents_and_variable
        )

        return self.log_gamma_term[num_parents] + log_term_r
