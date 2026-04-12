import math
from typing import Generic, TypeVar

import chex
import jax.numpy as jnp
from scipy.special import gammaln

from ..base import BaseRewardParams, TAction, TLogReward
from ..environment import DAGEnvParams, DAGEnvState


@chex.dataclass(frozen=True)
class BaseDAGLikelihoodParams(BaseRewardParams):
    pass


TDAGLikelihoodParams = TypeVar("TDAGLikelihoodParams", bound=BaseDAGLikelihoodParams)


class BaseDAGLikelihood(Generic[TDAGLikelihoodParams]):
    def init(self, rng_key: chex.PRNGKey, dummy_state: DAGEnvState) -> TDAGLikelihoodParams:
        """Initialize the likelihood.

        Args:
        - rng_key: chex.PRNGKey, random key
        - dummy_state: DAGEnvState, a single dummy state (no batch dim)
        """
        raise NotImplementedError

    def log_prob(
        self, state: DAGEnvState, likelihood_params: TDAGLikelihoodParams
    ) -> TLogReward:
        """Computes the log-likelihood of the data given the state - graph G:

            log P(D | G) = sum_j LocalScore(X_j | Pa_G(X_j))

        Args:
        - state: DAGEnvState, single state (no batch dim)
        - likelihood_params: params of the likelihood

        Returns:
        - scalar log-likelihood
        """
        num_variables = state.adjacency_matrix.shape[0]
        # Row i of adjacency_matrix.T gives the parents of node i
        parents = state.adjacency_matrix.T  # [num_variables, num_variables]
        variables = jnp.arange(num_variables)  # [num_variables]

        local_scores = self._local_score(variables, parents, likelihood_params)
        return jnp.sum(local_scores)  # scalar

    def delta_score(
        self,
        state: DAGEnvState,
        action: TAction,
        next_state: DAGEnvState,
        env_params: DAGEnvParams,
        likelihood_params: TDAGLikelihoodParams,
    ) -> TLogReward:
        """Computes the delta-score for adding an edge X_i -> X_j to some graph
        G, for a specific choice of local score. The delta-score is given by:

            LocalScore(X_j | Pa_G(X_j) U X_i) - LocalScore(X_j | Pa_G(X_j))

        Args:
        - state: DAGEnvState, single state (no batch dim)
        - action: DAGEnvAction, scalar action
        - next_state: DAGEnvState, single next state (no batch dim)
        - env_params: DAGEnvParams, params of environment (for num_variables)
        - likelihood_params: params of the likelihood

        Returns:
        - scalar delta-score
        """
        _source, target = jnp.divmod(action, env_params.num_variables)
        parents = state.adjacency_matrix[:, target]  # [num_variables]
        next_parents = next_state.adjacency_matrix[:, target]  # [num_variables]
        return self._local_score(
            target[None], next_parents[None], likelihood_params
        ).squeeze() - self._local_score(
            target[None], parents[None], likelihood_params
        ).squeeze()

    def _local_score(
        self,
        variables: chex.Array,
        parents: chex.Array,
        likelihood_params: TDAGLikelihoodParams,
    ) -> TLogReward:
        """Computes the local score LocalScore(X_j | Pa_G(X_j)).

        Args:
        - variables: chex.Array, shape [K], variables to score
        - parents: chex.Array, shape [K, num_variables], mask of parents per variable
        - likelihood_params: params of the likelihood

        Returns:
        - chex.Array, shape [K], local scores
        """
        raise NotImplementedError


class ZeroScore(BaseDAGLikelihood[BaseDAGLikelihoodParams]):
    def init(
        self, rng_key: chex.PRNGKey, dummy_state: DAGEnvState
    ) -> BaseDAGLikelihoodParams:
        return BaseDAGLikelihoodParams()

    def delta_score(
        self,
        state: DAGEnvState,
        action: TAction,
        next_state: DAGEnvState,
        env_params: DAGEnvParams,
        likelihood_params: BaseDAGLikelihoodParams,
    ) -> TLogReward:
        return jnp.zeros(())  # scalar

    def _local_score(
        self,
        variables: chex.Array,
        parents: chex.Array,
        likelihood_params: BaseDAGLikelihoodParams,
    ) -> TLogReward:
        return jnp.zeros(variables.shape[0])  # [K]


@chex.dataclass(frozen=True)
class LinearGaussianScoreParams(BaseDAGLikelihoodParams):
    data: chex.Array


class LinearGaussianScore(BaseDAGLikelihood[LinearGaussianScoreParams]):
    def __init__(
        self,
        data: chex.Array,
        prior_mean: float = 0.0,
        prior_scale: float = 1.0,
        obs_scale: float = math.sqrt(0.1),
    ):
        self.data = data
        self.prior_mean = prior_mean
        self.prior_scale = prior_scale
        self.obs_scale = obs_scale

    def init(
        self, rng_key: chex.PRNGKey, dummy_state: DAGEnvState
    ) -> LinearGaussianScoreParams:
        return LinearGaussianScoreParams(data=self.data)

    def _local_score(
        self,
        variables: chex.Array,
        parents: chex.Array,
        likelihood_params: LinearGaussianScoreParams,
    ) -> TLogReward:
        data = likelihood_params.data
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


@chex.dataclass(frozen=True)
class BGeScoreParams(BaseDAGLikelihoodParams):
    r_matrix: chex.Array
    log_gamma_term: chex.Array


class BGeScore(BaseDAGLikelihood[BGeScoreParams]):
    def __init__(
        self,
        data: chex.Array,
        mean_obs: chex.Array | None = None,
        alpha_mu: float = 1.0,
        alpha_w: float | None = None,
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

    def init(self, rng_key: chex.PRNGKey, dummy_state: DAGEnvState) -> BGeScoreParams:
        return BGeScoreParams(r_matrix=self.r_matrix, log_gamma_term=self.log_gamma_term)

    def _local_score(
        self,
        variables: chex.Array,
        parents: chex.Array,
        likelihood_params: BGeScoreParams,
    ) -> TLogReward:
        r_matrix = likelihood_params.r_matrix
        log_gamma_term = likelihood_params.log_gamma_term

        def _logdet(array: chex.Array, mask: chex.Array) -> chex.Array:
            mask = mask[:, None, :] * mask[:, :, None]
            array = mask * array + (1.0 - mask) * jnp.eye(self.num_variables)
            _, logdet = jnp.linalg.slogdet(array)
            return logdet

        num_parents = jnp.sum(parents, axis=1)  # [K]
        arange = jnp.arange(parents.shape[0])
        parents_and_variable = parents.at[arange, variables].set(True)

        factor = self.num_samples + self.alpha_w - self.num_variables + num_parents

        log_term_r = 0.5 * factor * _logdet(r_matrix, parents) - 0.5 * (factor + 1) * _logdet(
            r_matrix, parents_and_variable
        )

        return log_gamma_term[num_parents] + log_term_r
