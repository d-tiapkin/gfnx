"""DAG environment for Bayesian structure learning: add edges one at a time.

Rewards (graph priors and marginal likelihoods) live in
[`gfnx.reward.dag`][gfnx.reward.dag], [`gfnx.reward.dag_prior`][gfnx.reward.dag_prior] and
[`gfnx.reward.dag_likelihood`][gfnx.reward.dag_likelihood].
"""

from typing import Any

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool

from gfnx.base import (
    BaseEnvironment,
    BaseEnvParams,
    BaseEnvState,
    BaseRewardModule,
    TAction,
    TDone,
    TRewardParams,
)

from .. import spaces
from ..utils.dag import adj_to_index, construct_all_dags, get_all_adjacencies_flat_bits


@chex.dataclass(frozen=True)
class EnvState(BaseEnvState):
    r"""State of a single DAG environment.

    Attributes:
        adjacency_matrix: Boolean matrix of shape `[d, d]`; entry `[i, j]` is `True` iff
            the graph contains the edge $X_i \to X_j$.
        closure_T: Transitive closure of the *transposed* graph, shape `[d, d]`, with a
            `True` diagonal: entry `[i, j]` is `True` iff $X_j$ reaches $X_i$ in $G$
            (including $i = j$). Adding $X_i \to X_j$ creates a cycle iff
            `closure_T[i, j]` is `True`.
        is_terminal: `True` once the stop action has been taken.
        is_initial: `True` for the empty graph.
        is_pad: `True` for padding steps after termination.
    """

    adjacency_matrix: Bool[Array, " num_variables num_variables"]
    closure_T: Bool[Array, " num_variables num_variables"]
    is_terminal: Bool[Array, ""]
    is_initial: Bool[Array, ""]
    is_pad: Bool[Array, ""]


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    """DAG dynamics parameters.

    Attributes:
        num_variables: Number of nodes $d$. Also needed by `delta_score` to decode edge
            actions.
    """

    num_variables: int = 4


class DAGEnvironment(BaseEnvironment[EnvState, EnvParams]):
    r"""Sequential construction of directed acyclic graphs over $d$ variables.

    Following Deleu et al. (2022), a GFlowNet samples a DAG $G$ over variables
    $X_1, \ldots, X_d$ by starting from the empty graph and adding one edge at a time,
    so that its terminal distribution approximates the posterior $P(G \mid \mathcal{D})$.
    Every state is itself a valid DAG and can be terminated with the stop action.

    * **State** — the adjacency matrix plus the transitive closure of its transpose
      (`closure_T`), both `[d, d]` booleans. $s_0$ is the empty graph.
    * **Forward actions** — `d * d + 1` actions. Action `i * d + j` adds the edge
      $X_i \to X_j$; it is valid iff the edge is absent and does not close a cycle.
      Action `d * d` (`stop_action`) terminates the trajectory and is always valid.
    * **Backward actions** — same `d * d + 1` encoding: action `i * d + j` removes an
      existing edge $X_i \to X_j$; at a terminal state the only valid backward action is
      `d * d`, which undoes the stop. Forward and backward encodings coincide, so
      `get_forward_action` / `get_backward_action` are the identity.
    * **Observation** — the adjacency matrix, `[d, d]` booleans.
    * **Episode length** — at most `d * (d - 1) / 2 + 1` steps (a complete DAG, then stop).

    **Online acyclicity.** Adding $u \to v$ updates `closure_T` with a single outer
    product (every node that reaches $u$ can now reach every node reachable from $v$),
    an $O(d^2)$ update with no graph traversal. The action mask is
    `~(adjacency_matrix | closure_T)`. Removing an edge recomputes the closure by BFS.

    **Enumeration.** For `d < 6` (29,281 DAGs at `d = 5`) the environment enumerates
    every DAG and exposes the exact posterior, $\log Z$ (computed in log space) and the
    expected reward.

    Args:
        num_variables: Number of variables $d$.

    Attributes:
        stop_action: Index of the stop action (`d * d`).

    References:
        Deleu, T., Góis, A., Emezue, C., Rankawat, M., Lacoste-Julien, S., Bauer, S.,
        Bengio, Y. (2022). Bayesian structure learning with generative flow networks.
        UAI.
    """

    def __init__(
        self,
        num_variables: int,
    ) -> None:
        self.num_variables = num_variables
        self.stop_action = self.num_variables * self.num_variables

        if self.is_enumerable:
            # Helper array to convert adjacency matrix to index — computed once at init.
            self.all_adjacencies_flat_bits = get_all_adjacencies_flat_bits(self.num_variables)
            self.all_dags_num = self.all_adjacencies_flat_bits.shape[0]

    def reset(self) -> EnvState:
        """Returns the empty graph (identity `closure_T`)."""
        return EnvState(
            adjacency_matrix=jnp.zeros(
                (self.num_variables, self.num_variables),
                dtype=jnp.bool,
            ),
            closure_T=jnp.eye(self.num_variables, dtype=jnp.bool),
            is_terminal=jnp.bool_(False),
            is_initial=jnp.bool_(True),
            is_pad=jnp.bool_(False),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        """Returns `EnvParams(num_variables)`. The environment is deterministic."""
        return EnvParams(num_variables=self.num_variables)

    @property
    def max_steps_in_episode(self) -> int:
        """`d * (d - 1) / 2` edge additions (a complete DAG) plus the stop action."""
        return (self.num_variables * (self.num_variables - 1)) // 2 + 1

    def step(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
        r"""Adds the edge `divmod(action, d)` or stops.

        Args:
            state: Current state.
            action: Edge index `i * d + j` (adds $X_i \to X_j$), or `stop_action`.
            env_params: Environment parameters.

        Returns:
            `(obs, next_state, done, info)`; `done` is `True` after the stop action.
        """
        done = action == self.stop_action
        source, target = jnp.divmod(action, self.num_variables)

        def get_state_finished(_s: chex.Array, _t: chex.Array) -> EnvState:
            return state.replace(is_terminal=True, is_initial=False)

        def get_state_inter(s: chex.Array, t: chex.Array) -> EnvState:
            adjacency_matrix = state.adjacency_matrix.at[s, t].set(True)
            closure_T = state.closure_T
            outer_product = jnp.logical_and(
                jnp.expand_dims(closure_T[s], 0),
                jnp.expand_dims(closure_T[:, t], 1),
            )
            closure_T = jnp.logical_or(closure_T, outer_product)
            return state.replace(
                adjacency_matrix=adjacency_matrix,
                closure_T=closure_T,
                is_terminal=False,
                is_initial=False,
            )

        next_active = jax.lax.cond(done, get_state_finished, get_state_inter, source, target)
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

    def _single_source_bfs(self, adjacency_t: chex.Array, start: int) -> chex.Array:
        """BFS from `start` in the graph with adjacency matrix `adjacency_t`.

        Returns a boolean 1D array `visited` of shape `(d,)`, indicating which
        nodes are reachable from `start` in `adjacency_t`. Convention:
        `adjacency_t[i, j] = True` means there's an edge `i -> j` in G^T.
        """
        d = adjacency_t.shape[0]
        visited_init = jnp.zeros(d, dtype=jnp.bool).at[start].set(True)
        frontier_init = visited_init

        def cond_fun(carry):
            frontier, _visited = carry
            return jnp.any(frontier)  # continue while we have newly discovered nodes

        def body_fun(carry):
            frontier, visited = carry
            # adjacency_t & frontier[:, None] marks edges from any node in `frontier`.
            # Taking `any(..., axis=0)` merges them into which nodes we can discover next.
            neighbors = jnp.any(adjacency_t & frontier[:, None], axis=0)
            new_frontier = neighbors & jnp.logical_not(visited)
            new_visited = visited | new_frontier
            return (new_frontier, new_visited)

        _, visited_final = jax.lax.while_loop(cond_fun, body_fun, (frontier_init, visited_init))
        return visited_final

    def _single_compute_closure(self, adjacency: chex.Array) -> chex.Array:
        """Transitive closure of graph G via BFS-from-each-node.

        Given the adjacency matrix of G (shape `(d, d)`), compute its
        transitive closure. `closure[i, j] = True` iff i can reach j in G.
        The diagonal is forced True (i reaches i by convention).
        """
        d = adjacency.shape[0]
        closure = jax.vmap(lambda i: self._single_source_bfs(adjacency, i))(jnp.arange(d))
        return jnp.logical_or(closure, jnp.eye(d, dtype=jnp.bool))

    def backward_step(
        self,
        state: EnvState,
        backward_action: chex.Array,
        env_params: EnvParams,
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
        """Removes an edge, or undoes the stop action at a terminal state.

        Args:
            state: Current state.
            backward_action: Edge index `i * d + j` of an existing edge to remove, or
                `stop_action` at a terminal state.
            env_params: Environment parameters.

        Returns:
            `(obs, prev_state, done, info)`; `done` is `True` once the graph is empty.
            `closure_T` is recomputed from scratch by BFS after an edge removal.
        """
        unterminate = backward_action == self.stop_action

        def get_state_terminating() -> EnvState:
            return state.replace(
                is_terminal=False,
                is_initial=jnp.all(jnp.logical_not(state.adjacency_matrix)),
                is_pad=False,
            )

        def get_state_inter() -> EnvState:
            source, target = jnp.divmod(backward_action, self.num_variables)
            adjacency_matrix = state.adjacency_matrix.at[source, target].set(False)
            closure_T = self._single_compute_closure(adjacency_matrix.T)
            return state.replace(
                adjacency_matrix=adjacency_matrix,
                closure_T=closure_T,
                is_terminal=False,
                is_initial=jnp.all(jnp.logical_not(adjacency_matrix)),
                is_pad=False,
            )

        non_initial = jax.lax.cond(unterminate, get_state_terminating, get_state_inter)
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
        """Returns the adjacency matrix, shape `[d, d]`."""
        return state.adjacency_matrix

    def get_backward_action(
        self,
        _state: EnvState,
        forward_action: chex.Array,
        _next_state: EnvState,
        _params: EnvParams,
    ) -> chex.Array:
        """Identity: forward and backward actions share the same encoding."""
        # Forward and backward action spaces share the same indexing: edge
        # indices `i * N + j` for `i, j in [0, N)`, plus the stop/unterminate
        # action at index `N * N`. The two cases (edge add vs. termination)
        # collapse to the identity mapping.
        return forward_action

    def get_forward_action(
        self,
        _state: EnvState,
        backward_action: chex.Array,
        _prev_state: EnvState,
        _params: EnvParams,
    ) -> chex.Array:
        """Identity: forward and backward actions share the same encoding."""
        # See `get_backward_action` — the forward/backward action encodings
        # coincide, so inverting an edge removal or an un-terminate step
        # is the identity.
        return backward_action

    def get_action_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Valid forward-action mask (single state). [N*N+1].

        An edge ``(i, j)`` is valid iff it does not yet exist and adding it
        would not create a cycle. The stop action (last entry) is always valid.
        """
        invalid = jnp.logical_or(state.adjacency_matrix, state.closure_T).reshape(-1)
        return jnp.concatenate([jnp.logical_not(invalid), jnp.ones((1,), dtype=jnp.bool)], axis=0)

    def get_backward_action_mask(self, state: EnvState, _params: EnvParams) -> chex.Array:
        """Valid backward-action mask (single state). [N*N+1].

        Only existing edges can be removed; the stop action (un-terminate) is
        valid only at a terminal state.
        """
        return jax.lax.cond(
            state.is_terminal,
            lambda: jnp.append(
                jnp.zeros((self.num_variables**2,), dtype=jnp.bool),
                jnp.ones((1,), dtype=jnp.bool),
            ),
            lambda: jnp.append(
                state.adjacency_matrix.reshape(-1),
                jnp.zeros((1,), dtype=jnp.bool),
            ),
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return f"DAG-{self.num_variables}-v0"

    @property
    def action_space(self) -> spaces.Discrete:
        """`Discrete(d * d + 1)`: edges `i * d + j`, then the stop action."""
        return spaces.Discrete(self.num_variables * self.num_variables + 1)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """`Discrete(d * d + 1)`: edge removals, then un-stop."""
        return spaces.Discrete(self.num_variables * self.num_variables + 1)

    @property
    def observation_space(self) -> spaces.Box:
        """Boolean adjacency matrix of shape `[d, d]`."""
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.num_variables, self.num_variables),
            dtype=jnp.bool,
        )

    @property
    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
            "adjacency_matrix": spaces.Box(
                low=0,
                high=1,
                shape=(self.num_variables, self.num_variables),
                dtype=jnp.bool,
            ),
            "closure_T": spaces.Box(
                low=0,
                high=1,
                shape=(self.num_variables, self.num_variables),
                dtype=jnp.bool,
            ),
            "is_terminal": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
            "is_initial": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
            "is_pad": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
        })

    @property
    def is_enumerable(self) -> bool:
        """`True` for `d < 6`, where all DAGs can be listed."""
        return self.num_variables < 6

    def state_to_index(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Returns the index of a single DAG in `get_all_states` (enumerable only)."""
        return adj_to_index(state.adjacency_matrix, self.all_adjacencies_flat_bits)

    def get_all_states(self, env_params: EnvParams) -> EnvState:
        """Returns every DAG on `d` nodes as a batched `EnvState`, sorted by index."""
        all_adjacency_matrices = construct_all_dags(self.num_variables)
        sort_idx = jax.vmap(adj_to_index, in_axes=(0, None))(
            all_adjacency_matrices, self.all_adjacencies_flat_bits
        )
        all_adjacency_matrices = all_adjacency_matrices[sort_idx]
        return jax.vmap(
            lambda x: EnvState(
                adjacency_matrix=x,
                closure_T=self._single_compute_closure(x.T),
                is_terminal=jnp.bool_(False),
                is_initial=jnp.bool_(False),
                is_pad=jnp.bool_(False),
            )
        )(all_adjacency_matrices)

    def _get_states_rewards(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> chex.Array:
        """Returns (unnormalized) rewards for every DAG on `num_variables` nodes.

        NOTE: rewards are shifted by `max(log_reward)` before exponentiation
        (softmax trick), since raw log rewards can be ~-100 and would
        otherwise underflow.
        """
        log_reward = self._get_states_log_rewards(env_params, reward_module, reward_params)
        log_reward = log_reward - jnp.max(log_reward)  # softmax trick
        return jnp.exp(log_reward)

    def _get_states_log_rewards(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> chex.Array:
        """Returns unshifted log rewards for every DAG on `num_variables` nodes."""
        all_dags = self.get_all_states(env_params)
        return jax.vmap(reward_module.log_reward, in_axes=(0, None))(all_dags, reward_params)

    def get_true_distribution(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> chex.Array:
        """Returns the exact posterior $R(G)/Z$ over all DAGs, shape `[num_dags]`."""
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return rewards / rewards.sum()

    def get_empirical_distribution(self, states: EnvState, env_params: EnvParams) -> chex.Array:
        """Histogram of a batch of terminal DAGs, shape `[num_dags]`."""
        sample_idx = jax.vmap(adj_to_index, in_axes=(0, None))(
            states.adjacency_matrix, self.all_adjacencies_flat_bits
        )
        valid_mask = states.is_terminal.astype(jnp.float32)
        empirical_dist = jax.ops.segment_sum(
            valid_mask, sample_idx, num_segments=self.all_dags_num
        )
        empirical_dist /= empirical_dist.sum()
        return empirical_dist

    @property
    def is_expected_reward_tractable(self) -> bool:
        """`True` for `d < 6`."""
        return self.num_variables < 6

    def get_expected_reward(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> float:
        r"""Returns $\sum_G R(G)^2 / \sum_G R(G)$, computed in log space."""
        # E_{x ~ R/Z}[R(x)] = sum R^2 / sum R, computed in log space
        log_reward = self._get_states_log_rewards(env_params, reward_module, reward_params)
        return jnp.exp(jax.nn.logsumexp(2 * log_reward) - jax.nn.logsumexp(log_reward))

    @property
    def is_normalizing_constant_tractable(self) -> bool:
        """`True` for `d < 6`."""
        return self.num_variables < 6

    def get_normalizing_constant(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> float:
        r"""Returns $Z = \exp(\log Z)$.

        Raw DAG log-rewards are often far below float32's `exp` range, so prefer
        `get_log_normalizing_constant`.
        """
        return jnp.exp(self.get_log_normalizing_constant(env_params, reward_module, reward_params))

    def get_log_normalizing_constant(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> float:
        r"""Returns $\log Z = \operatorname{logsumexp}_G \log R(G)$ over all DAGs."""
        # Log space: raw DAG log rewards can be far below float32's exp range.
        log_reward = self._get_states_log_rewards(env_params, reward_module, reward_params)
        return jax.nn.logsumexp(log_reward)
