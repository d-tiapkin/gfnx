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
    adjacency_matrix: Bool[Array, " num_variables num_variables"]
    closure_T: Bool[Array, " num_variables num_variables"]
    is_terminal: Bool[Array, ""]
    is_initial: Bool[Array, ""]
    is_pad: Bool[Array, ""]


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    num_variables: int = 4


class DAGEnvironment(BaseEnvironment[EnvState, EnvParams]):
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
        return EnvParams(num_variables=self.num_variables)

    @property
    def max_steps_in_episode(self) -> int:
        return (self.num_variables * (self.num_variables - 1)) // 2 + 1

    def step(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> tuple[chex.Array, EnvState, TDone, dict[Any, Any]]:
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
        visited_init = jnp.zeros(d, dtype=bool).at[start].set(True)
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
        """Backward transition for the DAG environment.

        Removing an edge is equivalent to adding a 'phantom' edge in reverse.
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
        return state.adjacency_matrix

    def get_backward_action(
        self,
        _state: EnvState,
        forward_action: chex.Array,
        _next_state: EnvState,
        _params: EnvParams,
    ) -> chex.Array:
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
        # See `get_backward_action` — the forward/backward action encodings
        # coincide, so inverting an edge removal or an un-terminate step
        # is the identity.
        return backward_action

    def get_invalid_mask(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        """Invalid mask for forward actions (single state). [N*N+1].

        Constructed as a logical OR of the adjacency matrix and the transitive
        closure of the transposed adjacency matrix — an edge `(i, j)` is
        invalid if it already exists or if adding it would create a cycle.
        The stop action (last entry) is always valid.
        """
        mask = jnp.logical_or(state.adjacency_matrix, state.closure_T).reshape(-1)
        return jnp.concatenate(
            [mask, jnp.zeros((1,), dtype=jnp.bool)], axis=0
        )  # stop action == last action is always valid

    def get_invalid_backward_mask(self, state: EnvState, _params: EnvParams) -> chex.Array:
        """Invalid mask for backward actions (single state). [N*N+1].

        Inverts the adjacency matrix (we may only un-set existing edges) and
        allows the stop action only for a terminal state (un-terminate).
        """
        return jax.lax.cond(
            state.is_terminal,
            lambda: jnp.append(
                jnp.ones((self.num_variables**2,), dtype=jnp.bool),
                jnp.zeros((1,), dtype=jnp.bool),
            ),
            lambda: jnp.append(
                jnp.logical_not(state.adjacency_matrix).reshape(-1),
                jnp.ones((1,), dtype=jnp.bool),
            ),
        )

    @property
    def name(self) -> str:
        return f"DAG-{self.num_variables}-v0"

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.num_variables * self.num_variables + 1)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(self.num_variables * self.num_variables + 1)

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.num_variables, self.num_variables),
            dtype=jnp.bool,
        )

    @property
    def state_space(self) -> spaces.Dict:
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
        return self.num_variables < 6

    def state_to_index(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        return adj_to_index(state.adjacency_matrix, self.all_adjacencies_flat_bits)

    def get_all_states(self, env_params: EnvParams) -> EnvState:
        """Returns all DAG states (batched — this method is inherently batch-returning)."""
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
        all_dags = self.get_all_states(env_params)
        log_reward = jax.vmap(reward_module.log_reward, in_axes=(0, None))(all_dags, reward_params)
        log_reward = log_reward - jnp.max(log_reward)  # softmax trick
        return jnp.exp(log_reward)

    def get_true_distribution(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> chex.Array:
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return rewards / rewards.sum()

    def get_empirical_distribution(self, states: EnvState, env_params: EnvParams) -> chex.Array:
        """Extracts empirical distribution from a batch of states."""
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
        return self.num_variables < 6

    def get_expected_reward(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> float:
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return jnp.pow(rewards, 2).sum() / rewards.sum()

    @property
    def is_normalizing_constant_tractable(self) -> bool:
        return self.num_variables < 6

    def get_normalizing_constant(
        self, env_params: EnvParams, reward_module: BaseRewardModule, reward_params: TRewardParams
    ) -> float:
        rewards = self._get_states_rewards(env_params, reward_module, reward_params)
        return rewards.sum()
