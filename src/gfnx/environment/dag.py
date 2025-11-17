from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from gfnx.base import (
    BaseEnvParams,
    BaseEnvState,
    BaseVecEnvironment,
    TAction,
    TDone,
    TRewardModule,
)

from .. import spaces


@chex.dataclass(frozen=True)
class EnvState(BaseEnvState):
    adjacency_matrix: Bool[Array, " batch_size num_variables num_variables"]
    closure_T: Bool[Array, " batch_size num_variables num_variables"]
    time: Int[Array, " batch_size"]
    is_terminal: Bool[Array, " batch_size"]
    is_initial: Bool[Array, " batch_size"]
    is_pad: Bool[Array, " batch_size"]


@chex.dataclass(frozen=True)
class EnvParams(BaseEnvParams):
    num_variables: int = 4
    reward_params: Any = None


class DAGEnvironment(BaseVecEnvironment[EnvState, EnvParams]):
    def __init__(
        self,
        reward_module: TRewardModule,
        num_variables: int,
    ) -> None:
        super().__init__(reward_module)
        self.num_variables = num_variables
        self.stop_action = self.num_variables * self.num_variables

    def get_init_state(self, num_envs: int) -> EnvState:
        return EnvState(
            adjacency_matrix=jnp.zeros(
                (num_envs, self.num_variables, self.num_variables),
                dtype=jnp.bool,
            ),
            closure_T=jnp.tile(
                jnp.eye(self.num_variables, dtype=jnp.bool),
                (num_envs, 1, 1),
            ),
            time=jnp.zeros((num_envs,), dtype=jnp.int32),
            is_terminal=jnp.zeros((num_envs,), dtype=jnp.bool),
            is_initial=jnp.ones((num_envs,), dtype=jnp.bool),
            is_pad=jnp.zeros((num_envs,), dtype=jnp.bool),
        )

    def init(self, rng_key: chex.PRNGKey) -> EnvParams:
        dummy_state = self.get_init_state(1)
        reward_params = self.reward_module.init(rng_key, dummy_state)
        return EnvParams(
            num_variables=self.num_variables,
            reward_params=reward_params,
        )

    @property
    def max_steps_in_episode(self) -> int:
        return (self.num_variables * (self.num_variables - 1)) // 2 + 1

    def _single_transition(
        self,
        state: EnvState,
        action: TAction,
        env_params: EnvParams,
    ) -> Tuple[EnvState, TDone, Dict[Any, Any]]:
        is_terminal = state.is_terminal
        time = state.time

        def get_state_terminal() -> EnvState:
            return state.replace(is_pad=True)

        def get_state_nonterminal() -> EnvState:
            done = action == self.stop_action
            source, target = jnp.divmod(action, self.num_variables)
            return jax.lax.cond(
                done, get_state_finished, get_state_inter, source, target
            )

        def get_state_finished(
            source: chex.Array, target: chex.Array
        ) -> EnvState:
            return state.replace(
                time=time + 1, is_terminal=True, is_initial=False
            )

        def get_state_inter(
            source: chex.Array, target: chex.Array
        ) -> EnvState:
            adjacency_matrix = state.adjacency_matrix.at[source, target].set(
                True
            )
            closure_T = state.closure_T
            outer_product = jnp.logical_and(
                jnp.expand_dims(closure_T[source], 0),
                jnp.expand_dims(closure_T[:, target], 1),
            )
            closure_T = jnp.logical_or(closure_T, outer_product)
            return state.replace(
                adjacency_matrix=adjacency_matrix,
                closure_T=closure_T,
                time=time + 1,
                is_terminal=False,
                is_initial=False,
            )

        next_state = jax.lax.cond(
            is_terminal, get_state_terminal, get_state_nonterminal
        )

        return next_state, next_state.is_terminal, {}

    def _single_source_bfs(
        self, adjacency_t: chex.Array, start: int
    ) -> chex.Array:
        """
        Returns a boolean 1D array 'visited' of shape (d,),
        indicating which nodes are reachable from 'start' in adjacency_t.

        adjacency_t[i, j] = True means there's an edge i->j in G^T.
        """
        d = adjacency_t.shape[0]
        visited_init = jnp.zeros(d, dtype=bool).at[start].set(True)
        frontier_init = visited_init

        def cond_fun(carry):
            frontier, visited = carry
            return jnp.any(
                frontier
            )  # continue while we have newly discovered nodes

        def body_fun(carry):
            frontier, visited = carry
            # adjacency_t & frontier[:, None] marks edges from any node in `frontier`.
            # Taking "any(..., axis=0)" merges them into which nodes we can discover next
            neighbors = jnp.any(adjacency_t & frontier[:, None], axis=0)
            new_frontier = neighbors & jnp.logical_not(visited)
            new_visited = visited | new_frontier
            return (new_frontier, new_visited)

        _, visited_final = jax.lax.while_loop(
            cond_fun, body_fun, (frontier_init, visited_init)
        )
        return visited_final

    def _single_compute_closure_t(self, adjacency_t: chex.Array) -> chex.Array:
        """
        Given the adjacency matrix of G^T (shape (d, d)),
        compute its transitive closure via BFS-from-each-node.
        closure_t[i, j] = True if i can reach j in the transpose graph (G^T).
        """
        d = adjacency_t.shape[0]
        closure = jax.vmap(lambda i: self._single_source_bfs(adjacency_t, i))(
            jnp.arange(d)
        )
        # Force the diagonal True (i can reach i by convention)
        closure = jnp.logical_or(closure, jnp.eye(d, dtype=jnp.bool))
        return closure

    def _single_backward_transition(
        self,
        state: EnvState,
        backward_action: chex.Array,
        env_params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[Any, Any]]:
        """Backward transition for DAG environment.
        Removing an edge is equivalent to adding a 'phantom' edge.
        """
        is_initial = state.is_initial
        time = state.time

        def get_state_initial() -> EnvState:
            return state.replace(is_pad=True)

        def get_state_non_initial() -> EnvState:
            unterminate = backward_action == self.stop_action
            return jax.lax.cond(
                unterminate, get_state_terminating, get_state_inter
            )

        def get_state_terminating() -> EnvState:
            return state.replace(
                time=time - 1,
                is_terminal=False,
                is_initial=jnp.all(jnp.logical_not(state.adjacency_matrix)),
                is_pad=False,
            )

        def get_state_inter() -> EnvState:
            source, target = jnp.divmod(backward_action, self.num_variables)
            adjacency_matrix = state.adjacency_matrix.at[source, target].set(
                False
            )
            adjacency_t = adjacency_matrix.T
            closure_T = self._single_compute_closure_t(adjacency_t)
            return state.replace(
                adjacency_matrix=adjacency_matrix,
                closure_T=closure_T,
                time=time - 1,
                is_terminal=False,
                is_initial=jnp.all(jnp.logical_not(adjacency_matrix)),
                is_pad=False,
            )

        prev_state = jax.lax.cond(
            is_initial, get_state_initial, get_state_non_initial
        )
        return prev_state, prev_state.is_initial, {}

    def get_obs(self, state: EnvState, env_params: EnvParams) -> chex.Array:
        return state.adjacency_matrix

    def get_backward_action(
        self,
        state: EnvState,
        forward_action: chex.Array,
        next_state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        return forward_action

    def get_forward_action(
        self,
        state: EnvState,
        backward_action: chex.Array,
        prev_state: EnvState,
        params: EnvParams,
    ) -> chex.Array:
        return backward_action

    def get_invalid_mask(
        self, state: EnvState, env_params: EnvParams
    ) -> chex.Array:
        """Invalid mask for forward actions.
        Constructed as a logical or of adjacency matrix and
        transitive closure of transposed adjacency matrix.
        """
        num_envs = state.time.shape[0]
        mask = jnp.logical_or(state.adjacency_matrix, state.closure_T).reshape(
            num_envs, -1
        )
        mask = jnp.concatenate(
            [mask, jnp.zeros((num_envs, 1), dtype=jnp.bool)], axis=1
        )  # stop action == last action is always valid
        return mask

    def get_invalid_backward_mask(
        self, state: EnvState, params: EnvParams
    ) -> chex.Array:
        """Invalid mask for backward actions.
        Invert adjacency matrix and allow stop action only for the terminal state.
        """

        def _single_get_invalid_backward_mask(state: EnvState) -> chex.Array:
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

        return jax.vmap(_single_get_invalid_backward_mask)(state)

    @property
    def name(self) -> str:
        return f"DAG-{self.num_variables}-v0"

    @property
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_variables * self.num_variables + 1)

    @property
    def backward_action_space(self) -> spaces.Discrete:
        """Backward action space of the environment."""
        return spaces.Discrete(self.num_variables * self.num_variables + 1)

    @property
    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
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
            "time": spaces.Discrete(self.max_steps_in_episode),
            "is_terminal": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
            "is_initial": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
            "is_pad": spaces.Box(low=0, high=1, shape=(), dtype=jnp.bool),
        })
