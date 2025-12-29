"""Single-file implementation for Modified Detailed Balance in DAG environment.

Run the script with the following command:
```bash
python baselines/mdb_dag.py
```

Also see https://jax.readthedocs.io/en/latest/gpu_performance_tips.html for
performance tips when running on GPU, i.e., XLA flags.

"""

import functools
import logging
import os
from typing import NamedTuple

import chex
import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import jraph
import optax
from jax_tqdm import loop_tqdm
from omegaconf import OmegaConf
from utils.checkpoint import save_checkpoint
from utils.logger import Writer

import gfnx
from gfnx.metrics import (
    ExactDistributionMetricsModule,
    MultiMetricsModule,
    MultiMetricsState,
    TestCorrelationMetricsModule,
)
from gfnx.utils.dag import get_markov_blanket

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
writer = Writer()


class GNNPolicy(eqx.Module):
    """
    A policy module that uses a Graph Neural Network (GNN) to generate
    forward and backward action logits.
    """

    num_layers: int
    n_bwd_actions: int
    node_embeddings: eqx.nn.Embedding
    edge_embeddings: eqx.nn.Embedding
    global_embeddings: eqx.nn.Embedding
    node_mlp: eqx.nn.MLP
    edge_mlp: eqx.nn.MLP
    global_mlp: eqx.nn.MLP
    nodes_layer_norm: eqx.nn.LayerNorm
    edges_layer_norm: eqx.nn.LayerNorm
    globals_layer_norm: eqx.nn.LayerNorm
    attention_proj: eqx.nn.Linear
    attention: eqx.nn.MultiheadAttention
    post_attention_nodes_norm: eqx.nn.LayerNorm
    post_attention_globals_norm: eqx.nn.LayerNorm
    senders_mlp: eqx.nn.MLP
    receivers_mlp: eqx.nn.MLP
    stop_mlp: eqx.nn.MLP

    def __init__(
        self,
        num_nodes: int,
        num_layers: int,
        mlp_num_layers: int,
        emb_size: int,
        num_heads: int,
        n_bwd_actions: int,
        rng_key: chex.PRNGKey,
    ):
        (
            rng_key,
            node_emb_key,
            edge_emb_key,
            global_emb_key,
            node_mlp_key,
            edge_mlp_key,
            global_mlp_key,
            self_attention_proj_key,
            self_attention_key,
            senders_mlp_key,
            receivers_mlp_key,
            stop_mlp_key,
        ) = jax.random.split(rng_key, 12)

        self.num_layers = num_layers
        self.n_bwd_actions = n_bwd_actions

        self.node_embeddings = eqx.nn.Embedding(
            num_embeddings=num_nodes, embedding_size=emb_size, key=node_emb_key
        )
        truncated_normal = jax.nn.initializers.truncated_normal(stddev=1.0)
        self.edge_embeddings = eqx.nn.Embedding(
            weight=truncated_normal(edge_emb_key, (1, emb_size))
        )
        self.global_embeddings = eqx.nn.Embedding(
            weight=truncated_normal(global_emb_key, (1, emb_size))
        )
        self.node_mlp = eqx.nn.MLP(
            in_size=4 * emb_size,  # concat [Nodes, Senders, Receivers, Globals]
            out_size=emb_size,
            width_size=emb_size,
            depth=mlp_num_layers,
            key=node_mlp_key,
        )
        self.edge_mlp = eqx.nn.MLP(
            in_size=4 * emb_size,  # concat [Edges, Senders, Receivers, Globals]
            out_size=emb_size,
            width_size=emb_size,
            depth=mlp_num_layers,
            key=edge_mlp_key,
        )
        self.global_mlp = eqx.nn.MLP(
            in_size=3 * emb_size,  # concat [Nodes, Edges, Globals]
            out_size=emb_size,
            width_size=emb_size,
            depth=mlp_num_layers,
            key=global_mlp_key,
        )
        self.nodes_layer_norm = eqx.nn.LayerNorm(emb_size)
        self.edges_layer_norm = eqx.nn.LayerNorm(emb_size)
        self.globals_layer_norm = eqx.nn.LayerNorm(emb_size)

        self.attention_proj = eqx.nn.Linear(emb_size, emb_size * 3, key=self_attention_proj_key)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=emb_size,
            key_size=emb_size,
            value_size=emb_size,
            output_size=emb_size,
            key=self_attention_key,
        )
        self.post_attention_nodes_norm = eqx.nn.LayerNorm(emb_size)
        self.post_attention_globals_norm = eqx.nn.LayerNorm(emb_size)

        self.senders_mlp = eqx.nn.MLP(
            in_size=emb_size,
            out_size=emb_size,
            width_size=emb_size,
            depth=mlp_num_layers,
            key=senders_mlp_key,
        )
        self.receivers_mlp = eqx.nn.MLP(
            in_size=emb_size,
            out_size=emb_size,
            width_size=emb_size,
            depth=mlp_num_layers,
            key=receivers_mlp_key,
        )
        self.stop_mlp = eqx.nn.MLP(
            in_size=emb_size,
            out_size=1,
            width_size=emb_size,
            depth=mlp_num_layers,
            key=stop_mlp_key,
        )

    def __call__(self, adjacency: chex.Array) -> chex.Array:
        batch_size, num_variables, num_variables = adjacency.shape  # Batch, Nodes
        # Convert batch of adjacency matrices to a graph tuple
        graph_tuple = self._to_graph_tuple(adjacency)
        features = graph_tuple._replace(
            nodes=jax.vmap(self.node_embeddings, in_axes=(0,))(graph_tuple.nodes),
            edges=jax.vmap(self.edge_embeddings, in_axes=(0,))(graph_tuple.edges),
            globals=jax.vmap(self.global_embeddings, in_axes=(0,))(graph_tuple.globals),
        )

        @jraph.concatenated_args
        def update_node_fn(features):
            return jax.vmap(self.node_mlp, in_axes=(0,))(features)

        @jraph.concatenated_args
        def update_edge_fn(features):
            return jax.vmap(self.edge_mlp, in_axes=(0,))(features)

        @jraph.concatenated_args
        def update_global_fn(features):
            return jax.vmap(self.global_mlp, in_axes=(0,))(features)

        jraph_net = jraph.GraphNetwork(
            update_node_fn=update_node_fn,
            update_edge_fn=update_edge_fn,
            update_global_fn=update_global_fn,
        )

        for _ in range(self.num_layers):
            updates = jraph_net(features)
            features = features._replace(
                nodes=jax.vmap(self.nodes_layer_norm, in_axes=(0,))(
                    features.nodes + updates.nodes
                ),
                edges=jax.vmap(self.edges_layer_norm, in_axes=(0,))(
                    features.edges + updates.edges
                ),
                globals=jax.vmap(self.globals_layer_norm, in_axes=(0,))(
                    features.globals + updates.globals
                ),
            )

        # Self-attention layer
        node_features = jax.vmap(self.attention_proj, in_axes=(0,))(features.nodes)
        node_features = node_features[: batch_size * num_variables].reshape(
            batch_size, num_variables, -1
        )  # Exclude padding and reshape
        q, k, v = jnp.split(node_features, 3, axis=-1)
        node_features = jax.vmap(self.attention, in_axes=(0, 0, 0))(q, k, v)
        node_features = node_features.reshape(batch_size * num_variables, -1)  # [B * N, emb_size]
        node_features = jax.vmap(self.post_attention_nodes_norm, in_axes=(0,))(node_features)
        global_features = jax.vmap(self.post_attention_globals_norm, in_axes=(0,))(
            features.globals[:batch_size]
        )

        senders = jax.vmap(self.senders_mlp, in_axes=(0))(node_features)  # [B*N, emb_size]
        senders = senders.reshape(batch_size, num_variables, -1)  # [B, N, emb_size]
        receivers = jax.vmap(self.receivers_mlp, in_axes=(0))(node_features)  # [B*N, emb_size]
        receivers = receivers.reshape(batch_size, num_variables, -1)  # [B, N, emb_size]

        logits = jax.lax.batch_matmul(senders, receivers.transpose(0, 2, 1))
        logits = logits.reshape(batch_size, -1)

        stop = jax.vmap(self.stop_mlp, in_axes=(0,))(global_features)

        fwd_logits = jnp.concatenate([logits, stop], axis=-1)
        bwd_logits = jnp.zeros((batch_size, self.n_bwd_actions))

        return {
            "forward_logits": fwd_logits,
            "backward_logits": bwd_logits,
        }

    def _to_graph_tuple(self, adjacency: chex.Array) -> jraph.GraphsTuple:
        """Convert adjacency matrix to a graph tuple to be used in jraph.GraphNetwork."""
        # Number of graphs and variables (nodes in each graph)
        num_graphs, num_variables = adjacency.shape[:2]
        # All number of edges rounded up to the nearest power of 2
        total_num_edges = jnp.sum(adjacency)
        # Max number of edges in a batch of DAGs
        size = num_graphs * num_variables * (num_variables - 1) // 2
        # Indices for each edge: (graph_idx, source_idx, target_idx)
        counts, sources, targets = jnp.nonzero(
            adjacency,
            size=size,
            fill_value=0,
        )

        n_node = jnp.full((num_graphs + 1,), num_variables, dtype=jnp.int32)
        n_node = n_node.at[-1].set(1)  # Padding

        n_edge = jnp.sum(adjacency, axis=(1, 2))
        n_edge = jnp.append(n_edge, size - total_num_edges)  # Padding

        nodes = jnp.tile(jnp.arange(num_variables, dtype=jnp.int32), num_graphs)
        nodes = jnp.append(nodes, 0)

        edges = jnp.zeros((size,), dtype=jnp.int32)

        valid_mask = jnp.arange(size) < total_num_edges
        valid_senders = sources + counts * num_variables
        senders = jnp.where(valid_mask, valid_senders, num_graphs * num_variables)
        valid_receivers = targets + counts * num_variables
        receivers = jnp.where(valid_mask, valid_receivers, num_graphs * num_variables)

        globals = jnp.zeros((num_graphs + 1,), dtype=jnp.int32)

        return jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=globals,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
        )


# Define the train state that will be used in the training loop
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.DAGEnvironment
    env_params: chex.Array
    exploration_schedule: optax.Schedule
    model: GNNPolicy
    target_model: GNNPolicy
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    metrics_module: MultiMetricsModule
    metrics_state: MultiMetricsState


@eqx.filter_jit
def train_step(idx: int, train_state: TrainState) -> TrainState:
    rng_key = train_state.rng_key
    num_envs = train_state.config.num_envs
    env = train_state.env
    env_params = train_state.env_params
    # Step 1. Generate a batch of trajectories and split to transitions
    rng_key, sample_traj_key = jax.random.split(train_state.rng_key)
    # Split the model to pass into forward rollout
    policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)

    cur_eps = train_state.exploration_schedule(idx)

    # Define the policy function suitable for gfnx.utils.forward_rollout
    def fwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = policy(env_obs)
        logits = policy_outputs["forward_logits"]

        # Apply the exploration schedule
        rng_key, exploration_key = jax.random.split(rng_key)
        batch_size = logits.shape[0]
        exploration_mask = jax.random.bernoulli(exploration_key, cur_eps, (batch_size,))
        logits = jnp.where(exploration_mask[..., None], 0, logits)
        policy_outputs = eqx.tree_at(lambda x: x["forward_logits"], policy_outputs, logits)
        return logits, policy_outputs

    # Generating the trajectory and splitting it into transitions
    traj_data, log_info = gfnx.utils.forward_rollout(
        rng_key=sample_traj_key,
        num_envs=num_envs,
        policy_fn=fwd_policy_fn,
        policy_params=policy_params,
        env=train_state.env,
        env_params=train_state.env_params,
    )
    transitions = gfnx.utils.split_traj_to_transitions(traj_data)
    bwd_actions = train_state.env.get_backward_action(
        transitions.state,
        transitions.action,
        transitions.next_state,
        train_state.env_params,
    )
    # Compute the RL reward / ELBO (for logging purposes)
    _, log_pb_traj = gfnx.utils.forward_trajectory_log_probs(env, traj_data, env_params)
    rl_reward = log_pb_traj + log_info["log_gfn_reward"] + log_info["entropy"]

    # Step 2. Compute the loss
    def loss_fn(model: GNNPolicy) -> chex.Array:
        # Call the network to get the logits
        policy_outputs = model(transitions.obs)
        # Compute the forward log-probs
        fwd_logits = policy_outputs["forward_logits"]
        invalid_mask = env.get_invalid_mask(transitions.state, env_params)
        masked_fwd_logits = gfnx.utils.mask_logits(fwd_logits, invalid_mask)
        fwd_all_log_probs = jax.nn.log_softmax(masked_fwd_logits, axis=-1)
        sink_logprobs = fwd_all_log_probs[:, -1]
        fwd_logprobs = jnp.take_along_axis(
            fwd_all_log_probs,
            jnp.expand_dims(transitions.action, axis=-1),
            axis=-1,
        ).squeeze(-1)

        # Compute the stats for the next state
        next_policy_outputs = train_state.target_model(transitions.next_obs)
        next_fwd_logits = next_policy_outputs["forward_logits"]
        next_fwd_invalid_mask = env.get_invalid_mask(transitions.next_state, env_params)
        masked_next_fwd_logits = gfnx.utils.mask_logits(next_fwd_logits, next_fwd_invalid_mask)
        next_fwd_all_log_probs = jax.nn.log_softmax(masked_next_fwd_logits, axis=-1)
        next_sink_logprobs = next_fwd_all_log_probs[:, -1]

        bwd_logits = next_policy_outputs["backward_logits"]
        next_bwd_invalid_mask = env.get_invalid_backward_mask(transitions.next_state, env_params)
        masked_bwd_logits = gfnx.utils.mask_logits(bwd_logits, next_bwd_invalid_mask)
        bwd_all_log_probs = jax.nn.log_softmax(masked_bwd_logits, axis=-1)
        bwd_logprobs = jnp.take_along_axis(
            bwd_all_log_probs, jnp.expand_dims(bwd_actions, axis=-1), axis=-1
        ).squeeze(-1)
        delta_score = env.reward_module.delta_score(
            transitions.state,
            transitions.action,
            transitions.next_state,
            env_params,
        )

        # Compute the MDB loss with masking
        done_or_pad = transitions.done | transitions.pad
        error = next_sink_logprobs + fwd_logprobs - sink_logprobs - bwd_logprobs - delta_score
        error = jnp.where(done_or_pad, 0.0, error)
        loss = optax.huber_loss(error).sum() / jnp.logical_not(done_or_pad).sum()

        return loss

    mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(train_state.model)
    # Step 3. Update the model with grads
    updates, opt_state = train_state.optimizer.update(
        grads,
        train_state.opt_state,
        eqx.filter(train_state.model, eqx.is_array),
    )
    model = eqx.apply_updates(train_state.model, updates)

    # Update the target model
    model_params, _ = eqx.partition(model, eqx.is_array)
    target_params, target_static = eqx.partition(train_state.target_model, eqx.is_array)
    new_target_params = jax.lax.cond(
        idx % train_state.config.agent.target_update_every == 0,
        lambda: jax.tree.map(lambda x: x, model_params),
        lambda: target_params,
    )
    new_target_model = eqx.combine(new_target_params, target_static)

    metrics_state = train_state.metrics_state
    metrics_module = train_state.metrics_module

    rng_key, eval_rng_key = jax.random.split(rng_key)
    # Perform evaluation computations if needed
    is_eval_step = idx % train_state.config.logging.track_each == 0
    is_eval_step = is_eval_step | (idx + 1 == train_state.config.num_train_steps)

    metrics_state = jax.lax.cond(
        is_eval_step,
        lambda kwargs: metrics_module.process(**kwargs),
        lambda kwargs: kwargs["metrics_state"],  # Do nothing if not eval step
        {
            "metrics_state": metrics_state,
            "rng_key": eval_rng_key,
            "args": metrics_module.ProcessArgs(
                metrics_args={
                    "exact_distribution": ExactDistributionMetricsModule.ProcessArgs(
                        policy_params=model_params,
                        env_params=train_state.env_params,
                    ),
                    "reward_corr": TestCorrelationMetricsModule.ProcessArgs(
                        policy_params=model_params,
                        env_params=train_state.env_params,
                    ),
                    "edge_corr": TestCorrelationMetricsModule.ProcessArgs(
                        policy_params=model_params,
                        env_params=train_state.env_params,
                    ),
                    "path_corr": TestCorrelationMetricsModule.ProcessArgs(
                        policy_params=model_params,
                        env_params=train_state.env_params,
                    ),
                    "markov_blanket_corr": TestCorrelationMetricsModule.ProcessArgs(
                        policy_params=model_params,
                        env_params=train_state.env_params,
                    ),
                }
            ),
        },
    )

    # Perform the logging via JAX debug callback
    def logging_callback(idx: int, train_info: dict, metrics_state: MultiMetricsState, cfg):
        train_info = {f"train/{key}": float(value) for key, value in train_info.items()}

        if idx % cfg.logging.eval_each == 0 or idx + 1 == cfg.num_train_steps:
            log.info(f"Step {idx}")
            log.info(train_info)
            eval_info = metrics_module.get(metrics_state)
            eval_info = {f"eval/{key}": float(value) for key, value in eval_info.items()}
            log.info(eval_info)
            if cfg.logging.use_writer:
                writer.log(eval_info, commit=False)

        if cfg.logging.use_writer and idx % cfg.logging.track_each == 0:
            writer.log(train_info)

    jax.debug.callback(
        logging_callback,
        idx,
        {
            "mean_loss": mean_loss,
            "entropy": log_info["entropy"].mean(),
            "grad_norm": optax.tree_utils.tree_l2_norm(grads),
            "mean_reward": jnp.exp(log_info["log_gfn_reward"]).mean(),
            "mean_log_reward": log_info["log_gfn_reward"].mean(),
            "rl_reward": rl_reward.mean(),
        },
        metrics_state,
        train_state.config,
        ordered=True,
    )

    # Return the updated train state
    return train_state._replace(
        rng_key=rng_key,
        model=model,
        target_model=new_target_model,
        opt_state=opt_state,
        metrics_state=metrics_state,
    )


@hydra.main(config_path="configs/", config_name="mdb_dag", version_base=None)
def run_experiment(cfg: OmegaConf) -> None:
    # Log the configuration
    log.info(OmegaConf.to_yaml(cfg))

    rng_key = jax.random.PRNGKey(cfg.seed)
    # This key is needed to initialize the environment
    env_init_key = jax.random.PRNGKey(cfg.env_seed)
    # This key is needed to initialize the evaluation process
    # i.e., generate random test set.
    eval_init_key = jax.random.PRNGKey(cfg.eval_seed)

    # Load the samples
    train_samples = gfnx.utils.dag.generate_dataset(
        data_seed=cfg.env_seed,
        num_variables=cfg.environment.num_variables,
        num_edges_per_node=cfg.environment.num_edges_per_node,
        num_train_samples=cfg.environment.num_train_samples,
        loc_edges=cfg.environment.loc_edges,
        scale_edges=cfg.environment.scale_edges,
        obs_scale=cfg.environment.obs_scale,
        folder=cfg.logging.log_dir,
    )
    # Define the reward function for the environment
    if cfg.environment.prior.type == "uniform":
        prior = gfnx.reward.UniformDAGPrior(
            num_variables=cfg.environment.num_variables,
        )
    else:
        raise ValueError(f"Unknown prior type: {cfg.environment.prior.type}")

    if cfg.environment.likelihood.type == "linear_gaussian_score":
        likelihood = gfnx.reward.LinearGaussianScore(
            data=train_samples,
            prior_mean=cfg.environment.likelihood.prior_mean,
            prior_scale=cfg.environment.likelihood.prior_scale,
            obs_scale=cfg.environment.likelihood.obs_scale,
        )
    elif cfg.environment.likelihood.type == "bge_score":
        likelihood = gfnx.reward.BGeScore(
            data=train_samples,
            mean_obs=cfg.environment.likelihood.mean_obs,
            alpha_mu=cfg.environment.likelihood.alpha_mu,
            alpha_w=cfg.environment.likelihood.alpha_w,
        )
    else:
        raise ValueError(f"Unknown likelihood type: {cfg.environment.likelihood.type}")
    reward_module = gfnx.DAGRewardModule(prior=prior, likelihood=likelihood)

    # Initialize the environment and its inner parameters
    env = gfnx.environment.DAGEnvironment(
        reward_module,
        num_variables=cfg.environment.num_variables,
    )
    env_params = env.init(env_init_key)

    rng_key, net_init_key = jax.random.split(rng_key)
    # Initialize the network
    model = GNNPolicy(
        num_nodes=cfg.environment.num_variables,
        num_layers=cfg.network.num_layers,
        mlp_num_layers=cfg.network.mlp_num_layers,
        emb_size=cfg.network.embedding_size,
        num_heads=cfg.network.num_heads,
        n_bwd_actions=env.backward_action_space.n,
        rng_key=net_init_key,
    )
    target_model = jax.tree.map(lambda x: x, model)  # Copy of the model
    exploration_schedule = optax.linear_schedule(
        init_value=cfg.agent.start_eps,
        end_value=cfg.agent.end_eps,
        transition_steps=cfg.agent.exploration_steps,
    )

    # Initialize the optimizer
    optimizer = optax.adam(learning_rate=cfg.agent.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    # Initialize the backward policy function for correlation computation
    policy_static = eqx.filter(model, eqx.is_array, inverse=True)

    def fwd_policy_fn(
        rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params
    ) -> tuple[chex.Array, dict[str, chex.Array]]:
        del rng_key
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = policy(env_obs)
        return policy_outputs["forward_logits"], policy_outputs

    def bwd_policy_fn(
        rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params
    ) -> tuple[chex.Array, dict[str, chex.Array]]:
        del rng_key
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = policy(env_obs)
        return policy_outputs["backward_logits"], policy_outputs

    def edge_score_transform_fn(env_state: gfnx.DAGEnvState, log_score: chex.Array) -> chex.Array:
        adjacency_matrix = env_state.adjacency_matrix
        eye = jnp.eye(env_state.adjacency_matrix.shape[1], dtype=jnp.bool)[None, :, :]
        log_score = jnp.expand_dims(log_score, axis=(-1, -2))
        # logP(E) = logsumexp_G[logP(E | G) + logP(G)]
        log_score = jnp.where(adjacency_matrix | eye, log_score, -jnp.inf)
        edge_log_score = jax.nn.logsumexp(log_score, axis=0)
        return edge_log_score.reshape(-1)

    def path_score_transform_fn(env_state: gfnx.DAGEnvState, log_score: chex.Array) -> chex.Array:
        closure = jnp.transpose(env_state.closure_T, (0, 2, 1))
        eye = jnp.eye(env_state.closure_T.shape[1], dtype=jnp.bool)[None, :, :]
        log_score = jnp.expand_dims(log_score, axis=(-1, -2))
        # logP(path(i, j)) = logsumexp_G[logP(path(i, j) | G) + logP(G)]
        log_score = jnp.where(closure | eye, log_score, -jnp.inf)
        path_log_score = jax.nn.logsumexp(log_score, axis=0)
        return path_log_score.reshape(-1)

    def markov_blanket_transform_fn(
        env_state: gfnx.DAGEnvState, log_score: chex.Array
    ) -> chex.Array:
        markov_blanket = jax.vmap(lambda x: get_markov_blanket(x), in_axes=0)(
            env_state.adjacency_matrix
        )  # (num_graphs, num_variables, num_variables)
        eye = jnp.eye(env_state.adjacency_matrix.shape[1], dtype=jnp.bool)[None, :, :]
        log_score = jnp.expand_dims(log_score, axis=(-1, -2))
        # logP(mb(i, j)) = logsumexp_G[logP(mb(i, j) | G) + logP(G)]
        log_score = jnp.where(markov_blanket | eye, log_score, -jnp.inf)
        markov_blanket_log_score = jax.nn.logsumexp(log_score, axis=0)
        return markov_blanket_log_score.reshape(-1)

    metrics_module = MultiMetricsModule({
        "exact_distribution": ExactDistributionMetricsModule(
            metrics=["tv", "kl", "jsd"],
            env=env,
            fwd_policy_fn=fwd_policy_fn,
            batch_size=cfg.metrics.batch_size,
        ),
        "reward_corr": TestCorrelationMetricsModule(
            env=env,
            bwd_policy_fn=bwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds,
            batch_size=cfg.metrics.batch_size,
        ),
        "edge_corr": TestCorrelationMetricsModule(
            env=env,
            bwd_policy_fn=bwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds,
            batch_size=cfg.metrics.batch_size,
            transform_fn=edge_score_transform_fn,
        ),
        "path_corr": TestCorrelationMetricsModule(
            env=env,
            bwd_policy_fn=bwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds,
            batch_size=cfg.metrics.batch_size,
            transform_fn=path_score_transform_fn,
        ),
        "markov_blanket_corr": TestCorrelationMetricsModule(
            env=env,
            bwd_policy_fn=bwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds,
            batch_size=cfg.metrics.batch_size,
            transform_fn=markov_blanket_transform_fn,
        ),
    })

    # Construct the test set
    test_set_adjacency_matrix = gfnx.utils.dag.construct_all_dags(cfg.environment.num_variables)
    test_set_closure_T = jax.vmap(lambda x: env._single_compute_closure(x.T), in_axes=0)(
        test_set_adjacency_matrix
    )
    num_graphs = test_set_adjacency_matrix.shape[0]
    test_set_states = gfnx.DAGEnvState(
        adjacency_matrix=test_set_adjacency_matrix,
        closure_T=test_set_closure_T,
        is_terminal=jnp.ones(num_graphs, dtype=jnp.bool),
        is_initial=jnp.zeros(num_graphs, dtype=jnp.bool),
        is_pad=jnp.zeros(num_graphs, dtype=jnp.bool),
    )

    metrics_state = metrics_module.init(
        eval_init_key,
        metrics_module.InitArgs(
            metrics_args={
                "exact_distribution": ExactDistributionMetricsModule.InitArgs(
                    env_params=env_params
                ),
                "reward_corr": TestCorrelationMetricsModule.InitArgs(
                    env_params=env_params,
                    test_set=test_set_states,
                ),
                "edge_corr": TestCorrelationMetricsModule.InitArgs(
                    env_params=env_params,
                    test_set=test_set_states,
                ),
                "path_corr": TestCorrelationMetricsModule.InitArgs(
                    env_params=env_params,
                    test_set=test_set_states,
                ),
                "markov_blanket_corr": TestCorrelationMetricsModule.InitArgs(
                    env_params=env_params,
                    test_set=test_set_states,
                ),
            }
        ),
    )

    train_state = TrainState(
        rng_key=rng_key,
        config=cfg,
        env=env,
        env_params=env_params,
        exploration_schedule=exploration_schedule,
        model=model,
        target_model=target_model,
        optimizer=optimizer,
        opt_state=opt_state,
        metrics_module=metrics_module,
        metrics_state=metrics_state,
    )
    # Split train state into parameters and static parts to make jit work.
    train_state_params, train_state_static = eqx.partition(train_state, eqx.is_array)

    @functools.partial(jax.jit, donate_argnums=(1,))
    @loop_tqdm(cfg.num_train_steps, print_rate=cfg.logging["tqdm_print_rate"])
    def train_step_wrapper(idx: int, train_state_params):
        # Wrapper to use a usual jit in jax, since it is required by fori_loop.
        train_state = eqx.combine(train_state_params, train_state_static)
        train_state = train_step(idx, train_state)
        train_state_params, _ = eqx.partition(train_state, eqx.is_array)
        return train_state_params

    if cfg.logging.use_writer:
        log.info("Initialize writer")
        log_dir = (
            cfg.logging.log_dir
            if cfg.logging.log_dir
            else os.path.join(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, f"run_{os.getpid()}/"
            )
        )
        writer.init(
            writer_type=cfg.writer.writer_type,
            save_locally=cfg.writer.save_locally,
            log_dir=log_dir,
            entity=cfg.writer.entity,
            project=cfg.writer.project,
            tags=["MDB", env.name.upper()],
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    log.info("Start training")
    # Run the training loop via jax.lax.fori_loop
    train_state_params = jax.lax.fori_loop(
        lower=0,
        upper=cfg.num_train_steps,
        body_fun=train_step_wrapper,
        init_val=train_state_params,
    )
    jax.block_until_ready(train_state_params)

    # Save the final model
    train_state = eqx.combine(train_state_params, train_state_static)
    dir = (
        cfg.logging.checkpoint_dir
        if cfg.logging.checkpoint_dir
        else os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            f"checkpoints_{os.getpid()}/",
        )
    )
    save_checkpoint(os.path.join(dir, "train_state"), train_state)


if __name__ == "__main__":
    run_experiment()
