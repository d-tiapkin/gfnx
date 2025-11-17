"""Single-file implementation for Detailed Balance in phylogenetic tree environment.

Run the script with the following command:
```bash
python baselines/db_phylo.py
```

Also see https://jax.readthedocs.io/en/latest/gpu_performance_tips.html for
performance tips when running on GPU, i.e., XLA flags.
"""

import functools
import logging
import os
from functools import partial
from typing import Literal, NamedTuple, Optional

import chex
import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from jax_tqdm import loop_tqdm
from omegaconf import OmegaConf

import gfnx
from gfnx.environment.phylogenetic_tree import PhyloTreeEnvironment
from gfnx.reward.phylogenetic_tree import PhyloTreeRewardModule
from gfnx.utils import (
    ExplorationState,
    apply_epsilon_greedy_vmap,
    create_exploration_schedule,
    get_phylo_initialization_args,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class TransformerPolicy(eqx.Module):
    """
    A policy module that uses a Transformer to generate forward and backward action logits
    as well as a flow. The transformer processes sequences of DNA/RNA data and outputs
    actions for building phylogenetic trees.

    Args:
        n_fwd_actions (int): Number of forward actions.
        n_bwd_actions (int): Number of backward actions.
        train_backward_policy (bool): Whether to train the backward policy.
        encoder_params (dict): Parameters for the encoder.
        key (chex.PRNGKey): Random key for initialization.
    """

    pos_embeddings: chex.Array = eqx.field(static=True)
    embedder_block: eqx.nn.MLP
    emb2hidden: eqx.nn.Linear
    layers: list[gfnx.networks.TransformerLayer]
    forward_mlp: eqx.nn.MLP
    flow_mlp: eqx.nn.MLP
    backward_mlp: Optional[eqx.nn.MLP] = None
    train_backward_policy: bool
    n_fwd_actions: int
    n_bwd_actions: int
    row: chex.Array = eqx.field(static=True)
    col: chex.Array = eqx.field(static=True)

    def __init__(
        self,
        num_nodes: int,
        n_fwd_actions: int,
        n_bwd_actions: int,
        train_backward_policy: bool,
        emb_params: dict,
        transformer_params: dict,
        mlp_params: dict,
        *,
        key: chex.PRNGKey,
    ):
        (
            emb_key,
            pos_emb_key,
            transformer_key,
            forward_mlp_key,
            flow_mlp_key,
            backward_mlp_key,
        ) = jax.random.split(key, num=6)
        initializer = jax.nn.initializers.truncated_normal(stddev=0.02)
        self.pos_embeddings = initializer(
            pos_emb_key, (num_nodes, emb_params["embedding_size"])
        )
        self.train_backward_policy = train_backward_policy
        self.n_fwd_actions = n_fwd_actions
        self.n_bwd_actions = n_bwd_actions

        self.embedder_block = eqx.nn.MLP(
            in_size=emb_params["in_size"],
            out_size=emb_params["embedding_size"],
            width_size=emb_params["embedding_width"],
            depth=emb_params["embedding_depth"],
            activation=jax.nn.leaky_relu,
            key=emb_key,
        )
        self.emb2hidden = eqx.nn.Linear(
            in_features=emb_params["embedding_size"],
            out_features=transformer_params["hidden_size"],
            key=emb_key,
        )

        layer_keys = jax.random.split(
            transformer_key, num=transformer_params["num_layers"]
        )
        self.layers = []
        for layer_key in layer_keys:
            self.layers.append(
                gfnx.networks.TransformerLayer(
                    hidden_size=transformer_params["hidden_size"],
                    intermediate_size=transformer_params["intermediate_size"],
                    num_heads=transformer_params["num_heads"],
                    dropout_rate=transformer_params["dropout_rate"],
                    attention_dropout_rate=transformer_params[
                        "attention_dropout_rate"
                    ],
                    key=layer_key,
                )
            )

        self.row, self.col = jnp.triu_indices(num_nodes, k=1)
        self.forward_mlp = eqx.nn.MLP(
            in_size=transformer_params["hidden_size"],
            out_size=1,
            width_size=mlp_params["width_size"],
            depth=mlp_params["depth"],
            activation=jax.nn.leaky_relu,
            key=forward_mlp_key,
        )
        self.flow_mlp = eqx.nn.MLP(
            in_size=num_nodes * transformer_params["hidden_size"],
            out_size=1,
            width_size=mlp_params["width_size"],
            depth=mlp_params["depth"],
            activation=jax.nn.leaky_relu,
            key=flow_mlp_key,
        )
        if train_backward_policy:
            self.backward_mlp = eqx.nn.MLP(
                in_size=transformer_params["hidden_size"],
                out_size=1,
                width_size=mlp_params["width_size"],
                depth=mlp_params["depth"],
                activation=jax.nn.leaky_relu,
                key=backward_mlp_key,
            )

    def __call__(
        self,
        input: chex.Array,
        *,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> chex.Array:
        # [num_nodes, sequence_length, bits_per_seq_elem]
        N, S, BPSE = input.shape
        # [num_nodes, sequence_length * bits_per_seq_elem]
        input = input.reshape(N, -1)
        mask = jnp.any(input, axis=-1)  # [num_nodes]

        token_embeddings = jax.vmap(self.embedder_block, in_axes=(0,))(
            input
        )  # [num_nodes, embedding_size]
        input_embeddings = token_embeddings + self.pos_embeddings

        x = jax.vmap(self.emb2hidden, in_axes=(0,))(
            input_embeddings
        )  # [num_nodes, hidden_size]
        l_key = key
        for layer in self.layers:
            cl_key, l_key = (
                (None, None) if l_key is None else jax.random.split(l_key)
            )
            x = layer(
                x, mask, enable_dropout=enable_dropout, key=cl_key
            )  # [num_nodes, hidden_size]

        encodings_combination = (
            x[self.row] + x[self.col]
        )  # [2 * num_nodes - 1, hidden_size]
        # Tree topology MLP
        forward_logits = jax.vmap(self.forward_mlp, in_axes=(0,))(
            encodings_combination
        ).squeeze(-1)  # [2 * num_nodes - 1]
        flow = self.flow_mlp(jnp.hstack(x)).squeeze(-1)

        if self.train_backward_policy:
            backward_logits = jax.vmap(self.backward_mlp, in_axes=(0,))(
                x[:-1]
            ).squeeze(-1)  # [num_nodes - 1]
        else:
            backward_logits = jnp.zeros(
                shape=(self.n_bwd_actions,), dtype=jnp.float32
            )  # [num_nodes - 1]

        return {
            "forward_logits": forward_logits,  # [2 * num_nodes - 1]
            "log_flow": flow,  # [1]
            "backward_logits": backward_logits,  # [num_nodes - 1]
        }


# Define the train state that will be used in the training loop
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.PhyloTreeEnvironment
    env_params: chex.Array
    model: TransformerPolicy
    target_model: (
        TransformerPolicy  # New: target network copy used for bootstrapping
    )
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    exploration_state: ExplorationState
    metrics_module: dict
    metrics: dict
    learning_rate_schedule: optax.Schedule


def get_policy_fn(
    policy_static: TransformerPolicy,
    policy_type: Literal["fwd", "bwd"],
    enable_dropout: bool = False,
    use_exploration: bool = False,
    exploration_state: ExplorationState = None,
):
    def policy_fn(
        rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params
    ) -> chex.Array:
        dropout_key, eps_key = jax.random.split(rng_key)
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = jax.vmap(
            partial(policy, enable_dropout=enable_dropout, key=dropout_key),
            in_axes=(0),
        )(env_obs)
        logits = (
            policy_outputs["forward_logits"]
            if policy_type == "fwd"
            else policy_outputs["backward_logits"]
        )
        if use_exploration:
            epsilon = exploration_state.schedule(exploration_state.step)
            logits = apply_epsilon_greedy_vmap(eps_key, logits, epsilon)
        return logits, policy_outputs

    return policy_fn


@eqx.filter_jit
def train_step(idx: int, train_state: TrainState) -> TrainState:
    rng_key = train_state.rng_key
    num_envs = train_state.config.num_envs
    env = train_state.env
    env_params = train_state.env_params
    # Step 1. Generate a batch of trajectories and split to transitions
    rng_key, sample_traj_key = jax.random.split(train_state.rng_key)
    # Split the model to pass into forward rollout
    policy_params, policy_static = eqx.partition(
        train_state.model, eqx.is_array
    )

    # Define the policy function suitable for gfnx.utils.forward_rollout
    fwd_policy_fn = get_policy_fn(
        policy_static,
        "fwd",
        enable_dropout=True,
        use_exploration=True,
        exploration_state=train_state.exploration_state,
    )

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
    delta_score = train_state.env.reward_module.delta_score(
        transitions.next_state
    )

    # Step 2. Compute the loss
    def loss_fn(model: TransformerPolicy) -> chex.Array:
        # Call the network to get the logits
        policy_outputs = jax.vmap(model, in_axes=(0,))(transitions.obs)
        # Compute the forward log-probs
        fwd_logits = policy_outputs["forward_logits"]
        invalid_mask = env.get_invalid_mask(transitions.state, env_params)
        masked_fwd_logits = gfnx.utils.mask_logits(fwd_logits, invalid_mask)
        fwd_all_log_probs = jax.nn.log_softmax(masked_fwd_logits, axis=-1)
        fwd_logprobs = jnp.take_along_axis(
            fwd_all_log_probs,
            jnp.expand_dims(transitions.action, axis=-1),
            axis=-1,
        ).squeeze(-1)
        log_flow = policy_outputs["log_flow"]

        # Use the target network for next state
        next_policy_outputs = jax.vmap(train_state.target_model, in_axes=(0,))(
            transitions.next_obs
        )
        bwd_logits = next_policy_outputs["backward_logits"]
        next_bwd_invalid_mask = env.get_invalid_backward_mask(
            transitions.next_state, env_params
        )
        masked_bwd_logits = gfnx.utils.mask_logits(
            bwd_logits, next_bwd_invalid_mask
        )
        bwd_all_log_probs = jax.nn.log_softmax(masked_bwd_logits, axis=-1)
        bwd_logprobs = jnp.take_along_axis(
            bwd_all_log_probs, jnp.expand_dims(bwd_actions, axis=-1), axis=-1
        ).squeeze(-1)
        next_log_flow = next_policy_outputs["log_flow"]
        # In forward-looking DB, the flow is zero for the terminal state
        next_log_flow = jnp.where(transitions.done, 0.0, next_log_flow)
        target = jax.lax.stop_gradient(
            bwd_logprobs + next_log_flow + delta_score
        )
        loss = optax.huber_loss(
            jnp.where(transitions.pad, 0.0, fwd_logprobs + log_flow),
            jnp.where(transitions.pad, 0.0, target),
        ).mean()
        return loss

    mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(train_state.model)
    # Step 3. Update the model with grads
    updates, opt_state = train_state.optimizer.update(
        grads,
        train_state.opt_state,
        eqx.filter(train_state.model, eqx.is_array),
    )
    new_model = eqx.apply_updates(train_state.model, updates)

    def update_target_network(online, target, config):
        # Partition the online and target networks into arrays and static (non‑array) parts
        online_arrays, online_static = eqx.partition(online, eqx.is_array)
        target_arrays, _ = eqx.partition(target, eqx.is_array)

        tau = config.agent.target_update.exponential_tau
        updated_arrays = jax.tree_util.tree_map(
            lambda o, t: tau * o + (1 - tau) * t,
            online_arrays,
            target_arrays,
        )

        # Recombine the updated arrays with the static parts
        return eqx.combine(updated_arrays, online_static)

    new_target_model = update_target_network(
        new_model, train_state.target_model, train_state.config
    )
    exploration_state = train_state.exploration_state.replace(step=idx + 1)

    metric_state = {
        key: module.update(
            train_state.metrics[key], log_info["final_env_state"], env_params
        )
        for key, module in train_state.metrics_module.items()
    }

    # Compute the evaluation info if needed
    def evaluation(args):
        rng_key, metric_states, policy_params, env_params = args
        corr_dict = train_state.metrics_module["corr"].compute(
            rng_key, metric_states["corr"], policy_params, env_params
        )
        return corr_dict

    rng_key, eval_rng_key = jax.random.split(rng_key)
    eval_info = jax.lax.cond(
        (idx % train_state.config.logging.track_each == 0)
        | (idx + 1 == train_state.config.num_train_steps),
        evaluation,
        lambda _: {
            "pearson_corr": 0.0,
        },
        (eval_rng_key, metric_state, policy_params, env_params),
    )

    # Perform the logging via JAX debug callback
    def logging_callback(idx: int, train_info: dict, eval_info: dict):
        if (
            idx % train_state.config.logging.track_each == 0
            or idx + 1 == train_state.config.num_train_steps
        ):
            log.info(f"Step {idx}")
            log.info({key: float(value) for key, value in train_info.items()})
            log.info({key: float(value) for key, value in eval_info.items()})
            if train_state.config.logging.use_wandb:
                wandb.log(eval_info, commit=False)

        if train_state.config.logging.use_wandb:
            wandb.log(train_info)

    # Get current learning rate
    current_lr = train_state.learning_rate_schedule(idx)

    jax.debug.callback(
        logging_callback,
        idx,
        {
            "train/mean_loss": mean_loss,
            "train/entropy": log_info["entropy"].mean(),
            "train/grad_norm": optax.tree_utils.tree_l2_norm(grads),
            "train/learning_rate": current_lr,
        },
        {f"eval/{key}": value for key, value in eval_info.items()},
        ordered=True,
    )

    return train_state._replace(
        rng_key=rng_key,
        model=new_model,
        target_model=new_target_model,
        opt_state=opt_state,
        exploration_state=exploration_state,
    )


@hydra.main(config_path="configs/", config_name="db_phylo", version_base=None)
def run_experiment(cfg: OmegaConf) -> None:
    # Log the configuration
    log.info(OmegaConf.to_yaml(cfg))

    rng_key = jax.random.PRNGKey(cfg.seed)
    # This key is needed to initialize the environment
    env_init_key = jax.random.PRNGKey(cfg.env_init_seed)
    # This key is needed to initialize the evaluation process
    # i.e., generate random test set.
    eval_init_key = jax.random.PRNGKey(cfg.eval_init_seed)

    # Initialize the environment
    env_kwargs, reward_kwargs = get_phylo_initialization_args(
        cfg.environment.dataset, cfg.environment.data_folder
    )
    reward_module = PhyloTreeRewardModule(**reward_kwargs)
    env = PhyloTreeEnvironment(**env_kwargs, reward_module=reward_module)
    env_params = env.init(env_init_key)

    rng_key, net_init_key = jax.random.split(rng_key)
    # Initialize the network
    model = TransformerPolicy(
        num_nodes=env_params.num_nodes,
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        train_backward_policy=cfg.agent.train_backward,
        emb_params={
            "in_size": env_params.sequence_length
            * env_params.bits_per_seq_elem,
            **(OmegaConf.to_container(cfg.network.emb_params)),
        },
        transformer_params=OmegaConf.to_container(
            cfg.network.transformer_params
        ),
        mlp_params=OmegaConf.to_container(cfg.network.mlp_params),
        key=net_init_key,
    )

    # Initialize the target network as an exact copy
    target_model = model

    # Create learning rate schedule
    total_steps = cfg.num_train_steps
    warmup_steps = cfg.agent.learning_rate.warmup_steps

    # Create transition steps schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.agent.learning_rate.peak_value,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=cfg.agent.learning_rate.final_value,
    )

    # Initialize the optimizer with the schedule
    optimizer = optax.chain(
        # optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    # Initialize the backward policy function for correlation computation
    policy_static = eqx.filter(model, eqx.is_array, inverse=True)

    fwd_policy_fn = get_policy_fn(policy_static, "fwd")
    bwd_policy_fn = get_policy_fn(policy_static, "bwd")

    exploration_state = ExplorationState(
        schedule=create_exploration_schedule(**cfg.agent.exploration),
        step=jnp.array(0, dtype=jnp.int32),
    )

    metrics_module = {
        "corr": gfnx.metrics.PhyloTreeCorrelationMetric(
            n_rounds=cfg.metrics.n_rounds,
            n_terminal_states=cfg.metrics.n_terminal_states,
            batch_size=cfg.metrics.batch_size,
            fwd_policy_fn=fwd_policy_fn,
            bwd_policy_fn=bwd_policy_fn,
            env=env,
        ),
    }
    # Fill the initial states of metrics
    metrics = {}
    for key, module in metrics_module.items():
        eval_init_key, new_key = jax.random.split(eval_init_key)
        metrics[key] = module.init(new_key, env_params)

    train_state = TrainState(
        rng_key=rng_key,
        config=cfg,
        env=env,
        env_params=env_params,
        model=model,
        target_model=target_model,  # Include the target network here.
        optimizer=optimizer,
        opt_state=opt_state,
        exploration_state=exploration_state,
        metrics_module=metrics_module,
        metrics=metrics,
        learning_rate_schedule=lr_schedule,
    )
    # Split train state into parameters and static parts to make jit work.
    train_state_params, train_state_static = eqx.partition(
        train_state, eqx.is_array
    )

    @functools.partial(jax.jit, donate_argnums=(1,))
    @loop_tqdm(cfg.num_train_steps, print_rate=cfg.logging.tqdm_print_rate)
    def train_step_wrapper(idx: int, train_state_params):
        # Wrapper to use a usual jit in jax, since it is required by fori_loop.
        train_state = eqx.combine(train_state_params, train_state_static)
        train_state = train_step(idx, train_state)
        train_state_params, _ = eqx.partition(train_state, eqx.is_array)
        return train_state_params

    if cfg.logging.use_wandb:
        log.info("Initialize wandb")
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            tags=cfg.wandb.tags,
        )
        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

    log.info("Start training")
    # Run the training loop via jax lax.fori_loop
    train_state_params = jax.lax.fori_loop(
        lower=0,
        upper=cfg.num_train_steps,
        body_fun=train_step_wrapper,
        init_val=train_state_params,
    )
    jax.block_until_ready(train_state_params)

    # Save the final model
    path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    cwd = os.path.join(path, "model")
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckptr.save(cwd, train_state_params)
    ckptr.wait_until_finished()


if __name__ == "__main__":
    run_experiment()
