"""Single-file implementation for Soft DQN in hypergrid environment.

Run the script with the following command:
```bash
python baselines/soft_dqn_hypergrid.py
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
import numpy as np
import optax
from jax_tqdm import loop_tqdm
from omegaconf import OmegaConf

import gfnx
from gfnx.metrics.new import ApproxDistributionMetricsModule, ApproxDistributionMetricsState

from utils.logger import Writer
from utils.checkpoint import save_checkpoint

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
writer = Writer()

class MLPPolicy(eqx.Module):
    """
    A policy module that uses a Multi-Layer Perceptron (MLP) to generate
    forward and backward action logits.

    Args:
        input_size (int): The size of the input features.
        n_fwd_actions (int): Number of forward actions.
        n_bwd_actions (int): Number of backward actions.
        hidden_size (int): The size of the hidden layers in the MLP.
        train_backward_policy (bool): Flag indicating whether to train
            the backward policy.
        depth (int): The number of layers in the MLP.
        rng_key (chex.PRNGKey): Random key for initializing the MLP.

    Methods:
        __call__(x: chex.Array) -> chex.Array:
            Forward pass through the MLP network. Returns a dictionary
            containing forward logits and backward logits.
    """

    network: eqx.nn.MLP
    dueling: bool
    n_fwd_actions: int

    def __init__(
        self,
        input_size: int,
        n_fwd_actions: int,
        hidden_size: int,
        dueling: bool,
        depth: int,
        rng_key: chex.PRNGKey,
    ):
        self.dueling = dueling
        self.n_fwd_actions = n_fwd_actions

        output_size = self.n_fwd_actions
        if dueling:
            output_size += 1 # for the value logit

        self.network = eqx.nn.MLP(
            in_size=input_size,
            out_size=output_size,
            width_size=hidden_size,
            depth=depth,
            key=rng_key,
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        x = self.network(x)
        if self.dueling:
            unmasked_advantage_logits, value_logits = jnp.split(x, [self.n_fwd_actions], axis=-1)
            return {"unmasked_advantage_logits": unmasked_advantage_logits, 
                    "value_logits": value_logits}
        else:
            return {"raw_qvalue_logits": x,
                    "value_logits": jnp.zeros(shape=(1,), dtype=jnp.float32)}


# Define the train state that will be used in the training loop
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.HypergridEnvironment
    env_params: chex.Array
    model: MLPPolicy
    target_model: MLPPolicy
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    metrics_module: ApproxDistributionMetricsModule
    metrics_state: ApproxDistributionMetricsState

@eqx.filter_jit
def train_step(idx: int, train_state: TrainState) -> TrainState:
    rng_key = train_state.rng_key
    num_envs = train_state.config.num_envs
    env = train_state.env
    env_params = train_state.env_params
    metrics_module = train_state.metrics_module
    # Step 1. Generate a batch of trajectories and split to transitions
    rng_key, sample_traj_key = jax.random.split(train_state.rng_key)
    # Split the model to pass into forward rollout
    policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)

    def fwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = jax.vmap(policy, in_axes=(0,))(env_obs)
        if train_state.config.agent.dueling:
            fwd_logits = policy_outputs["unmasked_advantage_logits"]
        else:
            fwd_logits = policy_outputs["raw_qvalue_logits"]
        return fwd_logits, policy_outputs

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

    def loss_fn(model, target_model) -> chex.Array:
        num_transition = transitions.pad.shape[0]
        not_pad_mask = jnp.logical_not(transitions.pad)
        
        # Step 1. Compute the Q-value
        policy_outputs = jax.vmap(model, in_axes=(0,))(transitions.obs)
        invalid_mask = env.get_invalid_mask(transitions.state, env_params)
        if train_state.config.agent.dueling:
            raw_advantage = policy_outputs["unmasked_advantage_logits"]
            value = policy_outputs["value_logits"]
            advantage = gfnx.utils.mask_logits(raw_advantage, invalid_mask)
            qvalue = value + jax.nn.log_softmax(advantage, axis=-1)
        else:
            qvalue = policy_outputs["raw_qvalue_logits"]
            qvalue = gfnx.utils.mask_logits(qvalue, invalid_mask)
            value = jax.nn.logsumexp(qvalue, axis=-1)

        qvalue = jnp.take_along_axis(
            qvalue, jnp.expand_dims(transitions.action, axis=-1), axis=-1
            ).squeeze(-1)
        padded_q_value = jnp.where(transitions.pad, 0.0, qvalue)

        # Step 2.1: Compute the target Q-value
        target_policy_outputs = jax.vmap(target_model, in_axes=(0,))(transitions.next_obs)
        next_invalid_actions_mask = env.get_invalid_mask(transitions.next_state, env_params)
        if train_state.config.agent.dueling:
            raw_next_advantage = target_policy_outputs["unmasked_advantage_logits"]
            target_next_value = target_policy_outputs["value_logits"]
            next_advantage = gfnx.utils.mask_logits(raw_next_advantage, next_invalid_actions_mask)
            target_next_qvalue = target_next_value + jax.nn.log_softmax(next_advantage, axis=-1) 
            target_next_value = target_next_value.squeeze(-1) # should be (N,)
        else:
            target_next_qvalue = target_policy_outputs["raw_qvalue_logits"]
            target_next_qvalue = gfnx.utils.mask_logits(target_next_qvalue, 
                                                        next_invalid_actions_mask)
            target_next_value = jax.nn.logsumexp(target_next_qvalue, axis=-1)
        
        # Step 2.2: Compute intermidiate rewards.
        bwd_logits = jnp.zeros(shape=(num_transition, env.backward_action_space.n), 
                               dtype=jnp.float32)
        next_bwd_invalid_mask = env.get_invalid_backward_mask(transitions.next_state, env_params)
        masked_bwd_logits = gfnx.utils.mask_logits(bwd_logits, next_bwd_invalid_mask)
        bwd_all_log_probs = jax.nn.log_softmax(masked_bwd_logits, axis=-1)
        bwd_logprobs = jnp.take_along_axis(
            bwd_all_log_probs, jnp.expand_dims(bwd_actions, axis=-1), axis=-1
        ).squeeze(-1) # (N,)
        
        target = jnp.where(
            transitions.done,
            transitions.log_gfn_reward,
            bwd_logprobs + target_next_value # (N,) + (N,) = (N,)
        )
        padded_target = jnp.where(
            transitions.pad, 0.0, target
        )

        # Step 4. Compute the loss
        local_losses = optax.losses.huber_loss(padded_q_value, padded_target)
        local_losses = jnp.where(
            transitions.done, local_losses * train_state.config.agent.leaf_coeff, local_losses
        )
        local_losses = jnp.sum(local_losses * not_pad_mask) / jnp.sum(not_pad_mask)
        return local_losses

    mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(train_state.model,
                                                          train_state.target_model)
    updates, opt_state = train_state.optimizer.update(
        grads,
        train_state.opt_state,
        eqx.filter(train_state.model, eqx.is_array),
    )
    new_model = eqx.apply_updates(train_state.model, updates)

    metrics_state = metrics_module.update(
        train_state.metrics_state,
        rng_key=jax.random.key(0),  # This key is not used in the update method
        args=metrics_module.UpdateArgs(states=log_info["final_env_state"]),
    )

    is_target_update = idx % train_state.config.agent.target_update_every == 0
    trgt_model_params, trgt_model_static = eqx.partition(train_state.target_model, eqx.is_array)
    model_params, model_static = eqx.partition(new_model, eqx.is_array)

    new_trgt_model_params = jax.lax.cond(
        is_target_update,
        lambda: optax.incremental_update(model_params,
                                        trgt_model_params, 
                                        train_state.config.agent.target_update_tau),
        lambda: trgt_model_params,
    )
    new_trgt_model = eqx.combine(new_trgt_model_params, trgt_model_static)


    # Perform evaluation computations if needed
    is_eval_step = idx % train_state.config.logging.eval_each == 0
    is_eval_step = is_eval_step | (idx + 1 == train_state.config.num_train_steps)

    metrics_state = jax.lax.cond(
        is_eval_step,
        lambda kwargs: metrics_module.process(**kwargs),
        lambda kwargs: kwargs["metrics_state"],  # Do nothing if not eval step
        {
            "metrics_state": metrics_state,
            "rng_key": jax.random.key(0),  # This key is not used in the process method
            "args": metrics_module.ProcessArgs(env_params=env_params),
        },
    )

    # Perform the logging via JAX debug callback
    def logging_callback(
        idx: int,
        train_info: dict,
        metrics_state: ApproxDistributionMetricsState,
        cfg,
    ):
        train_info = {f"train/{key}": float(value) for key, value in train_info.items()}
        if idx % cfg.logging.eval_each == 0 or idx + 1 == cfg.num_train_steps:
            log.info(f"Step {idx}")
            log.info(train_info)
            # Get the evaluation metrics
            eval_info = metrics_module.get(metrics_state)
            eval_info = {f"eval/{key}": value for key, value in eval_info.items()}

            log.info({
                key: float(value)
                for key, value in eval_info.items()
                if key not in ["eval/2d_marginal_distribution"]
            })
            if cfg.logging.use_writer:
                marginal_dist = eval_info["eval/2d_marginal_distribution"]
                marginal_dist = (marginal_dist - marginal_dist.min()) / (
                    marginal_dist.max() - marginal_dist.min()
                )
                eval_info["eval/2d_marginal_distribution"] = writer.Image(
                    np.array(
                        255.0 * marginal_dist,
                        dtype=np.int32,
                    )
                )
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
            "rl_reward": log_info["log_gfn_reward"].mean() + log_info["entropy"].mean(),
        },
        metrics_state,
        train_state.config,
        ordered=True,
    )
    return train_state._replace(
        rng_key=rng_key,
        model=new_model,
        target_model=new_trgt_model,
        opt_state=opt_state,
        metrics_state=metrics_state,
    )


@hydra.main(config_path="configs/", config_name="soft_dqn_hypergrid", version_base=None)
def run_experiment(cfg: OmegaConf) -> None:
    # Log the configuration
    log.info(OmegaConf.to_yaml(cfg))

    rng_key = jax.random.PRNGKey(cfg.seed)
    # This key is needed to initialize the environment
    env_init_key = jax.random.PRNGKey(cfg.env_init_seed)
    # This key is needed to initialize the evaluation process
    # i.e., generate random test set.
    eval_init_key = jax.random.PRNGKey(cfg.eval_init_seed)

    # Define the reward function for the environment
    reward_module_factory = {
        "easy": gfnx.EasyHypergridRewardModule,
        "hard": gfnx.HardHypergridRewardModule,
    }[cfg.environment.reward]
    reward_module = reward_module_factory()

    # Initialize the environment and its inner parameters
    env = gfnx.environment.HypergridEnvironment(
        reward_module, dim=cfg.environment.dim, side=cfg.environment.side
    )
    env_params = env.init(env_init_key)

    rng_key, net_init_key = jax.random.split(rng_key)
    # Initialize the network
    model = MLPPolicy(
        input_size=env.observation_space.shape[0],
        n_fwd_actions=env.action_space.n,
        hidden_size=cfg.network.hidden_size,
        dueling=cfg.agent.dueling,
        depth=cfg.network.depth,
        rng_key=net_init_key,
    )
    import copy
    target_model = copy.deepcopy(model)

    # Initialize the optimizer
    optimizer = optax.adam(learning_rate=cfg.agent.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    metrics_module = ApproxDistributionMetricsModule(
        metrics=["tv", "kl", "2d_marginal_distribution"],
        env=env,
        buffer_size=cfg.logging.metric_buffer_size,
    )
    # Initialize the metrics state
    eval_init_key, new_eval_init_key = jax.random.split(eval_init_key)
    metrics_state = metrics_module.init(
        new_eval_init_key, metrics_module.InitArgs(env_params=env_params)
    )

    train_state = TrainState(
        rng_key=rng_key,
        config=cfg,
        env=env,
        env_params=env_params,
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
        log_dir = cfg.logging.log_dir if cfg.logging.log_dir else os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, f"run_{os.getpid()}/"
        )
        writer.init(
            writer_type=cfg.writer.writer_type,
            save_locally=cfg.writer.save_locally,
            log_dir=log_dir,
            entity=cfg.writer.entity,
            project=cfg.writer.project,
            tags=["SoftDQN", env.name.upper()],
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
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
    train_state = eqx.combine(train_state_params, train_state_static)
    dir = cfg.logging.checkpoint_dir if cfg.logging.checkpoint_dir else os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        f"checkpoints_{os.getpid()}/",
    )
    save_checkpoint(os.path.join(dir, "train_state"), train_state)
    save_checkpoint(os.path.join(dir, "model"), train_state.model)
    save_checkpoint(os.path.join(dir, "target_model"), train_state.target_model)


if __name__ == "__main__":
    run_experiment()

