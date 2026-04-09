"""Single-file implementation for Soft DQN in hypergrid environment.

Run the script with the following command:
```bash
python baselines/soft_dqn_hypergrid.py
```

Also see https://jax.readthedocs.io/en/latest/gpu_performance_tips.html for
performance tips when running on GPU, i.e., XLA flags.

"""

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
from omegaconf import OmegaConf
from utils.checkpoint import save_checkpoint
from utils.logger import Writer

import gfnx
from gfnx.metrics import ApproxDistributionMetricsModule, ApproxDistributionMetricsState

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
            output_size += 1  # for the value logit

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
            return {
                "unmasked_advantage_logits": unmasked_advantage_logits,
                "value_logits": value_logits,
            }
        else:
            return {
                "raw_qvalue_logits": x,
                "value_logits": jnp.zeros(shape=(1,), dtype=jnp.float32),
            }


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
    exploration_schedule: optax.Schedule
    eval_info: dict
    reward_module: gfnx.GeneralHypergridRewardModule
    reward_params: chex.Array


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

    cur_epsilon = train_state.exploration_schedule(idx)

    def fwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = policy(env_obs)
        if train_state.config.agent.dueling:
            fwd_logits = policy_outputs["unmasked_advantage_logits"]
        else:
            fwd_logits = policy_outputs["raw_qvalue_logits"]
        do_explore = jax.random.bernoulli(rng_key, cur_epsilon)
        fwd_logits = jnp.where(do_explore, 0, fwd_logits)
        return fwd_logits, policy_outputs

    rng_keys = jax.random.split(sample_traj_key, num_envs)
    traj_data, final_states, info = jax.vmap(
        lambda rng: gfnx.utils.forward_rollout(
            rng, fwd_policy_fn, policy_params, train_state.env, train_state.env_params
        )
    )(rng_keys)
    transitions = jax.tree.map(
        lambda x: x.reshape((-1,) + x.shape[2:]),
        jax.vmap(gfnx.utils.split_traj_to_transitions)(traj_data),
    )
    bwd_actions = train_state.env.get_backward_action_batch(
        transitions.state,
        transitions.action,
        transitions.next_state,
        train_state.env_params,
    )
    # Compute rewards for terminal states
    log_rewards = jax.vmap(train_state.reward_module.log_reward, in_axes=(0, None))(
        final_states, train_state.reward_params
    )
    T_steps = transitions.done.shape[0] // num_envs
    traj_rewards_flat = jnp.repeat(log_rewards, T_steps)  # [B*T]

    # Compute the RL reward / ELBO (for logging purposes)
    log_pb_traj = jax.vmap(
        lambda td: gfnx.utils.forward_trajectory_log_probs(env, td, env_params)
    )(traj_data)[1]
    rl_reward = log_pb_traj + log_rewards + info["entropy"]

    def loss_fn(model, target_model, current_traj_rewards_flat) -> chex.Array:
        num_transition = transitions.pad.shape[0]
        not_pad_mask = jnp.logical_not(transitions.pad)

        # Step 1. Compute the Q-value
        policy_outputs = jax.vmap(model)(transitions.obs)
        invalid_mask = env.get_invalid_mask_batch(transitions.state, env_params)
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
        target_policy_outputs = jax.vmap(target_model)(transitions.next_obs)
        next_invalid_actions_mask = env.get_invalid_mask_batch(transitions.next_state, env_params)
        if train_state.config.agent.dueling:
            raw_next_advantage = target_policy_outputs["unmasked_advantage_logits"]
            target_next_value = target_policy_outputs["value_logits"]
            next_advantage = gfnx.utils.mask_logits(raw_next_advantage, next_invalid_actions_mask)
            target_next_qvalue = target_next_value + jax.nn.log_softmax(next_advantage, axis=-1)
            target_next_value = target_next_value.squeeze(-1)  # should be (N,)
        else:
            target_next_qvalue = target_policy_outputs["raw_qvalue_logits"]
            target_next_qvalue = gfnx.utils.mask_logits(
                target_next_qvalue, next_invalid_actions_mask
            )
            target_next_value = jax.nn.logsumexp(target_next_qvalue, axis=-1)

        # Step 2.2: Compute intermidiate rewards.
        bwd_logits = jnp.zeros(
            shape=(num_transition, env.backward_action_space.n), dtype=jnp.float32
        )
        next_bwd_invalid_mask = env.get_invalid_backward_mask_batch(
            transitions.next_state, env_params
        )
        bwd_logprobs = gfnx.utils.compute_action_log_probs(
            bwd_logits, bwd_actions, next_bwd_invalid_mask
        )

        target = jnp.where(
            transitions.done,
            current_traj_rewards_flat,
            bwd_logprobs + target_next_value,  # (N,) + (N,) = (N,)
        )
        padded_target = jnp.where(transitions.pad, 0.0, target)

        # Step 4. Compute the loss
        local_losses = optax.losses.huber_loss(padded_q_value, padded_target)
        local_losses = jnp.where(
            transitions.done, local_losses * train_state.config.agent.leaf_coeff, local_losses
        )
        return jnp.sum(local_losses * not_pad_mask) / jnp.sum(not_pad_mask)

    mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(
        train_state.model, train_state.target_model, traj_rewards_flat
    )
    updates, opt_state = train_state.optimizer.update(
        grads,
        train_state.opt_state,
        eqx.filter(train_state.model, eqx.is_array),
    )
    new_model = eqx.apply_updates(train_state.model, updates)

    is_target_update = idx % train_state.config.agent.target_update_every == 0
    trgt_model_params, trgt_model_static = eqx.partition(train_state.target_model, eqx.is_array)
    model_params, _model_static = eqx.partition(new_model, eqx.is_array)

    new_trgt_model_params = jax.lax.cond(
        is_target_update,
        lambda: optax.incremental_update(
            model_params, trgt_model_params, train_state.config.agent.target_update_tau
        ),
        lambda: trgt_model_params,
    )
    new_trgt_model = eqx.combine(new_trgt_model_params, trgt_model_static)

    # Perform all the required logging
    metrics_state, eval_info = metrics_module.step(
        idx=idx,
        metrics_state=train_state.metrics_state,
        rng_key=jax.random.key(0),
        update_args=metrics_module.UpdateArgs(states=final_states),
        process_args=metrics_module.ProcessArgs(env_params=env_params),
        eval_each=train_state.config.logging.eval_each,
        num_train_steps=train_state.config.num_train_steps,
        prev_eval_info=train_state.eval_info,
    )

    # Perform the logging via JAX debug callback
    def logging_callback(
        idx: int,
        train_info: dict,
        eval_info: dict,
        cfg,
    ):
        train_info = {f"train/{key}": float(value) for key, value in train_info.items()}
        if idx % cfg.logging.eval_each == 0 or idx + 1 == cfg.num_train_steps:
            log.info(f"Step {idx}")
            log.info(train_info)
            # Get the evaluation metrics
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
            "entropy": info["entropy"].mean(),
            "grad_norm": optax.tree_utils.tree_l2_norm(grads),
            "mean_reward": jnp.exp(log_rewards).mean(),
            "mean_log_reward": log_rewards.mean(),
            "rl_reward": rl_reward.mean(),
        },
        eval_info,
        train_state.config,
        ordered=True,
    )
    return train_state._replace(
        rng_key=rng_key,
        model=new_model,
        target_model=new_trgt_model,
        opt_state=opt_state,
        metrics_state=metrics_state,
        eval_info=eval_info,
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
    reward_module = reward_module_factory(side=cfg.environment.side)

    # Initialize the environment and its inner parameters
    env = gfnx.environment.HypergridEnvironment(dim=cfg.environment.dim, side=cfg.environment.side)
    env_params = env.init(env_init_key)
    reward_params = reward_module.init(env_init_key, env.get_init_state())

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
    model_params, model_static = eqx.partition(model, eqx.is_array)
    target_model_params = jax.tree_util.tree_map(jnp.copy, model_params)
    target_model = eqx.combine(target_model_params, model_static)
    # Initialize the exploration schedule
    exploration_schedule = optax.linear_schedule(
        init_value=cfg.agent.start_eps,
        end_value=cfg.agent.end_eps,
        transition_steps=cfg.agent.exploration_steps,
    )
    # Initialize the optimizer
    optimizer = optax.adam(learning_rate=cfg.agent.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    metrics_module = ApproxDistributionMetricsModule(
        metrics=["tv", "kl", "2d_marginal_distribution"],
        env=env,
        reward_module=reward_module,
        buffer_size=cfg.logging.metric_buffer_size,
    )
    # Initialize the metrics state
    eval_init_key, new_eval_init_key = jax.random.split(eval_init_key)
    metrics_state = metrics_module.init(
        new_eval_init_key,
        metrics_module.InitArgs(env_params=env_params, reward_params=reward_params),
    )
    eval_info = metrics_module.get(metrics_state)

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
        exploration_schedule=exploration_schedule,
        eval_info=eval_info,
        reward_module=reward_module,
        reward_params=reward_params,
    )
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
            tags=["SoftDQN", env.name.upper()],
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    log.info("Start training")
    train_state = gfnx.utils.run_training_loop(
        train_step, train_state, cfg.num_train_steps, cfg.logging["tqdm_print_rate"]
    )
    dir = (
        cfg.logging.checkpoint_dir
        if cfg.logging.checkpoint_dir
        else os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            f"checkpoints_{os.getpid()}/",
        )
    )
    save_checkpoint(os.path.join(dir, "train_state"), train_state)
    save_checkpoint(os.path.join(dir, "model"), train_state.model)
    save_checkpoint(os.path.join(dir, "target_model"), train_state.target_model)


if __name__ == "__main__":
    run_experiment()
