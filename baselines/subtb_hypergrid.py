"""Single-file implementation for Sub-Trajectory Balance in hypergrid environment.

Run the script with the following command:
```bash
python baselines/subtb_hypergrid.py
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
            containing forward logits, backward logits, and log_flow.
    """

    network: eqx.nn.MLP
    train_backward_policy: bool
    n_fwd_actions: int
    n_bwd_actions: int

    def __init__(
        self,
        input_size: int,
        n_fwd_actions: int,
        n_bwd_actions: int,
        hidden_size: int,
        train_backward_policy: bool,
        depth: int,
        rng_key: chex.PRNGKey,
    ):
        self.train_backward_policy = train_backward_policy
        self.n_fwd_actions = n_fwd_actions
        self.n_bwd_actions = n_bwd_actions

        if train_backward_policy:
            output_size = n_fwd_actions + n_bwd_actions + 1
        else:
            output_size = n_fwd_actions + 1
        self.network = eqx.nn.MLP(
            in_size=input_size,
            out_size=output_size,
            width_size=hidden_size,
            depth=depth,
            key=rng_key,
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        x = self.network(x)
        if self.train_backward_policy:
            forward_logits, backward_logits, log_flow = jnp.split(
                x, [self.n_fwd_actions, self.n_fwd_actions + self.n_bwd_actions], axis=-1
            )
        else:
            forward_logits, log_flow = jnp.split(x, [self.n_fwd_actions], axis=-1)
            backward_logits = jnp.zeros(shape=(self.n_bwd_actions,), dtype=jnp.float32)
        return {
            "forward_logits": forward_logits,
            "backward_logits": backward_logits,
            "log_flow": log_flow,
        }


# Define the train state that will be used in the training loop
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.HypergridEnvironment
    env_params: chex.Array
    reward_module: gfnx.GeneralHypergridRewardModule
    reward_params: chex.Array
    model: MLPPolicy
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    metrics_module: ApproxDistributionMetricsModule
    metrics_state: ApproxDistributionMetricsState
    exploration_schedule: optax.Schedule
    eval_info: dict


@eqx.filter_jit
def train_step(idx: int, train_state: TrainState) -> TrainState:
    rng_key = train_state.rng_key
    num_envs = train_state.config.num_envs
    env = train_state.env
    env_params = train_state.env_params

    # Get model parameters and static parts
    policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)

    # Step 1. Generate a batch of trajectories
    rng_key, sample_traj_key = jax.random.split(rng_key)
    cur_epsilon = train_state.exploration_schedule(idx)

    # Define the policy function suitable for gfnx.utils.forward_rollout
    def fwd_policy_fn(
        fwd_rng_key: chex.PRNGKey,
        env_obs: gfnx.TObs,
        current_policy_params,  # current_policy_params are network params
    ) -> chex.Array:
        del fwd_rng_key
        # Recombine the network parameters with the static parts of the model
        current_model = eqx.combine(current_policy_params, policy_static)
        policy_outputs = current_model(env_obs)
        do_explore = jax.random.bernoulli(rng_key, cur_epsilon)
        forward_logits = jnp.where(do_explore, 0, policy_outputs["forward_logits"])
        return forward_logits, policy_outputs

    rng_keys = jax.random.split(sample_traj_key, num_envs)
    traj_data, final_states, info = jax.vmap(
        lambda rng: gfnx.utils.forward_rollout(rng, fwd_policy_fn, policy_params, env, env_params)
    )(rng_keys)
    # Compute the RL reward / ELBO (for logging purposes)
    log_pb_traj = jax.vmap(
        lambda td: gfnx.utils.forward_trajectory_log_probs(env, td, env_params)
    )(traj_data)[1]
    log_rewards = jax.vmap(train_state.reward_module.log_reward, in_axes=(0, None))(
        final_states, train_state.reward_params
    )
    rl_reward = log_pb_traj + log_rewards + info["entropy"]

    # Step 2. Compute the loss
    def loss_fn(
        current_all_params: dict,
        static_model_parts: MLPPolicy,
        current_traj_data: gfnx.utils.TrajectoryData,
        current_env: gfnx.HypergridEnvironment,
        current_env_params: gfnx.HypergridEnvParams,
        current_log_rewards: jnp.ndarray,
    ):
        model_learnable_params = current_all_params["model_params"]
        model_to_call = eqx.combine(model_learnable_params, static_model_parts)
        # Compute policy outputs for the whole trajectory
        policy_outputs_traj = jax.vmap(jax.vmap(model_to_call))(current_traj_data.obs)
        fwd_logits_traj = policy_outputs_traj["forward_logits"]
        bwd_logits_traj = policy_outputs_traj["backward_logits"]
        log_flow_traj = policy_outputs_traj["log_flow"].squeeze(-1)

        _batch_size, traj_len_plus1 = current_traj_data.action.shape
        traj_len = traj_len_plus1 - 1

        # Masks
        forward_invalid_mask = jax.vmap(current_env.get_invalid_mask_batch, in_axes=(0, None))(
            current_traj_data.state, current_env_params
        )
        backward_invalid_mask = jax.vmap(
            current_env.get_invalid_backward_mask_batch, in_axes=(0, None)
        )(current_traj_data.state, current_env_params)
        forward_action = current_traj_data.action[:, :-1]
        # Compute backward actions
        prev_states = jax.tree.map(lambda x: x[:, :-1], current_traj_data.state)
        fwd_actions = current_traj_data.action[:, :-1]
        curr_states = jax.tree.map(lambda x: x[:, 1:], current_traj_data.state)
        bwd_actions_traj = jax.vmap(
            current_env.get_backward_action_batch,
            in_axes=(0, 0, 0, None),
        )(prev_states, fwd_actions, curr_states, current_env_params)
        pad_mask = current_traj_data.pad[:, :-1]
        done_mask = current_traj_data.done[:, :-1]

        # Forward log-probs
        log_pf_along_traj = gfnx.utils.compute_action_log_probs(
            fwd_logits_traj[:, :-1], forward_action, forward_invalid_mask[:, :-1], pad_mask
        )

        # Backward log-probs
        log_pb_along_traj = gfnx.utils.compute_action_log_probs(
            bwd_logits_traj[:, 1:], bwd_actions_traj, backward_invalid_mask[:, 1:], pad_mask
        )

        # log_flow
        log_flow_traj = log_flow_traj.at[:, 1:].set(
            jnp.where(done_mask, current_log_rewards[:, jnp.newaxis], log_flow_traj[:, 1:])
        )
        log_flow_traj = log_flow_traj.at[:, 1:].set(jnp.where(pad_mask, 0.0, log_flow_traj[:, 1:]))

        def process_one_traj(log_pf, log_pb, log_flow, done, pad):
            def process_pair_idx(i, j, log_pf, log_pb, log_flow, done, pad):
                def fn():
                    mask = jnp.logical_and(i <= jnp.arange(traj_len), jnp.arange(traj_len) < j)
                    weight = jnp.power(train_state.config.agent.lmbd, j - i)
                    log_pf_subtraj = log_flow[i] + (log_pf * mask).sum()
                    log_pb_subtraj = log_flow[j] + (log_pb * mask).sum()
                    loss = optax.losses.squared_error(log_pf_subtraj, log_pb_subtraj)
                    return weight * loss, weight

                return jax.lax.cond(pad[j - 1], lambda: (0.0, 0.0), fn)

            i, j = jnp.triu_indices(traj_len + 1, k=1)
            weighted_loss, weighted_norm = jax.vmap(
                process_pair_idx, in_axes=(0, 0, None, None, None, None, None)
            )(i, j, log_pf, log_pb, log_flow, done, pad)
            return weighted_loss.sum() / weighted_norm.sum()

        return jax.vmap(process_one_traj)(
            log_pf_along_traj,
            log_pb_along_traj,
            log_flow_traj,
            done_mask,
            pad_mask,
        ).mean()

    # Prepare parameters for the loss function and gradient calculation
    # policy_params are model network parameters
    # policy_static are model static parts.
    params_for_loss = {"model_params": policy_params}

    mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(
        params_for_loss, policy_static, traj_data, env, env_params, log_rewards
    )

    # Step 3. Update parameters (model network)
    # `grads` is a dict {'model_params': ...}
    # `optax_params_for_update` should match the structure given
    # to optimizer.init
    optax_params_for_update = policy_params
    updates, new_opt_state = train_state.optimizer.update(
        grads["model_params"], train_state.opt_state, optax_params_for_update
    )
    new_model = eqx.apply_updates(train_state.model, updates)

    # Perform all the required logging
    metrics_state, eval_info = train_state.metrics_module.step(
        idx=idx,
        metrics_state=train_state.metrics_state,
        rng_key=jax.random.key(0),
        update_args=train_state.metrics_module.UpdateArgs(states=final_states),
        process_args=train_state.metrics_module.ProcessArgs(env_params=env_params),
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

    # Return the updated train state
    return train_state._replace(
        rng_key=rng_key,
        model=new_model,
        opt_state=new_opt_state,
        metrics_state=metrics_state,
        eval_info=eval_info,
    )


@hydra.main(config_path="configs/", config_name="subtb_hypergrid", version_base=None)
def run_experiment(cfg: OmegaConf) -> None:
    # Log the configuration
    log.info(OmegaConf.to_yaml(cfg))

    rng_key = jax.random.PRNGKey(cfg.seed)
    env_init_key = jax.random.PRNGKey(cfg.env_init_seed)
    eval_init_key = jax.random.PRNGKey(cfg.eval_init_seed)

    reward_module_factory = {
        "easy": gfnx.EasyHypergridRewardModule,
        "hard": gfnx.HardHypergridRewardModule,
    }[cfg.environment.reward]
    reward_module = reward_module_factory(side=cfg.environment.side)

    env = gfnx.environment.HypergridEnvironment(dim=cfg.environment.dim, side=cfg.environment.side)
    env_params = env.init(env_init_key)
    reward_params = reward_module.init(env_init_key, env.get_init_state())

    rng_key, net_init_key = jax.random.split(rng_key)
    model = MLPPolicy(
        input_size=env.observation_space.shape[0],
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        hidden_size=cfg.network.hidden_size,
        train_backward_policy=cfg.agent.train_backward,
        depth=cfg.network.depth,
        rng_key=net_init_key,
    )
    # Initialize the exploration schedule
    exploration_schedule = optax.linear_schedule(
        init_value=cfg.agent.start_eps,
        end_value=cfg.agent.end_eps,
        transition_steps=cfg.agent.exploration_steps,
    )
    # Prepare parameters for Optax
    model_params_init = eqx.filter(model, eqx.is_array)
    optimizer = optax.adam(learning_rate=cfg.agent.learning_rate)
    opt_state = optimizer.init(model_params_init)

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
        reward_module=reward_module,
        reward_params=reward_params,
        model=model,
        optimizer=optimizer,
        opt_state=opt_state,
        metrics_module=metrics_module,
        metrics_state=metrics_state,
        exploration_schedule=exploration_schedule,
        eval_info=eval_info,
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
            tags=["SubTB", env.name.upper()],
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    log.info("Start training")
    final_train_state = gfnx.utils.run_training_loop(
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
    save_checkpoint(os.path.join(dir, "model"), final_train_state.model)


if __name__ == "__main__":
    run_experiment()
