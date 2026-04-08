"""Single-file implementation for Trajectory Balance in AMP environment.

Run the script with the following command:
```bash
python baselines/tb_amp.py
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
import optax
from jaxtyping import Array, Int
from omegaconf import OmegaConf
from utils.checkpoint import save_checkpoint
from utils.logger import Writer

import gfnx
from gfnx.metrics import MultiMetricsModule, MultiMetricsState, TopKMetricsModule

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
writer = Writer()


class TransformerPolicy(eqx.Module):
    """
    A policy module that uses a simple transformer model to generate
    forward and backward action logits as well as a flow.
    """

    encoder: gfnx.networks.Encoder
    pooler: eqx.nn.Linear
    train_backward_policy: bool
    n_fwd_actions: int
    n_bwd_actions: int

    def __init__(
        self,
        n_fwd_actions: int,
        n_bwd_actions: int,
        train_backward_policy: bool,
        encoder_params: dict,
        *,
        key: chex.PRNGKey,
    ):
        self.train_backward_policy = train_backward_policy
        self.n_fwd_actions = n_fwd_actions
        self.n_bwd_actions = n_bwd_actions

        output_size = self.n_fwd_actions + 1  # +1 for flow
        if train_backward_policy:
            output_size += n_bwd_actions

        encoder_key, pooler_key = jax.random.split(key)
        self.encoder = gfnx.networks.Encoder(key=encoder_key, **encoder_params)
        self.pooler = eqx.nn.Linear(
            in_features=encoder_params["hidden_size"],
            out_features=output_size,
            key=pooler_key,
        )

    def __call__(
        self,
        obs_ids: Int[Array, " seq_len"],
        *,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> chex.Array:
        pos_ids = jnp.arange(obs_ids.shape[0])
        encoded_obs = self.encoder(obs_ids, pos_ids, enable_dropout=enable_dropout, key=key)[
            "layers_out"
        ][-1]  # [seq_len, hidden_size]
        encoded_obs = encoded_obs.mean(axis=0)  # Average pooling
        output = self.pooler(encoded_obs)
        if self.train_backward_policy:
            # The TB loss does not use the flow term from the policy.
            # We expect fwd_logits and bwd_logits only.
            # So, we will ignore the flow term here.
            fwd_logits, _, bwd_logits = jnp.split(
                output, [self.n_fwd_actions, self.n_fwd_actions + 1], axis=-1
            )
        else:
            # Similarly, ignore flow if not training backward policy.
            fwd_logits, _ = jnp.split(output, [self.n_fwd_actions], axis=-1)
            bwd_logits = jnp.zeros(shape=(self.n_bwd_actions,), dtype=jnp.float32)
        return {
            "forward_logits": fwd_logits,
            "backward_logits": bwd_logits,
        }


# Define the train state that will be used in the training loop
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.AMPEnvironment
    env_params: chex.Array
    reward_module: gfnx.EqxProxyAMPRewardModule
    reward_params: chex.Array
    model: TransformerPolicy
    logZ: chex.Array  # Added logZ here
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    metrics_module: MultiMetricsModule
    metrics_state: MultiMetricsState
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

    # Define the policy function suitable for gfnx.utils.forward_rollout.
    # This function is called per-environment step to get action logits.
    # - fwd_rng_key: PRNGKey for potential stochasticity in the policy
    #   (e.g., dropout).
    # - env_obs: Current environment observation
    #   (for TransformerPolicy, these are obs_ids).
    # - current_policy_params: Learnable parameters of the policy network.
    def fwd_policy_fn(
        fwd_rng_key: chex.PRNGKey,
        env_obs: gfnx.TObs,
        current_policy_params,
    ) -> chex.Array:
        current_model = eqx.combine(current_policy_params, policy_static)

        dropout_key, exploration_key = jax.random.split(fwd_rng_key)
        policy_outputs = current_model(env_obs, enable_dropout=True, key=dropout_key)
        # Apply epsilon exploration to logits
        do_explore = jax.random.bernoulli(exploration_key, cur_epsilon)
        forward_logits = jnp.where(do_explore, 0, policy_outputs["forward_logits"])
        return forward_logits, policy_outputs

    # Generate a batch of trajectories using the defined forward policy.
    # policy_params here are the learnable parameters of train_state.model.
    rng_keys = jax.random.split(sample_traj_key, num_envs)
    traj_data, final_states, info = jax.vmap(
        lambda rng: gfnx.utils.forward_rollout(rng, fwd_policy_fn, policy_params, env, env_params)
    )(rng_keys)
    # Compute log rewards on terminal states
    log_rewards = jax.vmap(train_state.reward_module.log_reward, in_axes=(0, None))(
        final_states, train_state.reward_params
    )
    # Compute the RL reward / ELBO (for logging purposes)
    log_pb_traj = jax.vmap(
        lambda td: gfnx.utils.forward_trajectory_log_probs(env, td, env_params)
    )(traj_data)[1]
    rl_reward = log_pb_traj + log_rewards + info["entropy"]

    # Step 2. Compute the loss.
    # The loss_fn takes all learnable parameters (model and logZ)
    # and static model parts.
    def loss_fn(
        current_all_params: dict,
        static_model_parts: TransformerPolicy,
        current_traj_data: gfnx.utils.TrajectoryData,
        current_env: gfnx.AMPEnvironment,
        current_env_params: gfnx.AMPEnvParams,
        current_log_rewards: jnp.ndarray,
    ):
        # Extract model's learnable parameters and logZ from the input
        model_learnable_params = current_all_params["model_params"]
        logZ_val = current_all_params["logZ"]

        # Reconstruct the callable model using its learnable parameters
        model_to_call = eqx.combine(model_learnable_params, static_model_parts)
        dropout_keys = jax.random.split(rng_key, current_traj_data.obs.shape[:2])
        # Get policy outputs for the entire trajectory
        policy_outputs_traj = jax.vmap(
            jax.vmap(
                lambda obs, key: model_to_call(obs, enable_dropout=True, key=key),
            ),
        )(current_traj_data.obs, dropout_keys)

        fwd_logits_traj = policy_outputs_traj["forward_logits"]

        # Calculate forward masks.
        # jax.vmap is used to apply get_invalid_mask over the time dimension 1
        # of current_traj_data.state. Leaves in current_traj_data.state
        # are expected to have shape (batch_size, time, ...).
        invalid_fwd_mask_batch_time_actions = jax.vmap(
            current_env.get_invalid_mask_batch,
            in_axes=(0, None),
        )(current_traj_data.state, current_env_params)
        # Resulting shape: (batch_size, time, num_fwd_actions)

        fwd_logprobs_traj = gfnx.utils.compute_action_log_probs(
            fwd_logits_traj,
            current_traj_data.action,
            invalid_fwd_mask_batch_time_actions,
            current_traj_data.pad,
        )
        sum_log_pf_along_traj = fwd_logprobs_traj.sum(axis=1)
        log_pf_traj = logZ_val + sum_log_pf_along_traj  # Use extracted logZ_val

        # Calculate backward actions.
        # Slicing trajectory data to get seq. of (state, action, next_state).
        # prev_states_for_bwd: states from t=0 to T-1.
        # Shape: (batch, max_len, ...)
        prev_states_for_bwd = jax.tree.map(lambda x: x[:, :-1], current_traj_data.state)
        # fwd_actions_for_bwd: actions from t=0 to T-1.
        # Shape: (batch, max_len)
        fwd_actions_for_bwd = current_traj_data.action[:, :-1]
        # curr_states_for_bwd: states from t=1 to T.
        # Shape: (batch, max_len, ...)
        curr_states_for_bwd = jax.tree.map(lambda x: x[:, 1:], current_traj_data.state)

        # jax.vmap is used to apply get_backward_action
        # over the time dimension (axis 1).
        bwd_actions_traj = jax.vmap(
            current_env.get_backward_action_batch,
            in_axes=(0, 0, 0, None),
        )(
            prev_states_for_bwd,
            fwd_actions_for_bwd,
            curr_states_for_bwd,
            current_env_params,
        )
        # Resulting bwd_actions_traj shape: (batch_size, max_len)
        chex.assert_rank(bwd_actions_traj, 2)

        bwd_logits_traj = policy_outputs_traj["backward_logits"]
        bwd_logits_for_pb = bwd_logits_traj[:, 1:]  # Logits for P_B(s_t+1 | s_t)

        # Calculate backward masks using curr_states_for_bwd
        # (states from t=1 to T).
        # These are the states *from which* backward actions are taken.
        # jax.vmap maps get_invalid_backward_mask over the time dimension (1).
        invalid_bwd_mask_batch_time_actions = jax.vmap(
            current_env.get_invalid_backward_mask_batch,
            in_axes=(0, None),
        )(curr_states_for_bwd, current_env_params)
        # Resulting shape: (batch_size, max_len, num_bwd_actions)
        # Note: max_len here refers to the length of curr_states_for_bwd (T).
        # This matches the length of bwd_logits_for_pb (logits from t=1 to T).

        log_pb_selected = gfnx.utils.compute_action_log_probs(
            bwd_logits_for_pb,
            bwd_actions_traj,
            invalid_bwd_mask_batch_time_actions,
            current_traj_data.pad[:, :-1],
        )
        log_pb_sum = jnp.sum(log_pb_selected, axis=1)
        target = log_pb_sum + current_log_rewards

        return optax.losses.squared_error(log_pf_traj, target).mean()

    # Prepare parameters for the loss function and gradient calculation
    # policy_params are model network parameters
    # policy_static are model static parts.
    params_for_loss = {"model_params": policy_params, "logZ": train_state.logZ}

    mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(
        params_for_loss, policy_static, traj_data, env, env_params, log_rewards
    )

    # Step 3. Update parameters (model network and logZ)
    # `grads` is a dict {'model_params': ..., 'logZ': ...}
    # `optax_params_for_update` should match the structure given
    # to optimizer.init
    optax_params_for_update = {
        "model_params": policy_params,
        "logZ": train_state.logZ,
    }
    updates, new_opt_state = train_state.optimizer.update(
        grads, train_state.opt_state, optax_params_for_update
    )

    # Apply updates
    # updates contains the deltas for the model's learnable parameters.
    new_model = eqx.apply_updates(train_state.model, updates["model_params"])
    new_logZ = eqx.apply_updates(train_state.logZ, updates["logZ"])

    # Get model parameters for evaluation
    current_policy_params = eqx.filter(new_model, eqx.is_array)

    rng_key, eval_rng_key = jax.random.split(rng_key)

    # Perform all the required logging
    metrics_state, eval_info = train_state.metrics_module.step(
        idx=idx,
        metrics_state=train_state.metrics_state,
        rng_key=eval_rng_key,
        update_args=train_state.metrics_module.UpdateArgs(
            metrics_args={"topk": TopKMetricsModule.UpdateArgs()}
        ),
        process_args=train_state.metrics_module.ProcessArgs(
            metrics_args={
                "topk": TopKMetricsModule.ProcessArgs(
                    policy_params=current_policy_params, env_params=env_params
                )
            }
        ),
        eval_each=train_state.config.logging.eval_each,
        num_train_steps=train_state.config.num_train_steps,
        prev_eval_info=train_state.eval_info,
    )

    # Logging via JAX debug callback for train and evaluation info.
    def logging_callback(idx: int, train_info: dict, eval_info: dict, cfg):
        train_info = {f"train/{key}": float(value) for key, value in train_info.items()}

        if idx % cfg.logging.eval_each == 0 or idx + 1 == cfg.num_train_steps:
            log.info(f"Step {idx}")
            log.info(train_info)
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
            "entropy": info["entropy"].mean(),
            "grad_norm": optax.tree_utils.tree_l2_norm(grads),
            "logZ": new_logZ,
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
        logZ=new_logZ,
        opt_state=new_opt_state,
        metrics_state=metrics_state,
        eval_info=eval_info,
    )


@hydra.main(config_path="configs/", config_name="tb_amp", version_base=None)
def run_experiment(cfg: OmegaConf) -> None:
    # Log the configuration
    log.info(OmegaConf.to_yaml(cfg))

    rng_key = jax.random.PRNGKey(cfg.seed)
    env_init_key = jax.random.PRNGKey(cfg.env_init_seed)
    eval_init_key = jax.random.PRNGKey(cfg.eval_init_seed)

    # Define the reward function for the environment
    reward_module = gfnx.EqxProxyAMPRewardModule(
        proxy_config_path=cfg.environment.proxy_config_path,
        pretrained_proxy_path=cfg.environment.pretrained_proxy_path,
        reward_exponent=cfg.environment.reward_exponent,
        min_reward=cfg.environment.min_reward,
    )
    # Initialize the environment and its inner parameters
    env = gfnx.AMPEnvironment()
    env_params = env.init(env_init_key)
    reward_params = reward_module.init(env_init_key, env.get_init_state())

    rng_key, net_init_key = jax.random.split(rng_key)
    # Initialize the network
    model = TransformerPolicy(
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        train_backward_policy=cfg.agent.train_backward,
        encoder_params={
            "pad_id": env.pad_token,
            "vocab_size": env.ntoken,
            "max_length": env.max_length + 1,  # +1 for BOS token
            **OmegaConf.to_container(cfg.network),
        },
        key=net_init_key,
    )
    # Initialize the exploration schedule
    exploration_schedule = optax.linear_schedule(
        init_value=cfg.agent.start_eps,
        end_value=cfg.agent.end_eps,
        transition_steps=cfg.agent.exploration_steps,
    )

    # Initialize logZ separately
    logZ = jnp.array(150.0)

    # Prepare parameters for Optax
    model_params_init = eqx.filter(model, eqx.is_array)
    initial_optax_params = {"model_params": model_params_init, "logZ": logZ}

    # Define parameter labels for multi_transform
    param_labels = {
        "model_params": jax.tree.map(lambda _: "network_lr", model_params_init),
        "logZ": "logZ_lr",
    }

    optimizer_defs = {
        "network_lr": optax.adamw(
            learning_rate=cfg.agent.learning_rate, weight_decay=cfg.agent.weight_decay
        ),
        "logZ_lr": optax.adam(learning_rate=cfg.agent.logZ_learning_rate),
    }
    optimizer = optax.multi_transform(optimizer_defs, param_labels)
    opt_state = optimizer.init(initial_optax_params)

    # Initialize the policy function for metrics computation
    # This requires policy_static part of the model.
    _, policy_static_for_metrics = eqx.partition(model, eqx.is_array)

    def fwd_policy_fn_for_metrics(
        fwd_rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params
    ) -> chex.Array:
        # Recombine the network parameters with the static parts of the model
        current_model_for_metrics = eqx.combine(policy_params, policy_static_for_metrics)
        # The TransformerPolicy expects obs_ids.
        # Dropout is disabled for metrics, so no key is needed.
        del fwd_rng_key  # fwd_rng_key is not used when dropout is disabled
        policy_outputs_for_metrics = current_model_for_metrics(env_obs, enable_dropout=False)
        return policy_outputs_for_metrics["forward_logits"], policy_outputs_for_metrics

    def amp_distance_fn(lhs_state: gfnx.AMPEnvState, rhs_state: gfnx.AMPEnvState) -> chex.Array:
        """Compute the distance between two AMP states."""
        return gfnx.utils.distances.levenshtein_distance(
            lhs_state.tokens, rhs_state.tokens, eos_id=env.eos_token, pad_id=env.pad_token
        )

    metrics_module = MultiMetricsModule({
        "topk": TopKMetricsModule(
            fwd_policy_fn=fwd_policy_fn_for_metrics,
            env=env,
            num_traj=cfg.metrics.num_traj,
            batch_size=cfg.metrics.batch_size,  # Ignored for a moment
            top_k=[10, 50, 100],
            distance_fn=amp_distance_fn,
        )
    })
    metrics_state = metrics_module.init(
        eval_init_key,
        metrics_module.InitArgs(metrics_args={"topk": TopKMetricsModule.InitArgs()}),
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
        logZ=logZ,
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
            tags=["TB", env.name.upper()],
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
    save_checkpoint(
        os.path.join(dir, "model_and_logZ"),
        {
            "model": final_train_state.model,
            "logZ": final_train_state.logZ,
        },
    )


if __name__ == "__main__":
    run_experiment()
