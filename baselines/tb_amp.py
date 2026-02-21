"""Single-file implementation for Trajectory Balance in AMP environment.

Run the script with the following command:
```bash
python baselines/tb_amp.py
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
import optax
from jax_tqdm import loop_tqdm
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

        num_samples = env_obs.shape[0]
        dropout_key, exploration_key = jax.random.split(fwd_rng_key)
        dropout_keys = jax.random.split(dropout_key, num_samples)
        policy_outputs = jax.vmap(
            lambda obs, dkey: current_model(obs, enable_dropout=True, key=dkey),
            in_axes=(0,0),
        )(env_obs, dropout_keys)
        # Apply epsilon exploration to logits
        do_explore = jax.random.bernoulli(exploration_key, cur_epsilon, shape=(env_obs.shape[0],))
        forward_logits = jnp.where(
            do_explore[..., jnp.newaxis], 0, policy_outputs["forward_logits"]
        )
        return forward_logits, policy_outputs

    # Generate a batch of trajectories using the defined forward policy.
    # policy_params here are the learnable parameters of train_state.model.
    traj_data, aux_info = gfnx.utils.forward_rollout(
        rng_key=sample_traj_key,
        num_envs=num_envs,
        policy_fn=fwd_policy_fn,
        policy_params=policy_params,
        env=env,
        env_params=env_params,
    )
    # Compute the RL reward / ELBO (for logging purposes)
    _, log_pb_traj = gfnx.utils.forward_trajectory_log_probs(
        env, traj_data, env_params
    )
    rl_reward = log_pb_traj + aux_info["log_gfn_reward"] + aux_info["entropy"]

    # Step 2. Compute the loss.
    # The loss_fn takes all learnable parameters (model and logZ)
    # and static model parts.
    def loss_fn(
        current_all_params: dict,
        static_model_parts: TransformerPolicy,
        current_traj_data: gfnx.utils.TrajectoryData,
        current_env: gfnx.AMPEnvironment,
        current_env_params: gfnx.AMPEnvParams,
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
            current_env.get_invalid_mask,
            in_axes=(
                1,
                None,
            ),  # Map over axis 1 of states, env_params is fixed.
            out_axes=1,  # Output mask also has time as axis 1.
        )(current_traj_data.state, current_env_params)
        # Resulting shape: (batch_size, time, num_fwd_actions)

        masked_fwd_logits_traj = gfnx.utils.mask_logits(
            fwd_logits_traj, invalid_fwd_mask_batch_time_actions
        )
        fwd_all_log_probs_traj = jax.nn.log_softmax(masked_fwd_logits_traj, axis=-1)

        fwd_logprobs_traj = jnp.take_along_axis(
            fwd_all_log_probs_traj,
            jnp.expand_dims(current_traj_data.action, axis=-1),
            axis=-1,
        ).squeeze(-1)

        fwd_logprobs_traj = jnp.where(current_traj_data.pad, 0.0, fwd_logprobs_traj)
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
            current_env.get_backward_action,
            in_axes=(
                1,
                1,
                1,
                None,
            ),  # Map over time axis of states and actions.
            out_axes=1,  # Output also has time as axis 1.
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
            current_env.get_invalid_backward_mask,
            in_axes=(1, None),  # Map over time axis of states.
            out_axes=1,  # Output also has time as axis 1.
        )(curr_states_for_bwd, current_env_params)
        # Resulting shape: (batch_size, max_len, num_bwd_actions)
        # Note: max_len here refers to the length of curr_states_for_bwd (T).
        # This matches the length of bwd_logits_for_pb (logits from t=1 to T).

        masked_bwd_logits_traj = gfnx.utils.mask_logits(
            bwd_logits_for_pb, invalid_bwd_mask_batch_time_actions
        )
        bwd_all_log_probs_traj = jax.nn.log_softmax(masked_bwd_logits_traj, axis=-1)

        log_pb_selected = jnp.take_along_axis(
            bwd_all_log_probs_traj,
            jnp.expand_dims(bwd_actions_traj, axis=-1),  # Unconditionally expand
            axis=-1,
        ).squeeze(-1)

        pad_mask_for_bwd = current_traj_data.pad[:, :-1]
        log_pb_selected = jnp.where(pad_mask_for_bwd, 0.0, log_pb_selected)

        log_rewards_at_steps = current_traj_data.log_gfn_reward[:, :-1]
        masked_log_rewards_at_steps = jnp.where(pad_mask_for_bwd, 0.0, log_rewards_at_steps)

        log_pb_plus_rewards_along_traj = log_pb_selected + masked_log_rewards_at_steps
        target = jnp.sum(log_pb_plus_rewards_along_traj, axis=1)

        return optax.losses.squared_error(log_pf_traj, target).mean()

    # Prepare parameters for the loss function and gradient calculation
    # policy_params are model network parameters
    # policy_static are model static parts.
    params_for_loss = {"model_params": policy_params, "logZ": train_state.logZ}

    mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(
        params_for_loss, policy_static, traj_data, env, env_params
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

    # Perform metrics updates using the new MultiMetricsModule.
    metrics_state = train_state.metrics_module.update(
        train_state.metrics_state,
        rng_key=jax.random.key(0),  # not used, but required by the API
        args=train_state.metrics_module.UpdateArgs(
            metrics_args={"topk": TopKMetricsModule.UpdateArgs()}
        ),
    )

    rng_key, eval_rng_key = jax.random.split(rng_key)
    # Perform evaluation computations if needed
    is_eval_step = idx % train_state.config.logging.eval_each == 0
    is_eval_step = is_eval_step | (idx + 1 == train_state.config.num_train_steps)

    # Get model parameters for evaluation
    current_policy_params = eqx.filter(new_model, eqx.is_array)

    metrics_state = jax.lax.cond(
        is_eval_step,
        lambda kwargs: train_state.metrics_module.process(**kwargs),
        lambda kwargs: kwargs["metrics_state"],  # Do nothing if not eval step
        {
            "metrics_state": metrics_state,
            "rng_key": eval_rng_key,
            "args": train_state.metrics_module.ProcessArgs(
                metrics_args={
                    "topk": TopKMetricsModule.ProcessArgs(
                        policy_params=current_policy_params, env_params=env_params
                    )
                }
            ),
        },
    )
    eval_info = jax.lax.cond(
        is_eval_step,
        lambda metrics_state: train_state.metrics_module.get(metrics_state),
        lambda metrics_state: train_state.eval_info,  # Do nothing if not eval step
        metrics_state,
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
            "entropy": aux_info["entropy"].mean(),
            "grad_norm": optax.tree_utils.tree_l2_norm(grads),
            "logZ": new_logZ,
            "mean_reward": jnp.exp(aux_info["log_gfn_reward"]).mean(),
            "mean_log_reward": aux_info["log_gfn_reward"].mean(),
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
    env = gfnx.AMPEnvironment(reward_module)
    env_params = env.init(env_init_key)

    rng_key, net_init_key = jax.random.split(rng_key)
    # Initialize the network
    model = TransformerPolicy(
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        train_backward_policy=cfg.agent.train_backward,
        encoder_params={
            "pad_id": env.pad_token,
            "vocab_size": env.ntoken,
            "max_length": env.max_length + 1, # +1 for BOS token
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
            learning_rate=cfg.agent.learning_rate,
            weight_decay=cfg.agent.weight_decay
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
        policy_outputs_for_metrics = jax.vmap(
            lambda model, obs: model(obs, enable_dropout=False),
            in_axes=(
                None,
                0,
            ),  # current_model_for_metrics is fixed, map over env_obs
        )(current_model_for_metrics, env_obs)
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
        model=model,
        logZ=logZ,
        optimizer=optimizer,
        opt_state=opt_state,
        metrics_module=metrics_module,
        metrics_state=metrics_state,
        exploration_schedule=exploration_schedule,
        eval_info=eval_info,
    )

    # Partition the initial TrainState into dynamic (jittable) and static parts
    train_state_params, train_state_static = eqx.partition(train_state, eqx.is_array)

    @functools.partial(jax.jit, donate_argnums=(1,))  # train_state_params is arg 1 (0-indexed)
    @loop_tqdm(cfg.num_train_steps, print_rate=cfg.logging["tqdm_print_rate"])
    def train_step_wrapper(idx: int, current_train_state_params) -> TrainState:  # Input is params
        # Recombine static and dynamic parts to get the full TrainState
        current_train_state = eqx.combine(current_train_state_params, train_state_static)
        # Call the original JITted train_step
        updated_train_state = train_step(idx, current_train_state)
        # Partition again before returning for the next iteration of the loop
        new_train_state_params, _ = eqx.partition(updated_train_state, eqx.is_array)
        return new_train_state_params

    # Initial train_state_params for the loop
    loop_init_val = train_state_params

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
            tags=["TB", env.name.upper()],
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    log.info("Start training")
    # Run the training loop via jax lax.fori_loop
    final_train_state_params = jax.lax.fori_loop(  # Result will be params
        lower=0,
        upper=cfg.num_train_steps,
        body_fun=train_step_wrapper,  # body_fun now expects and returns params
        init_val=loop_init_val,  # Pass only the JAX array parts
    )
    jax.block_until_ready(final_train_state_params)

    # Save the final model
    final_train_state = eqx.combine(final_train_state_params, train_state_static)
    dir = cfg.logging.checkpoint_dir if cfg.logging.checkpoint_dir else os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        f"checkpoints_{os.getpid()}/",
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
