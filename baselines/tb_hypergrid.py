"""Single-file implementation for Trajectory Balance in hypergrid environment.

Run the script with the following command:
```bash
python baselines/tb_hypergrid.py
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
import orbax.checkpoint as ocp
import wandb
from jax_tqdm import loop_tqdm
from omegaconf import OmegaConf

import gfnx

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class MLPPolicy(eqx.Module):
    """
    A policy module that uses a Multi-Layer Perceptron (MLP) to generate
    forward and backward action logits. It also includes logZ, the log-partition function.

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

        output_size = self.n_fwd_actions
        if train_backward_policy:
            output_size += n_bwd_actions
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
            forward_logits, backward_logits = jnp.split(
                x, [self.n_fwd_actions], axis=-1
            )
        else:
            forward_logits = x
            backward_logits = jnp.zeros(
                shape=(self.n_bwd_actions,), dtype=jnp.float32
            )
        return {
            "forward_logits": forward_logits,
            "backward_logits": backward_logits,
        }


# Define the train state that will be used in the training loop
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.HypergridEnvironment
    env_params: chex.Array
    model: MLPPolicy
    logZ: chex.Array  # Added logZ here
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    metrics_module: gfnx.metrics.HypergridMetricModule
    metrics: gfnx.metrics.HypergridMetricState


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
    
    # Define the policy function suitable for gfnx.utils.forward_rollout
    # Note: policy_params for this function are only the MLPPolicy's network parameters
    def fwd_policy_fn(
        fwd_rng_key: chex.PRNGKey, env_obs: gfnx.TObs, current_policy_params # current_policy_params are model network params
    ) -> chex.Array:
        del fwd_rng_key
        # Recombine the network parameters with the static parts of the model to form a callable MLPPolicy
        current_model = eqx.combine(current_policy_params, policy_static)
        policy_outputs = jax.vmap(current_model, in_axes=(0,))(env_obs)
        return policy_outputs["forward_logits"], policy_outputs

    traj_data, log_info = gfnx.utils.forward_rollout(
        rng_key=sample_traj_key,
        num_envs=num_envs,
        policy_fn=fwd_policy_fn,
        policy_params=policy_params, # Pass only network parameters
        env=env,
        env_params=env_params,
    )

    # Step 2. Compute the loss
    # loss_fn now takes combined parameters (model_params and logZ)
    # It also needs static parts of the model to reconstruct it.
    def loss_fn(current_all_params: dict, static_model_parts: MLPPolicy, 
                  current_traj_data: gfnx.utils.TrajectoryData, # Corrected type hint
                  current_env: gfnx.HypergridEnvironment, 
                  current_env_params: chex.Array):
        
        # Extract model's learnable parameters and logZ from the input dictionary
        model_learnable_params = current_all_params['model_params']
        logZ_val = current_all_params['logZ']

        # Reconstruct the callable model using its learnable parameters and static parts
        model_to_call = eqx.combine(model_learnable_params, static_model_parts)
        
        policy_outputs_traj = jax.vmap(jax.vmap(model_to_call))(current_traj_data.obs)

        fwd_logits_traj = policy_outputs_traj["forward_logits"]
        
        def get_fwd_masks_per_step(states_at_t_step, env_params_):
            return current_env.get_invalid_mask(states_at_t_step, env_params_)

        # Transpose state PyTree from (batch, time, features) to (time, batch, features)
        # to allow vmapping over the time dimension for mask calculation.
        state_transposed_for_fwd_masking = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), current_traj_data.state)
        # Vmap get_fwd_masks_per_step over the time dimension. For each time step t,
        # it processes the batch of states state[t, :, ...].
        invalid_fwd_mask_transposed = jax.vmap(get_fwd_masks_per_step, in_axes=(0, None))(state_transposed_for_fwd_masking, current_env_params)
        # Transpose the resulting mask (time, batch, actions) back to (batch, time, actions).
        invalid_fwd_mask_batch_time_actions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), invalid_fwd_mask_transposed)

        masked_fwd_logits_traj = gfnx.utils.mask_logits(fwd_logits_traj, invalid_fwd_mask_batch_time_actions)
        fwd_all_log_probs_traj = jax.nn.log_softmax(masked_fwd_logits_traj, axis=-1)
        
        fwd_logprobs_traj = jnp.take_along_axis(
            fwd_all_log_probs_traj, jnp.expand_dims(current_traj_data.action, axis=-1), axis=-1
        ).squeeze(-1)
        
        fwd_logprobs_traj = jnp.where(current_traj_data.pad, 0.0, fwd_logprobs_traj)
        sum_log_pf_along_traj = fwd_logprobs_traj.sum(axis=1)
        log_pf_traj = logZ_val + sum_log_pf_along_traj # Use extracted logZ_val

        def get_bwd_actions_for_traj(single_traj_state, single_traj_action, single_traj_pad):
            max_len = single_traj_state.state.shape[0] # Assuming .state.shape[0] is the fix
            
            def body_fn(carry, t):
                prev_state_t = jax.tree.map(lambda x: x[t - 1], single_traj_state)
                state_t = jax.tree.map(lambda x: x[t], single_traj_state)
                fwd_action_taken = single_traj_action[t - 1]
                bwd_action_t = current_env.get_backward_action(
                    state_t, fwd_action_taken, prev_state_t, current_env_params
                )
                return carry, bwd_action_t
            ts = jnp.arange(1, max_len)
            _, bwd_actions_one_traj = jax.lax.scan(body_fn, None, ts)
            return bwd_actions_one_traj

        bwd_actions_traj = jax.vmap(get_bwd_actions_for_traj)(current_traj_data.state, current_traj_data.action, current_traj_data.pad)
        
        bwd_logits_traj = policy_outputs_traj["backward_logits"]
        bwd_logits_for_pb = bwd_logits_traj[:, 1:]

        def get_bwd_masks_per_step(states_at_t_step, env_params_):
            return current_env.get_invalid_backward_mask(states_at_t_step, env_params_)
        
        states_for_bwd_mask = jax.tree.map(lambda x: x[:, 1:], current_traj_data.state)
        # Transpose state PyTree (batch, time-1, features) to (time-1, batch, features)
        # for vmapping over the time dimension (excluding initial state).
        state_transposed_for_bwd_masking = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), states_for_bwd_mask)
        # Vmap get_bwd_masks_per_step over the time dimension.
        invalid_bwd_mask_transposed = jax.vmap(get_bwd_masks_per_step, in_axes=(0, None))(state_transposed_for_bwd_masking, current_env_params)
        # Transpose the resulting mask (time-1, batch, actions) back to (batch, time-1, actions).
        invalid_bwd_mask_batch_time_actions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), invalid_bwd_mask_transposed)
        
        masked_bwd_logits_traj = gfnx.utils.mask_logits(bwd_logits_for_pb, invalid_bwd_mask_batch_time_actions)
        bwd_all_log_probs_traj = jax.nn.log_softmax(masked_bwd_logits_traj, axis=-1)
        
        log_pb_selected = jnp.take_along_axis(
            bwd_all_log_probs_traj, jnp.expand_dims(bwd_actions_traj, axis=-1), axis=-1
        ).squeeze(-1)
        
        pad_mask_for_bwd = current_traj_data.pad[:, :-1]
        log_pb_selected = jnp.where(pad_mask_for_bwd, 0.0, log_pb_selected)

        log_rewards_at_steps = current_traj_data.log_gfn_reward[:, :-1]
        masked_log_rewards_at_steps = jnp.where(pad_mask_for_bwd, 0.0, log_rewards_at_steps)
        
        log_pb_plus_rewards_along_traj = log_pb_selected + masked_log_rewards_at_steps
        target = jnp.sum(log_pb_plus_rewards_along_traj, axis=1) 
        
        loss = optax.losses.squared_error(log_pf_traj, target).mean()
        # Return log_info from outer scope, not the one passed into loss_fn
        return loss, log_info 

    # Prepare parameters for the loss function and gradient calculation
    # policy_params are model network parameters, policy_static are model static parts.
    params_for_loss = {'model_params': policy_params, 'logZ': train_state.logZ}

    (mean_loss, aux_info), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(params_for_loss, policy_static, traj_data, env, env_params) # Pass static parts and other args

    # Step 3. Update parameters (model network and logZ)
    # `grads` is a dict {'model_params': ..., 'logZ': ...}
    # `optax_params_for_update` should match the structure given to optimizer.init
    optax_params_for_update = {'model_params': policy_params, 'logZ': train_state.logZ}
    updates, new_opt_state = train_state.optimizer.update(grads, train_state.opt_state, optax_params_for_update)
    
    # Apply updates
    # updates['model_params'] contains the deltas for the model's learnable parameters.
    # train_state.model contains the model with its parameters *before* this update.
    new_model = eqx.apply_updates(train_state.model, updates['model_params'])
    
    # Correctly apply the updates to logZ.
    # For standard optimizers like Adam, updates['logZ'] is the delta.
    # optax.apply_updates adds this delta to the current train_state.logZ.
    new_logZ = optax.apply_updates(train_state.logZ, updates['logZ'])
    
    # Perform all the required logging
    metric_state = train_state.metrics_module.update(
        train_state.metrics, aux_info["final_env_state"], env_params
    )
    # Compute the evaluation info if needed
    eval_info = jax.lax.cond(
        (idx % train_state.config.logging.track_each == 0)
        | (idx + 1 == train_state.config.num_train_steps),
        train_state.metrics_module.get,
        lambda x: {
            "tv": -1.0,
            "kl": -1.0,
            "empirical_distribution": x.true_dist,
        },
        metric_state,
    )

    # Perform the logging via JAX debug callback
    # logging_callback now accepts the OmegaConf object directly
    def logging_callback(idx: int, train_info: dict, eval_info: dict, 
                         cfg_from_train_state: OmegaConf): # Reverted signature
        if idx % cfg_from_train_state.logging.track_each == 0 or idx + 1 == cfg_from_train_state.num_train_steps:
            log.info(f"Step {idx}")
            log.info({key: float(value) for key, value in train_info.items()})
            log.info({
                key: float(value)
                for key, value in eval_info.items()
                if key != "eval/empirical_distribution"
            })
            if cfg_from_train_state.logging.use_wandb: # Use cfg object
                empirical_dist = eval_info["eval/empirical_distribution"]
                ndim = len(empirical_dist.shape)
                index_tuple = (slice(None), slice(None), *([0] * (ndim - 2)))
                # Take a slice of the empirical distribution to visualize it
                empirical_dist = empirical_dist[index_tuple]
                empirical_dist = (empirical_dist - empirical_dist.min()) / (
                    empirical_dist.max() - empirical_dist.min()
                )
                eval_info["eval/empirical_distribution"] = wandb.Image(
                    np.array(255.0 * empirical_dist, dtype=np.int32)
                )
                wandb.log(eval_info, commit=False)

        if cfg_from_train_state.logging.use_wandb: # Use cfg object
            wandb.log(train_info)

    jax.debug.callback(
        logging_callback,
        idx,
        {
            "train/mean_loss": mean_loss,
            "train/entropy": aux_info["entropy"].mean(), 
            "train/grad_norm": optax.tree_utils.tree_l2_norm(grads), 
            "train/logZ": new_logZ, 
        },
        {f"eval/{key}": value for key, value in eval_info.items()},
        train_state.config, # Pass OmegaConf object directly
        ordered=True,
    )

    # Return the updated train state
    return train_state._replace(
        rng_key=rng_key, model=new_model, logZ=new_logZ, opt_state=new_opt_state, metrics=metric_state
    )


@hydra.main(
    config_path="configs/", config_name="tb_hypergrid", version_base=None
)
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
    reward_module = reward_module_factory()

    env = gfnx.environment.HypergridEnvironment(
        reward_module, dim=cfg.environment.dim, side=cfg.environment.side
    )
    env_params = env.init(env_init_key)

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

    # Initialize logZ separately
    logZ = jnp.array(0.0)

    # Prepare parameters for Optax
    model_params_init = eqx.filter(model, eqx.is_array)
    initial_optax_params = {'model_params': model_params_init, 'logZ': logZ}

    # Define parameter labels for multi_transform
    param_labels = {
        'model_params': jax.tree.map(lambda _: "network_lr", model_params_init),
        'logZ': "logZ_lr"
    }
    
    optimizer_defs = {
        "network_lr": optax.adam(learning_rate=cfg.agent.learning_rate),
        "logZ_lr": optax.adam(learning_rate=cfg.agent.logZ_learning_rate)
    }
    optimizer = optax.multi_transform(optimizer_defs, param_labels)
    opt_state = optimizer.init(initial_optax_params)

    metrics_module = gfnx.metrics.HypergridMetricModule(
        env, buffer_max_length=cfg.logging.metric_buffer_size
    )
    metrics = metrics_module.init(eval_init_key, env_params)

    train_state = TrainState(
        rng_key=rng_key,
        config=cfg, # Use the original OmegaConf object
        env=env,
        env_params=env_params,
        model=model,
        logZ=logZ, # Add logZ to TrainState
        optimizer=optimizer,
        opt_state=opt_state,
        metrics_module=metrics_module,
        metrics=metrics,
    )
    
    # Partition the initial TrainState into dynamic (JAX arrays) and static parts
    train_state_params, train_state_static = eqx.partition(train_state, eqx.is_array)

    @functools.partial(jax.jit, donate_argnums=(1,)) # train_state_params is arg 1 (0-indexed)
    @loop_tqdm(cfg.num_train_steps, print_rate=cfg.logging["tqdm_print_rate"])
    def train_step_wrapper(idx: int, current_train_state_params) -> TrainState: # Input is params
        # Recombine static and dynamic parts to get the full TrainState
        current_train_state = eqx.combine(current_train_state_params, train_state_static)
        # Call the original JITted train_step
        updated_train_state = train_step(idx, current_train_state)
        # Partition again before returning for the next iteration of the loop
        new_train_state_params, _ = eqx.partition(updated_train_state, eqx.is_array)
        return new_train_state_params

    # Initial train_state_params for the loop
    loop_init_val = train_state_params

    if cfg.logging.use_wandb:
        log.info("Initialize wandb")
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            tags=["TB", env.name.upper()], # Updated to TB
        )
        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

    log.info("Start training")
    # Run the training loop via jax lax.fori_loop
    final_train_state_params = jax.lax.fori_loop( # Result will be params
        lower=0,
        upper=cfg.num_train_steps,
        body_fun=train_step_wrapper, # body_fun now expects and returns params
        init_val=loop_init_val, # Pass only the JAX array parts
    )
    jax.block_until_ready(final_train_state_params)

    # Reconstruct the final TrainState from the dynamic and static parts
    final_train_state = eqx.combine(final_train_state_params, train_state_static)

    # Save the final model and logZ
    # We need to save model parameters and logZ.
    # Orbax can save PyTrees; initial_optax_params is a good structure.
    final_model_params = eqx.filter(final_train_state.model, eqx.is_array)
    final_params_to_save = {'model_params': final_model_params, 'logZ': final_train_state.logZ}
    
    path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    cwd = os.path.join(path, "model_and_logZ") # Changed path name
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    # Save the dictionary containing model parameters and logZ
    ckptr.save(cwd, args=ocp.args.StandardSave(final_params_to_save))
    ckptr.wait_until_finished()


if __name__ == "__main__":
    run_experiment()
