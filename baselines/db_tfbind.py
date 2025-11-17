"""Single-file implementation for Detailed Balance in TFBind-8 environment.

Run the script with the following command:
```bash
python baselines/db_tfbind.py
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

import gfnx
from gfnx.metrics import (
    ApproxDistributionMetricsModule,
    MultiMetricsModule,
    MultiMetricsState,
    SWMeanRewardSWMetricsModule,
)

from utils.logger import Writer
from utils.checkpoint import save_checkpoint

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
writer = Writer()


class MLPPolicy(eqx.Module):
    """
    A policy module that uses a Multi-Layer Perceptron (MLP) to generate
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
        self.encoder = eqx.nn.MLP(
            in_size=5 * 8,
            out_size=encoder_params["hidden_size"],
            width_size=encoder_params["hidden_size"],
            depth=encoder_params["depth"],
            key=encoder_key,
        )
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
        obs_ids = jax.nn.one_hot(obs_ids[1:], 5).reshape(-1)
        encoded_obs = self.encoder(obs_ids)
        output = self.pooler(encoded_obs)
        if self.train_backward_policy:
            forward_logits, flow, backward_logits = jnp.split(
                output, [self.n_fwd_actions, self.n_fwd_actions + 1], axis=-1
            )
        else:
            forward_logits, flow = jnp.split(output, [self.n_fwd_actions], axis=-1)
            backward_logits = jnp.zeros(shape=(self.n_bwd_actions,), dtype=jnp.float32)
        return {
            "forward_logits": forward_logits,
            "log_flow": flow.squeeze(-1),
            "backward_logits": backward_logits,
        }


# Define the train state that will be used in the training loop
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.TFBind8Environment
    env_params: chex.Array
    model: MLPPolicy
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
    # Step 1. Generate a batch of trajectories and split to transitions
    rng_key, sample_traj_key = jax.random.split(train_state.rng_key)
    # Split the model to pass into forward rollout
    policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)

    cur_eps = train_state.exploration_schedule(idx)

    # Define the policy function suitable for gfnx.utils.forward_rollout
    def fwd_policy_fn(
        fwd_rng_key: chex.PRNGKey,
        env_obs: gfnx.TObs,
        current_policy_params,  # current_policy_params are network params
        train=True,
    ) -> chex.Array:
        # Recombine the network parameters with the static parts of the model
        current_model = eqx.combine(current_policy_params, policy_static)
        policy_outputs = jax.vmap(current_model, in_axes=(0,))(env_obs)

        # Get forward logits
        fwd_logits = policy_outputs["forward_logits"]

        # Apply epsilon exploration to logits
        batch_size = fwd_logits.shape[0]
        exploration_mask = jax.random.bernoulli(fwd_rng_key, cur_eps, (batch_size,))
        fwd_logits = jnp.where(exploration_mask[..., jnp.newaxis], 0, fwd_logits)
        # Update policy outputs with modified logits
        return fwd_logits, policy_outputs

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
    _, log_pb_traj = gfnx.utils.forward_trajectory_log_probs(
        env, traj_data, env_params
    )
    rl_reward = log_pb_traj + log_info["log_gfn_reward"] + log_info["entropy"]

    # Step 2. Compute the loss
    def loss_fn(model: MLPPolicy) -> chex.Array:
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

        # Compute the stats for the next state
        next_policy_outputs = jax.vmap(model, in_axes=(0,))(transitions.next_obs)
        bwd_logits = next_policy_outputs["backward_logits"]
        next_bwd_invalid_mask = env.get_invalid_backward_mask(transitions.next_state, env_params)
        masked_bwd_logits = gfnx.utils.mask_logits(bwd_logits, next_bwd_invalid_mask)
        bwd_all_log_probs = jax.nn.log_softmax(masked_bwd_logits, axis=-1)
        bwd_logprobs = jnp.take_along_axis(
            bwd_all_log_probs, jnp.expand_dims(bwd_actions, axis=-1), axis=-1
        ).squeeze(-1)
        next_log_flow = next_policy_outputs["log_flow"]
        # Replace the target with the log_gfn_reward if the episode is done
        target = jnp.where(
            transitions.done,
            bwd_logprobs + transitions.log_gfn_reward,
            bwd_logprobs + next_log_flow,
        )

        # Compute the DB loss with masking
        num_transition = jnp.logical_not(transitions.pad).sum()
        loss = optax.l2_loss(
            jnp.where(transitions.pad, 0.0, fwd_logprobs + log_flow),
            jnp.where(transitions.pad, 0.0, target),
        ).sum()
        return loss / num_transition

    mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(train_state.model)
    # Step 3. Update the model with grads
    updates, opt_state = train_state.optimizer.update(
        grads,
        train_state.opt_state,
        eqx.filter(train_state.model, eqx.is_array),
    )
    model = eqx.apply_updates(train_state.model, updates)
    # Peform all the requied logging
    rewards = jnp.exp(log_info["log_gfn_reward"])
    metrics_state = train_state.metrics_module.update(
        train_state.metrics_state,
        rng_key=jax.random.key(0),  # not used, but required by the API
        args=train_state.metrics_module.UpdateArgs(
            metrics_args={
                "distribution": ApproxDistributionMetricsModule.UpdateArgs(
                    states=log_info["final_env_state"]
                ),
                "rd": SWMeanRewardSWMetricsModule.UpdateArgs(
                    rewards=rewards,
                ),
            }
        ),
    )

    rng_key, eval_rng_key = jax.random.split(rng_key)
    # Perform evaluation computations if needed
    is_eval_step = idx % train_state.config.logging.eval_each == 0
    is_eval_step = is_eval_step | (idx + 1 == train_state.config.num_train_steps)

    metrics_state = jax.lax.cond(
        is_eval_step,
        lambda kwargs: train_state.metrics_module.process(**kwargs),
        lambda kwargs: kwargs["metrics_state"],  # Do nothing if not eval step
        {
            "metrics_state": metrics_state,
            "rng_key": eval_rng_key,
            "args": train_state.metrics_module.ProcessArgs(
                metrics_args={
                    "distribution": ApproxDistributionMetricsModule.ProcessArgs(
                        env_params=env_params
                    ),
                    "rd": SWMeanRewardSWMetricsModule.ProcessArgs(),
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

    # Perform the logging via JAX debug callback
    def logging_callback(
        idx: int, train_info: dict, eval_info: dict, cfg
    ):
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
            "entropy": log_info["entropy"].mean(),
            "grad_norm": optax.tree_utils.tree_l2_norm(grads),
            "mean_reward": jnp.exp(log_info["log_gfn_reward"]).mean(),
            "mean_log_reward": log_info["log_gfn_reward"].mean(),
            "rl_reward": rl_reward.mean(),
        },
        eval_info,
        train_state.config,
        ordered=True,
    )

    # Return the updated train state
    return train_state._replace(
        rng_key=rng_key,
        model=model,
        opt_state=opt_state,
        metrics_state=metrics_state,
        eval_info=eval_info,
    )


@hydra.main(config_path="configs/", config_name="db_tfbind", version_base=None)
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
    reward_module = gfnx.TFBind8RewardModule()
    # Initialize the environment and its inner parameters
    env = gfnx.TFBind8Environment(reward_module)
    env_params = env.init(env_init_key)

    rng_key, net_init_key = jax.random.split(rng_key)
    # Initialize the network
    model = MLPPolicy(
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        train_backward_policy=cfg.agent.train_backward,
        encoder_params={
            "pad_id": env.pad_token,
            "vocab_size": env.ntoken,
            "max_length": env.max_length,
            **OmegaConf.to_container(cfg.network),
        },
        key=net_init_key,
    )

    exploration_schedule = optax.linear_schedule(
        init_value=cfg.agent.start_eps,
        end_value=cfg.agent.end_eps,
        transition_steps=cfg.agent.exploration_steps,
    )

    # Initialize the optimizer
    optimizer = optax.adam(learning_rate=cfg.agent.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    metrics_module = MultiMetricsModule(
        metrics={
            "distribution": ApproxDistributionMetricsModule(
                metrics=["tv", "kl"],
                env=env,
                buffer_size=cfg.logging.metric_buffer_size,
            ),
            "rd": SWMeanRewardSWMetricsModule(
                env=env,
                env_params=env_params,
                buffer_size=cfg.logging.metric_buffer_size,
            ),
        }
    )
    # Fill the initial states of metrics
    metrics_state = metrics_module.init(
        rng_key=eval_init_key,
        args=metrics_module.InitArgs(
            metrics_args={
                "distribution": ApproxDistributionMetricsModule.InitArgs(env_params=env_params),
                "rd": SWMeanRewardSWMetricsModule.InitArgs(),
            }
        ),
    )
    eval_info = metrics_module.get(metrics_state)

    train_state = TrainState(
        rng_key=rng_key,
        config=cfg,
        env=env,
        env_params=env_params,
        model=model,
        optimizer=optimizer,
        opt_state=opt_state,
        metrics_module=metrics_module,
        metrics_state=metrics_state,
        exploration_schedule=exploration_schedule,
        eval_info=eval_info,
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
            tags=["DB", env.name.upper()],
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


if __name__ == "__main__":
    run_experiment()
