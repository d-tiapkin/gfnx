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
import orbax.checkpoint as ocp
import wandb
from jax_tqdm import loop_tqdm
from jaxtyping import Array, Int
from omegaconf import OmegaConf

import gfnx

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


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
        #self.encoder = gfnx.networks.Encoder(key=encoder_key, **encoder_params)
        self.encoder = eqx.nn.MLP(in_size=5 * 8, out_size=encoder_params["hidden_size"], width_size=256, depth=2, key=encoder_key)
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
        # pos_ids = jnp.arange(obs_ids.shape[0])
        # encoded_obs = self.encoder(
        #     obs_ids, pos_ids, enable_dropout=enable_dropout, key=key
        # )["layers_out"][-1]  # [seq_len, hidden_size]
        #encoded_obs = encoded_obs.mean(axis=0)  # Average pooling
        output = self.pooler(encoded_obs)
        if self.train_backward_policy:
            forward_logits, flow, backward_logits = jnp.split(
                output, [self.n_fwd_actions, self.n_fwd_actions + 1], axis=-1
            )
        else:
            forward_logits, flow = jnp.split(
                output, [self.n_fwd_actions], axis=-1
            )
            backward_logits = jnp.zeros(
                shape=(self.n_bwd_actions,), dtype=jnp.float32
            )
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
    model: TransformerPolicy
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    metrics_module: gfnx.metrics.TFBindMetricModule  # dict with metric modules
    metrics: gfnx.metrics.TFBindMetricState  # dict with metric states


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
    def fwd_policy_fn(
        rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params
    ) -> chex.Array:
        del rng_key
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = jax.vmap(policy, in_axes=(0,))(env_obs)
        return policy_outputs["forward_logits"], policy_outputs

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

        # Compute the stats for the next state
        next_policy_outputs = jax.vmap(model, in_axes=(0,))(
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
        # Replace the target with the log_gfn_reward if the episode is done
        target = jnp.where(
            transitions.done,
            transitions.log_gfn_reward,
            bwd_logprobs + next_log_flow,
        )

        # Compute the DB loss with masking
        loss = optax.l2_loss(
            jnp.where(transitions.pad, 0.0, fwd_logprobs + log_flow),
            jnp.where(transitions.pad, 0.0, target),
        ).mean()
        return loss, log_info

    (mean_loss, log_info), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(train_state.model)
    # Step 3. Update the model with grads
    updates, opt_state = train_state.optimizer.update(
        grads,
        train_state.opt_state,
        eqx.filter(train_state.model, eqx.is_array),
    )
    model = eqx.apply_updates(train_state.model, updates)
    # Peform all the requied logging
    metric_state = train_state.metrics_module.update(
        train_state.metrics, log_info["final_env_state"], env_params
    )
    # Compute the evaluation info if needed
    eval_info = jax.lax.cond(
        (idx % train_state.config.logging.track_each == 0)
        | (idx + 1 == train_state.config.num_train_steps),
        train_state.metrics_module.get,
        lambda a, b: {"tv": -1.0, "kl": -1.0, "reward_delta": -1.0},
        metric_state,
        env_params
    )

    # Perform the logging via JAX debug callback
    def logging_callback(idx: int, train_info: dict, eval_info: dict, cfg):
        if idx % cfg.logging.track_each == 0 or idx + 1 == cfg.num_train_steps:
            log.info(f"Step {idx}")
            log.info({key: float(value) for key, value in train_info.items()})
            log.info({key: float(value) for key, value in eval_info.items()})
            if cfg.logging.use_wandb:
                wandb.log(eval_info, commit=False)

        if cfg.logging.use_wandb:
            wandb.log(train_info)

    jax.debug.callback(
        logging_callback,
        idx,
        {
            "train/mean_loss": mean_loss,
            "train/entropy": log_info["entropy"].mean(),
            "train/grad_norm": optax.tree_utils.tree_l2_norm(grads),
        },
        {f"eval/{key}": value for key, value in eval_info.items()},
        train_state.config,
        ordered=True,
    )

    # Return the updated train state
    return train_state._replace(
        rng_key=rng_key, model=model, opt_state=opt_state, metrics=metric_state
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
    model = TransformerPolicy(
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        train_backward_policy=False,
        encoder_params={
            "pad_id": env.pad_token,
            "vocab_size": env.ntoken,
            "max_length": env.max_length,
            **OmegaConf.to_container(cfg.network),
        },
        key=net_init_key,
    )
    # Initialize the optimizer
    optimizer = optax.adam(learning_rate=cfg.agent.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    metrics_module = gfnx.metrics.TFBindMetricModule(
        env, buffer_max_length=cfg.logging.metric_buffer_size
    )
    # Fill the initial states of metrics
    metrics = metrics_module.init(eval_init_key, env_params)

    train_state = TrainState(
        rng_key=rng_key,
        config=cfg,
        env=env,
        env_params=env_params,
        model=model,
        optimizer=optimizer,
        opt_state=opt_state,
        metrics_module=metrics_module,
        metrics=metrics,
    )
    # Split train state into parameters and static parts to make jit work.
    train_state_params, train_state_static = eqx.partition(
        train_state, eqx.is_array
    )

    @functools.partial(jax.jit, donate_argnums=(1,))
    @loop_tqdm(cfg.num_train_steps, print_rate=cfg.logging["tqdm_print_rate"])
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
            tags=["DB", env.name.upper()],
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
    ckptr.save(cwd, args=ocp.args.StandardSave(train_state_params))
    ckptr.wait_until_finished()


if __name__ == "__main__":
    run_experiment()
