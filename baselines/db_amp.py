"""Single-file implementation for Detailed Balance in AMP environment.

Run the script with the following command:
```bash
python baselines/db_amp.py
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
    env: gfnx.AMPEnvironment
    env_params: chex.Array
    reward_module: gfnx.EqxProxyAMPRewardModule
    reward_params: chex.Array
    model: TransformerPolicy
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
    def fwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        rng_key, explore_key = jax.random.split(rng_key)
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = policy(env_obs, enable_dropout=True, key=rng_key)
        # With probability cur_eps, return zero logits and the same policy outputs
        do_explore = jax.random.bernoulli(explore_key, cur_eps)
        forward_logits = jnp.where(do_explore, 0, policy_outputs["forward_logits"])
        return forward_logits, policy_outputs

    # Generating the trajectory and splitting it into transitions
    rng_keys = jax.random.split(sample_traj_key, num_envs)
    traj_data, final_states, info = jax.vmap(
        lambda rng: gfnx.utils.forward_rollout(rng, fwd_policy_fn, policy_params, env, env_params)
    )(rng_keys)
    # Compute log rewards on terminal states, then broadcast to match transitions layout
    log_rewards = jax.vmap(train_state.reward_module.log_reward, in_axes=(0, None))(
        final_states, train_state.reward_params
    )
    transitions = jax.tree.map(
        lambda x: x.reshape((-1,) + x.shape[2:]),
        jax.vmap(gfnx.utils.split_traj_to_transitions)(traj_data),
    )
    T_steps = transitions.done.shape[0] // num_envs
    traj_rewards_flat = jnp.repeat(log_rewards, T_steps)  # [B*T]
    bwd_actions = env.get_backward_action_batch(
        transitions.state,
        transitions.action,
        transitions.next_state,
        env_params,
    )
    # Compute the RL reward / ELBO (for logging purposes)
    _, log_pb_traj = jax.vmap(
        lambda td: gfnx.utils.forward_trajectory_log_probs(env, td, env_params)
    )(traj_data)
    rl_reward = log_pb_traj + log_rewards + info["entropy"]

    # Step 2. Compute the loss
    def loss_fn(model: TransformerPolicy, current_traj_rewards_flat: jnp.ndarray) -> chex.Array:
        # Call the network to get the logits
        batch_size = transitions.obs.shape[0]
        policy_outputs = jax.vmap(
            lambda x, key: model(x, enable_dropout=True, key=key), in_axes=(0, 0)
        )(transitions.obs, jax.random.split(rng_key, batch_size))
        # Compute the forward log-probs
        fwd_logits = policy_outputs["forward_logits"]
        invalid_mask = env.get_invalid_mask_batch(transitions.state, env_params)
        fwd_logprobs = gfnx.utils.compute_action_log_probs(
            fwd_logits, transitions.action, invalid_mask
        )
        log_flow = policy_outputs["log_flow"]

        # Compute the stats for the next state
        next_policy_outputs = jax.vmap(
            lambda x, key: model(x, enable_dropout=True, key=key), in_axes=(0, 0)
        )(transitions.next_obs, jax.random.split(rng_key, batch_size))
        bwd_logits = next_policy_outputs["backward_logits"]
        next_bwd_invalid_mask = env.get_invalid_backward_mask_batch(
            transitions.next_state, env_params
        )
        bwd_logprobs = gfnx.utils.compute_action_log_probs(
            bwd_logits, bwd_actions, next_bwd_invalid_mask
        )
        next_log_flow = next_policy_outputs["log_flow"]
        # Replace the target with the log reward if the episode is done
        target = jnp.where(
            transitions.done,
            bwd_logprobs + current_traj_rewards_flat,
            bwd_logprobs + next_log_flow,
        )

        # Compute the DB loss with masking
        transition = jnp.logical_not(transitions.pad)
        done = transitions.done
        not_done = jnp.logical_not(transitions.done)

        loss = optax.l2_loss(
            jnp.where(transitions.pad, 0.0, fwd_logprobs + log_flow),
            jnp.where(transitions.pad, 0.0, target),
        )

        leaf_loss = (loss * done).sum() / (jnp.logical_and(transition, done)).sum()
        flow_loss = (loss * not_done).sum() / (jnp.logical_and(transition, not_done)).sum()

        return leaf_loss * 25 + flow_loss

    mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(train_state.model, traj_rewards_flat)
    # Step 3. Update the model with grads
    updates, opt_state = train_state.optimizer.update(
        grads,
        train_state.opt_state,
        eqx.filter(train_state.model, eqx.is_array),
    )
    model = eqx.apply_updates(train_state.model, updates)
    # Peform all the requied updates of metrics
    rng_key, eval_rng_key = jax.random.split(rng_key)

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
                    policy_params=policy_params, env_params=env_params
                )
            }
        ),
        eval_each=train_state.config.logging.eval_each,
        num_train_steps=train_state.config.num_train_steps,
        prev_eval_info=train_state.eval_info,
    )

    # Perform the logging via JAX debug callback
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

    generation_length = jnp.array(
        (final_states.tokens != env.pad_token) & (final_states.tokens != env.eos_token),
        dtype=jnp.float32,
    ).sum(axis=-1)

    jax.debug.callback(
        logging_callback,
        idx,
        {
            "mean_loss": mean_loss,
            "entropy": info["entropy"].mean(),
            "grad_norm": optax.tree_utils.tree_l2_norm(grads),
            "mean_length": generation_length.mean(),
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
        model=model,
        opt_state=opt_state,
        metrics_state=metrics_state,
        eval_info=eval_info,
    )


@hydra.main(config_path="configs/", config_name="db_amp", version_base=None)
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
            "max_length": env.max_length + 2,  # +1 for BOS token
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
    # Initialize the optimizer
    optimizer = optax.adam(learning_rate=cfg.agent.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    # Initialize the backward policy function for correlation computation
    policy_static = eqx.filter(model, eqx.is_array, inverse=True)

    def fwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        del rng_key
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = policy(env_obs)
        return policy_outputs["forward_logits"], policy_outputs

    def amp_distance_fn(lhs_state: gfnx.AMPEnvState, rhs_state: gfnx.AMPEnvState) -> chex.Array:
        """Compute the distance between two AMP states."""
        return gfnx.utils.distances.levenshtein_distance(
            lhs_state.tokens, rhs_state.tokens, eos_id=env.eos_token, pad_id=env.pad_token
        )

    metrics_module = MultiMetricsModule({
        "topk": TopKMetricsModule(
            fwd_policy_fn=fwd_policy_fn,
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
            tags=["DB", env.name.upper()],
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


if __name__ == "__main__":
    run_experiment()
