"""Single-file implementation for Trajectory Balance in bitseq environment.

Run the script with the following command:
```bash
python baselines/tb_bitseq.py
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
from gfnx.metrics import (
    AccumulatedModesMetricsModule,
    MultiMetricsModule,
    MultiMetricsState,
    TestCorrelationMetricsModule,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
writer = Writer()


class TransformerPolicy(eqx.Module):
    """
    A policy module that uses a simple transformer model to generate
    forward and backward action logits as well as a flow.
    """

    encoder: gfnx.networks.Encoder
    forward_pooler: eqx.nn.Linear
    backward_pooler: eqx.nn.Linear
    train_backward_policy: bool
    n_fwd_actions: int
    n_bwd_actions: int
    env_max_length: int

    def __init__(
        self,
        n_fwd_actions: int,
        n_bwd_actions: int,
        env_max_length: int,
        train_backward_policy: bool,
        encoder_params: dict,
        *,
        key: chex.PRNGKey,
    ):
        self.train_backward_policy = train_backward_policy
        self.n_fwd_actions = n_fwd_actions
        self.n_bwd_actions = n_bwd_actions
        self.env_max_length = env_max_length

        encoder_key, pooler_key = jax.random.split(key)
        self.encoder = gfnx.networks.Encoder(key=encoder_key, **encoder_params)

        self.forward_pooler = eqx.nn.Linear(
            in_features=encoder_params["hidden_size"],
            out_features=self.n_fwd_actions // self.env_max_length,
            key=pooler_key,
        )

        self.backward_pooler = eqx.nn.Linear(
            in_features=encoder_params["hidden_size"],
            out_features=self.n_bwd_actions // self.env_max_length,
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

        forward_logits = jnp.ravel(jax.vmap(self.forward_pooler)(encoded_obs[1:]))

        if self.train_backward_policy:
            backward_logits = jnp.ravel(jax.vmap(self.backward_pooler)(encoded_obs[1:]))
        else:
            backward_logits = jnp.zeros(shape=(self.n_bwd_actions,), dtype=jnp.float32)

        return {
            "forward_logits": forward_logits,
            "backward_logits": backward_logits,
        }


# Define the train state that will be used in the training loop
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.BitseqEnvironment
    env_params: chex.Array
    reward_module: gfnx.BitseqRewardModule
    reward_params: chex.Array
    model: TransformerPolicy
    logZ: chex.Array  # Added logZ for TB
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

    # Get epsilon exploration value from config
    cur_eps = train_state.exploration_schedule(idx)

    # Define the policy function suitable for gfnx.utils.forward_rollout
    def fwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        rng_key, explore_key, dropout_key = jax.random.split(rng_key, 3)
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = policy(env_obs, enable_dropout=True, key=dropout_key)
        # With probability cur_eps, return zero logits and the same policy outputs
        do_explore = jax.random.bernoulli(explore_key, cur_eps)
        forward_logits = jnp.where(do_explore, 0, policy_outputs["forward_logits"])
        return forward_logits, policy_outputs

    rng_keys = jax.random.split(sample_traj_key, num_envs)
    traj_data, final_states, info = jax.vmap(
        lambda rng: gfnx.utils.forward_rollout(
            rng, fwd_policy_fn, policy_params, train_state.env, train_state.env_params
        )
    )(rng_keys)
    # Compute log rewards on terminal states
    log_rewards = jax.vmap(train_state.reward_module.log_reward, in_axes=(0, None))(
        final_states, train_state.reward_params
    )
    final_env_state = aux_info["final_env_state"]
    # Compute the RL reward / ELBO (for logging purposes)
    log_pb_traj = jax.vmap(
        lambda td: gfnx.utils.forward_trajectory_log_probs(env, td, env_params)
    )(traj_data)[1]
    rl_reward = log_pb_traj + log_rewards + info["entropy"]

    # Step 2. Compute the TB loss
    def loss_fn(
        current_all_params: dict,
        static_model_parts: TransformerPolicy,
        current_traj_data: gfnx.utils.TrajectoryData,
        current_env: gfnx.BitseqEnvironment,
        current_env_params: chex.Array,
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

        # Step 2.1 Compute forward actions and log probabilities
        fwd_logits_traj = policy_outputs_traj["forward_logits"]

        # Vmap get_invalid_mask over the time dimension
        invalid_fwd_mask = jax.vmap(current_env.get_invalid_mask_batch, in_axes=(0, None))(
            current_traj_data.state, current_env_params
        )

        fwd_logprobs_traj = gfnx.utils.compute_action_log_probs(
            fwd_logits_traj, current_traj_data.action, invalid_fwd_mask, current_traj_data.pad
        )
        sum_log_pf_along_traj = fwd_logprobs_traj.sum(axis=1)
        # Use extracted logZ_val
        log_pf_traj = logZ_val + sum_log_pf_along_traj

        # Step 2.2 Compute backward actions and log probabilities
        prev_states = jax.tree.map(lambda x: x[:, :-1], current_traj_data.state)
        fwd_actions = current_traj_data.action[:, :-1]
        curr_states = jax.tree.map(lambda x: x[:, 1:], current_traj_data.state)

        bwd_actions_traj = jax.vmap(
            current_env.get_backward_action_batch,
            in_axes=(0, 0, 0, None),
        )(prev_states, fwd_actions, curr_states, current_env_params)

        bwd_logits_traj = policy_outputs_traj["backward_logits"]
        bwd_logits_for_pb = bwd_logits_traj[:, 1:]
        # Vmap get_invalid_backward_mask over the time dimension
        invalid_bwd_mask = jax.vmap(
            current_env.get_invalid_backward_mask_batch,
            in_axes=(0, None),
        )(curr_states, current_env_params)

        log_pb_selected = gfnx.utils.compute_action_log_probs(
            bwd_logits_for_pb, bwd_actions_traj, invalid_bwd_mask, current_traj_data.pad[:, :-1]
        )
        log_pb_sum = jnp.sum(log_pb_selected, axis=1)
        target = log_pb_sum + current_log_rewards

        return optax.losses.squared_error(log_pf_traj, target).mean()

    # Prepare parameters for the loss function and gradient calculation
    params_for_loss = {"model_params": policy_params, "logZ": train_state.logZ}

    mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(
        params_for_loss, policy_static, traj_data, env, env_params, log_rewards
    )

    # Step 3. Update parameters (model network and logZ)
    optax_params_for_update = {
        "model_params": policy_params,
        "logZ": train_state.logZ,
    }
    updates, new_opt_state = train_state.optimizer.update(
        grads, train_state.opt_state, optax_params_for_update
    )

    # Apply updates
    new_model = eqx.apply_updates(train_state.model, updates["model_params"])
    new_logZ = eqx.apply_updates(train_state.logZ, updates["logZ"])

    # Perform all the required updates of metrics
    transitions = jax.tree.map(
        lambda x: x.reshape((-1,) + x.shape[2:]),
        jax.vmap(gfnx.utils.split_traj_to_transitions)(traj_data),
    )
    rng_key, eval_rng_key = jax.random.split(rng_key)

    # Perform all the required logging
    metrics_state, eval_info = train_state.metrics_module.step(
        idx=idx,
        metrics_state=train_state.metrics_state,
        rng_key=eval_rng_key,
        update_args=train_state.metrics_module.UpdateArgs(
            metrics_args={
                "modes": AccumulatedModesMetricsModule.UpdateArgs(
                    states=final_states,
                ),
            }
        ),
        process_args=train_state.metrics_module.ProcessArgs(
            metrics_args={
                "correlation": TestCorrelationMetricsModule.ProcessArgs(
                    policy_params=policy_params,
                    env_params=train_state.env_params,
                    reward_params=train_state.reward_params,
                ),
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

    jax.debug.callback(
        logging_callback,
        idx,
        {
            "mean_loss": mean_loss,
            "entropy": info["entropy"].mean(),
            "grad_norm": optax.tree_utils.tree_l2_norm(grads),
            "logZ": new_logZ,
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
        logZ=new_logZ,
        eval_info=eval_info,
    )


@hydra.main(config_path="configs/", config_name="tb_bitseq", version_base=None)
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
    reward_module = gfnx.BitseqRewardModule(
        sentence_len=cfg.environment.n,
        k=cfg.environment.k,
        mode_set_size=cfg.environment.num_modes,
        reward_exponent=cfg.environment.reward_exponent,
    )
    # Initialize the environment and its inner parameters
    env = gfnx.BitseqEnvironment(n=cfg.environment.n, k=cfg.environment.k)
    env_params = env.init(env_init_key)
    reward_params = reward_module.init(env_init_key, env.get_init_state())

    rng_key, net_init_key = jax.random.split(rng_key)
    # Initialize the network
    model = TransformerPolicy(
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        env_max_length=env.max_length,
        train_backward_policy=cfg.agent.train_backward,
        encoder_params={
            "pad_id": -1,  # No masking in this setup
            "vocab_size": env.ntoken,
            "max_length": env.max_length + 1,  # +1 for BOS token
            **OmegaConf.to_container(cfg.network),
        },
        key=net_init_key,
    )

    exploration_schedule = optax.linear_schedule(
        init_value=cfg.agent.start_eps,
        end_value=cfg.agent.end_eps,
        transition_steps=cfg.agent.exploration_steps,
    )

    # Initialize logZ separately
    logZ = jnp.array(0.0)

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

    # Initialize the backward policy function for correlation computation
    policy_static = eqx.filter(model, eqx.is_array, inverse=True)

    def bwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        del rng_key
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = policy(env_obs)
        return policy_outputs["backward_logits"], policy_outputs

    metrics_module = MultiMetricsModule({
        "correlation": TestCorrelationMetricsModule(
            env=env,
            bwd_policy_fn=bwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds,
            batch_size=cfg.metrics.batch_size,
        ),
        "modes": AccumulatedModesMetricsModule(
            env=env,
            distance_fn=lambda x, y: gfnx.utils.distances.hamming_distance(
                gfnx.utils.bitseq.detokenize(x.tokens, env.k),
                gfnx.utils.bitseq.detokenize(y.tokens, env.k),
            ),
            distance_threshold=cfg.metrics.mode_threshold,
        ),
    })

    eval_init_key, correlation_init_key = jax.random.split(eval_init_key)
    binary_test_set = gfnx.utils.bitseq.construct_binary_test_set(
        correlation_init_key, reward_params["mode_set"]
    )
    vector_tokenize = jax.vmap(lambda x: gfnx.utils.bitseq.tokenize(x, env.k))
    test_set_tokens = vector_tokenize(binary_test_set)
    test_set_states = gfnx.BitseqEnvState.from_tokens(test_set_tokens)
    # Initialize the metrics
    mode_set = reward_params["mode_set"]
    mode_set_tokens = vector_tokenize(mode_set)
    modes_states = gfnx.BitseqEnvState.from_tokens(mode_set_tokens)

    # Here we need to pass the initial parameters for all  metrics
    metrics_state = metrics_module.init(
        eval_init_key,
        metrics_module.InitArgs(
            metrics_args={
                "correlation": TestCorrelationMetricsModule.InitArgs(
                    env_params=env_params, test_set=test_set_states, reward_params=reward_params
                ),
                "modes": AccumulatedModesMetricsModule.InitArgs(modes=modes_states),
            }
        ),
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
        logZ=logZ,
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
    save_checkpoint(
        os.path.join(dir, "model_and_logZ"),
        {
            "model": train_state.model,
            "logZ": train_state.logZ,
        },
    )


if __name__ == "__main__":
    run_experiment()
