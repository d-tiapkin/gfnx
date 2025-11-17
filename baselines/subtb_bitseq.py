"""Single-file implementation for SubTrajectory Balance in bitseq environment.

Run the script with the following command:
```bash
python baselines/subtb_bitseq.py
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
from optax.losses import squared_error

import gfnx
from gfnx.metrics.new import (
    AccumulatedModesMetricsModule,
    MultiMetricsModule,
    MultiMetricsState,
    TestCorrelationMetricsModule,
)

from utils.logger import Writer
from utils.checkpoint import save_checkpoint

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
    flow_pooler: eqx.nn.Linear
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

        self.flow_pooler = eqx.nn.Linear(
            in_features=encoder_params["hidden_size"],
            out_features=1,
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
        flow = self.flow_pooler(encoded_obs[0])
        #jax.debug.print("hello {bar}", bar=flow.shape)

        if self.train_backward_policy:
            backward_logits = jnp.ravel(jax.vmap(self.backward_pooler)(encoded_obs[1:]))
        else:
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
    env: gfnx.BitseqEnvironment
    env_params: chex.Array
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

    # Get model parameters and static parts
    policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)

    # Step 1. Generate a batch of trajectories
    rng_key, sample_traj_key = jax.random.split(rng_key)
    cur_eps = train_state.exploration_schedule(idx)

    # Define the policy function suitable for gfnx.utils.forward_rollout
    def fwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        rng_key, explore_key = jax.random.split(rng_key)
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = jax.vmap(
            lambda obs, key: policy(obs, enable_dropout=True, key=key), in_axes=(0, 0)
        )(env_obs, jax.random.split(rng_key, env_obs.shape[0]))
        # With probability cur_eps, return zero logits and the same policy outputs
        explore_mask = jax.random.bernoulli(explore_key, cur_eps, (env_obs.shape[0],))
        forward_logits = jnp.where(explore_mask[..., None], 0, policy_outputs["forward_logits"])
        return forward_logits, policy_outputs

    traj_data, aux_info = gfnx.utils.forward_rollout(
        rng_key=sample_traj_key,
        num_envs=num_envs,
        policy_fn=fwd_policy_fn,
        policy_params=policy_params,  # Pass only network parameters
        env=env,
        env_params=env_params,
    )

    # Step 2. Compute the SubTB loss
    def loss_fn(
        current_all_params: dict,
        static_model_parts: TransformerPolicy,
        current_traj_data: gfnx.utils.TrajectoryData,
        current_env: gfnx.BitseqEnvironment,
        current_env_params: chex.Array,
    ):
        model_learnable_params = current_all_params["model_params"]
        model_to_call = eqx.combine(model_learnable_params, static_model_parts)
        # Compute policy outputs for the whole trajectory
        dropout_keys = jax.random.split(rng_key, current_traj_data.obs.shape[:2])
        # Get policy outputs for the entire trajectory
        policy_outputs_traj = jax.vmap(
            jax.vmap(
                lambda obs, key: model_to_call(obs, enable_dropout=True, key=key),
            ),
        )(current_traj_data.obs, dropout_keys)

        fwd_logits_traj = policy_outputs_traj["forward_logits"]
        bwd_logits_traj = policy_outputs_traj["backward_logits"]
        log_flow_traj = policy_outputs_traj["log_flow"]

        batch_size, traj_len_plus1 = current_traj_data.action.shape
        traj_len = traj_len_plus1 - 1

        # Masks
        forward_invalid_mask = jax.vmap(
            current_env.get_invalid_mask, in_axes=(1, None), out_axes=1
        )(current_traj_data.state, current_env_params)
        backward_invalid_mask = jax.vmap(
            current_env.get_invalid_backward_mask, in_axes=(1, None), out_axes=1
        )(current_traj_data.state, current_env_params)
        forward_action = current_traj_data.action[:, :-1]
        # Compute backward actions
        prev_states = jax.tree.map(lambda x: x[:, :-1], current_traj_data.state)
        fwd_actions = current_traj_data.action[:, :-1]
        curr_states = jax.tree.map(lambda x: x[:, 1:], current_traj_data.state)
        bwd_actions_traj = jax.vmap(
            current_env.get_backward_action,
            in_axes=(1, 1, 1, None),
            out_axes=1,
        )(prev_states, fwd_actions, curr_states, current_env_params)
        pad_mask = current_traj_data.pad[:, :-1]
        done_mask = current_traj_data.done[:, :-1]

        # Forward log-probs
        pf_logits = gfnx.utils.mask_logits(fwd_logits_traj, forward_invalid_mask)
        log_pf = jax.nn.log_softmax(pf_logits, axis=-1)[:, :-1]
        log_pf_along_traj = jnp.take_along_axis(
            log_pf, jnp.expand_dims(forward_action, axis=-1), axis=-1
        ).squeeze(-1)
        log_pf_along_traj = jnp.where(pad_mask, 0.0, log_pf_along_traj)

        # Backward log-probs
        pb_logits = gfnx.utils.mask_logits(bwd_logits_traj, backward_invalid_mask)
        log_pb = jax.nn.log_softmax(pb_logits, axis=-1)[:, 1:]
        log_pb_along_traj = jnp.take_along_axis(
            log_pb, jnp.expand_dims(bwd_actions_traj, axis=-1), axis=-1
        ).squeeze(-1)
        log_pb_along_traj = jnp.where(pad_mask, 0.0, log_pb_along_traj)
        log_pb_along_traj = log_pb_along_traj + jnp.where(
            pad_mask, 0.0, current_traj_data.log_gfn_reward[:, :-1]
        )

        # log_flow
        log_flow = jnp.where(current_traj_data.pad, 0.0, log_flow_traj)

        def process_one_traj(log_pf, log_pb, log_flow, done, pad):
            def process_pair_idx(i, j, log_pf, log_pb, log_flow, done, pad):
                def fn():
                    mask = jnp.logical_and(i <= jnp.arange(traj_len), jnp.arange(traj_len) < j)
                    weight = jnp.power(train_state.config.agent.lmbd, j - i)
                    log_flow_start = log_flow[i]
                    log_pf_subtraj = log_flow_start + (log_pf * mask).sum()
                    log_flow_finish = jnp.where(done[j - 1], 0.0, log_flow[j])
                    log_pb_subtraj = log_flow_finish + (log_pb * mask).sum()
                    return weight * squared_error(log_pf_subtraj, log_pb_subtraj), weight

                return jax.lax.cond(pad[j - 1], lambda: (0.0, 0.0), fn)

            i, j = jnp.triu_indices(traj_len + 1, k=1)
            weighted_loss, weighted_norm = jax.vmap(
                process_pair_idx, in_axes=(0, 0, None, None, None, None, None)
            )(i, j, log_pf, log_pb, log_flow, done, pad)
            return weighted_loss.sum() / weighted_norm.sum()

        loss = jax.vmap(process_one_traj)(
            log_pf_along_traj,
            log_pb_along_traj,
            log_flow,
            done_mask,
            pad_mask,
        ).mean()
        return loss

    # Prepare parameters for the loss function and gradient calculation
    params_for_loss = {"model_params": policy_params}

    mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(
        params_for_loss, policy_static, traj_data, env, env_params
    )

    # Step 3. Update parameters (model network)
    optax_params_for_update = policy_params
    updates, new_opt_state = train_state.optimizer.update(
        grads["model_params"], train_state.opt_state, optax_params_for_update
    )
    new_model = eqx.apply_updates(train_state.model, updates)

    # Perform all the required updates of metrics
    # Peform all the requied updates of metrics
    transitions = gfnx.utils.split_traj_to_transitions(traj_data)
    metrics_state = train_state.metrics_module.update(
        metrics_state=train_state.metrics_state,
        rng_key=jax.random.key(0),  # not used, but required by the API
        args=train_state.metrics_module.UpdateArgs(
            metrics_args={
                "modes": AccumulatedModesMetricsModule.UpdateArgs(
                    states=transitions.state,
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
                    "correlation": TestCorrelationMetricsModule.ProcessArgs(
                        policy_params=policy_params,
                        env_params=train_state.env_params,
                    ),
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
            "mean_reward": jnp.exp(aux_info["log_gfn_reward"]).mean(),
            "mean_log_reward": aux_info["log_gfn_reward"].mean(),
            "rl_reward": aux_info["log_gfn_reward"].mean() + aux_info["entropy"].mean(),
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


@hydra.main(config_path="configs/", config_name="subtb_bitseq", version_base=None)
def run_experiment(cfg: OmegaConf) -> None:
    # Log the configuration
    log.info(OmegaConf.to_yaml(cfg))

    rng_key = jax.random.PRNGKey(cfg.seed)
    env_init_key = jax.random.PRNGKey(cfg.env_init_seed)
    eval_init_key = jax.random.PRNGKey(cfg.eval_init_seed)

    # Define the reward function for the environment
    reward_module = gfnx.BitseqRewardModule(
        sentence_len=cfg.environment.n,
        k=cfg.environment.k,
        mode_set_size=cfg.environment.num_modes,
        reward_exponent=cfg.environment.reward_exponent,
    )
    # Initialize the environment and its inner parameters
    env = gfnx.BitseqEnvironment(reward_module, n=cfg.environment.n, k=cfg.environment.k)
    env_params = env.init(env_init_key)

    rng_key, net_init_key = jax.random.split(rng_key)
    # Initialize the network
    model = TransformerPolicy(
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        env_max_length=env.max_length,
        train_backward_policy=cfg.agent.train_backward,
        encoder_params={
            "pad_id": -1,  # Do not mask any token
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

    # Prepare parameters for Optax
    model_params_init = eqx.filter(model, eqx.is_array)
    optimizer = optax.adam(learning_rate=cfg.agent.learning_rate)
    opt_state = optimizer.init(model_params_init)

    # Initialize the backward policy function for correlation computation
    policy_static = eqx.filter(model, eqx.is_array, inverse=True)

    def bwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        del rng_key
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = jax.vmap(policy, in_axes=(0,))(env_obs)
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
        correlation_init_key, env_params.reward_params["mode_set"]
    )
    vector_tokenize = jax.vmap(lambda x: gfnx.utils.bitseq.tokenize(x, env.k))
    test_set_tokens = vector_tokenize(binary_test_set)
    test_set_states = gfnx.BitseqEnvState.from_tokens(test_set_tokens)
    # Initialize the metrics
    mode_set = env_params.reward_params["mode_set"]
    mode_set_tokens = vector_tokenize(mode_set)
    modes_states = gfnx.BitseqEnvState.from_tokens(mode_set_tokens)

    # Here we need to pass the initial parameters for all  metrics
    metrics_state = metrics_module.init(
        eval_init_key,
        metrics_module.InitArgs(
            metrics_args={
                "correlation": TestCorrelationMetricsModule.InitArgs(
                    env_params=env_params, test_set=test_set_states
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
        model=model,
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
