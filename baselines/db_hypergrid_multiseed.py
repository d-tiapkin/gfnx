"""Vmap-over-seeds implementation for Detailed Balance in hypergrid environment.

Runs the full training loop in parallel over multiple random seeds using jax.vmap.
No per-step logging callbacks (jax.debug.callback is not vmap-compatible); instead,
metric history (mean ± std) over seeds is printed after training completes.

The training loop uses a two-level jax.lax.scan structure:
  - Outer scan: over num_evals "epochs"
  - Each epoch: (1) evaluate metrics unconditionally, then (2) run steps_per_eval
    training steps via an inner jax.lax.scan
  - Returns metric_history[num_evals] across all seeds

This avoids jax.lax.cond for eval gating (both branches execute under vmap,
causing extreme slowdown). Instead, evals happen unconditionally at fixed intervals.

Run the script with the following command:
```bash
python baselines/db_hypergrid_multiseed.py
```

Also see https://jax.readthedocs.io/en/latest/gpu_performance_tips.html for
performance tips when running on GPU, i.e., XLA flags.

"""

import csv
import logging
from pathlib import Path
from typing import NamedTuple

import chex
import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf

import gfnx
from gfnx.metrics import ExactDistributionMetricsModule, ExactDistributionMetricsState

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class MLPPolicy(eqx.Module):
    """MLP generating forward and backward action logits and a log-flow."""

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

        output_size = self.n_fwd_actions + 1  # +1 for log_flow
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
            forward_logits, log_flow, backward_logits = jnp.split(
                x, [self.n_fwd_actions, self.n_fwd_actions + 1], axis=-1
            )
        else:
            forward_logits, log_flow = jnp.split(x, [self.n_fwd_actions], axis=-1)
            backward_logits = jnp.zeros(shape=(self.n_bwd_actions,), dtype=jnp.float32)
        return {
            "forward_logits": forward_logits,
            "log_flow": log_flow.squeeze(-1),
            "backward_logits": backward_logits,
        }


# Dynamic (JAX array) part of the training state — this is what gets vmapped over seeds.
class TrainStateParams(NamedTuple):
    rng_key: chex.PRNGKey
    model_params: chex.Array  # eqx.filter(model, eqx.is_array)
    opt_state: optax.OptState
    reward_params: chex.Array


@hydra.main(config_path="configs/", config_name="db_hypergrid_multiseed", version_base=None)
def run_experiment(cfg: OmegaConf) -> None:
    log.info(OmegaConf.to_yaml(cfg))

    env_init_key = jax.random.PRNGKey(cfg.env_init_seed)

    # Build reward module and environment (shared across seeds — static)
    reward_module_factory = {
        "easy": gfnx.EasyHypergridRewardModule,
        "hard": gfnx.HardHypergridRewardModule,
    }[cfg.environment.reward]
    reward_module = reward_module_factory(side=cfg.environment.side)

    env = gfnx.environment.HypergridEnvironment(dim=cfg.environment.dim, side=cfg.environment.side)
    env_params = env.init(env_init_key)

    # Build a template model to obtain static parts (shared across seeds)
    template_key = jax.random.PRNGKey(0)
    template_model = MLPPolicy(
        input_size=env.observation_space.shape[0],
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        hidden_size=cfg.network.hidden_size,
        train_backward_policy=cfg.agent.train_backward,
        depth=cfg.network.depth,
        rng_key=template_key,
    )
    policy_static = eqx.filter(template_model, eqx.is_array, inverse=True)

    optimizer = optax.adam(learning_rate=cfg.agent.learning_rate)

    exploration_schedule = optax.linear_schedule(
        init_value=cfg.agent.start_eps,
        end_value=cfg.agent.end_eps,
        transition_steps=cfg.agent.exploration_steps,
    )

    # -------------------------------------------------------------------------
    # Metrics setup — ExactDistributionMetricsModule (no replay buffer needed)
    # -------------------------------------------------------------------------
    steps_per_eval = cfg.num_train_steps // cfg.num_evals

    # Forward policy function for exact distribution computation (uses closure)
    def fwd_policy_fn_for_metrics(
        _fwd_rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params
    ) -> chex.Array:
        current_model = eqx.combine(policy_params, policy_static)
        policy_outputs = current_model(env_obs)
        return policy_outputs["forward_logits"], policy_outputs

    metrics_module = ExactDistributionMetricsModule(
        metrics=["tv", "kl"],
        env=env,
        fwd_policy_fn=fwd_policy_fn_for_metrics,
        reward_module=reward_module,
        batch_size=cfg.metrics.batch_size,
    )

    # Canonical reward_params for metrics init (same for all seeds)
    canonical_reward_params = reward_module.init(env_init_key, env.get_init_state())
    init_metrics_state = metrics_module.init(
        jax.random.key(0),
        metrics_module.InitArgs(env_params=env_params, reward_params=canonical_reward_params),
    )
    # Tile the single metrics state across all seeds
    all_init_metrics: ExactDistributionMetricsState = jax.tree.map(
        lambda x: jnp.stack([x] * cfg.num_seeds), init_metrics_state
    )

    # -------------------------------------------------------------------------
    # Per-seed initializer: takes a scalar seed, returns TrainStateParams arrays
    # -------------------------------------------------------------------------
    def make_init_params(seed: chex.Array) -> TrainStateParams:
        rng_key = jax.random.PRNGKey(seed)
        rng_key, net_key, reward_key = jax.random.split(rng_key, 3)
        model = MLPPolicy(
            input_size=env.observation_space.shape[0],
            n_fwd_actions=env.action_space.n,
            n_bwd_actions=env.backward_action_space.n,
            hidden_size=cfg.network.hidden_size,
            train_backward_policy=cfg.agent.train_backward,
            depth=cfg.network.depth,
            rng_key=net_key,
        )
        model_params = eqx.filter(model, eqx.is_array)
        reward_params = reward_module.init(reward_key, env.get_init_state())
        opt_state = optimizer.init(model_params)
        return TrainStateParams(
            rng_key=rng_key,
            model_params=model_params,
            opt_state=opt_state,
            reward_params=reward_params,
        )

    # -------------------------------------------------------------------------
    # Single training step (operates on TrainStateParams only; static parts
    # are captured via closure)
    # -------------------------------------------------------------------------
    def train_step(idx: int, state: TrainStateParams) -> TrainStateParams:
        rng_key, sample_key = jax.random.split(state.rng_key)
        cur_eps = exploration_schedule(idx)

        def fwd_policy_fn(
            fwd_rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params
        ) -> chex.Array:
            current_model = eqx.combine(policy_params, policy_static)
            policy_outputs = current_model(env_obs)
            fwd_logits = policy_outputs["forward_logits"]
            _rng_key, expl_key = jax.random.split(fwd_rng_key)
            do_explore = jax.random.bernoulli(expl_key, cur_eps)
            fwd_logits = jnp.where(do_explore, 0, fwd_logits)
            return fwd_logits, policy_outputs

        rng_keys = jax.random.split(sample_key, cfg.num_envs)
        traj_data, final_states, _info = jax.vmap(
            lambda rng: gfnx.utils.forward_rollout(
                rng, fwd_policy_fn, state.model_params, env, env_params
            )
        )(rng_keys)
        # Post-rollout reward: [num_envs] terminal rewards broadcast to [num_envs * T]
        log_rewards = jax.vmap(reward_module.log_reward, in_axes=(0, None))(
            final_states, state.reward_params
        )
        transitions = jax.tree.map(
            lambda x: x.reshape((-1,) + x.shape[2:]),
            jax.vmap(gfnx.utils.split_traj_to_transitions)(traj_data),
        )
        T_steps = transitions.done.shape[0] // cfg.num_envs
        traj_rewards_flat = jnp.repeat(log_rewards, T_steps)  # [num_envs * T]

        bwd_actions = env.get_backward_action_batch(
            transitions.state,
            transitions.action,
            transitions.next_state,
            env_params,
        )

        def loss_fn(model_params, current_traj_rewards_flat):
            m = eqx.combine(model_params, policy_static)
            # Current state
            policy_outputs = jax.vmap(m)(transitions.obs)
            fwd_logits = policy_outputs["forward_logits"]
            log_flow = policy_outputs["log_flow"]
            invalid_mask = env.get_invalid_mask_batch(transitions.state, env_params)
            fwd_logprobs = gfnx.utils.compute_action_log_probs(
                fwd_logits, transitions.action, invalid_mask
            )

            # Next state
            next_policy_outputs = jax.vmap(m)(transitions.next_obs)
            next_log_flow = next_policy_outputs["log_flow"]
            bwd_logits = next_policy_outputs["backward_logits"]
            next_bwd_invalid_mask = env.get_invalid_backward_mask_batch(
                transitions.next_state, env_params
            )
            bwd_logprobs = gfnx.utils.compute_action_log_probs(
                bwd_logits, bwd_actions, next_bwd_invalid_mask
            )

            target = jnp.where(
                transitions.done,
                bwd_logprobs + current_traj_rewards_flat,
                bwd_logprobs + next_log_flow,
            )
            not_pad = jnp.logical_not(transitions.pad)
            loss = optax.losses.squared_error(
                jnp.where(transitions.pad, 0.0, fwd_logprobs + log_flow),
                jnp.where(transitions.pad, 0.0, target),
            )
            return loss.sum() / not_pad.sum()

        _mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(
            state.model_params, traj_rewards_flat
        )
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.model_params)
        new_model_params = eqx.apply_updates(state.model_params, updates)

        return TrainStateParams(
            rng_key=rng_key,
            model_params=new_model_params,
            opt_state=new_opt_state,
            reward_params=state.reward_params,
        )

    # -------------------------------------------------------------------------
    # Epoch function: evaluate then train for steps_per_eval steps.
    # Outer scan iterates over num_evals epochs.
    # -------------------------------------------------------------------------
    def epoch_fn(carry, epoch_idx):
        state, metrics_state = carry

        # 1. Evaluate: compute exact distribution → compute metrics
        processed = metrics_module.process(
            metrics_state,
            jax.random.key(0),
            metrics_module.ProcessArgs(
                policy_params=state.model_params, env_params=env_params
            ),
        )
        eval_info = metrics_module.get(processed)

        # 2. Train for steps_per_eval steps via inner scan
        def inner_step(carry, global_idx):
            state = carry
            new_state = train_step(global_idx, state)
            return new_state, None

        global_indices = epoch_idx * steps_per_eval + jnp.arange(steps_per_eval)
        state, _ = jax.lax.scan(inner_step, state, global_indices)

        return (state, metrics_state), eval_info

    # -------------------------------------------------------------------------
    # Vmap over seeds, outer scan over epochs
    # -------------------------------------------------------------------------
    seeds = jnp.arange(cfg.num_seeds)
    all_init_params = jax.vmap(make_init_params)(seeds)

    @jax.jit
    def run_all_seeds(all_params, all_metrics):
        def run_one_seed(params, metrics):
            _, metric_history = jax.lax.scan(
                epoch_fn, (params, metrics), jnp.arange(cfg.num_evals)
            )
            return metric_history

        return jax.vmap(run_one_seed)(all_params, all_metrics)

    log.info(
        f"Starting training: {cfg.num_seeds} seeds × {cfg.num_train_steps} steps, "
        f"{cfg.num_evals} evals every {steps_per_eval} steps"
    )
    # all_histories: dict[str, Array[num_seeds, num_evals]]
    all_histories = jax.block_until_ready(run_all_seeds(all_init_params, all_init_metrics))

    # -------------------------------------------------------------------------
    # Log metric history and save per-metric CSVs
    # -------------------------------------------------------------------------
    seed_cols = [f"seed_{i}" for i in range(cfg.num_seeds)]
    log.info("=== Metric history (mean ± std over seeds) ===")
    for key, values in all_histories.items():
        log.info(f"--- {key} ---")
        csv_path = Path(f"{key}.csv")
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + seed_cols)
            for i in range(cfg.num_evals):
                step = i * steps_per_eval
                row = [step] + [float(values[s, i]) for s in range(cfg.num_seeds)]
                writer.writerow(row)
                mean = float(values[:, i].mean())
                std = float(values[:, i].std())
                log.info(f"  step {step:6d}: {mean:.4f} ± {std:.4f}")
        log.info(f"  saved → {csv_path.resolve()}")


if __name__ == "__main__":
    run_experiment()
