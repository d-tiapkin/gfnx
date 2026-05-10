# Walkthrough

This guide dissects the Detailed Balance (DB) baseline for the Hypergrid environment. The complete, runnable script lives in `baselines/db_hypergrid.py`; the sections below emphasize the design choices so you can extend the baseline to new settings. A second, shorter walkthrough at the end of this page covers a *vmapped-over-seeds* baseline (`baselines/tb_hypergrid_multiseed.py`) and shows how to scale the same pattern to many seeds in parallel.

### What you’ll learn

- How the DB objective is implemented in practice.
- How the policy network, optimizer, metrics, and training loop fit together.
- How to structure JAX/Equinox code so it compiles cleanly under `jax.jit`.
- How to vmap the entire training loop over random seeds for free.

### Before you start

- Install the project with baseline extras: `pip install -e '.[baselines]'`.
- Open `baselines/db_hypergrid.py` for the full reference while following this walkthrough.

## Recap: Detailed Balance objective

The DB loss enforces pairwise flow consistency between forward and backward transitions:

$$
\mathcal{L}_{\text{DB}}(\theta; s \rightarrow s') =
\left[
  \log \frac{P_F(s' \mid s; \theta)\,\mathcal{F}(s; \theta)}
           {P_B(s \mid s'; \theta)\,\mathcal{F}(s'; \theta)}
\right]^2,
$$

where for terminal $s'$ we swap $\mathcal{F}(s'; \theta)$ for the reward $R(s')$, and $\theta$ denotes the policy-network parameters.

## Step&nbsp;1 – Policy network

We need a network that parametrizes (1) the forward policy $P_F$, (2) the log-flow $\log \mathcal{F}$, and (3) optionally the backward policy $P_B$. To keep everything JAX-friendly we rely on [Equinox](https://github.com/patrick-kidger/equinox), which lets us define a module once and treat its parameters as a PyTree throughout the training loop.

The module below produces all three heads in a single forward pass and operates on a **single** observation — the rollout / loss code vmaps it externally where needed.

```python
class MLPPolicy(eqx.Module):
    """Shared MLP head for forward policy, log-flow, and optional backward policy."""

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

        output_size = self.n_fwd_actions + 1  # +1 for log-flow
        if train_backward_policy:
            output_size += n_bwd_actions
        self.network = eqx.nn.MLP(
            in_size=input_size,
            out_size=output_size,
            width_size=hidden_size,
            depth=depth,
            key=rng_key,
        )

    def __call__(self, x: chex.Array) -> dict[str, chex.Array]:
        x = self.network(x)
        if self.train_backward_policy:
            forward_logits, log_flow, backward_logits = jnp.split(
                x, [self.n_fwd_actions, self.n_fwd_actions + 1], axis=-1
            )
        else:
            forward_logits, log_flow = jnp.split(x, [self.n_fwd_actions], axis=-1)
            backward_logits = jnp.zeros((self.n_bwd_actions,), dtype=jnp.float32)
        return {
            "forward_logits": forward_logits,
            "log_flow": log_flow.squeeze(-1),
            "backward_logits": backward_logits,
        }
```

## Step&nbsp;2 – Initialization and train-state definition

Next we recreate the `run_experiment` setup from `baselines/db_hypergrid.py`. This step wires together the reward, environment, model, optimizer, metrics, and the train state that will be threaded through `jax.jit`.

**JAX RNG hygiene.** Always split PRNG keys before reusing them: `rng_key, subkey = jax.random.split(rng_key)`. Treat keys as immutable tokens; otherwise subtle stochastic bugs creep in.

### 2.1 Reward and environment

Environments are **decoupled from rewards**: the env constructor takes no `reward_module`, and reward parameters are produced by a separate `reward_module.init(rng_key, dummy_state)` call. The two halves can then be evolved independently — handy when the reward is large (a lookup table, a proxy network) or trainable (e.g. the Ising J matrix).

```python
# Build the reward module first — hypergrid rewards need `side` to compute coordinates.
reward_module = gfnx.EasyHypergridRewardModule(side=cfg.environment.side)

# Build the environment with no reward attached.
env = gfnx.environment.HypergridEnvironment(
    dim=cfg.environment.dim, side=cfg.environment.side
)
env_init_key = jax.random.PRNGKey(cfg.env_init_seed)
env_init_key, reward_init_key = jax.random.split(env_init_key)
env_params = env.init(env_init_key)

# Use a separate key for reward init — reward and env may have independent stochastic state.
reward_params = reward_module.init(reward_init_key, env.reset())
```

### 2.2 Policy network

```python
rng_key, net_init_key = jax.random.split(rng_key)
model = MLPPolicy(
    input_size=env.observation_space.shape[0],
    n_fwd_actions=env.action_space.n,
    n_bwd_actions=env.backward_action_space.n,
    hidden_size=256,
    train_backward_policy=True,
    depth=3,
    rng_key=net_init_key,
)
```

Equinox stores parameters inside `model`, so no extra init call is required.

### 2.3 Optimizer and exploration schedule

```python
exploration_schedule = optax.linear_schedule(
    init_value=1.0,
    end_value=0.0,
    transition_steps=1_000,
)
optimizer = optax.adam(learning_rate=3e-4)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
```

### 2.4 Metrics module

Distribution-based metrics need both the reward module (passed at construction) and the current `reward_params` (passed via `InitArgs`) to compute the ground-truth distribution.

```python
metrics_module = ApproxDistributionMetricsModule(
    metrics=["tv", "kl", "2d_marginal_distribution"],
    env=env,
    reward_module=reward_module,
    buffer_size=200_000,
)
eval_init_key = jax.random.PRNGKey(cfg.eval_init_seed)
eval_init_key, new_eval_init_key = jax.random.split(eval_init_key)
metrics_state = metrics_module.init(
    new_eval_init_key,
    metrics_module.InitArgs(env_params=env_params, reward_params=reward_params),
)
eval_info = metrics_module.get(metrics_state)
```

### 2.5 Combine everything into `TrainState`

`TrainState` bundles all training state into a single object so it can be threaded through `jax.jit` and `jax.lax.fori_loop`. Its fields split into two categories:

- **Static** (non-array Python objects — captured in the JIT closure via `eqx.partition`, never traced): `config`, `env`, `reward_module`, `model`, `optimizer`, `metrics_module`, `exploration_schedule`
- **Dynamic** (JAX array trees — threaded through `fori_loop` as traced values): `rng_key`, `env_params`, `reward_params`, `opt_state`, `metrics_state`, `eval_info`

`eqx.partition(train_state, eqx.is_array)` performs this split automatically; `eqx.combine` reconstructs the full object at the start of each training step (see section 2.6).

```python
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.HypergridEnvironment
    env_params: gfnx.HypergridEnvParams
    reward_module: gfnx.GeneralHypergridRewardModule  # static
    reward_params: gfnx.HypergridRewardParams        # dynamic
    model: MLPPolicy
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    metrics_module: ApproxDistributionMetricsModule
    metrics_state: ApproxDistributionMetricsState
    exploration_schedule: optax.Schedule
    eval_info: dict


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
```

### 2.6 Run the training loop

Most baselines wrap the standard `eqx.partition` / `jax.lax.fori_loop` / `block_until_ready` boilerplate behind a single helper:

```python
train_state = gfnx.utils.run_training_loop(
    train_step,
    train_state,
    cfg.num_train_steps,
    cfg.logging["tqdm_print_rate"],
)
```

Internally it splits the train state into JIT-friendly arrays (`eqx.partition(..., eqx.is_array)`), runs `train_step` under `jax.lax.fori_loop`, and recombines the static parts on the way out.

## Step&nbsp;3 – Implement `train_step`

`train_step` is decorated with `@eqx.filter_jit`, so the entire body — rollouts, loss, gradient update, metrics — compiles into one XLA program. Pull out the components we'll reuse:

```python
@eqx.filter_jit
def train_step(idx: int, train_state: TrainState) -> TrainState:
    rng_key = train_state.rng_key
    num_envs = 16
    env = train_state.env
    env_params = train_state.env_params
    metrics_module = train_state.metrics_module
```

### 3.1 Generate trajectories

`gfnx.utils.forward_rollout` runs **one** environment under a given policy and pads the result to `env.max_steps_in_episode + 1` steps. To collect `num_envs` trajectories in parallel we split the RNG key and `jax.vmap` over the leading axis. Reward is computed *post*-rollout on the terminal states — `TrajectoryData` no longer carries a `log_gfn_reward` field.

```python
rng_key, sample_traj_key = jax.random.split(train_state.rng_key)
policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)
cur_epsilon = train_state.exploration_schedule(idx)


def fwd_policy_fn(rng_key, env_obs, policy_params):
    policy = eqx.combine(policy_params, policy_static)
    policy_outputs = policy(env_obs)  # SINGLE obs — no batch dim
    do_explore = jax.random.bernoulli(rng_key, cur_epsilon)
    forward_logits = jnp.where(do_explore, 0, policy_outputs["forward_logits"])
    return forward_logits, policy_outputs


rng_keys = jax.random.split(sample_traj_key, num_envs)
traj_data, final_states, info = jax.vmap(
    lambda rng: gfnx.utils.forward_rollout(
        rng, fwd_policy_fn, policy_params, env, env_params
    )
)(rng_keys)

# Reward is decoupled — compute it on the batch of terminal states.
log_rewards = jax.vmap(
    train_state.reward_module.log_reward, in_axes=(0, None)
)(final_states, train_state.reward_params)  # [B]
```

The DB loss works on transitions, so we split each trajectory into single-step samples (then flatten the batch and time axes together) and request the matching backward actions. The library's `_batch` helpers come straight from `BaseEnvironment` — no need to vmap manually:

```python
transitions = jax.tree.map(
    lambda x: x.reshape((-1,) + x.shape[2:]),
    jax.vmap(gfnx.utils.split_traj_to_transitions)(traj_data),
)  # [B*T, ...]
T_steps = transitions.done.shape[0] // num_envs
traj_rewards_flat = jnp.repeat(log_rewards, T_steps)  # [B*T]
bwd_actions = env.get_backward_action_batch(
    transitions.state,
    transitions.action,
    transitions.next_state,
    env_params,
)
```

For logging we also estimate the RL/ELBO reward on each trajectory:

```python
_, log_pb_traj = jax.vmap(
    lambda td: gfnx.utils.forward_trajectory_log_probs(env, td, env_params)
)(traj_data)
rl_reward = log_pb_traj + log_rewards + info["entropy"]
```

### 3.2 Loss function

Two helpers from `gfnx.utils` cut the ceremony to one line each:

- `get_action_mask_batch` / `get_backward_action_mask_batch` follow the **`True = valid`** convention — pass them straight to `jax.nn.log_softmax(..., where=mask)`.
- `gfnx.utils.compute_action_log_probs(logits, actions, action_mask, step_mask=None)` performs masked `log_softmax` + `take_along_axis` + (optional) zero-out of padding steps in one call.
- `transitions.valid` (= `~transitions.pad`) gives the per-step `True = real step` mask.

```python
def loss_fn(model: MLPPolicy, current_traj_rewards_flat: jnp.ndarray) -> chex.Array:
    policy_outputs = jax.vmap(model, in_axes=(0,))(transitions.obs)
    fwd_logits = policy_outputs["forward_logits"]
    action_mask = env.get_action_mask_batch(transitions.state, env_params)
    fwd_logprobs = gfnx.utils.compute_action_log_probs(
        fwd_logits, transitions.action, action_mask
    )
    log_flow = policy_outputs["log_flow"]

    next_policy_outputs = jax.vmap(model, in_axes=(0,))(transitions.next_obs)
    bwd_logits = next_policy_outputs["backward_logits"]
    next_backward_action_mask = env.get_backward_action_mask_batch(
        transitions.next_state, env_params
    )
    bwd_logprobs = gfnx.utils.compute_action_log_probs(
        bwd_logits, bwd_actions, next_backward_action_mask
    )
    next_log_flow = next_policy_outputs["log_flow"]

    # Replace the target with log R(s') at terminal transitions.
    target = jnp.where(
        transitions.done,
        bwd_logprobs + current_traj_rewards_flat,
        bwd_logprobs + next_log_flow,
    )

    # Masked DB loss: only count real (non-padding) transitions.
    valid = transitions.valid
    num_transition = valid.sum()
    loss = optax.l2_loss(
        jnp.where(valid, fwd_logprobs + log_flow, 0.0),
        jnp.where(valid, target, 0.0),
    ).sum()
    return loss / num_transition
```

### 3.3 Perform the gradient update

```python
mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(
    train_state.model, traj_rewards_flat
)
updates, opt_state = train_state.optimizer.update(
    grads,
    train_state.opt_state,
    eqx.filter(train_state.model, eqx.is_array),
)
model = eqx.apply_updates(train_state.model, updates)
```

### 3.4 Evaluation and logging

`BaseMetricsModule.step(...)` runs the cheap `update` every step and the expensive `process` + `get` only on eval steps (`jax.lax.cond` internally — stays JIT-compatible). On non-eval steps it returns `prev_eval_info` unchanged. This collapses the historical update / `cond(process)` / `cond(get)` boilerplate into a single call:

```python
metrics_state, eval_info = metrics_module.step(
    idx=idx,
    metrics_state=train_state.metrics_state,
    rng_key=jax.random.key(0),  # not used by ApproxDistribution
    update_args=metrics_module.UpdateArgs(states=final_states),
    process_args=metrics_module.ProcessArgs(env_params=env_params),
    eval_each=train_state.config.logging.eval_each,
    num_train_steps=train_state.config.num_train_steps,
    prev_eval_info=train_state.eval_info,
)
```

To log scalar summaries from inside the JIT we rely on `jax.debug.callback`. Setting `ordered=True` ensures host-side logging respects device execution order even with asynchronous execution:

```python
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
```

### 3.5 Final update of the train state

Because `train_step` is functional we finish by returning an updated `TrainState`:

```python
return train_state._replace(
    rng_key=rng_key,
    model=model,
    opt_state=opt_state,
    metrics_state=metrics_state,
    eval_info=eval_info,
)
```

From here you can add checkpointing, richer loggers, or alternate objectives without changing the core training flow. Refer back to `baselines/db_hypergrid.py` for the exact Hydra configuration and CLI entry point.

## Vmapping training over seeds (proof of concept)

`baselines/tb_hypergrid_multiseed.py` shows how to scale the same pattern to many random seeds *in parallel* by `jax.vmap`-ing the entire training loop. Conceptually nothing changes — same Trajectory Balance loss, same env, same metric — but a few JAX constraints have to be respected. This section highlights what's *new* relative to the single-seed walkthrough; refer to the script for the full code.

### What can't go inside a vmap

- **`jax.debug.callback`** — host-side I/O does not vectorise. The multiseed scripts drop per-step logging entirely and emit a CSV of metric history (mean ± std over seeds) once training finishes.
- **`jax.lax.cond` for eval gating** — under `vmap` *both* branches execute, which destroys the speed-up. The trick is to schedule evaluations at fixed intervals via a two-level `jax.lax.scan` (outer = `num_evals` epochs, inner = `steps_per_eval` train steps) so the eval branch runs unconditionally.
- **Replay buffers tied to a single trajectory** — `ApproxDistributionMetricsModule` is fine for single-seed runs but becomes awkward to vmap. The multiseed script swaps to `ExactDistributionMetricsModule`, which evaluates the policy distribution by power iteration on the enumerated state graph — no buffer needed.

### Carry only JAX arrays through `vmap`

Split static and dynamic state explicitly. Static parts (`env`, `reward_module`, `policy_static`, `optimizer`) are captured by closure; dynamic parts live in a `chex.dataclass` that contains *only* JAX arrays:

```python
@chex.dataclass
class TrainStateParams:
    rng_key: chex.PRNGKey
    model_params: chex.ArrayTree  # eqx.filter(model, eqx.is_array)
    logZ: chex.Array
    opt_state: optax.OptState
    reward_params: gfnx.HypergridRewardParams
```

A per-seed initializer builds the model, reward params, and optimizer state for one seed; `jax.vmap` then tiles it across all seeds:

```python
def make_init_params(seed: chex.Array) -> TrainStateParams:
    rng_key = jax.random.PRNGKey(seed)
    rng_key, net_key, reward_key = jax.random.split(rng_key, 3)
    model = MLPPolicy(...)
    model_params = eqx.filter(model, eqx.is_array)
    reward_params = reward_module.init(reward_key, env.reset())
    opt_state = optimizer.init({"model_params": model_params, "logZ": jnp.array(0.0)})
    return TrainStateParams(
        rng_key=rng_key,
        model_params=model_params,
        logZ=jnp.array(0.0),
        opt_state=opt_state,
        reward_params=reward_params,
    )


seeds = jnp.arange(cfg.num_seeds)
all_init_params = jax.vmap(make_init_params)(seeds)
```

### Two-level scan: evaluate, then train

The body of one seed's training loop nests two `lax.scan`s. The outer one runs `num_evals` epochs; each epoch first evaluates the metric (always, no `cond`) and then runs `steps_per_eval` training steps via the inner scan:

```python
def epoch_fn(carry, epoch_idx):
    state, metrics_state = carry

    # 1. Evaluate.
    processed = metrics_module.process(
        metrics_state,
        jax.random.key(0),
        metrics_module.ProcessArgs(
            policy_params=state.model_params, env_params=env_params
        ),
    )
    eval_info = metrics_module.get(processed)

    # 2. Run a chunk of training steps.
    def inner_step(carry, global_idx):
        return train_step(global_idx, carry), None

    global_indices = epoch_idx * steps_per_eval + jnp.arange(steps_per_eval)
    state, _ = jax.lax.scan(inner_step, state, global_indices)

    return (state, metrics_state), eval_info
```

`epoch_fn` returns the per-epoch `eval_info`; the outer scan collects it into a `[num_evals, ...]` history, and a final post-training evaluation is appended to the history before returning.

### Vmap over seeds, jit the whole thing

```python
@jax.jit
def run_all_seeds(all_params, all_metrics):
    def run_one_seed(params, metrics):
        final_carry, metric_history = jax.lax.scan(
            epoch_fn, (params, metrics), jnp.arange(cfg.num_evals)
        )
        # Append a final post-training evaluation so the last chunk is covered.
        ...
        return metric_history  # dict of [num_evals + 1, ...]

    return jax.vmap(run_one_seed)(all_params, all_metrics)


all_histories = jax.block_until_ready(run_all_seeds(all_init_params, all_init_metrics))
# all_histories: dict[str, Array[num_seeds, num_evals + 1]]
```

Aggregating mean ± std over the `num_seeds` axis happens *after* `block_until_ready` — purely host-side Python, free of vmap constraints.

### When to use this pattern

The vmapped pattern shines when individual seeds are cheap relative to compile time, when the metric does not require side-effectful logging, and when the environment supports an exact (or otherwise vmap-friendly) evaluation procedure. Production training loops on accelerators happily fit dozens of seeds inside one device, turning a sweep of independent runs into one `block_until_ready`. For everything else the single-seed script in the previous sections remains the right starting point.
