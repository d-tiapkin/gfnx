# Walkthrough

This guide dissects the Detailed Balance (DB) baseline for the Hypergrid environment. The complete, runnable script lives in `baselines/db_hypergrid.py`; the sections below emphasize the design choices so you can extend the baseline to new settings.

### What you’ll learn

- How the DB objective is implemented in practice.  
- How the policy network, optimizer, metrics, and training loop fit together.  
- How to structure JAX/Eqinox code so it compiles cleanly under `jax.jit`.

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

where terminal $s'$ swap $\mathcal{F}(s'; \theta)$ for the reward $R(s')$ and $\theta$ denotes the policy-network parameters.

## Step&nbsp;1 – Policy network

As a first step we need a network that parametrizes (1) the forward policy $P_F$, (2) the log-flow $\log \mathcal{F}$, and (3) optionally the backward policy $P_B$. To keep everything JAX-friendly we rely on [Equinox](https://github.com/patrick-kidger/equinox), which lets us define a module once and treat its parameters as a PyTree throughout the training loop.

The module below is exactly what `baselines/db_hypergrid.py` uses; it produces all three heads in one pass so we do not have to juggle multiple networks.

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

```python
reward_module = gfnx.EasyHypergridRewardModule()
env = gfnx.environment.HypergridEnvironment(
    reward_module, dim=cfg.environment.dim, side=cfg.environment.side
)
env_init_key = jax.random.PRNGKey(cfg.env_seed)
env_params = env.init(env_init_key)  # dummy for Hypergrid, non-trivial elsewhere
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

```python
metrics_module = ApproxDistributionMetricsModule(
    metrics=["tv", "kl", "2d_marginal_distribution"],
    env=env,
    buffer_size=200_000,
)
eval_init_key = jax.random.PRNGKey(cfg.eval_init_seed)
eval_init_key, new_eval_init_key = jax.random.split(eval_init_key)
metrics_state = metrics_module.init(
    new_eval_init_key,
    metrics_module.InitArgs(env_params=env_params),
)
eval_info = metrics_module.get(metrics_state)
```

`InitArgs` keeps the initialization signature explicit even when multiple metrics require different inputs; it avoids a tangle of `**kwargs` and makes IDE type checking happier.

### 2.5 Combine everything into `TrainState`

```python
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.HypergridEnvironment
    env_params: chex.Array
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
    model=model,
    optimizer=optimizer,
    opt_state=opt_state,
    metrics_module=metrics_module,
    metrics_state=metrics_state,
    exploration_schedule=exploration_schedule,
    eval_info=eval_info,
)
```

### 2.6 Define the training loop

At this point we assume there exists a `train_step` function that takes `(idx, train_state)` and returns an updated `TrainState`. We want to iterate it `cfg.num_train_steps` times using `jax.lax.fori_loop`, but we **cannot** feed the entire `TrainState` directly because it contains non-JIT-friendly objects (modules, configs, callables). `fori_loop` JIT-compiles `body_fun` under the hood, so we first partition the state into JAX arrays and static leftovers:

```python
train_state_params, train_state_static = eqx.partition(train_state, eqx.is_array)

@functools.partial(jax.jit, donate_argnums=(1,))
def train_step_wrapper(idx: int, state_params):
    state = eqx.combine(state_params, train_state_static)
    state = train_step(idx, state)
    state_params, _ = eqx.partition(state, eqx.is_array)
    return state_params

train_state_params = jax.lax.fori_loop(
    lower=0,
    upper=cfg.num_train_steps,
    body_fun=train_step_wrapper,
    init_val=train_state_params,
)
train_state_params = jax.block_until_ready(train_state_params)
train_state = eqx.combine(train_state_params, train_state_static)
```

## Step&nbsp;3 – Implement `train_step`

Before diving into the per-step logic we pull out the components we need:

```python
num_envs = 16
env = train_state.env
env_params = train_state.env_params
metrics_module = train_state.metrics_module
```

### 3.1 Generate trajectories

To gather data we reuse `gfnx.utils.forward_rollout`, which generate the `num_envs` parallel trajectories, using a corresponding policy function. Imporantly, it pads trajectories to the maximum possible length and is agnostic to the NN framework. The only requirement is to provide a pure `policy_fn` that emits logits (and auxiliary info) for the current batch of observations.

```python
rng_key, sample_traj_key = jax.random.split(train_state.rng_key)
policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)
cur_epsilon = train_state.exploration_schedule(idx)

def fwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params):
    policy = eqx.combine(policy_params, policy_static)
    policy_outputs = jax.vmap(policy, in_axes=(0,))(env_obs)
    do_explore = jax.random.bernoulli(rng_key, cur_epsilon, shape=(env_obs.shape[0],))
    forward_logits = jnp.where(
        do_explore[..., jnp.newaxis], 0, policy_outputs["forward_logits"]
    )
    return forward_logits, policy_outputs

traj_data, log_info = gfnx.utils.forward_rollout(
    rng_key=sample_traj_key,
    num_envs=num_envs,
    policy_fn=fwd_policy_fn,
    policy_params=policy_params,
    env=env,
    env_params=env_params,
)
```

The DB loss works on state transitions, so we split the padded trajectory into single-step samples and compute the matching backward actions:

```python
transitions = gfnx.utils.split_traj_to_transitions(traj_data)
bwd_actions = env.get_backward_action(
    transitions.state,
    transitions.action,
    transitions.next_state,
    env_params,
)
```

For logging, we also estimate the RL/ELBO reward:

```python
_, log_pb_traj = gfnx.utils.forward_trajectory_log_probs(env, traj_data, env_params)
rl_reward = log_pb_traj + log_info["log_gfn_reward"] + log_info["entropy"]
```

### 3.2 Loss function

We define the DB loss using an Equinox-style closure: the only argument is the model, everything else closes over the current batch. There is no need to JIT the loss separately—it gets traced as part of the surrounding training loop.

```python
def loss_fn(model: MLPPolicy) -> chex.Array:
    policy_outputs = jax.vmap(model, in_axes=(0,))(transitions.obs)
    fwd_logits = policy_outputs["forward_logits"]
    invalid_mask = env.get_invalid_mask(transitions.state, env_params)
    fwd_all_log_probs = jax.nn.log_softmax(
        fwd_logits, where=jnp.logical_not(invalid_mask), axis=-1
    )
    fwd_logprobs = jnp.take_along_axis(
        fwd_all_log_probs,
        jnp.expand_dims(transitions.action, axis=-1),
        axis=-1,
    ).squeeze(-1)
    log_flow = policy_outputs["log_flow"]

    next_policy_outputs = jax.vmap(model, in_axes=(0,))(transitions.next_obs)
    bwd_logits = next_policy_outputs["backward_logits"]
    next_bwd_invalid_mask = env.get_invalid_backward_mask(
        transitions.next_state, env_params
    )
    bwd_all_log_probs = jax.nn.log_softmax(
        bwd_logits, where=jnp.logical_not(next_bwd_invalid_mask), axis=-1
    )
    bwd_logprobs = jnp.take_along_axis(
        bwd_all_log_probs, jnp.expand_dims(bwd_actions, axis=-1), axis=-1
    ).squeeze(-1)
    next_log_flow = next_policy_outputs["log_flow"]

    target = jnp.where(
        transitions.done,
        bwd_logprobs + transitions.log_gfn_reward,
        bwd_logprobs + next_log_flow,
    )
    num_transition = jnp.logical_not(transitions.pad).sum()
    loss = optax.l2_loss(
        jnp.where(transitions.pad, 0.0, fwd_logprobs + log_flow),
        jnp.where(transitions.pad, 0.0, target),
    ).sum()
    return loss / num_transition
```

### 3.3 Perform the gradient update

```python
mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(train_state.model)
updates, opt_state = train_state.optimizer.update(
    grads,
    train_state.opt_state,
    eqx.filter(train_state.model, eqx.is_array),
)
model = eqx.apply_updates(train_state.model, updates)
```

### 3.4 Evaluation and logging

The metrics stack has two tiers: `update` (cheap, every step) and `process` (expensive, only on evaluation steps). We first apply the lightweight update using the final states from the rollout:

```python
metrics_state = metrics_module.update(
    train_state.metrics_state,
    rng_key=jax.random.key(0),  # not used in this module
    args=metrics_module.UpdateArgs(states=log_info["final_env_state"]),
)
```

Then we optionally run the heavy evaluation pass and collect metrics:

```python
is_eval_step = idx % train_state.config.logging.eval_each == 0
is_eval_step = is_eval_step | (idx + 1 == train_state.config.num_train_steps)

metrics_state = jax.lax.cond(
    is_eval_step,
    lambda kwargs: metrics_module.process(**kwargs),
    lambda kwargs: kwargs["metrics_state"],
    {
        "metrics_state": metrics_state,
        "rng_key": jax.random.key(0),  # not used here either
        "args": metrics_module.ProcessArgs(env_params=env_params),
    },
)
eval_info = jax.lax.cond(
    is_eval_step,
    lambda state: metrics_module.get(state),
    lambda state: train_state.eval_info,
    metrics_state,
)
```

To log scalar summaries from inside the JIT we rely on `jax.debug.callback`. Setting `ordered=True` ensures that host-side logging respects device execution order even with asynchronous execution:

```python
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
```

### 3.5 Final update of the train state

Since `train_step` is written in a functional style, we finish by returning an updated `TrainState`:

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
