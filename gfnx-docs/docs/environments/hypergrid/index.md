# Hypergrid Environment

The hypergrid environment is a classic GFlowNet benchmark: you build a point on
a $D$-dimensional grid by incrementing coordinates one step at a time and then
decide when to stop. Because every transition is discrete and the search space
is finite and enumerable, the environment is great for debugging objectives and
policies before moving on to more complicated domains.

!!! note "Single source of truth"
    The specification below (states, actions, rewards, parameters) is rendered
    directly from the docstrings of
    [`HypergridEnvironment`][gfnx.environment.hypergrid.HypergridEnvironment] and
    [`GeneralHypergridRewardModule`][gfnx.reward.hypergrid.GeneralHypergridRewardModule],
    so it is also what `help(gfnx.HypergridEnvironment)` prints.

## Specification

::: gfnx.environment.hypergrid.HypergridEnvironment
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false
      heading_level: 3

## Reward

::: gfnx.reward.hypergrid.GeneralHypergridRewardModule
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false
      heading_level: 3

Two presets fix $(R_0, R_1, R_2)$:

- [`EasyHypergridRewardModule(side=...)`][gfnx.reward.hypergrid.EasyHypergridRewardModule]
  — $(10^{-3}, 0.5, 2.0)$.
- [`HardHypergridRewardModule(side=...)`][gfnx.reward.hypergrid.HardHypergridRewardModule]
  — $(10^{-4}, 1.0, 3.0)$: a much lower background reward, so the modes are
  harder to discover.

You can plug in your own reward by subclassing
[`BaseRewardModule`][gfnx.base.BaseRewardModule]; it only needs `init`,
`log_reward` and `reward` on a single state.

## Quick start

```python
import jax
import jax.numpy as jnp
import gfnx

# 1. Build the reward module. The hypergrid rewards need to know the grid
#    size in order to compute coordinates, so `side` is passed at construction.
reward_module = gfnx.EasyHypergridRewardModule(side=5)

# 2. Build the environment (no reward module is passed in).
env = gfnx.HypergridEnvironment(dim=3, side=5)
env_params = env.init(jax.random.PRNGKey(0))

# 3. Initialise reward parameters using a dummy state.
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

# 4. Reset to get a single initial state (no batch dimension).
state = env.reset()

# 5. Take a forward step (increment coordinate 0).
action = jnp.int32(0)
obs, state, done, _ = env.step(state, action, env_params)

# 6. Stop when you are ready to terminate the trajectory. The stop action is
#    the last action in the action space.
stop = jnp.int32(env.action_space.n - 1)
obs, state, done, _ = env.step(state, stop, env_params)

# 7. Reward is computed on the terminal state via the reward module.
log_reward = reward_module.log_reward(state, reward_params)
print("Terminal?", bool(state.is_terminal))
print("Log reward:", float(log_reward))
```

All environment methods operate on a **single** state — to roll out multiple
trajectories in parallel, wrap them with `jax.vmap`. For example:

```python
rng_keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
traj_data, final_states, info = jax.vmap(
    lambda rng: gfnx.utils.forward_rollout(
        rng, policy_fn, policy_params, env, env_params
    )
)(rng_keys)
log_rewards = jax.vmap(reward_module.log_reward, in_axes=(0, None))(
    final_states, reward_params
)
```

## Exploring the grid

The hypergrid exposes utilities that are handy for analysis and evaluation.
Because rewards are decoupled, the enumerable helpers take both the reward
module and its parameters explicitly:

```python
# Enumerate the full state space.
all_states = env.get_all_states(env_params)

# Compute the exact partition function and reward-proportional distribution.
Z = env.get_normalizing_constant(env_params, reward_module, reward_params)
true_dist = env.get_true_distribution(
    env_params, reward_module, reward_params
)  # shape = (side,)*dim

# Draw samples directly from the ground-truth distribution.
gt_state = env.get_ground_truth_sampling(
    rng_key=jax.random.PRNGKey(1),
    batch_size=4,
    env_params=env_params,
    reward_module=reward_module,
    reward_params=reward_params,
)
```

These helpers are invaluable for sanity checks (does your policy match the true
distribution?) and for tracking metrics such as the mean reward or the KL
divergence to the ground truth — see
[`ExactDistributionMetricsModule`](../../metrics/distribution_based/exact.md).

## API references

- [Environment](environment_api.md)
- [Reward module](reward_api.md)
