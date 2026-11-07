# Hypergrid Environment

The hypergrid environment is a classic GFlowNet benchmark: you build a point on
a $D$-dimensional grid by incrementing coordinates one step at a time and then
decide when to stop. The terminal state is the final grid cell you land on, and
its reward is given by a user-chosen hypergrid reward module. Because every
transition is discrete and the search space is finite, the environment is great
for debugging objectives and policies before moving on to more complicated
domains.

## Intuition

- **State**: a vector of length `dim`, with each coordinate in
  `[0, side - 1]`. The all-zero vector is the initial (empty) state.
- **Actions**: choose a coordinate to increment or issue a *stop* action.
  Coordinates that already reached `side - 1` are automatically clamped, so
  the agent cannot walk off the grid.
- **Trajectory**: a sequence of increments that ends with the stop action.
- **Reward**: determined by a separate hypergrid reward module; higher values
  make the corresponding grid cells more likely under an ideal GFlowNet.

## Key parameters

- `dim`: number of dimensions (default `4`). Increasing it makes the grid grow
  exponentially; keep this small for quick experiments.
- `side`: number of discrete positions per dimension (default `20`).

The reward module is constructed and initialised separately — environments are
fully decoupled from rewards (see the library overview for the rationale).

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

## Reward options

- By default the reward assigned to a terminal state $s = (s^1, \ldots, s^D)$ with side length
  `side = H` follows

  $$
  \mathcal{R}(s) = R_0
  + R_1 \prod_{i=1}^D \mathbb{I}\left[0.25 < \left|\frac{s^i}{H-1}-0.5\right|\right]
  + R_2 \prod_{i=1}^D \mathbb{I}\left[0.3 < \left|\tfrac{s^i}{H-1}-0.5\right| < 0.4\right].
  $$

  The indicator products carve out $2^D$ symmetric modes: instead of peaking at the grid centre,
  the reward places mass on annuli that sit away from the middle, making exploration highly
  multimodal. Adjusting $(R_0, R_1, R_2)$ changes how prominent each ring of modes is.

- `gfnx.EasyHypergridRewardModule(side=...)` – baseline reward with a gentle
  peak near the centre of the mode.
- `gfnx.HardHypergridRewardModule(side=...)` – sharper peaks that make
  exploration and credit assignment harder.
- `gfnx.GeneralHypergridRewardModule(side=..., R0=..., R1=..., R2=...)` –
  customise how wide the reward plateaus are by tuning the coefficients.

You can plug your own reward by subclassing `BaseRewardModule`. It only needs to
implement `init`, `log_reward`, and `reward` on a single state.

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
distribution?) and for tracking metrics such as the mean reward or KL divergence
to ground truth. For a deeper dive into the environment and reward APIs, check
the companion pages in this section of the docs.

## API references:

- [Environment](environment_api.md)
- [Reward module](reward_api.md)

## References

- Bengio, E. *et&nbsp;al.* (2021). *Flow network based generative models for non-iterative diverse candidate generation.* 
  Advances in Neural Information Processing Systems (NeurIPS).
