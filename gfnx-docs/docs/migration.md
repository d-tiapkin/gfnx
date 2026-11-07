# Migrating from 0.0.1

The reward-decoupling refactor ([#5](https://github.com/d-tiapkin/gfnx/pull/5)) decouples rewards from environments and moves every environment
primitive to **single states** (no batch dimension). Batching is done with
`jax.vmap` by the caller. Most of the changes below follow from these two
decisions.

## At a glance

```python
# 0.0.1
reward_module = gfnx.EasyHypergridRewardModule()
env = gfnx.HypergridEnvironment(reward_module, dim=4, side=20)
env_params = env.init(key)  # also initialised reward params inside env_params
traj_data, info = gfnx.utils.forward_rollout(
    key, num_envs, policy_fn, policy_params, env, env_params
)
log_rewards = info["log_gfn_reward"]

# current
reward_module = gfnx.EasyHypergridRewardModule(side=20)
env = gfnx.HypergridEnvironment(dim=4, side=20)
env_params = env.init(env_key)                               # dynamics only
reward_params = reward_module.init(reward_key, env.reset())  # single dummy state
traj_data, final_states, info = jax.vmap(
    lambda k: gfnx.utils.forward_rollout(k, policy_fn, policy_params, env, env_params)
)(jax.random.split(key, num_envs))
log_rewards = jax.vmap(reward_module.log_reward, in_axes=(0, None))(final_states, reward_params)
```

## Environments

| 0.0.1 | Current |
|-------|-------|
| `HypergridEnvironment(reward_module, dim=..., side=...)` (same for every environment) | `HypergridEnvironment(dim=..., side=...)`: constructors no longer take `reward_module` |
| `env.init(key)` also initialised the reward; `env_params.reward_params` | `env.init(key)` returns dynamics params only; call `reward_module.init(key, env.reset())` and keep `reward_params` yourself. `BaseEnvParams.reward_params` is gone |
| `obs, state = env.reset(num_envs, env_params)` | `state = env.reset()`: one initial state, no batch dim; get the observation with `env.get_obs(state, env_params)` |
| `obs, next_state, log_reward, done, info = env.step(state, action, env_params)` | `obs, next_state, done, info = env.step(state, action, env_params)`: no reward |
| `obs, prev_state, log_reward, done, info = env.backward_step(state, bwd_action, env_params)` | `obs, prev_state, done, info = env.backward_step(state, bwd_action, env_params)` |
| `BaseVecEnvironment` (batched methods) | `BaseEnvironment`: every method takes a single state; `jax.vmap` over states for batches |
| `env.get_invalid_mask(states, env_params)` (batched, `True` = **invalid**) | `env.get_action_mask(state, env_params)` (single state, `True` = **valid**), or `env.get_action_mask_batch(states, env_params)` |
| `env.get_invalid_backward_mask(states, env_params)` (`True` = **invalid**) | `env.get_backward_action_mask(state, env_params)` / `get_backward_action_mask_batch` (`True` = **valid**) |
| `env.get_backward_action(...)` / `env.get_forward_action(...)` on batches | Single-transition methods; use `get_backward_action_batch` / `get_forward_action_batch` on `[B, ...]` inputs |
| `env.sample_action(key, logits)` / `env.sample_backward_action(...)` | Removed; rollouts sample internally. Use `jax.random.categorical` if you need it elsewhere |
| `env.get_mean_reward(env_params)` / `env.is_mean_reward_tractable` | `env.get_expected_reward(env_params, reward_module, reward_params)` / `env.is_expected_reward_tractable` |
| `env.get_true_distribution(env_params)`, `get_normalizing_constant(env_params)`, `get_ground_truth_sampling(key, n, env_params)` | Same methods with `reward_module, reward_params` as extra trailing arguments |
| — | New: `env.get_log_normalizing_constant(env_params, reward_module, reward_params)` returns log Z directly |

The mask flip is the easiest change to miss: anything that did
`jnp.logical_not(env.get_invalid_mask(...))` now passes `env.get_action_mask(...)` as is.

## Rewards

| 0.0.1 | Current |
|-------|-------|
| `reward_module.log_reward(states, env_params)` (batched) | `reward_module.log_reward(state, reward_params)` (single state); batch with `jax.vmap(reward_module.log_reward, in_axes=(0, None))(states, reward_params)` |
| `reward_module.init(key, batch_of_dummy_states)` | `reward_module.init(key, env.reset())`: a single dummy state; returns a `*RewardParams` dataclass (e.g. `HypergridRewardParams`, `TFBind8RewardParams`) |
| `EasyHypergridRewardModule()`, `HardHypergridRewardModule()`, `GeneralHypergridRewardModule(R0=..., ...)` | Keyword-only `side` is required: `EasyHypergridRewardModule(side=cfg.environment.side)` |
| Rewards returned by `env.step` and accumulated in rollouts | Compute once per trajectory on the terminal states returned by `forward_rollout` |

## Rollouts and masking utilities

| 0.0.1 | Current |
|-------|-------|
| `traj_data, info = forward_rollout(key, num_envs, policy_fn, params, env, env_params)` | `traj_data, final_state, info = forward_rollout(key, policy_fn, params, env, env_params)`: one trajectory `[T+1, ...]`; `jax.vmap` over split keys for `[B, T+1, ...]` |
| `info["final_env_state"]` | The second return value, `final_state` |
| `info["log_gfn_reward"]`, `TrajectoryData.log_gfn_reward`, `TransitionData.log_gfn_reward` | Removed; compute `log_rewards` from `final_states` (see above) |
| `backward_rollout(...)` returned `(traj_data, info)` | Returns `(traj_data, final_state, info)` for a single trajectory |
| `policy_fn(key, batched_obs, params)` | `policy_fn(key, obs, params)` receives a single observation and returns `[n_actions]` logits |
| `generic_rollout(...)` | Removed from the public API; use `forward_rollout` / `backward_rollout` |
| `split_traj_to_transitions(traj_data)` on `[B, T+1, ...]`, returned `[B*T, ...]` | Single trajectory `[T+1, ...]` → `[T, ...]`; use `jax.vmap(split_traj_to_transitions)(traj_data)` and flatten with `jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), ...)` |
| `mask_logits(logits, invalid_mask)` + `log_softmax` + `take_along_axis` | `compute_action_log_probs(logits, actions, action_mask, step_mask)` (`True` = valid / real step) |
| — | New: `masked_sum`, `masked_mean` (`True` = include), `state.is_valid`, `traj_data.valid`, `run_training_loop` |

## Metrics

| 0.0.1 | Current |
|-------|-------|
| `MeanRewardMetricsModule`, `MeanRewardMetricsState` | `ExpectedRewardMetricsModule`, `ExpectedRewardMetricsState` |
| `SWMeanRewardSWMetricsModule`, `SWMeanRewardMetricsState` | `SWExpectedRewardMetricsModule`, `SWExpectedRewardMetricsState` |
| `MeanRewardMetricsModule(env, env_params)` | `ExpectedRewardMetricsModule(env, env_params, reward_module, reward_params)` |
| `UpdateArgs(log_rewards=log_rewards)` | `UpdateArgs(rewards=jnp.exp(log_rewards))`: linear scale |
| Reward-dependent metrics (ELBO, EUBO, correlation, top-k, approximate / exact distribution) got rewards through the environment | Pass `reward_module=` to the constructor and the current `reward_params=` in `ProcessArgs` at each evaluation |
| — | New: `metrics_module.step(...)` combines `update` with eval-gated `process` + `get` |

## Custom environments

Subclass `BaseEnvironment` instead of `BaseVecEnvironment` and drop the
`reward_module` argument from `__init__`. Replace `get_init_state(num_envs)`
with `reset()`, and `_single_transition` / `_single_backward_transition` with
`step` / `backward_step` returning `(obs, next_state, done, info)`. Implement
`get_action_mask` / `get_backward_action_mask` for a single state with
`True` = valid; the `_batch` variants come from the base class.
