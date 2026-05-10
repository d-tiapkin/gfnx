# Mean Reward & Reward Delta

The reward-delta metrics track how the empirical mean reward compares to the
true mean reward of the environment. They are useful whenever the environment
can provide ground-truth reward expectation (`env.is_expected_reward_tractable = True`).

## Intuition

- Accumulate rewards (log or linear) over time to estimate the empirical mean
  and compare it against the ground-truth mean supplied by the environment.
- Large absolute or relative deltas highlight when the current policy drifts
  away from the environment’s reward-weighted distribution, making it a simple
  health check during training.
- The sliding-window variant keeps a short-term view of recent rewards, while
  the global metric continues to aggregate from the beginning of training.
- Keep the reward scale consistent (log or linear) across updates so the metric
  reports meaningful averages and deltas.

## Modules

- `ExpectedRewardMetricsModule`: maintains running sums of collected rewards and
  reports the global mean, absolute delta, and relative delta.
- `SWExpectedRewardMetricsModule`: keeps a sliding window of the most recent
  rewards using a [`flashbax`](https://github.com/instadeepai/flashbax) buffer, offering the same statistics 
  but focused on recent performance.

Both modules return a dictionary with keys `mean_reward`, `reward_delta`, and
`rel_reward_delta`.

| Key | Meaning |
| --- | --- |
| `mean_reward` | Empirical mean of the collected rewards (use log rewards or linear rewards consistently). |
| `reward_delta` | Absolute difference between empirical and ground-truth means in the same scale. |
| `rel_reward_delta` | Relative error (absolute delta divided by ground-truth mean). |

## Inputs

- `env`, `env_params`: used during construction to query the ground-truth mean.
- arrays of sampled rewards in the scale you want to monitor (log rewards or
  linear rewards both work, as long as you stay consistent).
- an optional `buffer_size` for the sliding window metric.

## Quick start

> **Environment requirement:** the environment must expose a tractable ground-truth mean reward (`env.is_expected_reward_tractable = True`) so deltas are meaningful.

```python
import jax
import gfnx

reward_module = gfnx.EasyHypergridRewardModule(side=20)
env = gfnx.HypergridEnvironment()
env_params = env.init(jax.random.PRNGKey(0))
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

mean_metric = gfnx.metrics.ExpectedRewardMetricsModule(
    env=env,
    env_params=env_params,
    reward_module=reward_module,
    reward_params=reward_params,
)
state = mean_metric.init(jax.random.PRNGKey(1), mean_metric.InitArgs())

# During training: accumulate rewards from rollouts.
# `UpdateArgs.rewards` accepts whichever scale you collect (log or linear);
# stay consistent so deltas remain meaningful.
state = mean_metric.update(
    state,
    jax.random.PRNGKey(2),
    mean_metric.UpdateArgs(rewards=batch_rewards),
)

scores = mean_metric.get(state)
print(scores["mean_reward"], scores["reward_delta"], scores["rel_reward_delta"])
```
