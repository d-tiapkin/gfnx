# Top-K Discovery Metrics

`TopKMetricsModule` measures the quality and diversity of the highest-reward
samples drawn from the learned forward policy. It repeatedly samples trajectories,
picks the top `K` rewards, and reports aggregate statistics for each requested
`K`.

## Intuition

- Monitor whether the forward policy can surface high-reward terminals by
  sampling a large batch, exponentiating log rewards, and keeping the best `k`
  entries per evaluation run.
- The diversity statistic highlights mode collapse: a low value with high
  reward means the policy keeps finding the same solution; a rising diversity
  with stable reward signals broader coverage.
- Pairing Top-K metrics with ELBO/EUBO inside a `MultiMetricsModule` lets you
  track normalisation accuracy alongside sampling quality.
- Choose a fast distance function since the module computes pairwise
  distances across the top set each time you process new trajectories.

## What you get

| Metric key | Meaning |
| --- | --- |
| `top_{k}_reward` | Mean reward of the best `k` samples (rewards are converted to linear scale by exponentiating the logged rewards). |
| `top_{k}_diversity` | Average pairwise distance between the top `k` terminal states, measured with `distance_fn`. |

## Configuration

- `num_traj`: number of trajectories to sample each time you call `process`
  (should be ≥ max(top_k));
- `batch_size`: how many trajectories to roll out in parallel
- `top_k`: list of integers (e.g. `[10, 50, 100]`);
- `distance_fn`: callable that measures diversity between two terminal states.

## Quick start

> **Environment requirement:** supply a forward policy and distance function so batches of terminal states can be sampled and compared.

```python
import jax
import jax.numpy as jnp
import gfnx
from gfnx.utils.distances import hamming_distance

env = gfnx.HypergridEnvironment()
reward_module = gfnx.EasyHypergridRewardModule()
env_params = env.init(jax.random.PRNGKey(0))
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

policy_params = {
    "forward_num_actions": env.action_space.n,
    "backward_num_actions": env.backward_action_space.n,
}


def uniform_forward_policy(rng_key, obs, policy_params):
    batch = obs.shape[0]
    forward_logits = jnp.zeros((batch, policy_params["forward_num_actions"]), dtype=jnp.float32)
    backward_logits = jnp.zeros((batch, policy_params["backward_num_actions"]), dtype=jnp.float32)
    info = {"forward_logits": forward_logits, "backward_logits": backward_logits}
    return forward_logits, info


def grid_hamming(lhs_state, rhs_state):
    return hamming_distance(lhs_state.state, rhs_state.state)


metrics = gfnx.metrics.TopKMetricsModule(
    env=env,
    reward_module=reward_module,
    fwd_policy_fn=uniform_forward_policy,
    num_traj=4096,
    batch_size=256,
    top_k=[10, 50, 200],
    distance_fn=grid_hamming,
)
state = metrics.init(jax.random.PRNGKey(1), metrics.InitArgs())

state = metrics.process(
    state,
    jax.random.PRNGKey(2),
    metrics.ProcessArgs(
      policy_params=policy_params,
      reward_params=reward_params,
      env_params=params
    ),
)
report = metrics.get(state)
print(report["top_50_reward"], report["top_50_diversity"])
```
