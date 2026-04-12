# On-Policy Correlation

Correlation metrics quantify how well learned flows (log ratios recovered via backward rollouts) align with terminal log rewards. The on-policy variant generates fresh terminal states by rolling out the current forward policy, so the score reflects the sampler’s current distribution.

## Intuition

- Measures whether the policy puts mass in high-reward regions without requiring exact enumeration.
- Fresh samples every evaluation make this sensitive to training progress (and noise).
- Set `batch_size` based on accelerator memory; terminal states are processed in `(n_rounds, batch_size, ...)` chunks for backward rollouts.
- Use a `transform_fn` to marginalise or bin complex terminals (e.g., sequences, grids) before correlation.

## Key parameters

- `n_rounds`: number of backward-rollout repetitions (averaging reduces variance);
- `n_terminal_states`: total terminal states generated per evaluation (must be divisible by `batch_size`);
- `batch_size`: trajectories per rollout batch; trades throughput for memory;
- `fwd_policy_fn` / `bwd_policy_fn`: policy callables returning logits and an aux dict with both forward & backward logits;
- `transform_fn`: optional projector applied to states & log values before correlation;
- `env`: environment instance supplying rollout primitives.

### Lifecycle arguments

| Dataclass | Fields | Purpose |
| --- | --- | --- |
| `InitArgs` | `env_params` | Supplies environment parameters so the module can allocate dummy terminal states. |
| `UpdateArgs` | _empty_ | Correlation metrics skip streaming updates; pass nothing. |
| `ProcessArgs` | `policy_params`, `env_params` | Provides the current policy parameters and environment settings for each evaluation round. |

## Quick start

> **Environment requirement:** must support parallel forward/backward rollouts (with RNG-safe policy fns returning logits + aux info) so the metric can generate terminal states and compute log ratios.

```python
import jax
import jax.numpy as jnp
import gfnx

env = gfnx.HypergridEnvironment()
reward_module=gfnx.EasyHypergridRewardModule()
params = env.init(jax.random.PRNGKey(0))
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


def uniform_backward_policy(rng_key, obs, policy_params):
    batch = obs.shape[0]
    backward_logits = jnp.zeros((batch, policy_params["backward_num_actions"]), dtype=jnp.float32)
    forward_logits = jnp.zeros((batch, policy_params["forward_num_actions"]), dtype=jnp.float32)
    info = {"forward_logits": forward_logits, "backward_logits": backward_logits}
    return backward_logits, info


metrics = gfnx.metrics.OnPolicyCorrelationMetricsModule(
    n_rounds=8,
    n_terminal_states=1024,
    batch_size=128,
    fwd_policy_fn=uniform_forward_policy,
    bwd_policy_fn=uniform_backward_policy,
    env=env,
    reward_module=reward_module,
)
state = metrics.init(jax.random.PRNGKey(1), metrics.InitArgs(env_params=params))

state = metrics.process(
    state,
    jax.random.PRNGKey(2),
    metrics.ProcessArgs(
        policy_params=policy_params,
        env_params=params,
        reward_params=reward_params,
    ),
)
scores = metrics.get(state)
print(float(scores["pearson"]), float(scores["spearman"]))
```
