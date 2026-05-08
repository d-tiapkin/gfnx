# Exact Distribution Metrics

`ExactDistributionMetricsModule` compares the terminal states distribution
induced by the current policy with the ground truth distribution. The full state graph is considered
and the policy propagates through it to recover the exact terminal distribution it induces.

## Intuition

- Always use this metric when the environment can enumerate its terminal space, provide a true distribution, and time overhead is acceptable.
- Prefer approximate distribution metric when evaluation of this metric is time-consuming.
- For non-enumarable or too large environments, we recommend to use ELBO and correlation-based metrics instead due to large computational costs.

## Key parameters

- `metrics`: List of metric names to compute, choose from `{"tv", "kl", "jsd","2d_marginal_distribution"}`.
- `env`: Enumerable environment for which to compute metrics.
- `fwd_policy_fn`: Forward policy function for generating trajectories.
- `reward_module`: Reward module used to compute the ground-truth distribution
  via `env.get_true_distribution(env_params, reward_module, reward_params)`.
- `batch_size`: Batch size used when evaluating policy over states.
- `tol_epsilon`: Tolerance for convergence in distribution computation.

The environment must be enumerable and support `get_all_states`, `state_to_index`, and `get_true_distribution`.

## Quick start

> **Environment requirement:** must enumerate the full state graph (supporting `get_all_states`, `state_to_index`, `get_true_distribution`) so exact terminal probabilities can be computed.

```python
import jax
import jax.numpy as jnp
import gfnx

reward_module = gfnx.EasyHypergridRewardModule(side=20)
env = gfnx.HypergridEnvironment()
env_params = env.init(jax.random.PRNGKey(0))
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

policy_params = {
    "forward_num_actions": env.action_space.n,
    "backward_num_actions": env.backward_action_space.n,
}


def uniform_forward_policy(rng_key, obs, policy_params):
    forward_logits = jnp.zeros((policy_params["forward_num_actions"],), dtype=jnp.float32)
    backward_logits = jnp.zeros((policy_params["backward_num_actions"],), dtype=jnp.float32)
    info = {"forward_logits": forward_logits, "backward_logits": backward_logits}
    return forward_logits, info


metrics = gfnx.metrics.ExactDistributionMetricsModule(
    metrics=["tv", "kl"],
    env=env,
    fwd_policy_fn=uniform_forward_policy,
    reward_module=reward_module,
    batch_size=512,
)
state = metrics.init(
    jax.random.PRNGKey(1),
    metrics.InitArgs(env_params=env_params, reward_params=reward_params),
)

state = metrics.process(
    state,
    jax.random.PRNGKey(2),
    metrics.ProcessArgs(policy_params=policy_params, env_params=env_params),
)
scores = metrics.get(state)
print(scores["tv"], scores["kl"])
```

- The helper policy surfaces both forward and backward logits so `gfnx.utils.rollout` can reconstruct the log ratios it expects during enumeration.
