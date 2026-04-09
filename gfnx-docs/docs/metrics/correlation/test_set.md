# Test-Set Correlation

The test-set correlation metric evaluates alignment between learned flow probabilities (via backward trajectory log-ratios) and terminal log rewards using a *fixed* batch of terminal states (`test_set`). This yields stable, comparable curves across checkpoints without on-policy sampling noise.

## Intuition

- A fixed evaluation set isolates model improvement from sampling variance.
- Best when you have a curated or diverse batch of terminal states representing important regions.
- Use the on-policy variant instead if you want to directly reflect current sampling distribution shifts.
- Provide a `transform_fn` when raw terminal representation is high dimensional.

## Key parameters

- `n_rounds`: number of backward-rollout repetitions (averaging reduces variance);
- `test_set`: fixed batch of terminal states (tree structure must match environment states);
- `batch_size`: terminal states per backward rollout sub-batch;
- `bwd_policy_fn`: backward policy callable returning logits and aux with forward/backward logits;
- `transform_fn`: optional projector applied to states & log values before correlation;
- `env`: environment instance.

### Lifecycle arguments

| Dataclass | Fields | Purpose |
| --- | --- | --- |
| `InitArgs` | `env_params`, `test_set` | Supplies the fixed terminal set and environment parameters so log rewards can be cached. |
| `UpdateArgs` | _empty_ | Streaming updates are not needed; pass nothing between evaluations. |
| `ProcessArgs` | `policy_params`, `env_params` | Provides the latest policy weights and environment settings for backward rollouts. |

## Quick start

> **Environment requirement:** must evaluate log rewards for arbitrary terminal states and support backward rollouts so cached test sets can be correlated with model predictions.

```python
import jax
import jax.numpy as jnp
import gfnx

env = gfnx.HypergridEnvironment(reward_module=gfnx.EasyHypergridRewardModule())
params = env.init(jax.random.PRNGKey(0))

# Prepare any fixed batch of terminal states (e.g., enumerated dataset or samples from the ground truth).
test_set = env.get_ground_truth_sampling(jax.random.PRNGKey(123), 512, params)

policy_params = {
    "forward_num_actions": env.action_space.n,
    "backward_num_actions": env.backward_action_space.n,
}


def uniform_backward_policy(rng_key, obs, policy_params):
    batch = obs.shape[0]
    backward_logits = jnp.zeros((batch, policy_params["backward_num_actions"]), dtype=jnp.float32)
    forward_logits = jnp.zeros((batch, policy_params["forward_num_actions"]), dtype=jnp.float32)
    info = {"forward_logits": forward_logits, "backward_logits": backward_logits}
    return backward_logits, info


metrics = gfnx.metrics.TestCorrelationMetricsModule(
    n_rounds=8,
    bwd_policy_fn=uniform_backward_policy,
    env=env,
    batch_size=128,
)
state = metrics.init(
    jax.random.PRNGKey(1),
    metrics.InitArgs(env_params=params, test_set=test_set),
)
state = metrics.process(
    state,
    jax.random.PRNGKey(2),
    metrics.ProcessArgs(policy_params=policy_params, env_params=params),
)
scores = metrics.get(state)
print(float(scores["pearson"]), float(scores["spearman"]))
```
