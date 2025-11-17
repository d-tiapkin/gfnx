# Evidence Upper Bound (EUBO)

`EUBOMetricsModule` estimates the evidence upper bound by sampling backward
trajectories that start from ground-truth terminal states. It complements other metrics and helps understand how well the sampler covers different modes. Formally, it is defined as:

$$
\begin{aligned}
\mathrm{EUBO} &= \mathrm{\overline{EUBO}} - \log Z \geq 0
\\
\mathrm{\overline{EUBO}} &= \mathbb{E}_{\tau \sim \frac{R}{Z} \cdot P_B} \Bigg[ \log R(s_T) + \sum_{t=1}^T \log P_B(s_{t-1} \mid s_t) - \sum_{t=1}^T \log P_F(s_t \mid s_{t-1}) \Bigg] \geq \log Z
\end{aligned}
$$

## Intuition

- Lower values indicate better sampling quality and that the sampler covers different modes.
- However, $\text{EUBO}$ can reach low values even if the policy fails to match within-mode distribution. 
- Treat this metric as a measure of a global mode coverage, and use it with $\text{ELBO}$ or distribution-based metrics.
- The test set of terminal states is sampled from a true distribution and fixed, hence, the metric is comparable throughout training.
- If $\log Z$ (a true normalising constant) is accessible, the metric is normalised and $\text{EUBO}$ is reported. In this case, the perfect value is 0.
- If $\log Z$ is unknown for environment, the metric is unnormalised and $\mathrm{\overline{EUBO}}$ is reported. In this case, the perfect value is $\log Z$.

## Key parameters

- `env`: Environment for which metric is computed.
- `env_params`: Environment parameters used for trajectory generation.
- `bwd_policy_fn`: Backward policy function for generating trajectories starting from terminal states.
- `n_rounds`: Number of sampling rounds for statistical stability.
- `batch_size`: Batch size used when evaluating policy over states.
- `rng_key`: Key used for pseudo random.

## Quick start

> **Environment requirement:** provide a dataset sampled from the ground-truth terminal distribution (or the ability to resample it) plus a pure `bwd_policy_fn` that returns backward logits and any auxiliary info required for diagnostics.

```python
import jax
import jax.numpy as jnp
import gfnx

env = gfnx.HypergridEnvironment(reward_module=gfnx.EasyHypergridRewardModule())
params = env.init(jax.random.PRNGKey(0))

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


metrics = gfnx.metrics.EUBOMetricsModule(
    env=env,
    env_params=params,
    bwd_policy_fn=uniform_backward_policy,
    n_rounds=16,
    batch_size=256,
    rng_key=jax.random.PRNGKey(42),
)
state = metrics.init(jax.random.PRNGKey(1), metrics.InitArgs())

state = metrics.process(
    state,
    jax.random.PRNGKey(2),
    metrics.ProcessArgs(policy_params=policy_params, env_params=params),
)
eubo = metrics.get(state)["eubo"]
print(float(eubo))
```

The backward helper surfaces both logits so `gfnx.utils.rollout` can recompute the
trajectory probabilities required for the EUBO estimate.
