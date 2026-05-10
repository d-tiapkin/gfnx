# Evidence Lower Bound (ELBO)

`ELBOMetricsModule` estimates the evidence lower bound of the target
distribution under the learned forward policy. It measures how closely the
forward policy aligns with the backward distribution. Formally, it is defined as:

$$
\begin{aligned}
\mathrm{ELBO} &= \mathrm{\overline{ELBO}} - \log Z \leq 0
\\
\mathrm{\overline{ELBO}} &= 
\mathbb{E}_{\tau \sim P_F} \Bigg[ \log R(s_T) + \sum_{t=1}^T \log P_B(s_{t-1} \mid s_t) - \sum_{t=1}^T \log P_F(s_t \mid s_{t-1}) \Bigg] \leq \\ &\leq \log Z
\end{aligned}
$$

## Intuition

- Higher values indicate better sampling quality and that a learned forward policy matches a backward distribution.
- However, $\text{ELBO}$ can reach high values even if the policy concentrates on a single mode. 
- Treat this metric as a measure of within-mode quality rather than global coverage, and use with $\text{EUBO}$ or correlation metrics.
- Increase `n_rounds` if the estimate is too noisy. Each round performs a new set of forward rollouts;
- If  $\log Z$ (a true log-normalising constant) is accessible, $\text{ELBO}$ is reported. In this case, the perfect value is 0.
- If  $\log Z$ is unknown for environment, the metric is unnormalised and $\mathrm{\overline{ELBO}}$ is reported. In this case, the perfect value is $\log Z$.

## Key parameters

- `env`: Environment for which metric is computed.
- `env_params`: Environment parameters used for trajectory generation.
- `reward_module` / `reward_params`: Reward used to compute log rewards and (when
  tractable) the normalising constant `log Z` cached at construction time. The
  current `reward_params` must also be passed via `ProcessArgs` at every
  evaluation step (they may be trainable, e.g. for Ising).
- `fwd_policy_fn`: Forward policy function for generating trajectories. The
  policy operates on a **single** observation (no batch dim) — `gfnx.utils.forward_rollout`
  vmaps it internally for the metric.
- `n_rounds`: Number of sampling rounds for statistical stability.
- `batch_size`: Batch size used when evaluating policy over states.

## Quick start

> **Environment requirement:** must provide `log_reward` (and optionally `logZ`) so the ELBO objective can be evaluated. Supply a pure `fwd_policy_fn` that returns forward logits plus auxiliary info used for diagnostics.

```python
import jax
import jax.numpy as jnp
import gfnx

# Environment and reward are fully decoupled.
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


metrics = gfnx.metrics.ELBOMetricsModule(
    env=env,
    env_params=env_params,
    reward_module=reward_module,
    reward_params=reward_params,
    fwd_policy_fn=uniform_forward_policy,
    n_rounds=16,
    batch_size=128,
)
state = metrics.init(jax.random.PRNGKey(1), metrics.InitArgs())

state = metrics.process(
    state,
    jax.random.PRNGKey(2),
    metrics.ProcessArgs(
        policy_params=policy_params,
        env_params=env_params,
        reward_params=reward_params,
    ),
)
elbo = metrics.get(state)["elbo"]
print(float(elbo))
```
