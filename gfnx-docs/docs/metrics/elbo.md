# Evidence Lower Bound (ELBO)

`ELBOMetricsModule` estimates the evidence lower bound of the target
distribution under the learned forward policy. It measures how closely the
forward policy aligns with the backward distribution. Formally, it is defined as:

$$
\begin{aligned}
\mathrm{ELBO} &= \overline{\mathrm{ELBO}} - \log Z \leq 0
\\
\mathrm{\overline{ELBO}} &= 
\mathbb{E}_{\tau \sim P_F} \Bigg[ \log R(s_T) + \sum_{t=1}^T \log P_B(s_{t-1} \mid s_t) - \sum_{t=1}^T \log P_F(s_t \mid s_{t-1}) \Bigg] \leq \log Z
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
- `fwd_policy_fn`: Forward policy function for generating trajectories.
- `n_rounds`: Number of sampling rounds for statistical stability.
- `batch_size`: Batch size used when evaluating policy over states.

## Quick start

> **Environment requirement:** must provide `log_reward` (and optionally `logZ`) so the ELBO objective can be evaluated. Supply a pure `fwd_policy_fn` that returns forward logits plus auxiliary info used for diagnostics.

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


def uniform_forward_policy(rng_key, obs, policy_params):
    batch = obs.shape[0]
    forward_logits = jnp.zeros((batch, policy_params["forward_num_actions"]), dtype=jnp.float32)
    backward_logits = jnp.zeros((batch, policy_params["backward_num_actions"]), dtype=jnp.float32)
    info = {"forward_logits": forward_logits, "backward_logits": backward_logits}
    return forward_logits, info


metrics = gfnx.metrics.ELBOMetricsModule(
    env=env,
    env_params=params,
    fwd_policy_fn=uniform_forward_policy,
    n_rounds=16,
    batch_size=128,
)
state = metrics.init(jax.random.PRNGKey(1), metrics.InitArgs())

state = metrics.process(
    state,
    jax.random.PRNGKey(2),
    metrics.ProcessArgs(policy_params=policy_params, env_params=params),
)
elbo = metrics.get(state)["elbo"]
print(float(elbo))
```
