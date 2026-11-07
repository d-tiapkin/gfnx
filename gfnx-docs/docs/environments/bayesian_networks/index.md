# Bayesian Networks Structure Learning

This environment models Bayesian structure learning with Generative Flow
Networks, following Deleu *et&nbsp;al.* (2022). We search over directed acyclic
graphs (DAGs) that connect variables $\{X_1, \dots, X_d\}$ and learn a sampler
whose terminal distribution matches the posterior $P(G \mid \mathcal{D})$ for a
dataset $\mathcal{D}$. It is a strong benchmark for objectives that must
respect structural constraints and exploit factorised rewards.

The specification below is rendered from the docstrings of
[`DAGEnvironment`][gfnx.environment.dag.DAGEnvironment] and
[`DAGRewardModule`][gfnx.reward.dag.DAGRewardModule].

## Specification

::: gfnx.environment.dag.DAGEnvironment
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

## Reward structure and modularity

::: gfnx.reward.dag.DAGRewardModule
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

`gfnx` ships a uniform prior
([`UniformDAGPrior`][gfnx.reward.dag_prior.UniformDAGPrior]) and two marginal
likelihoods:

- [`LinearGaussianScore`][gfnx.reward.dag_likelihood.LinearGaussianScore] — the
  linear-Gaussian model of Deleu *et&nbsp;al.* (2022);
- [`BGeScore`][gfnx.reward.dag_likelihood.BGeScore] — the Bayesian Gaussian
  equivalent score of Geiger and Heckerman (1994).

Both, as well as the uniform prior, are modular, so the delta score of an edge
addition only re-scores the target node.

## Dataset generation

For experiments we follow the linear-Gaussian setting: ground-truth DAGs $G^*$
are sampled from an Erdős–Rényi model with expected in-degree $1$. Given $G^*$,
synthetic data are produced by ancestral sampling with weights
$w_{ij} \sim \mathcal{N}(0, 1)$ and fixed noise variance $\sigma_j^2 = 0.1$.
Each dataset contains $100$ samples. `gfnx.utils.dag.generate_dataset`
implements this pipeline (and optionally writes the graph, adjacency matrix and
CSV data to disk).

## Quick start

```python
import jax
import gfnx
from gfnx.utils.dag import generate_dataset

num_variables = 5

# 1. Sample a ground-truth DAG and 100 observations from it.
data = generate_dataset(data_seed=0, num_variables=num_variables, num_train_samples=100)

# 2. Build the reward (likelihood x prior) and the environment separately.
likelihood = gfnx.reward.BGeScore(data=data)  # or gfnx.reward.LinearGaussianScore(data=data)
prior = gfnx.reward.UniformDAGPrior(num_variables=num_variables)
reward_module = gfnx.DAGRewardModule(prior=prior, likelihood=likelihood)

env = gfnx.DAGEnvironment(num_variables=num_variables)
env_params = env.init(jax.random.PRNGKey(0))
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

# 3. Add the edge X_0 -> X_1 and read the delta score used by MDB.
state = env.reset()
action = 0 * num_variables + 1
_, next_state, _, _ = env.step(state, action, env_params)
delta = reward_module.delta_score(state, action, next_state, env_params, reward_params)

# 4. For num_variables < 6 the exact posterior is available (29,281 DAGs at d = 5).
posterior = env.get_true_distribution(env_params, reward_module, reward_params)
log_Z = env.get_log_normalizing_constant(env_params, reward_module, reward_params)
```

Use `jax.vmap` externally to explore multiple graphs in parallel. Modified
Detailed Balance (`baselines/mdb_dag.py`) uses `delta_score` to re-score only
the node whose parent set changed at each step, rather than the whole graph.

## API references

- [Environment](environment_api.md)
- [Reward modules](reward_api.md)

## References

- Deleu, T. *et&nbsp;al.* (2022). *Bayesian Structure Learning with Generative Flow Networks.*
  Proceedings of the 38th Conference on Uncertainty in Artificial Intelligence (UAI).
- Geiger, D. and Heckerman, D. (1994). *Learning Gaussian Networks.*
  Proceedings of the 10th Conference on Uncertainty in Artificial Intelligence (UAI).
