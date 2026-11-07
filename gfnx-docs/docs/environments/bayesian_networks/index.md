# Bayesian Networks Structure Learning

This environment models Bayesian structure learning with Generative Flow Networks,
following Deleu *et&nbsp;al.* (2022). We search over directed acyclic graphs (DAGs)
that connect variables $\{X_1, \dots, X_d\}$ and learn a sampler whose terminal
distribution matches the posterior $P(G \mid \mathcal{D})$ for a dataset
$\mathcal{D}$. The environment exposes the same incremental edge-building process
used in classic score-based structure learning, making it a strong benchmark for
objectives that must respect structural constraints and exploit factorised rewards.

## Intuition

- **State** – the current DAG under construction together with the transitive
  closure of its transpose. The initial state is the empty graph. Every
  intermediate state already corresponds to a valid (possibly sparse) DAG.
- **Action** – choose an ordered pair of nodes $(i, j)$ and add the edge
  $X_i \rightarrow X_j$, unless doing so would create a cycle. A dedicated stop
  action terminates the trajectory and yields the current graph.
- **Observation** – the adjacency matrix of the partially built DAG. Policies
  typically consume this matrix directly or embed it with a graph neural network.

Because any state can be treated as terminal, the environment naturally supports
Modified Detailed Balance (MDB) training as well as more standard objectives.

## Reward structure and modularity

The terminal reward is proportional to the posterior

$$
\log R(G) = \log P(\mathcal{D} \mid G) + \log P(G),
$$
where we assume i.i.d., fully observed samples and a uniform prior over structures.
GFNX ships two choices of marginal likelihoods $P(\mathcal{D} \mid G)$:

- `LinearGaussianScore` implements the linear-Gaussian model of Deleu *et&nbsp;al.* (2022),
  using data stored on disk.
- `BGeScore` implements the Bayesian Gaussian equivalent score of Geiger and Heckerman (1994).

Both scores, as well as the uniform prior, satisfy the modularity property:

$$
\log R(G) = \sum_{j=1}^{d} \mathrm{LocalScore}\left(X_j \mid \mathrm{Pa}_{G}(X_j)\right)
$$
When an edge $X_i \rightarrow X_j$ is added, only the term for $X_j$ changes, so
the delta score

$$
\Delta_j = \mathrm{LocalScore}\left(X_j \mid \mathrm{Pa}_G(X_j) \cup \{X_i\}\right)
          - \mathrm{LocalScore}\left(X_j \mid \mathrm{Pa}_G(X_j)\right)
$$
is cheap to evaluate. MDB relies on this delta to update flows efficiently, so
the environment exposes it via the reward module.

## Dataset generation

For experiments we follow the linear-Gaussian setting:
ground-truth DAGs $G^*$ are sampled from an Erdős–Rényi model
with expected in-degree $1$. Given $G^*$, synthetic data are produced by
ancestral sampling with weights $w_{ij} \sim \mathcal{N}(0, 1)$ and fixed noise
variance $\sigma_j^2 = 0.1$. Each dataset contains $100$ samples and we
generate $20$ such datasets (and associated true graphs) for evaluation.
Utilities in `gfnx.utils.dag` make it easy to replicate this pipeline or load
pre-generated CSV files.

## Efficient action masks

To enforce acyclicity online, the environment maintains two matrices per state:
the current adjacency matrix and the transitive closure of the transpose DAG.
When an edge $(u \rightarrow v)$ is added, the closure is updated with a single
outer-product operation that marks every node that can now reach $u$ or be
reached from $v$. The resulting mask excludes (i) existing edges and
(ii) edges that would complete a cycle, providing $O(d^2)$ updates without
resorting to repeated graph traversals.

## Quickstart

```python
from pathlib import Path

import jax
import gfnx
from gfnx.reward.dag import DAGRewardModule
from gfnx.reward.dag_likelihood import LinearGaussianScore
from gfnx.reward.dag_prior import UniformDAGPrior

num_variables = 5
data_path = Path("datasets/dag/train_data.csv")

# Build the reward (likelihood × prior) and the environment separately.
likelihood = LinearGaussianScore(data_path=str(data_path))
prior = UniformDAGPrior(num_variables=num_variables)
reward = DAGRewardModule(prior=prior, likelihood=likelihood)

env = gfnx.DAGEnvironment(num_variables=num_variables)
env_params = env.init(jax.random.PRNGKey(0))
reward_params = reward.init(jax.random.PRNGKey(0), env.reset())

# `env.reset()` returns a single initial state — no batch dim, no env_params.
state = env.reset()
```

Use `jax.vmap` externally to explore multiple graphs in parallel. For
`num_variables < 6` the environment can enumerate every DAG; the enumerable
helpers (`get_true_distribution`, `get_normalizing_constant`, …) take both
`reward_module` and `reward_params` explicitly.

`DAGRewardModule` additionally exposes a `delta_score(state, action,
next_state, env_params, reward_params)` method that exploits the modularity
of the score (only the local term of the affected node changes when an edge
is added). Modified Detailed Balance (`baselines/mdb_dag.py`) uses this for
$O(d^2)$ per-step updates rather than re-evaluating the full score.

## API references:

- [Environment](environment_api.md)
- [Reward module](reward_api.md)

## References

- Deleu, T. *et&nbsp;al.* (2022). *Bayesian Structure Learning with Generative Flow Networks.*
  Proceedings of the 38th Conference on Uncertainty in Artificial Intelligence (UAI).
