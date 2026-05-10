# Phylogenetic Trees Environment

This environment follows the PhyloGFN formulation of Zhou *et&nbsp;al.* (2024):
we represent the task of inferring a rooted binary phylogenetic tree as a
sequential decision process. Each node corresponds to a candidate tree topology
built by repeatedly merging species, making the environment a natural stress
test for GFlowNet objectives on structured combinatorial spaces.

## Intuition

- **State** – a forest of partially built trees. The initial state $s_0$
  contains $n$ singleton trees, one per species; intermediate states keep track
  of the current forest plus the merge history required to compute rewards.
- **Action** – pick two roots from the forest and combine them under a common
  ancestor. After $n-1$ merges you obtain a single rooted binary tree spanning
  all species and the episode terminates.
- **Observation** – one-hot Fitch features derived from the binary-encoded DNA
  (or RNA) sequences that appear at each root. These features are suitable for
  transformer-style policies and match the setup in the original PhyloGFN work.

## Reward structure

For a terminating tree $T$ with parsimony score $M(T)$ (the minimum number of
mutations required to explain the observed species), the raw reward follows a
Gibbs distribution:

$$
R(T) = \exp\left(- \frac{M(T)}{\alpha} \right),
$$
where $\alpha$ is a temperature hyperparameter. For training stability we
recenter this expression using a dataset-specific constant $C$:

$$
R(T) = \exp\left(\frac{C - M(T)}{\alpha}\right).
$$
This keeps rewards in a numerically friendly range while preserving the ranking
over tree topologies. Following Deleu *et&nbsp;al.* (2024) we set $\alpha = 4$
and choose $C$ per dataset:

- DS1: $C = 5800$
- DS2: $C = 8000$
- DS3: $C = 8800$
- DS4: $C = 3500$
- DS5: $C = 2300$
- DS6: $C = 2300$
- DS7: $C = 12500$
- DS8: $C = 2800$

You can supply your own scaling by constructing `PhyloTreeRewardModule` with
custom `C` and `scale` values.

## Loading datasets and building the environment

Datasets DS1–DS8 ship with the library in JSON form. Use
`gfnx.utils.get_phylo_initialization_args` to retrieve the encoded sequences and
reward parameters:

```python
from pathlib import Path

import jax
import gfnx
from gfnx.utils import get_phylo_initialization_args

data_dir = Path("path/to/phylo_datasets")
env_kwargs, reward_kwargs = get_phylo_initialization_args("DS1", data_dir)

# Build the reward module and the environment separately.
reward_module = gfnx.PhyloTreeRewardModule(**reward_kwargs)
env = gfnx.PhyloTreeEnvironment(**env_kwargs)
env_params = env.init(jax.random.PRNGKey(0))
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

# `env.reset()` returns a single initial state.
state = env.reset()
```

All environment methods operate on a single state — use `jax.vmap` to roll
out multiple forests in parallel. The log-reward is computed post-rollout
via `reward_module.log_reward(state, reward_params)`.


## API references:

- [Environment](environment_api.md)
- [Reward module](reward_api.md)

## References

- Zhou, M. *et&nbsp;al.* (2024). *PhyloGFN: Phylogenetic inference with generative flow networks.*
  The Twelfth International Conference on Learning Representations (ICLR).
- Deleu, T. *et&nbsp;al.* (2024). *Discrete Probabilistic Inference as Control in Multi-path Environments.*
  Proceedings of the 40th Conference on Uncertainty in Artificial Intelligence (UAI).
