# Phylogenetic Trees Environment

This environment follows the PhyloGFN formulation of Zhou *et&nbsp;al.* (2024):
inferring a rooted binary phylogenetic tree is cast as a sequential decision
process that repeatedly merges species into larger trees, making the
environment a natural stress test for GFlowNet objectives on structured
combinatorial spaces.

The specification below is rendered from the docstrings of
[`PhyloTreeEnvironment`][gfnx.environment.phylogenetic_tree.PhyloTreeEnvironment]
and [`PhyloTreeRewardModule`][gfnx.reward.phylogenetic_tree.PhyloTreeRewardModule].

## Specification

::: gfnx.environment.phylogenetic_tree.PhyloTreeEnvironment
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

## Reward

::: gfnx.reward.phylogenetic_tree.PhyloTreeRewardModule
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

## Loading datasets and building the environment

The DS1–DS8 benchmark datasets are downloaded into `datasets/` by the script at
the repository root:

```bash
bash download_ds.sh
```

`gfnx.utils.get_phylo_initialization_args` then encodes the sequences and
returns the matching environment and reward arguments ($C$ and $\alpha = 4$):

```python
from pathlib import Path

import jax
import gfnx
from gfnx.utils import get_phylo_initialization_args

env_kwargs, reward_kwargs = get_phylo_initialization_args("DS1", Path("datasets"))

# Build the reward module and the environment separately.
reward_module = gfnx.PhyloTreeRewardModule(**reward_kwargs)
env = gfnx.PhyloTreeEnvironment(**env_kwargs)
env_params = env.init(jax.random.PRNGKey(0))
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

# `env.reset()` returns a single initial state: n singleton trees.
state = env.reset()
```

All environment methods operate on a single state — use `jax.vmap` to roll
out multiple forests in parallel. The log-reward is computed post-rollout
via `reward_module.log_reward(state, reward_params)`; Forward-Looking Detailed
Balance (`baselines/fldb_phylo.py`) instead consumes the per-merge
`reward_module.delta_score(next_state)`.

## API references

- [Environment](environment_api.md)
- [Reward module](reward_api.md)

## References

- Zhou, M. *et&nbsp;al.* (2024). *PhyloGFN: Phylogenetic inference with generative flow networks.*
  The Twelfth International Conference on Learning Representations (ICLR).
- Deleu, T. *et&nbsp;al.* (2024). *Discrete Probabilistic Inference as Control in Multi-path Environments.*
  Proceedings of the 40th Conference on Uncertainty in Artificial Intelligence (UAI).
