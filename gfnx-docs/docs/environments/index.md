# Overview

Every `gfnx` environment describes only the *dynamics* of a generation DAG:
forward and backward transitions, action masks and observations. Rewards are
separate [reward modules](../api/rewards.md), built and parameterised on their
own and evaluated after the rollout on the terminal states:

```python
env = gfnx.HypergridEnvironment(dim=4, side=20)
env_params = env.init(rng_key)                                # dynamics only
reward_module = gfnx.EasyHypergridRewardModule(side=20)
reward_params = reward_module.init(reward_key, env.reset())   # single dummy state

traj_data, final_states, info = jax.vmap(
    lambda key: gfnx.utils.forward_rollout(key, policy_fn, policy_params, env, env_params)
)(jax.random.split(rng_key, num_envs))
log_rewards = jax.vmap(reward_module.log_reward, in_axes=(0, None))(final_states, reward_params)
```

All methods act on a single state; see
[`BaseEnvironment`][gfnx.base.BaseEnvironment] for the full contract.

## Implemented environments

| Family | Environment | Objects | Forward actions | Enumerable | Baselines |
| --- | --- | --- | --- | --- | --- |
| [Hypergrid](hypergrid/index.md) | `HypergridEnvironment` | cells of $\{0..H-1\}^D$ | increment a coordinate, stop | Yes | DB, TB, SubTB, Soft DQN |
| [Sequences](sequences/index.md) | `BitseqEnvironment` | $n$-bit strings | fill any empty $k$-bit word | No | DB, TB, SubTB |
| | `TFBind8Environment` | DNA 8-mers | append a nucleotide | Yes | DB, TB, SubTB |
| | `QM9SmallEnvironment` | 5-block molecules | prepend / append a block | Yes | DB, TB |
| | `AMPEnvironment` | peptides, length $\le 60$ | append an amino acid, stop | No | DB, TB |
| | `GFPEnvironment` (experimental) | proteins, length 237 | append an amino acid | No | DB (experimental) |
| [Bayesian networks](bayesian_networks/index.md) | `DAGEnvironment` | DAGs on $d$ nodes | add an edge, stop | For $d < 6$ | MDB |
| [Ising model](ising_model/index.md) | `IsingEnvironment` | spin configurations $\{\pm 1\}^D$ | assign a spin | No | TB (EB-GFN) |
| [Phylogenetic trees](phylogenetic_trees/index.md) | `PhyloTreeEnvironment` | rooted binary trees | merge two trees | No | FLDB |

Each family page has a narrative introduction and quick start, followed by an
*Environment API* and a *Reward API* page generated from the docstrings.
Enumerable environments additionally expose the exact target distribution,
$\log Z$, the expected reward and (where implemented) exact sampling, which the
[exact distribution metrics](../metrics/distribution_based/exact.md) use.

## Writing your own environment

Subclass [`BaseEnvState`][gfnx.base.BaseEnvState],
[`BaseEnvParams`][gfnx.base.BaseEnvParams] and
[`BaseEnvironment`][gfnx.base.BaseEnvironment], implementing `reset`, `init`,
`step`, `backward_step`, `get_obs`, the two action masks, the forward/backward
action conversions, the spaces and `max_steps_in_episode`. The
[custom environment notebook](../notebooks.md) walks through a complete
example; the existing environments in `src/gfnx/environment/` are short and
self-contained references.
