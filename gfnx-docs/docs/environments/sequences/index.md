# Sequence Environments

A collection of environments that generate sequences of tokens: bit strings,
DNA, small molecules and proteins. They all share one state layout — a padded
token buffer — and differ in *how* tokens are written into it.

The specifications on this page are rendered from the docstrings, so they are
also what `help(...)` prints.

## Generation regimes

::: gfnx.environment.sequence
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false

Environments and rewards are decoupled: build the reward module separately and
call `reward_module.init(rng_key, env.reset())` to obtain `reward_params`. The
environment constructor never takes a reward. All methods act on a **single**
state; wrap them with `jax.vmap` to run many trajectories in parallel.

!!! warning "Run from the repository root"
    The TFBind-8, QM9, AMP and GFP rewards load data files from `proxy/`
    (relative paths), so they need a clone of the repository with the working
    directory set to its root. The bit-sequence reward is self-contained.

## Bit sequences

::: gfnx.environment.bitseq.BitseqEnvironment
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

### Reward

::: gfnx.reward.bitseq.BitseqRewardModule
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

### Quick start

```python
import jax
import jax.numpy as jnp
import gfnx

# 1. Build the reward. `sentence_len` and `k` must match the environment.
reward_module = gfnx.BitseqRewardModule(
    sentence_len=8,
    k=2,
    mode_set_size=30,
    reward_exponent=3.0,
)

# 2. Build the environment.
env = gfnx.BitseqEnvironment(n=8, k=2)
env_params = env.init(jax.random.PRNGKey(0))
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

# 3. Fill every word: action = position * 2**k + word.
state = env.reset()
for position in range(env.max_length):
    action = jnp.int32(position * 2**env.k + 3)  # word "11"
    obs, state, done, _ = env.step(state, action, env_params)

# 4. The reward is computed on the terminal state via the reward module.
assert bool(state.is_terminal)
print(gfnx.utils.bitseq.detokenize(state.tokens, env.k))  # 8 ones
print(float(reward_module.log_reward(state, reward_params)))
```

## TFBind-8

::: gfnx.environment.tfbind.TFBind8Environment
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

### Reward

::: gfnx.reward.tfbind.TFBind8RewardModule
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

### Quick start

```python
import jax
import jax.numpy as jnp
import gfnx

# 1. Build the reward (a lookup table over all 4^8 sequences).
reward_module = gfnx.TFBind8RewardModule()

# 2. Build the environment.
env = gfnx.TFBind8Environment()
env_params = env.init(jax.random.PRNGKey(0))
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

# 3. Append eight nucleotides (A=0, C=1, G=2, T=3).
state = env.reset()
for nucleotide in [0, 1, 2, 3, 0, 1, 2, 3]:
    obs, state, done, _ = env.step(state, jnp.int32(nucleotide), env_params)

print(float(reward_module.reward(state, reward_params)))

# 4. The environment is enumerable: exact log Z and the target distribution.
log_Z = env.get_log_normalizing_constant(env_params, reward_module, reward_params)
true_dist = env.get_true_distribution(env_params, reward_module, reward_params)  # (4,)*8
```

## QM9 (small)

::: gfnx.environment.qm9_small.QM9SmallEnvironment
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

### Reward

::: gfnx.reward.qm9_small.QM9SmallRewardModule
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

### Quick start

```python
import jax
import jax.numpy as jnp
import gfnx

reward_module = gfnx.QM9SmallRewardModule()
env = gfnx.QM9SmallEnvironment()
env_params = env.init(jax.random.PRNGKey(0))
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

# Actions 0..10 prepend a block, actions 11..21 append one.
state = env.reset()
for action in [0, 11 + 1, 2, 11 + 3, 4]:
    obs, state, done, _ = env.step(state, jnp.int32(action), env_params)

print(state.tokens)  # [4, 2, 0, 1, 3]
print(float(reward_module.reward(state, reward_params)))
```

## AMP

::: gfnx.environment.amp.AMPEnvironment
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

### Reward

::: gfnx.reward.amp.EqxProxyAMPRewardModule
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

### Proxy reward model training

The code that trains the proxy with [Equinox](https://github.com/patrick-kidger/equinox)
lives in the `proxy` folder. Pretrained AMP weights are stored in
`proxy/weights/amp` as an `orbax` checkpoint. To retrain them, follow the
instructions in `proxy/datasets/amp.py` to install the extra packages, then run

```bash
python proxy/train_proxy.py --config-name amp
```

All the configuration is handled by Hydra, so every training hyperparameter can
be overridden from the command line.

### Quick start

Requires the `baselines` extra (`equinox`, `orbax-checkpoint`).

```python
import jax
import jax.numpy as jnp
import gfnx

# 1. Load the reward model.
reward_module = gfnx.EqxProxyAMPRewardModule(
    proxy_config_path="proxy/configs/amp.yaml",
    pretrained_proxy_path="proxy/weights/amp/model",
    reward_exponent=1.0,
    min_reward=1e-6,
)

# 2. Create the environment (no reward attached).
env = gfnx.AMPEnvironment()
env_params = env.init(jax.random.PRNGKey(0))

# 3. Initialise reward parameters — the proxy model weights live here.
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

# 4. Append a few amino acids, then stop (action 20 writes [EOS]).
state = env.reset()
for action in [0, 1, 2, 3, env.stop_action]:
    obs, state, done, _ = env.step(state, jnp.int32(action), env_params)

print(float(reward_module.reward(state, reward_params)))
```

`reward_params` carries the loaded proxy network weights — `env_params` only
holds environment dynamics (max length, vocabulary size, etc.).

## GFP (experimental)

::: gfnx.environment.gfp.GFPEnvironment
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

### Reward

::: gfnx.reward.gfp.EqxProxyGFPRewardModule
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

### Proxy reward model training

As for AMP, the proxy is trained with Equinox. We do not ship a pretrained GFP
proxy yet. Follow the instructions in `proxy/datasets/gfp.py` to install the
extra packages, then run

```bash
python proxy/train_proxy.py --config-name gfp
```

### Quick start

```python
import jax
import gfnx

# Placeholder weights until a pretrained GFP proxy is released.
reward_module = gfnx.EqxProxyGFPRewardModule(
    proxy_config_path="proxy/configs/dummy_gfp.yaml",
    pretrained_proxy_path="proxy/weights/dummy_gfp/model",
    reward_exponent=1.0,
    min_reward=1e-6,
)

env = gfnx.GFPEnvironment()
env_params = env.init(jax.random.PRNGKey(0))
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())
state = env.reset()
```

## API references

- [Environments](environment_api.md)
- [Reward modules](reward_api.md)

## References

- Malkin, N. *et&nbsp;al.* (2022). *Trajectory balance: Improved credit assignment in GFlowNets.*
  Advances in Neural Information Processing Systems (NeurIPS).
- Tiapkin, D. *et&nbsp;al.* (2024). *Generative Flow Networks as Entropy-Regularized RL.*
  International Conference on Artificial Intelligence and Statistics (AISTATS).
- Shen, M. *et&nbsp;al.* (2023). *Towards Understanding and Improving GFlowNet Training.*
  International Conference on Machine Learning (ICML).
- Madan, K. *et&nbsp;al.* (2023). *Learning GFlowNets from partial episodes for improved convergence and stability.*
  International Conference on Machine Learning (ICML).
- Pirtskhalava, M. *et&nbsp;al.* (2021). *DBAASP v3: database of antimicrobial/cytotoxic activity and structure of peptides as a resource for development of new therapeutics.*
  Nucleic Acids Research, 49.
- Sarkisyan, K. *et&nbsp;al.* (2016). *Local fitness landscape of the green fluorescent protein.*
  Nature, 533.
- Barrera, L. A. *et&nbsp;al.* (2016). *Survey of variation in human transcription factors reveals prevalent DNA binding changes.*
  Science, 351(6280).
