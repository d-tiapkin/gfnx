# Sequence Environments

A collection of environments centered around generating sequences of various nature.

# Bit Sequence environment

This task is to generate binary sequences of a fixed length $n$, using a vocabulary of $k$-bit blocks.  The state space of this environment corresponds to sequences of $n/k$ words, and each word in these sequences is either an empty word $\oslash$ or one of $2^k$ possible $k$-bit words. The initial state $s_0$ corresponds to a sequence of empty words. The possible actions in each state are replacing an empty word $\oslash$ with one of $2^k$ non-empty words in the vocabulary. The set of terminal states consists of sequences without empty words and corresponds to binary strings of length $n$. The reward function is defined as ${R(x) = \exp(-\beta \cdot \min_{x' \in M} d(x, x') / n)}$, where $M$ is a set of modes, $d$ is Hamming distance and $\beta$ is the reward exponent (inverse temperature). The environment was first presented in Malkin et al. (2022), and the version with arbitrary order generation implemented in `gfnx` was proposed in Tiapkin et al. (2024).

## Intuition

- **State**: a vector of length $n / k$, with each coordinate containing either a $k$-bit word or an empty word $\oslash$. 
- **Actions**: choose a coordinate containing $\oslash$ and a $k$-bit word to replace it with.
- **Trajectory**: a sequence of edits where each replaces an empty word.

## Environment parameters

- `n`: number of bits in the whole string.
- `k`: the number of bits in one block, choosing $k > 1$ results in working with sequences of length $n / k$.
  
## Reward parameters

- `mode_set_size`: number of modes $|M|$.
- `reward_exponent`: reward_exponent parameter, larger values result in a more peaky distribution.

## Quick start

Environments and rewards are decoupled in `gfnx`: build the reward module
separately, then call `reward_module.init(rng_key, env.reset())` to obtain
`reward_params`. The environment constructor never takes a reward.

```python
import jax
import jax.numpy as jnp
import gfnx

# 1. Build the reward.
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

# 3. Reset to obtain a single initial state.
state = env.reset()

# 4. Take a forward step:
#    action / 2^k corresponds to the position with empty word,
#    action mod 2^k corresponds to the k-bit word to put in its place.
action = jnp.int32(0)
obs, state, done, _ = env.step(state, action, env_params)

# 5. The reward is computed on the terminal state via the reward module.
log_reward = reward_module.log_reward(state, reward_params)
```

All environment methods operate on a **single** state — wrap them with
`jax.vmap` to run multiple trajectories in parallel.


# TFBind-8 environment

The task is to generate a string of length $8$ of nucleotides (A, C, G, T). The reward is wet-lab measured DNA binding
activity to a human transcription factor, SIX6. The generation is done autoregressively, each state is a string of length $\le 8$, and $4$ actions correspond to appending a nucleotide to its end. `gfnx` implements the autoregressive version of the environment from Shen et al. (2023).

## Quick start

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

# 3. Reset to a single initial state.
state = env.reset()

# 4. Take a forward step.
action = jnp.int32(0)
obs, state, done, _ = env.step(state, action, env_params)
```

# QM9 Small environment

The goal is to generate a small molecule graph. Rewards are taken from a proxy model trained to predict HOMO-LUMO gap. This is a small version of the environment that uses $11$ building blocks with $2$ stems, and generates $5$ blocks per molecule. As all the blocks have two stems, the environment is treated as a sequence prepend/append MDP. Thus, each state is a sequence of length $\le 5$, and each action adds one of $11$ blocks either to its beginning, or to its end. This version of the environment was presented in Shen et al. (2023).

## Quick start

```python
import jax
import jax.numpy as jnp
import gfnx

# 1. Build the reward.
reward_module = gfnx.QM9SmallRewardModule()

# 2. Build the environment.
env = gfnx.QM9SmallEnvironment()
env_params = env.init(jax.random.PRNGKey(0))
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

# 3. Reset to a single initial state.
state = env.reset()

# 4. Take a forward step.
action = jnp.int32(0)
obs, state, done, _ = env.step(state, action, env_params)
```


# AMP environment 

The goal is to generate a protein sequence of length $\le 60$ (where the vocabulary consists of $20$ amino acids and a special end-of-sequence token) with anti-microbial properties. The proxy reward model is a neural network classifier trained on the DBAASP database (Pirtskhalava et al. (2021)). Its probability output is used as $R(x)$. The generation is done autoregressively. `gfnx` implements the environment from Malkin et al. (2022).

## Proxy reward model training

We have implemented a code to train a proxy reward function with [Equinox](https://github.com/patrick-kidger/equinox) library for this environment in the `proxy` folder. The weights for the AMP model are already pretrained and stored in `/proxy/weights/amp` folder using `orbax` checkpointer. If you wish to perform training yourself, follow instructions in `proxy/datasets/amp.py` for installation of required packages, and then just run
```bash
python proxy/train_proxy.py --config-name amp
```

All the configuration is handled by Hydra, so it is possible to play with the training of your proxy network as you wish.

## Quick start

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

# 4. Reset to a single initial state.
state = env.reset()

# 5. Take a forward step.
action = jnp.int32(0)
obs, state, done, _ = env.step(state, action, env_params)
```

`reward_params` carries the loaded proxy network weights — `env_params` only
holds environment dynamics (max length, vocabulary size, etc.).

# GFP environment (experimental)

The goal is to generate a protein sequence of a fixed length $237$ (where the vocabulary consists of $20$ amino acids) with
fluorescence properties. The proxy reward model is a neural network regressor trained on a dataset of proteins with their fluorescence scores from Sarkisyan et al. (2016). The generation is done autoregressively. `gfnx` implements the environment from Madan et al. (2023).

## Proxy reward model training

Similarly to AMP, we have implemented to code to train the proxy reward function with [Equinox](https://github.com/patrick-kidger/equinox) library. The enviornment is **experimental** and we do not provide (yet) the pretrained proxy.
If you wish to perform training yourself, follow instructions in `proxy/datasets/gfp.py` for installation of required packages, and then just run

```bash
python proxy/train_proxy.py --config-name gfp
```


## Quick start


```python
import jax
import jax.numpy as jnp
import gfnx

# 1. Load the reward model. Currently we load a dummy reward model.
reward_module = gfnx.EqxProxyGFPRewardModule(
    proxy_config_path="proxy/configs/dummy_gfp.yaml",
    pretrained_proxy_path="proxy/weights/dummy_gfp/model",
    reward_exponent=1.0,
    min_reward=1e-6,
)

# 2. Create the environment.
env = gfnx.GFPEnvironment()
env_params = env.init(jax.random.PRNGKey(0))

# 3. Initialise reward parameters — proxy weights live here.
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

# 4. Reset to a single initial state.
state = env.reset()

# 5. Take a forward step.
action = jnp.int32(0)
obs, state, done, _ = env.step(state, action, env_params)
```


## API references:

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
