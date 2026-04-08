# AGENTS.md — Codebase Guide for AI Assistants

This file summarizes the architecture, patterns, and key design decisions in the `gfnx` codebase for the benefit of AI assistants (and human contributors). Keep it updated as the codebase evolves.

**Paper**: "gfnx: Fast and Scalable Library for Generative Flow Networks in JAX" — Tiapkin et al., arXiv:2511.16592

---

## Repository Layout

```
gfnx/
├── src/gfnx/              # Main library
│   ├── base.py            # Abstract base classes (environment, reward, renderer)
│   ├── spaces.py          # Discrete / Box / Dict space definitions
│   ├── environment/       # 9 concrete environments
│   ├── reward/            # Reward modules (one per environment family)
│   ├── networks/          # Neural network components (Transformer, MLP reward models)
│   ├── metrics/           # Evaluation metrics (TV, KL, ELBO, EUBO, correlation, …)
│   ├── utils/             # rollout.py, masking.py, training.py, exploration helpers, misc
│   └── visualize/         # Matplotlib visualizations
├── baselines/             # 20 self-contained training scripts + Hydra configs
├── proxy/                 # Proxy reward model training pipeline
└── tests/                 # Test suite
```

**Design philosophy**: follows CleanRL's "single-file" approach — each baseline is a self-contained, hackable script. Full end-to-end JIT compilation in the style of purejaxrl.

---

## GFlowNet Background

GFlowNets learn to sample from π(x) = R(x)/Z over a discrete space X, where R is the reward and Z is unknown. The generation process is a DAG G = (S, E) with one initial state s₀ and terminal states X. A trajectory τ = (s₀, s₁, …, s_nτ) ends at some x ∈ X.

**Forward policy** P_F(s'|s): distribution over children of s.
**Backward policy** P_B(s'|s): distribution over parents of s.

**Trajectory balance constraint** (the core GFlowNet objective):

```
∏ P_F(st | st-1) = R(s_nτ)/Z · ∏ P_B(st-1 | st)   ∀τ ∈ T
```

When satisfied, sampling with P_F yields terminal states with probability R(x)/Z.

### Training Objectives (all implemented)

**Detailed Balance (DB)** — per-transition loss:
```
L_DB(θ; s, s') = [log F_θ(s) P_F(s'|s,θ) / (F_θ(s') P_B(s|s',θ))]²
```
where F_θ(s) approximates the state flow function.

**Trajectory Balance (TB)** — per-trajectory loss:
```
L_TB(θ; τ) = [log Z_θ · ∏ P_F(st|st-1,θ) / (R(s_nτ) · ∏ P_B(st-1|st,θ))]²
```
Z_θ is a learnable scalar estimating the normalizing constant. Trained with a **separate (higher) learning rate** from the policy network.

**Sub-Trajectory Balance (SubTB)** — generalizes DB and TB:
```
L_SubTB(θ; τ) = Σ_{0≤j<k≤nτ} w_jk · [log F_θ(sj) ∏ P_F / (F_θ(sk) ∏ P_B)]²
```
Weights w_jk = λ^(k-j), normalized to sum to 1. λ = 0.9 by default. TB = SubTB taking only full trajectory; DB = SubTB taking only individual transitions.

**Forward-Looking Detailed Balance (FLDB)** — for energy-based rewards R(x) = exp(−E(x)):
```
L_FLDB(θ; s, s') = [log F̃_θ(s) P_F(s'|s,θ) / (F̃_θ(s') P_B(s|s',θ)) + E(s') − E(s)]²
```
where F̃_θ(s) = exp(E(s)) · F_θ(s). Used for phylogenetic trees.

**Modified Detailed Balance (MDB)** — used for DAG structure learning; exploits the delta-score structure of modular rewards for efficient per-transition updates.

---

## Core Abstractions (`src/gfnx/base.py`)

### `BaseEnvState`
Chex frozen dataclass. Every concrete state inherits from this and adds domain fields.
Always has: `is_terminal [B]`, `is_initial [B]`, `is_pad [B]`.

### `BaseEnvParams`
Chex frozen dataclass. Every concrete `EnvParams` inherits from this. Contains only dynamics parameters — **reward params are NOT stored here** (see reward decoupling below).

### `BaseEnvironment[TEnvState, TEnvParams]`
Abstract base. All environments operate on **single states** (no batch dimension). **Stateless**: all mutable data lives in `EnvState`. **Does not hold a reward module** — rewards are fully decoupled. `BaseVecEnvironment` is an alias for backward compatibility.

Use `jax.vmap` externally to run multiple environments in parallel:
```python
rng_keys = jax.random.split(rng_key, num_envs)
traj_data, final_states, info = jax.vmap(
    lambda rng: forward_rollout(rng, policy_fn, policy_params, env, env_params)
)(rng_keys)
```

Key methods (all single-instance):
| Method | Signature | Notes |
|--------|-----------|-------|
| `get_init_state()` | `-> TEnvState` | Returns a single initial state (no batch dim) |
| `init(rng_key)` | `-> TEnvParams` | Inits env dynamics params only (no reward) |
| `step(state, action, env_params)` | `-> (obs, next_state, done, info)` | Single step; no reward returned |
| `backward_step(state, bwd_action, env_params)` | `-> (obs, state, done, info)` | Single backward step |
| `_transition(state, action, env_params)` | `-> (next_state, done, info)` | **Implement this** in subclasses (unbatched) |
| `_backward_transition(state, bwd_action, env_params)` | `-> (next_state, done, info)` | **Implement this** in subclasses (unbatched) |
| `get_obs(state, env_params)` | `-> chex.ArrayTree` | Single state → single obs |
| `get_invalid_mask(state, env_params)` | `-> Bool[n_actions]` | Single state → mask |
| `get_invalid_backward_mask(state, env_params)` | `-> Bool[n_bwd_actions]` | Single state → mask |
| `get_backward_action(state, fwd_action, next_state, env_params)` | `-> Array` | Scalar backward action |
| `get_forward_action(state, bwd_action, prev_state, env_params)` | `-> Array` | Scalar forward action |

**Convenience batched methods** (defined in `BaseEnvironment`, not overridden):
| Method | Signature | Notes |
|--------|-----------|-------|
| `get_invalid_mask_batch(state, env_params)` | `-> Bool[B, n_actions]` | For use in `loss_fn` |
| `get_invalid_backward_mask_batch(state, env_params)` | `-> Bool[B, n_bwd_actions]` | For use in `loss_fn` |
| `get_backward_action_batch(state, fwd_action, next_state, env_params)` | `-> Array[B]` | For use in `loss_fn` |
| `get_forward_action_batch(state, bwd_action, prev_state, env_params)` | `-> Array[B]` | For use in `loss_fn` |

Enumerable environments additionally implement:
`get_all_states`, `state_to_index`, `get_true_distribution`, `get_empirical_distribution`,
`get_mean_reward`, `get_normalizing_constant`, `get_ground_truth_sampling`.
These methods accept `reward_module, reward_params` as explicit arguments.

### `BaseRewardModule[TEnvState, TRewardParams]`
Abstract base for reward functions. **Fully decoupled from environments**. Operates on **single states** — use `jax.vmap` for batches.

| Method | Signature | Notes |
|--------|-----------|-------|
| `init(rng_key, dummy_state)` | `-> TRewardParams` | Single dummy state (no batch dim) |
| `log_reward(state, reward_params)` | `-> Float[]` | Single state → scalar log reward |
| `reward(state, reward_params)` | `-> Float[]` | Single state → scalar reward |

Batched usage: `jax.vmap(reward_module.log_reward, in_axes=(0, None))(states, reward_params)`

**Reward decoupling design** (paper §2): rewards are decoupled from environment dynamics so that reward families can be swapped or learned during GFlowNet training without recompiling environment logic. The environment knows nothing about rewards; rewards know nothing about environment dynamics (only about terminal state structure).

**Initialization pattern** (post-decoupling):
```python
env = HypergridEnvironment(dim=4, side=20)  # no reward_module arg
env_params = env.init(rng_key)  # only dynamics params
reward_params = reward_module.init(reward_key, env.get_init_state())  # single dummy state
```

**Backward action abstraction** (paper §2, vs torchgfn): torchgfn defines a backward move as an inverse to every forward transition (e.g., "remove symbol at specific position"). gfnx abstracts to structural choices only (e.g., "remove any character at a particular position"). This makes it easy to reason about state reversibility and implement symmetric training objectives.

---

## Environments (`src/gfnx/environment/`)

| Class | File | Enumerable | Notes |
|-------|------|------------|-------|
| `HypergridEnvironment` | `hypergrid.py` | Yes | Grid of size `side^dim`; stop action = `dim` |
| `IsingEnvironment` | `ising.py` | No | Spin system; states ∈ {−1,+1,∅}^(N×N) |
| `NonAutoregressiveSequenceEnvironment` | `sequence.py` | No | Base for token-sequence envs |
| `BitseqEnvironment` | `bitseq.py` | No | Non-autoregressive bit strings |
| `GFPEnvironment` | `gfp.py` | No | Subclass of sequence |
| `AMPEnvironment` | `amp.py` | No | Autoregressive variable-length (≤60 tokens, vocab 20) |
| `TFBind8Environment` | `tfbind.py` | No | Autoregressive fixed-length 8, vocab 4 (A/C/G/T) |
| `DAGEnvironment` | `dag.py` | Conditionally | Sequential edge addition; acyclicity via transitive closure |
| `QM9SmallEnvironment` | `qm9_small.py` | No | Prepend/append, 11 blocks with 2 stems |
| `PhyloTreeEnvironment` | `phylogenetic_tree.py` | No | Sequential forest merges, n−1 steps |

### Sequence generation modes (paper §B.2)

Four distinct generation regimes implemented in the codebase:
- **Autoregressive fixed-length**: left-to-right, vocab size m, n steps
- **Autoregressive variable-length**: adds a stop action; P_B is degenerate
- **Prepend/append**: add symbols to beginning or end; action space = 2m
- **Non-autoregressive**: choose position + symbol at each step; action space = positions × vocab

---

## Reward Functions — Exact Formulas

### Hypergrid (paper Eq. 8)
```
R(s) = R0 + R1 × ∏_i 𝟙[0.25 < |s^i/(H−1) − 0.5|]
            + R2 × ∏_i 𝟙[0.3 < |s^i/(H−1) − 0.5| < 0.4]
```
Standard params: R0=10⁻³, R1=0.5, R2=2.0 (Easy); R0=10⁻⁴, R1=1.0, R2=3.0 (Hard).
There are 2^d high-reward regions near the corners of the grid.

### Bit Sequences (paper §B.2)
```
R(x) = exp(−β · min_{x'∈M} d(x, x')/n)
```
where d is Hamming distance, M is the mode set (|M|=60), β is reward exponent (β=3 default). Learner has access to R only, not M. Mode set H = {'00000000','11111111','11110000','00001111','00111100'}.

### TFBind8
Lookup table of wet-lab DNA binding activity to transcription factor SIX6. All 4^8 = 65536 entries stored in reward_params.

### QM9 (Small)
Proxy model predictions for HOMO-LUMO gap. Pre-trained weights loaded from reward_params.

### AMP
```
R_φ(x) = max(σ(f_φ(x)), r_min)
```
where f_φ is a classifier logit trained on DBAASP database (3219 AMP + 4611 non-AMP sequences).

### Phylogenetic Trees (paper §B.3)
```
R(T) = exp((C − M(T)) / α)
```
where M(T) is the parsimony score (minimum mutations), α=4 is temperature, C is dataset-specific constant {5800, 8000, 8800, 3500, 2300, 2300, 12500, 2800} for DS1–DS8.

### Bayesian Structure Learning / DAG (paper §B.4)
```
log R(G) = log P(D|G) + log P(G) = Σ_j LocalScore(X_j | Pa_G(X_j))
```
Implements both Linear-Gaussian and BGe scores. The score is **modular**: when adding edge X_i → X_j, only the local score of X_j changes (delta score). This is exploited for efficient MDB updates.

### Ising Model (paper §B.5)
```
E_J(x) = −x^T J x,  R(x) = exp(−E_J(x)),  J ∈ R^{N²×N²}
```
Joint learning of J and GFlowNet policy (EB-GFN algorithm). J initialized as toroidal lattice J = σ·A_N.

---

## Reward Modules (`src/gfnx/reward/`)

| Class | File | Notes |
|-------|------|-------|
| `GeneralHypergridRewardModule` | `hypergrid.py` | Analytical; takes `side: int` as constructor arg; `EasyHypergridRewardModule(side=...)`, `HardHypergridRewardModule(side=...)` are thin subclasses |
| `IsingRewardModule` | `ising.py` | Quadratic form x^T J x; J stored in `reward_params` as `IsingRewardParams(J=...)` |
| `EqxProxyAMPRewardModule` | `amp.py` | Pre-trained Equinox Transformer; model weights in `reward_params` |
| `EqxProxyGFPRewardModule` | `gfp.py` | Same pattern |
| `BitseqRewardModule` | `bitseq.py` | Mode-set Hamming distance; `reward_params["mode_set"]` holds the modes |
| `TFBind8RewardModule` | `tfbind.py` | Lookup table in `reward_params["rewards"]` |
| `QM9SmallRewardModule` | `qm9_small.py` | Lookup table in `reward_params["rewards"]` |
| `DAGRewardModule` | `dag.py` | Composes `BaseDAGPrior` + `BaseDAGLikelihood`; has `delta_score` for efficient incremental scoring |
| `PhyloTreeRewardModule` | `phylogenetic_tree.py` | Parsimony-based; stores `num_nodes` as Python attr; `reward_params = {}` |

**DAG special case**: `BaseDAGLikelihood` and `BaseDAGPrior` have a `delta_score` method not in `BaseRewardModule`:
```python
delta_score(state, action, next_state, env_params, reward_params)
```
`env_params` supplies geometry (e.g. `num_variables`); `reward_params` supplies data (likelihood/prior params). Used in `mdb_dag.py`.

**Ising special case**: The J matrix (stored in `reward_params`) is a **trainable** reward parameter jointly updated with the GFlowNet policy in `tb_ising.py`. After each EBM update step: `new_reward_params = IsingRewardParams(J=new_ebm.J)` replaces the old reward_params in `TrainState`.

**DAG acyclicity enforcement**: maintained online via adjacency matrix + transitive closure of G^T. When adding edge (u→v): set adjacency entry + update transitive closure via outer product of column v and row u (binary OR). O(d²) per step, no expensive cycle checks.

---

## Rollout Utilities (`src/gfnx/utils/rollout.py`)

### `TrajectoryData`
Chex dataclass with shape `[T+1, ...]` for a **single trajectory**:
- `obs`, `state`, `action`, `done`, `pad`, `info`
- After `jax.vmap(forward_rollout)(rng_keys)`, shape is `[B, T+1, ...]`.
- **No `log_gfn_reward` field** — rewards are computed post-rollout on terminal states.

### `TransitionData`
Shape `[T, ...]` for single trajectory after `split_traj_to_transitions`. Fields: `obs`, `state`, `action`, `next_obs`, `next_state`, `done`, `pad`.
After vmapping over B: `[B, T, ...]`. Flatten to `[B*T, ...]` with `jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), ...)`.

### Key functions

```python
# All functions operate on SINGLE trajectory/environment.
# Use jax.vmap externally for batches.

forward_rollout(rng_key, policy_fn, policy_params, env, env_params)
    -> (TrajectoryData[T+1, ...], final_state, info)
# info keys: "entropy" (scalar), "trajectory_length" (scalar)

backward_rollout(rng_key, init_state, policy_fn, policy_params, env, env_params)
    -> (TrajectoryData[T+1, ...], final_state, info)

split_traj_to_transitions(traj_data: TrajectoryData) -> TransitionData
# Single: [T+1, ...] -> [T, ...]; slices prev/next states

forward_trajectory_log_probs(env, fwd_traj_data, env_params) -> (log_pf, log_pb)  # scalars
backward_trajectory_log_probs(env, bwd_traj_data, env_params) -> (log_pf, log_pb)  # scalars
```

### Masking utilities (`src/gfnx/utils/masking.py`)

```python
mask_logits(logits, invalid_mask) -> logits
# Sets invalid action logits to -inf.

compute_action_log_probs(logits, actions, invalid_mask, pad_mask=None) -> Array
# Full pipeline: mask_logits → log_softmax → take_along_axis → optional zero-out pads.
# logits: [..., n_actions], actions: [...], invalid_mask: [..., n_actions]
# Returns: [...] log probabilities of selected actions; padded steps are set to 0.0.
```

Replaces the 4-line boilerplate in every baseline's loss function:
```python
# Before (4 lines):
masked_logits = gfnx.utils.mask_logits(logits, mask)
all_log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
selected = jnp.take_along_axis(all_log_probs, jnp.expand_dims(actions, -1), axis=-1).squeeze(-1)
selected = jnp.where(pad, 0.0, selected)

# After (1 line):
selected = gfnx.utils.compute_action_log_probs(logits, actions, mask, pad_mask)
```

### Training loop utility (`src/gfnx/utils/training.py`)

```python
run_training_loop(train_step_fn, init_state, num_steps, tqdm_print_rate=20) -> final_state
```

Wraps the standard `eqx.partition` / `jax.lax.fori_loop` / `jax.block_until_ready` boilerplate into one call. Used in all 18 single-seed baselines:
```python
train_state = gfnx.utils.run_training_loop(
    train_step, train_state, cfg.num_train_steps, cfg.logging["tqdm_print_rate"]
)
```
Not applicable to multiseed scripts (`*_multiseed.py`), which use `jax.vmap(fori_loop)` directly.

**Post-rollout pattern** (used in all training scripts):
```python
rng_keys = jax.random.split(sample_traj_key, num_envs)
traj_data, final_states, info = jax.vmap(
    lambda rng: gfnx.utils.forward_rollout(rng, fwd_policy_fn, policy_params, env, env_params)
)(rng_keys)
log_rewards = jax.vmap(reward_module.log_reward, in_axes=(0, None))(final_states, reward_params)
```

**Policy function contract** (for `forward_rollout`):
```python
def policy_fn(rng_key, env_obs, policy_params) -> (logits: Array[n_actions], info: dict)
# Single obs (no batch dim). info dict may contain "forward_logits", "backward_logits", "log_flow"
```

**Inner loop**: `jax.lax.scan` over `max_steps_in_episode + 1` steps. Output shape is `[T+1, ...]` (time-major, single env — no transpose needed).

---

## Training Scripts (`baselines/`)

### Structure (all 20 scripts follow the same pattern)

1. **`MLPPolicy(eqx.Module)`** — defined per-script; outputs vary by algorithm:
   - TB: `{"forward_logits", "backward_logits"}`
   - DB: `{"forward_logits", "log_flow", "backward_logits"}`

2. **`TrainState(NamedTuple)`** — holds all state for one training run:
   ```
   rng_key, config, env, env_params,
   reward_module,   # static — the reward module object
   reward_params,   # dynamic JAX arrays — e.g. lookup table, model weights, mode set
   model, optimizer, opt_state, metrics_module, metrics_state, eval_info, ...
   ```
   `reward_module` ends up in the static partition of `eqx.partition(train_state, eqx.is_array)`;
   `reward_params` ends up in the dynamic partition.

3. **`train_step(idx, train_state) -> TrainState`** — decorated with `@eqx.filter_jit`:
   - Split model: `policy_params, policy_static = eqx.partition(model, eqx.is_array)`
   - Rollout (vmapped single-env): `traj_data, final_states, info = jax.vmap(lambda rng: forward_rollout(rng, ...))(rng_keys)`
   - **Reward**: `log_rewards = jax.vmap(reward_module.log_reward, in_axes=(0, None))(final_states, reward_params)` → `[B]`
   - **Transitions**: `transitions = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), jax.vmap(split_traj_to_transitions)(traj_data))`
   - **Env methods in loss_fn**: use `env.get_invalid_mask_batch`, `env.get_invalid_backward_mask_batch`, `env.get_backward_action_batch`
   - **Log-probs in loss_fn**: `gfnx.utils.compute_action_log_probs(logits, actions, mask, pad_mask)` → scalar log prob per step
   - Loss: `eqx.filter_value_and_grad(loss_fn)(...)`
   - Update: `optimizer.update(grads, opt_state, params)`
   - Logging: `jax.debug.callback(...)` ← **not vmap-compatible**
   - **Metrics**: `metrics_state, eval_info = metrics_module.step(idx=idx, metrics_state=..., rng_key=..., update_args=..., process_args=..., eval_each=..., num_train_steps=..., prev_eval_info=...)`

4. **`run_experiment(cfg)`** (Hydra entry point):
   - Init env, reward module separately, model, optimizer, metrics
   - `train_state = gfnx.utils.run_training_loop(train_step, train_state, cfg.num_train_steps, cfg.logging["tqdm_print_rate"])`

### Reward application in losses

**TB loss** — `log_rewards` is `[B]`; used directly as terminal reward per trajectory:
```python
target = log_pb_sum + log_rewards  # both [B]
loss = squared_error(logZ + sum_log_pf, target).mean()
```

**DB loss** — `log_rewards` is `[B]`; must be broadcast to `[B*T]` (one per transition):
```python
T_steps = transitions.done.shape[0] // num_envs
traj_rewards_flat = jnp.repeat(log_rewards, T_steps)  # [B*T]
target = jnp.where(
    transitions.done, bwd_logprobs + traj_rewards_flat, bwd_logprobs + next_log_flow
)
```

**SubTB loss** — `log_rewards` is `[B]`; broadcast via `[:, jnp.newaxis]` when setting flow at terminal states:
```python
log_flow_traj = log_flow_traj.at[:, 1:].set(
    jnp.where(done_mask, log_rewards[:, jnp.newaxis], log_flow_traj[:, 1:])
)
```

### Optimizer pattern for TB (multi-param):
```python
# Separate learning rates for model and logZ
optimizer = optax.multi_transform(
    {"network_lr": optax.adam(lr), "logZ_lr": optax.adam(logZ_lr)}, param_labels
)
params_for_loss = {"model_params": policy_params, "logZ": logZ}
```

### Loss functions summary

| Algorithm | File | Loss formula |
|-----------|------|------|
| TB | `tb_hypergrid.py` | `(logZ + Σlog_pf − Σlog_pb − log_R)²` |
| DB | `db_hypergrid.py` | `(log_pf + log_F − log_pb − log_F_next)²` per transition; uses `log_R` at terminal |
| SubTB | `subtb_hypergrid.py` | Weighted sum over sub-trajectories; λ=0.9 |
| FLDB | `fldb_phylo.py` | `(log F̃(s)P_F − log F̃(s')P_B + E(s')−E(s))²`; delta_score via `reward_module.delta_score(state, action, next_state, env_params, reward_params)` |
| MDB | `mdb_dag.py` | DB with delta-score for modular rewards; per-edge update |

### Multi-seed vmap scripts

`tb_hypergrid_multiseed.py` and `db_hypergrid_multiseed.py` demonstrate vmapping the full training loop over random seeds:

```python
# TrainStateParams holds only JAX arrays (model_params, opt_state, reward_params, ...)
seeds = jnp.arange(cfg.num_seeds)
all_init_params = jax.vmap(make_init_params)(seeds)  # batched initialization


@jax.jit
def run_all_seeds(all_init_params):
    return jax.vmap(lambda init: jax.lax.fori_loop(0, cfg.num_train_steps, train_step, init))(
        all_init_params
    )


all_final_params = jax.block_until_ready(run_all_seeds(all_init_params))
# Log aggregate stats (mean ± std) over seeds after block_until_ready
```

Key constraints for vmap-compatible training loops:
- No `jax.debug.callback` inside the vmapped loop
- Static parts (`env`, `reward_module`, `policy_static`, `optimizer`) are captured via closure
- Only JAX arrays (`model_params`, `opt_state`, `reward_params`, `rng_key`, `logZ`) are batched

### Hyperparameters used in paper experiments

| Environment | Batch | LR | Z LR | Iterations | Architecture |
|-------------|-------|----|------|------------|--------------|
| Hypergrid | 16 | 1e-3 | 1e-1 | 1e6 | MLP, 2 layers, 256 hidden |
| Bit Sequences | 16 | 1e-3 | — | 5×10⁴ | Transformer, 3 layers, 64 hidden |
| TFBind8 / QM9 | 16 | 5e-4 | 0.05 | 1e6 | MLP, 2 layers, 256 hidden |
| AMP | 16 | 1e-3 | 0.64 | 2×10⁴ | Transformer, 3 layers, 64 hidden |
| Phylo Trees | 16–32 | 3e-4 | — | 3.2×10⁶ | Transformer, 6 layers, 32 hidden |
| DAG (MDB) | 128 | 1e-4 | — | 100000 | GNN+MLP+Transformer |
| Ising | 256 | — | — | 20000 | MLP, 4 layers, 256 hidden |

---

## JAX Patterns Used

| Pattern | Location | Purpose |
|---------|----------|---------|
| `jax.vmap(env.get_invalid_mask, in_axes=(0, None))` | `base.py:get_invalid_mask_batch` | Vectorize single-env mask method over batch; defined once in base class |
| `jax.vmap(jax.vmap(fn))` | training scripts | Double-vmap over `[B, T, ...]` trajectories |
| `jax.lax.scan` | `rollout.py:forward_rollout` | Unroll episode steps without Python loop |
| `jax.lax.cond` | `environment/*.py`, `metrics/base.py:step` | Conditional branching under JIT; used in both env transitions and metric eval gating |
| `jax.lax.fori_loop` | `utils/training.py:run_training_loop` | Main training loop — **vmappable**, unlike Python for loops |
| `jax.vmap(fori_loop(...))` | `*_multiseed.py` | Vmap full training over seeds |
| `eqx.filter_jit` / `eqx.partition` | `baselines/*:train_step` | JIT with Equinox modules |
| `eqx.filter_value_and_grad` | `baselines/*:loss_fn` | Grad through Equinox models |
| `jax.debug.callback` | `baselines/*:logging_callback` | Side-effectful logging inside JIT; **NOT vmap-compatible** |

---

## Metrics System (`src/gfnx/metrics/`)

All metrics follow a functional `init -> update -> process -> get` pattern, with a convenience `step()` method that combines update + conditional process + get into one call.

| Class | What it computes | How |
|-------|-----------------|-----|
| `ApproxDistributionMetricsModule` | TV / KL from rolling empirical buffer | Buffer of terminal states, flashbax |
| `ExactDistributionMetricsModule` | TV / KL vs. true distribution | Enumerable envs only; requires `get_true_distribution` |
| `ELBOMetricsModule` | ELBO = E[log R(x) − log P_θ(x)] | Forward rollouts |
| `EUBOMetricsModule` | EUBO = E[log P_B − log P_F + log R] | Backward rollouts from test set |
| `CorrelationMetricsModule` | Pearson/Spearman of P̂_θ(x) vs R(x) | Monte Carlo estimate: P̂_θ(x) = (1/N) Σ P_F(τ)/P_B(τ|x), τ~P_B(·|x) |
| `RewardDeltaMetricsModule` | Tracks mean reward shift | Running statistics |
| `MultiMetricsModule` | Composes any dict of the above | — |

**`step()` convenience method** (defined on `BaseMetricsModule`, available to all subclasses including `MultiMetricsModule`):
```python
metrics_state, eval_info = metrics_module.step(
    idx=idx,
    metrics_state=train_state.metrics_state,
    rng_key=rng_key,
    update_args=metrics_module.UpdateArgs(...),
    process_args=metrics_module.ProcessArgs(...),
    eval_each=train_state.config.logging.eval_each,
    num_train_steps=train_state.config.num_train_steps,
    prev_eval_info=train_state.eval_info,
)
```
Runs `update()` every step; runs `process()` + `get()` only on eval steps (using `jax.lax.cond` — stays JIT-compatible). Returns `prev_eval_info` unchanged on non-eval steps.

**Reward in metrics**: `EUBOMetricsModule` and `CorrelationMetricsModule` take `reward_module` at construction time (for one-shot operations like logZ estimation and test set sampling). `reward_params` is passed at each evaluation via `ProcessArgs` — it is NOT stored as a class field, since it can be large or trainable:
```python
EUBOMetricsModule(env=env, env_params=env_params,
                  reward_module=reward_module, reward_params=reward_params, ...)
# reward_params passed at each eval:
EUBOMetricsModule.ProcessArgs(policy_params=..., env_params=..., reward_params=train_state.reward_params)
```

**Pearson correlation metric** (paper §B.2): used for bitseq, AMP, phylo. Measures correlation between R(x) and P̂_θ(x) on a fixed test set. P̂_θ(x) = (1/N) Σ_{i=1}^{N} P_F(τ^i|θ)/P_B(τ^i|x), τ^i ~ P_B(τ|x), N=10 samples per state.

**JSD metric** (paper §B.4, Eq. 15): used for DAG. JSD(P||Q) = ½KL(P||M) + ½KL(Q||M), M = ½(P+Q). Exact posterior computable since |DAGs with d=5| = 29,281.

---

## Performance Benchmarks (paper Table 1)

All vs. torchgfn or author implementations:

| Environment | Device | Objective | Baseline | gfnx | Speedup |
|-------------|--------|-----------|----------|------|---------|
| Hypergrid 20⁴ | CPU | DB | 178 it/s | **1560 it/s** | ~9× |
| Hypergrid 20⁴ | CPU | TB | 220 it/s | **1463 it/s** | ~7× |
| Hypergrid 20⁴ | CPU | SubTB | 121 it/s | **596 it/s** | ~5× |
| Bitseq (n=120,k=8) | GPU | DB | 52 it/s | **1666 it/s** | ~32× |
| Bitseq (n=120,k=8) | GPU | TB | 54 it/s | **2434 it/s** | ~45× |
| TFBind8 | CPU | TB | 230 it/s | **6929 it/s** | ~30× |
| QM9 | CPU | TB | 162 it/s | **9062 it/s** | **~56×** |
| AMP | GPU | TB | 21 it/s | **413 it/s** | ~20× |
| Phylo Trees (DS-1) | GPU | FLDB | 13 it/s | **264 it/s** | ~20× |
| Structure Learning | GPU | MDB | 0.73 it/s | **58 it/s** | **~80×** |

Key insight: gfnx runs the full training pipeline (including environment) end-to-end on GPU/TPU via JAX JIT. Prior libraries (torchgfn, author implementations) run environment logic on CPU, causing repeated CPU↔GPU transfers.

---

## Known Limitations / Future Work (from paper §4)

- **No continuous action spaces** — only discrete. Phylogenetic trees with branch lengths would require this.
- **No non-acyclic environments** — permutation generation and similar problems need non-DAG state spaces.
- **No multi-objective support** — Pareto-optimal generation not implemented.
- **Missing baselines** — entropy-regularized RL algorithms, backward policy optimization, exploration techniques not yet added.

---

## Common Gotchas

- **`_transition` is unbatched** — implement it without vmap; it is the abstract method subclasses override. `step()` calls it directly (no internal vmap — the batch vmap is applied externally at the rollout site).
- **`env.max_steps_in_episode`** must be exact — `forward_rollout` allocates `T+1` steps. One extra step is always padding.
- **`TrajectoryData` is `[T+1, ...]` per single rollout** — after `jax.vmap(forward_rollout)(rng_keys)`, shape is `[B, T+1, ...]`. The last time step is always padding (`pad=True`).
- **`split_traj_to_transitions` is single-trajectory** — apply `jax.vmap(split_traj_to_transitions)(traj_data)` to get `[B, T, ...]`, then flatten to `[B*T, ...]` with `jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), ...)`.
- **`jax.debug.callback` does not work inside `jax.vmap`** — do not use it in training loops intended to be vmapped over seeds.
- **Equinox `eqx.partition`** splits a pytree into `(arrays, static)`. The static part must be shared / hashable. The dynamic arrays part is what gets vmapped/JIT'd.
- **`jax.lax.fori_loop` IS vmappable** — use it as the loop primitive in multi-seed training.
- **Reward is computed once per rollout**, not per step — `jax.vmap(reward_module.log_reward, in_axes=(0, None))(final_states, reward_params)` is called after `forward_rollout` returns, on the batch of terminal states.
- **`reward_params` must NOT be stored as a class field on metrics modules** — they can be large (e.g. full TFBind8 table) or trainable (Ising J matrix). Pass via `ProcessArgs` at each evaluation call instead.
- **Hypergrid reward modules require `side` at construction** — `EasyHypergridRewardModule(side=cfg.environment.side)`. The reward module needs to know the grid size to compute coordinates, but it is no longer passed `env_params` at call time.
- **DB terminal loss weighting** — for transition-level losses on AMP, terminal states are penalized more heavily (λ=25 penalty factor). This improves stability.
- **Phylo tree rewards use a stability constant C** — `R(T) = exp((C − M(T)) / α)`. Without C the exponent can overflow. Dataset-specific C values: {5800, 8000, 8800, 3500, 2300, 2300, 12500, 2800} for DS1–DS8.
- **ε-uniform exploration** is used across all environments to improve coverage. Schedules vary: constant ε=1e-3 (bitseq/AMP), linearly annealed 1.0→0.0 (TFBind8/QM9/phylo), linearly annealed 1.0→0.1 (DAG).
