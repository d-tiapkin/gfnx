# Flow Consistency in Sub-graphs (FCS)

FCS ([Silva et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a48f8928f78a58399ef0049453c14b02-Abstract-Conference.html))
is a goodness-of-fit metric that does not need the normalising constant or an
enumeration of $\mathcal{X}$. Rather than comparing the full sampling distribution
$p_T$ with $R/Z$, it compares both after *restricting* them to random subsets
$S \subseteq \mathcal{X}$ of terminal states:

$$
p_T^{(S)}(x) = \frac{\mathbb{1}\{x \in S\}\, p_T(x)}{\sum_{y \in S} p_T(y)},
\qquad
R^{(S)}(x) = \frac{\mathbb{1}\{x \in S\}\, R(x)}{\sum_{y \in S} R(y)},
\qquad
\mathrm{FCS}(p_T, R) = \mathbb{E}_{S \sim P_S}\Big[\mathrm{TV}\big(p_T^{(S)}, R^{(S)}\big)\Big].
$$

The marginal is estimated with backward rollouts,
$p_T(x) = \mathbb{E}_{\tau \sim P_B(\cdot \mid x)}\big[P_F(\tau) / P_B(\tau \mid x)\big]$,
using `n_rounds` trajectories per state
([`gfnx.utils.estimate_marginal_log_probs`](../api/utils.md#marginal-estimation)).

## Intuition

- FCS lies in $[0, 1]$ and is $0$ if and only if $p_T \propto R$; lower is better.
- The subset size $\beta$ (`subset_size`) interpolates between a ratio-matching
  metric ($\beta = 2$) and the full TV distance ($\beta = |\mathcal{X}|$). The paper
  sets $\beta$ to the training batch size.
- Only ratios within a subset matter, so $\log Z$ never has to be known.
- Within a subset, repeated states are counted once (`deduplicate=True`), so $S$ is a
  set as in the paper. Equality means every leaf of the state pytree matches; if two
  different representations encode the same object, they are treated as distinct.
- Cost per evaluation is `n_subsets * subset_size * n_rounds` backward rollouts
  (plus `n_subsets * subset_size` forward rollouts on-policy). In autoregressive
  environments $P_B$ is deterministic, so `n_rounds=1` gives exact $p_T$.

## Variants

### On-policy (`OnPolicyFCSMetricsModule`)

Each subset is the set of terminal states of a fresh batch of `subset_size` forward
rollouts of the current policy. This is the implementation used in the paper; it
focuses on the regions where the sampler puts its mass.

### Test set (`TestFCSMetricsModule`)

Each subset is a uniformly random `subset_size`-subset of a fixed test set of
terminal states. $P_S$ stays fixed across training, so scores are comparable between
evaluations. With `subset_size` equal to the test-set size, FCS is the TV distance
between $p_T$ and $R$ restricted to the test set.

## Key parameters

- `subset_size`: maximum subset size $\beta \geq 2$;
- `n_subsets`: subsets averaged per evaluation (default `32`, as in the paper);
- `n_rounds`: backward trajectories per terminal state for estimating $p_T$ (default `10`);
- `batch_size`: backward rollouts run in parallel; bounds memory (defaults to `subset_size`);
- `fwd_policy_fn` (on-policy only) / `bwd_policy_fn`: policy callables whose aux dict
  contains both `forward_logits` and `backward_logits`;
- `deduplicate`: count repeated states within a subset once (default `True`).

### Lifecycle arguments

| Dataclass | Fields | Purpose |
| --- | --- | --- |
| `InitArgs` | on-policy: _empty_; test set: `test_set` | The test-set variant stores the pool of terminal states to draw subsets from. |
| `UpdateArgs` | _empty_ | FCS is recomputed from scratch at each evaluation. |
| `ProcessArgs` | `policy_params`, `env_params`, `reward_params` | Current policy, environment and reward parameters. |

Both variants return `fcs`.

## In the baselines

Every baseline script that combines several metrics reports FCS as `eval/fcs/fcs`:

- the TB, DB and SubTB scripts for bit sequences and TFBind-8;
- the TB and DB scripts for QM9 and AMP;
- the DB script for GFP;
- `tb_hypergrid.py`, `mdb_dag.py`, `mdb_dag_replay_buffer.py` and `fldb_phylo.py`.

They use `OnPolicyFCSMetricsModule` with the current policy, without ε-exploration. The settings live under `metrics.fcs` in each config:

```yaml
metrics:
  fcs:
    subset_size: 128  # beta
    n_subsets: 32
    n_rounds: 10
    batch_size: 4096  # backward rollouts run in parallel
```

The defaults follow the paper (32 subsets of up to 128 states), with these exceptions:

- TFBind-8, AMP and GFP are autoregressive. Each sequence has a single trajectory, so
  `n_rounds: 1` already gives the exact $p_T$.
- AMP and GFP use fewer subsets (16 and 4), so that FCS samples as many sequences as the
  top-k metric does.
- Phylogenetic trees use 8 subsets of 32 trees with 8 backward rollouts each, because
  their rollouts are expensive.

FCS is usually the most expensive evaluation metric. At the defaults, one evaluation of
`tb_hypergrid.py` takes about 2 seconds on CPU. If evaluation dominates the run time,
lower `n_subsets` or `n_rounds`, or raise `logging.eval_each`.

## Quick start

```python
import jax
import jax.numpy as jnp
import gfnx

env = gfnx.HypergridEnvironment(dim=2, side=8)
env_params = env.init(jax.random.PRNGKey(0))
reward_module = gfnx.EasyHypergridRewardModule(side=8)
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())


def uniform_policy(rng_key, obs, policy_params):
    fwd = jnp.zeros((env.action_space.n,))
    bwd = jnp.zeros((env.backward_action_space.n,))
    return fwd, {"forward_logits": fwd, "backward_logits": bwd}


def uniform_backward_policy(rng_key, obs, policy_params):
    fwd = jnp.zeros((env.action_space.n,))
    bwd = jnp.zeros((env.backward_action_space.n,))
    return bwd, {"forward_logits": fwd, "backward_logits": bwd}


fcs = gfnx.metrics.OnPolicyFCSMetricsModule(
    env=env,
    reward_module=reward_module,
    fwd_policy_fn=uniform_policy,
    bwd_policy_fn=uniform_backward_policy,
    subset_size=16,
    n_subsets=32,
    n_rounds=10,
)
state = fcs.init(jax.random.PRNGKey(1))
state = fcs.process(
    state,
    jax.random.PRNGKey(2),
    fcs.ProcessArgs(policy_params=None, env_params=env_params, reward_params=reward_params),
)
print(float(fcs.get(state)["fcs"]))
```
