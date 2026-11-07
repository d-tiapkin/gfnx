# Ising Model Environment

This environment implements the Ising model as a discrete, energy-based sampling task in the GFlowNet framework. It follows the formulation introduced by Zhang *et al.* (2022), where generating lattice configurations corresponds to constructing spin assignments sequentially. The environment provides a structured setup for testing GFlowNets on probabilistic graphical models and energy-based rewards.

## Intuition

* **State** – a partial spin configuration represented by a 1D tensor of length $D$, where each entry corresponds to a site on the lattice.
  Each spin can take one of three values:

  * `-1`: unassigned site (denoted as $\emptyset$ in the theoretical formulation)
  * `0`: assigned spin −1
  * `1`: assigned spin +1

  The initial state is the empty configuration
  $$
  s_0 = [-1, -1, \dots, -1],
  $$
  and a full configuration $\mathbf{x} \in \{-1, +1\}^D$ is reached after all spins are assigned.

* **Action** – at each step, the policy selects one unassigned position and sets its spin.
  The action space has size $2D$:

  * Actions in `[0, D-1]` assign spin `0` (i.e., −1 in canonical form).
  * Actions in `[D, 2D-1]` assign spin `1` (i.e., +1 in canonical form).

  The environment terminates automatically when all positions are filled; there is no explicit “exit” action.

* **Backward action** – the inverse operation removes a previously assigned spin, replacing it with `-1`.
  The backward action space has size `D`, each action corresponding to a site index to clear.

* **Observation** – the current spin vector itself, a discrete tensor in $\{-1, 0, 1\}^D$.
  It fully specifies the partial configuration and serves as input to the policy and reward module.

* **Trajectory** – a sequence of states from the empty lattice to a complete spin configuration:
  $$
  s_0 \rightarrow s_1 \rightarrow \dots \rightarrow s_D = \mathbf{x}.
  $$
  Forward and backward trajectories are used symmetrically by the GFlowNet.

## Reward structure

At terminal states (full spin configurations), the reward corresponds to the Gibbs probability of the Ising model:

$$
R(\mathbf{x}) = \exp\big(-\mathcal{E}_J(\mathbf{x})\big),
$$

where the Ising energy is defined as

$$
\mathcal{E}_J(\mathbf{x}) = - \sum_{i=1}^D \sum_{j=1}^D J_{ij} , \mathbf{x}^i \mathbf{x}^j = -\mathbf{x}^\top J \mathbf{x}.
$$

Here:

* $\mathbf{x}^i \in \{-1, +1\}$ are the spin values (converted internally from $\{0, 1\}$ by $\mathbf{x} = 2s - 1$),
* $J \in \mathbb{R}^{D \times D}$ is the symmetric interaction matrix,
* Positive $J_{ij}$ values encourage aligned spins; negative values encourage anti-alignment.

The **log-reward** used in training is simply the negated energy:
$$
\log R(\mathbf{x}) = -\mathcal{E}_J(\mathbf{x}) = \mathbf{x}^\top J \mathbf{x}.
$$

Intermediate states (incomplete spin assignments) are typically assigned a reward of zero, so the total reward is only defined for terminal configurations.

### Reward module

The `IsingRewardModule` encapsulates this computation:

* The interaction matrix `J` lives inside `IsingRewardParams(J=...)` — i.e. as
  part of the **trainable** reward parameters, not as a constant attribute on
  the module.
* The log-reward is computed as

  ```python
  canonical = 2 * state.state - 1
  log_reward = canonical @ J @ canonical
  ```
* The full reward is obtained via `exp(log_reward)`.

Because `J` is part of `reward_params`, it can be jointly updated with the
GFlowNet policy during training. This is exactly what `baselines/tb_ising.py`
does: after each EBM step the train state replaces `reward_params` with the
updated matrix (`reward_params.replace(J=new_ebm.J)`).

## Example usage

```python
import jax
import gfnx
from gfnx import IsingEnvironment, IsingRewardModule

# 1. Build the reward module separately.
reward_module = IsingRewardModule()

# 2. Build the environment (no reward attached).
env = IsingEnvironment(dim=100)  # 10x10 lattice
env_params = env.init(jax.random.PRNGKey(0))

# 3. Initialise reward parameters — J is sampled / set here.
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())

# 4. Reset to a single initial state.
state = env.reset()
```

All environment methods operate on a single state — wrap them in `jax.vmap`
to roll out many configurations in parallel. The log-reward is computed
post-rollout via `reward_module.log_reward(state, reward_params)`.

## API references:

- [Environment](environment_api.md)
- [Reward module](reward_api.md)

## References

- Zhang, J. *et&nbsp;al.* (2022). *Generative flow networks for discrete probabilistic modeling*
  International Conference on Machine Learning, pages 26412–26428. PMLR, 2022.
- Ising, E. (1925). *Beitrag zur theorie des ferromagnetismus.*
  Zeitschrift für Physik, 31(1):253–258, 1925.
