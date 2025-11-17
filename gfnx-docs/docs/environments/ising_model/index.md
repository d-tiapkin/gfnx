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

* It maintains `J`, the interaction matrix, as part of its parameters.
* It computes the log-reward as

  ```python
  canonical = 2 * state.state - 1
  log_reward = jnp.einsum("bi,ij,bj->b", canonical, J, canonical)
  ```
* The full reward is obtained via `exp(log_reward)`.

This setup allows the energy model to be updated during training, enabling joint learning of the generative policy and the underlying energy function.

## Example usage

```python
import jax
import gfnx

from gfnx.environment.ising import IsingEnvironment, IsingRewardModule

reward = IsingRewardModule()
env = IsingEnvironment(reward_module=reward, dim=100)  # 10x10 lattice
params = env.init(jax.random.PRNGKey(0))

obs, state = env.reset(num_envs=1, env_params=params)
```

Just like other GFNX environments, `IsingEnvironment` is fully vectorised:
set `num_envs > 1` to roll out multiple forests in parallel. When a trajectory
terminates the returned `log_reward` corresponds to the expression above.

## API references:

- [Environment](environment_api.md)
- [Reward module](reward_api.md)

## References

- Zhang, J. *et&nbsp;al.* (2022). *Generative flow networks for discrete probabilistic modeling*
  International Conference on Machine Learning, pages 26412–26428. PMLR, 2022.
- Ising, E. (1925). *Beitrag zur theorie des ferromagnetismus.*
  Zeitschrift für Physik, 31(1):253–258, 1925.
