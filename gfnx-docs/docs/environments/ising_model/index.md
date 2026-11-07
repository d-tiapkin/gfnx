# Ising Model Environment

This environment implements the Ising model as a discrete, energy-based
sampling task in the GFlowNet framework. It follows the formulation introduced
by Zhang *et al.* (2022), where generating lattice configurations corresponds to
assigning spins one at a time. It is the environment used to demonstrate
*trainable* rewards in `gfnx`: the interaction matrix $J$ lives in the reward
parameters and can be learned jointly with the sampler (EB-GFN).

The specification below is rendered from the docstrings of
[`IsingEnvironment`][gfnx.environment.ising.IsingEnvironment] and
[`IsingRewardModule`][gfnx.reward.ising.IsingRewardModule].

## Specification

::: gfnx.environment.ising.IsingEnvironment
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

## Reward

::: gfnx.reward.ising.IsingRewardModule
    options:
      members: false
      show_root_heading: false
      show_root_toc_entry: false
      show_bases: false

Because `J` is part of `reward_params`, it can be updated during training. This
is exactly what `baselines/tb_ising.py` does: after each energy-model step the
train state replaces `reward_params` with the updated matrix
(`reward_params.replace(J=new_ebm.J)`), with no recompilation of the
environment.

## Quick start

```python
import jax
import gfnx

N, sigma = 10, 0.1  # 10 x 10 toroidal lattice

# 1. Build the reward module and the environment separately.
reward_module = gfnx.IsingRewardModule()
env = gfnx.IsingEnvironment(dim=N * N)
env_params = env.init(jax.random.PRNGKey(0))

# 2. `init` returns J = 0 (a uniform target); plug in a ground-truth lattice.
reward_params = reward_module.init(jax.random.PRNGKey(0), env.reset())
reward_params = reward_params.replace(J=gfnx.utils.get_true_ising_J(N, sigma))

# 3. Roll out a uniform policy and score the terminal configurations.
def uniform_policy(rng_key, obs, policy_params):
    return jax.numpy.zeros(env.action_space.n), {}

traj_data, final_states, info = jax.vmap(
    lambda key: gfnx.utils.forward_rollout(key, uniform_policy, None, env, env_params)
)(jax.random.split(jax.random.PRNGKey(1), 8))
log_rewards = jax.vmap(reward_module.log_reward, in_axes=(0, None))(final_states, reward_params)
```

## API references

- [Environment](environment_api.md)
- [Reward module](reward_api.md)

## References

- Zhang, D. *et&nbsp;al.* (2022). *Generative flow networks for discrete probabilistic modeling.*
  International Conference on Machine Learning (ICML), pages 26412–26428. PMLR.
- Ising, E. (1925). *Beitrag zur Theorie des Ferromagnetismus.*
  Zeitschrift für Physik, 31(1):253–258.
