# API reference

The API reference is generated from the docstrings in `src/gfnx`, so it is also
what `help(...)` prints in Python. If you are new to `gfnx`, read the
[Background](../background.md) and the [Walkthrough](../walkthrough.md) first.

| Section | Contents |
| --- | --- |
| [Core abstractions](core.md) | `BaseEnvironment`, `BaseRewardModule`, state / params base classes, spaces |
| [Rollouts & training utilities](utils.md) | `forward_rollout`, `backward_rollout`, trajectory log-probs, masking helpers, `run_training_loop`, exploration, replay buffer, distances |
| [Environments](environments.md) | Every environment, grouped by family |
| [Reward modules](rewards.md) | Every reward module, grouped by family |
| [Metrics](../metrics/api.md) | Evaluation metrics and the `init` / `update` / `process` / `get` lifecycle |

## Conventions

- **Single-state API.** Every environment, reward and rollout function acts on a
  single state with no batch dimension. Batch with `jax.vmap`.
- **Decoupled rewards.** Environments never hold a reward. Rewards are computed
  after the rollout with
  `jax.vmap(reward_module.log_reward, in_axes=(0, None))(final_states, reward_params)`.
- **Masks.** Every `*_mask` uses `True` = valid / include.
- **Pytrees everywhere.** States, environment parameters and reward parameters
  are frozen `chex` dataclasses, so they can be vmapped, scanned and passed
  through `jax.jit`.
