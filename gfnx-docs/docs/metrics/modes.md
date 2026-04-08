# Mode Coverage Metrics

`AccumulatedModesMetricsModule` tracks how many reference modes a policy has
visited so far. Supply a catalogue of known modes and a distance function; the
module marks a mode as discovered when any visited state falls within a
threshold radius.

## Great for

- benchmarking multimodal environments where you can enumerate or approximate
  high-reward states.
- monitoring exploration progress across training steps.
- computing interpretable coverage ratios alongside reward-based metrics.

## Inputs you provide

- `modes`: a batch of reference terminal states (typically from ground truth or
  prior knowledge).
- `distance_fn(lhs_state, rhs_state) -> float`: measures how close two terminal
  states are.
- `distance_threshold`: maximum distance for considering a mode as visited.

## Outputs

- `num_modes`: integer count of discovered modes.
- `percent_modes`: fraction of modes discovered (between 0 and 1).

## Typical usage

> **Environment requirement:** provide a batch of reference modes (terminal states) in the same PyTree structure as the environment states, along with a distance function and threshold.

```python
import jax
import jax.numpy as jnp
import gfnx

env = gfnx.HypergridEnvironment(reward_module=gfnx.EasyHypergridRewardModule())
params = env.init(jax.random.PRNGKey(0))

mode_states = env.get_ground_truth_sampling(jax.random.PRNGKey(1), 128, params)


def grid_distance(lhs_state, rhs_state):
    return jnp.linalg.norm(lhs_state.state - rhs_state.state)


metrics = gfnx.metrics.AccumulatedModesMetricsModule(
    env=env,
    distance_fn=grid_distance,
    distance_threshold=0.15,
)
state = metrics.init(jax.random.PRNGKey(2), metrics.InitArgs(modes=mode_states))

# After every batch of rollouts, update with the terminal states.
state = metrics.update(
    state,
    jax.random.PRNGKey(3),
    metrics.UpdateArgs(
        states=trajectory.final_env_state
    ),  # terminal states collected from your sampler
)

scores = metrics.get(state)
print(int(scores["num_modes"]), float(scores["percent_modes"]))
```

## Intuition

- Keep the reference modes in the same tree structure as the environment states
  (including padding flags) so that distance computations broadcast correctly.
- The metric is streaming: `process` is a no-op, so you can log coverage at any
  point without extra work.
- Provide a fast distance function; it runs over the full batch of
  visited states and reference modes.
