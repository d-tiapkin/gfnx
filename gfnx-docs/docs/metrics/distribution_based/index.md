# Distribution-Based Metrics

Distribution-based metrics compare the terminal states distribution
induced by the current policy or recent policies
with the ground truth distribution.
They come in two variants depending on how terminal states distribution is evaluated.

## Intuition

- Use this metric when the environment can enumerate its terminal space and provide a true distribution.
- Prefer the exact-distribution metric if produced time overhead is acceptable.
- Use ELBO for very large tasks, since distributional metrics need prohibitively many samples to be reliable.
- Add "2d_marginal_distribution" to view a coarse heatmap.

## Key parameters

### Approximate (`ApproxDistributionMetricsModule`)

- `metrics`: List of metric names to compute, choose from `{"tv", "kl", "jsd", "2d_marginal_distribution"}`.
- `env`: Enumerable environment for which to compute metrics.
- `buffer_size`: Maximum number of states to store in the replay buffer for empirical distribution computation.

### Exact (`ExactDistributionMetricsModule`)

- `metrics`: List of metric names to compute, choose from `{"tv", "kl", "2d_marginal_distribution"}`.
- `env`: Enumerable environment for which to compute metrics.
- `fwd_policy_fn`: Forward policy function for generating trajectories.
- `batch_size`: Batch size used when evaluating policy over states.
- `tol_epsilon`: Tolerance for convergence in distribution computation.

## Quick start

See the dedicated variant pages for runnable examples:

- [Approximate Distribution Metrics](approximate.md) - evaluates the approximation of a sampling distribution by storing recent terminal states in a first-in first-out buffer;
- [Exact Distribution Metrics](exact.md) -  evaluate the exact distribution produced by the current policy using dynamic programming;

## Returned scores (both variants)

- `tv`: Total Variation Distance.
- `kl`: KL-divergence.
- `jsd`: Jensen-Shannon divergence
- `2d_marginal_distribution`: Return a visualizable marginal distribution over first two axes.

We refer to the variant pages for key parameters and quick-start examples.
