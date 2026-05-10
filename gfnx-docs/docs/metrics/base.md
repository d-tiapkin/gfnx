# Lifecycle & Multimetrics

The metrics framework in `gfnx.metrics` gives every metric the same lifecycle and
makes it easy to compose several metrics in parallel. All metric modules inherit
from `BaseMetricsModule` and operate on immutable `MetricsState` dataclasses that
are safe to jit, pmap, or store on device.

## Lifecycle

- `init(rng_key, args)`: allocate a fresh state object (buffers, accumulators,
  cached ground-truth quantities);
- `update(state, rng_key, args)`: fold new observations into the state during
  training or evaluation loops;
- `process(state, rng_key, args)`: run any expensive post-processing before
  evaluation (often optional for streaming metrics);
- `get(state)`: return a flat `dict[str, Any]` of metric values ready for logging.

Each module defines `InitArgs`, `UpdateArgs`, and `ProcessArgs` classes so the
expected payload is explicit and type-checkable.

## Working with multiple metrics

`MultiMetricsModule` wraps a dictionary of metric modules and fans out every
lifecycle call. Results are namespaced as `{metric_name}/{key}` to avoid clashes.
Use it when you want a single object that drives several metrics on the same set
of trajectories.

```python
metrics = gfnx.metrics.MultiMetricsModule({
    "elbo": gfnx.metrics.ELBOMetricsModule(...),
    "tv": gfnx.metrics.ApproxDistributionMetricsModule(...),
})
state = metrics.init(rng_key, metrics.InitArgs(metrics_args={}))
state = metrics.update(state, rng_key, metrics.UpdateArgs(metrics_args={}))
state = metrics.process(state, rng_key, metrics.ProcessArgs(metrics_args={}))
values = metrics.get(state)
# values -> {"elbo/elbo": ..., "tv/tv": ...}
```

## Authoring a custom metric

- Subclass `BaseMetricsModule` and implement the four lifecycle methods;
- Define inner `InitArgs`, `UpdateArgs`, and `ProcessArgs` dataclasses (inherit
  the empty shells from `gfnx.metrics` if you have nothing to pass);
- Use pure functions and JAX arrays inside your state so it is compatible with
  JIT and parallel transforms;
- Return plain Python scalars or JAX arrays from `get`; higher-level tooling such
  as loggers or WandB adapters can handle the resulting dictionary directly.
