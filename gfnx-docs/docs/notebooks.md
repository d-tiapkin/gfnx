# Notebooks

Self-contained notebooks that run in Google Colab with nothing but a browser.
Each one installs `gfnx` from GitHub in its first cell; the sources live in
[`notebooks/`](https://github.com/d-tiapkin/gfnx/tree/main/notebooks).

| Notebook | What you will learn | Time (CPU) | |
| --- | --- | --- | --- |
| **1. Quickstart** | The decoupled environment / reward API, vmapped rollouts, a complete Trajectory Balance training loop compiled with `jax.jit`, and exact evaluation on the hypergrid. | ~1 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d-tiapkin/gfnx/blob/main/notebooks/01_quickstart.ipynb) |
| **2. Seeds and configurations in parallel** | Writing a training run as a pure function and vmapping it over a grid of random seeds × learning rates. | a few min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d-tiapkin/gfnx/blob/main/notebooks/02_vmap_seeds_and_configs.ipynb) |
| **3. Custom environment** | Implementing a new environment and reward module, checking that backward steps invert forward steps, and training a sampler with a learned backward policy. | ~1 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d-tiapkin/gfnx/blob/main/notebooks/03_custom_environment.ipynb) |

!!! note "Colab runtime"
    `gfnx` supports Python 3.12–3.14, which covers the current Colab runtimes.
    The first cell installs `gfnx` and, if that upgrades `numpy` or `jax` while
    an older version is already loaded, restarts the runtime once; run the
    notebook again from the top after the restart.

To run the notebooks locally instead, install `gfnx` together with `equinox`,
`optax` and Jupyter:

```bash
pip install -e ".[baselines]" jupyter
jupyter lab notebooks/
```

After the notebooks, the [Walkthrough](walkthrough.md) dissects the full
Detailed Balance baseline, and the [API reference](api/index.md) documents
every environment, reward module and utility.
