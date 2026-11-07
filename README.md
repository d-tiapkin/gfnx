# gfnx: Fast and Scalable Generative Flow Networks in jaX

<p align="center">
       <a href="https://www.python.org/doc/versions/">
        <img src="https://img.shields.io/badge/python-3.12--3.14-blue" /></a>
       <a href="https://pypi.org/project/gfnx/">
        <img src="https://img.shields.io/badge/pypi_package-0.0.1-brightgreen" /></a>
       <a href="https://arxiv.org/abs/2511.16592">
        <img src="https://img.shields.io/badge/arXiv-2511.16592-b31b1b" /></a>
       <a href="https://gfnx.readthedocs.io/en/latest/">
        <img src="https://img.shields.io/badge/docs-green" /></a>
</p>

`gfnx` is a JAX-native toolkit for building and studying Generative Flow Networks (GFlowNets). It brings together a collection of benchmark environments and reproducible baselines so you can iterate quickly on new ideas.

## Highlights

- End-to-end JAX implementations of GFlowNet building blocks (environments, reward modules, networks, and metrics).
- Ready-to-run baseline agents inspired by the [CleanRL](https://github.com/vwxyzjn/cleanrl) style of concise single-file experiments.
- Utilities for logging, checkpointing, and evaluation that make it easy to compare runs and extend the library with new research code.

## Documentation

The full documentation is hosted at **[gfnx.readthedocs.io](https://gfnx.readthedocs.io/en/latest/)**; its sources live in [`gfnx-docs/docs/`](gfnx-docs/docs/) and can be read directly on GitHub. First-time readers should go through:

1. [Background](gfnx-docs/docs/background.md) — GFlowNets, forward/backward policies and the training objectives.
2. [Walkthrough](gfnx-docs/docs/walkthrough.md) — a complete Detailed Balance baseline, line by line, including vmapping training over seeds.
3. [API reference](gfnx-docs/docs/api/index.md) — generated from the docstrings of [environments](gfnx-docs/docs/environments/index.md), reward modules, rollout utilities and [metrics](gfnx-docs/docs/metrics/index.md).

The [notebooks](notebooks/) open in Google Colab and need nothing but a browser; [baselines](gfnx-docs/docs/baselines.md) lists every ready-to-run training script. To build the docs site locally, see [Building the docs](gfnx-docs/docs/building_docs.md).

## Installation

### Requirements

- Python 3.12, 3.13 or 3.14.
- A working JAX installation. CPU works out of the box; for GPU/TPU accelerators follow the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

### Install the latest version

```bash
pip install git+https://github.com/d-tiapkin/gfnx.git
```

This installs the current `main` branch, which the documentation describes. The `gfnx` package on PyPI (`pip install gfnx`, version 0.0.1) predates the decoupled environment / reward API.

For NVIDIA GPUs, add the CUDA 12 extra:

```bash
pip install "gfnx[cuda12] @ git+https://github.com/d-tiapkin/gfnx.git"
```

Verify the install with:

```bash
python -c "import gfnx; print('gfnx import OK')"
```

### Develop locally and run baselines

```bash
git clone https://github.com/d-tiapkin/gfnx.git
cd gfnx
pip install -e ".[baselines]"
```

The quotes around `".[baselines]"` matter in `zsh` (the default macOS shell). The editable install keeps your local changes in sync with the Python package, while the optional `baselines` extra pulls in the dependencies required by the reference training scripts. In line with the CleanRL philosophy, the baselines are not meant to be imported: they serve as reference implementations. The TFBind-8, QM9, AMP and GFP rewards read data from `proxy/`, and the phylogenetic datasets are fetched with `bash download_ds.sh`, so run those from the repository root.

## Quickstart

Kick off a short training run of Detailed Balance in the Hypergrid environment:

```bash
python baselines/db_hypergrid.py num_train_steps=1_000 logging.tqdm_print_rate=100
```

The script is powered by Hydra, so you can override any configuration value on the command line (for example, picking another logging backend or playing with hyperparameters of the method). Baseline outputs, checkpoints, and Hydra logs default to `tmp/<date>/<time>/`; point the `logging.log_dir` or `logging.checkpoint_dir` fields to custom paths when running longer experiments.

## Support

- Open an issue on [GitHub](https://github.com/d-tiapkin/gfnx/issues) for bugs or feature requests.
- Start a discussion or reach out via pull requests if you would like to contribute improvements. Contributions with reproducible experiments and clear documentation get merged fastest.

## License

`gfnx` is released under the [MIT License](https://github.com/d-tiapkin/gfnx/blob/main/LICENCE). Feel free to use it in academic and commercial projects; please attribute the original authors when you publish results built on this codebase.

## Influences

`gfnx` stands on the shoulders of several excellent open-source projects:

- [torchgfn](https://torchgfn.readthedocs.io/en/latest/) – PyTorch-first GFlowNet library that shaped our environment design.
- [CleanRL](https://github.com/vwxyzjn/cleanrl) – taught us the value of single-file baselines and reproducible experiment configs.
- [purejaxrl](https://github.com/luchris429/purejaxrl/tree/main) and [JaxMARL](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl) – reference points for idiomatic, accelerator-ready JAX reinforcement learning code.

## Citation

```bibtex
@article{tiapkin2025gfnx,
  title={gfnx: Fast and Scalable Library for Generative Flow Networks in JAX},
  author={Tiapkin, Daniil and Agarkov, Artem and Morozov, Nikita and Maksimov, Ian and Tsyganov, Askar and Gritsaev, Timofei and Samsonov, Sergey},
  journal={arXiv preprint arXiv:2511.16592},
  year={2025}
}
```
