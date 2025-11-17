# Baselines

The `baselines/` directory collects reproducible training scripts for canonical GFlowNet objectives and a few closely related algorithms. Each script follows the CleanRL-style single-file pattern, relies on Hydra for configuration management, and logs to the shared `tmp/` directory unless overridden.

## Implementation matrix

The table below summarizes which method-environment combinations currently ship with the repository. Cells marked with &#x2705; point to a ready-to-run script in `baselines/`; &#x274C; indicates that the pairing has not been implemented yet.

| Method / Environment | Hypergrid | BitSeq | TFBind-8 | QM9 Small | AMP | GFP |
| --- | --- | --- | --- | --- | --- | --- |
| Detailed Balance (DB)&nbsp;[1] | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; |
| Trajectory Balance (TB)&nbsp;[2] | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x2705; | &#x274C; |
| Sub-Trajectory Balance (SubTB)&nbsp;[3] | &#x2705; | &#x2705; | &#x2705; | &#x274C; | &#x274C; | &#x274C; |

## Special-purpose scripts

Some environments require bespoke objectives or training tweaks. These live alongside the standard baselines:

- `baselines/soft_dqn_hypergrid.py` – Online SoftDQN baseline for Hypergrid&nbsp;[4].
- `baselines/mdb_dag.py` – Modified Detailed Balance for Bayesian network structure learning&nbsp;[5].
- `baselines/fldb_phylo.py` – Forward-Looking Detailed Balance for phylogenetic tree generation&nbsp;[6].
- `baselines/tb_ising.py` – Energy-based modeling of the Ising system using a TB objective&nbsp;[7].

## How to run a baseline

1. Install the package with baseline extras: `pip install -e '.[baselines]'`.
2. Launch the desired script, optionally overriding Hydra configs. For example:

```bash
python baselines/db_hypergrid.py num_train_steps=1_000 logging.tqdm_print_rate=100
```

Each script documents the relevant configuration group in its module docstring. Point `logging.log_dir` or `logging.checkpoint_dir` to persistent storage when running long jobs.


## References

1. Bengio, Y. *et&nbsp;al.* (2023). *GFlowNet Foundations.*  
   Journal of Machine Learning Research (JMLR).
2. Malkin, N. *et&nbsp;al.* (2022). *Trajectory Balance: Improved Credit Assignment in GFlowNets.*  
   Advances in Neural Information Processing Systems (NeurIPS).
3. Madan, K. *et&nbsp;al.* (2023). *Learning GFlowNets from Partial Episodes for Improved Convergence and Stability.*  
   International Conference on Machine Learning (ICML).
4. Tiapkin, D. *et&nbsp;al.* (2024). *Generative Flow Networks as Entropy-Regularized RL.*  
   International Conference on Artificial Intelligence and Statistics (AISTATS).
5. Deleu, T. *et&nbsp;al.* (2022). *Bayesian Structure Learning with Generative Flow Networks.*  
   Conference on Uncertainty in Artificial Intelligence (UAI).
6. Pan, L. *et&nbsp;al.* (2023). *Better Training of GFlowNets with Local Credit and Incomplete Trajectories.*  
   International Conference on Machine Learning (ICML).
7. Zhang, D. *et&nbsp;al.* (2022). *Generative Flow Networks for Discrete Probabilistic Modeling.*  
   International Conference on Machine Learning (ICML).
