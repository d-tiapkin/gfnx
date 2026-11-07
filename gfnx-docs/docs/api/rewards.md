# Reward modules

Reward modules are documented on the page of their environment family.

| Reward module | Environment | Reward | Family page |
| --- | --- | --- | --- |
| [`GeneralHypergridRewardModule`][gfnx.reward.hypergrid.GeneralHypergridRewardModule], `EasyHypergridRewardModule`, `HardHypergridRewardModule` | Hypergrid | Closed-form, $2^D$ modes | [Hypergrid](../environments/hypergrid/reward_api.md) |
| [`BitseqRewardModule`][gfnx.reward.bitseq.BitseqRewardModule] | Bitseq | Hamming distance to a hidden mode set | [Sequences](../environments/sequences/reward_api.md) |
| [`TFBind8RewardModule`][gfnx.reward.tfbind.TFBind8RewardModule] | TFBind-8 | Wet-lab lookup table | [Sequences](../environments/sequences/reward_api.md) |
| [`QM9SmallRewardModule`][gfnx.reward.qm9_small.QM9SmallRewardModule] | QM9 (small) | Proxy lookup table | [Sequences](../environments/sequences/reward_api.md) |
| [`EqxProxyAMPRewardModule`][gfnx.reward.amp.EqxProxyAMPRewardModule] | AMP | Pretrained transformer classifier | [Sequences](../environments/sequences/reward_api.md) |
| [`EqxProxyGFPRewardModule`][gfnx.reward.gfp.EqxProxyGFPRewardModule] | GFP | Transformer regressor (experimental) | [Sequences](../environments/sequences/reward_api.md) |
| [`DAGRewardModule`][gfnx.reward.dag.DAGRewardModule] + priors / likelihoods | DAG | $\log P(\mathcal{D} \mid G) + \log P(G)$, with `delta_score` | [Bayesian networks](../environments/bayesian_networks/reward_api.md) |
| [`IsingRewardModule`][gfnx.reward.ising.IsingRewardModule] | Ising | $\mathbf{x}^\top J \mathbf{x}$, trainable $J$ | [Ising model](../environments/ising_model/reward_api.md) |
| [`PhyloTreeRewardModule`][gfnx.reward.phylogenetic_tree.PhyloTreeRewardModule] | Phylogenetic trees | Parsimony, with `delta_score` | [Phylogenetic trees](../environments/phylogenetic_trees/reward_api.md) |
