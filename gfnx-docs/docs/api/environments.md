# Environments

Environment classes are documented on the page of their family, next to the
corresponding reward modules.

| Environment | Module | Family page | Enumerable |
| --- | --- | --- | --- |
| [`HypergridEnvironment`][gfnx.environment.hypergrid.HypergridEnvironment] | `gfnx.environment.hypergrid` | [Hypergrid](../environments/hypergrid/environment_api.md) | Yes |
| [`BitseqEnvironment`][gfnx.environment.bitseq.BitseqEnvironment] | `gfnx.environment.bitseq` | [Sequences](../environments/sequences/environment_api.md) | No |
| [`TFBind8Environment`][gfnx.environment.tfbind.TFBind8Environment] | `gfnx.environment.tfbind` | [Sequences](../environments/sequences/environment_api.md) | Yes |
| [`QM9SmallEnvironment`][gfnx.environment.qm9_small.QM9SmallEnvironment] | `gfnx.environment.qm9_small` | [Sequences](../environments/sequences/environment_api.md) | Yes |
| [`AMPEnvironment`][gfnx.environment.amp.AMPEnvironment] | `gfnx.environment.amp` | [Sequences](../environments/sequences/environment_api.md) | No |
| [`GFPEnvironment`][gfnx.environment.gfp.GFPEnvironment] | `gfnx.environment.gfp` | [Sequences](../environments/sequences/environment_api.md) | No |
| [`DAGEnvironment`][gfnx.environment.dag.DAGEnvironment] | `gfnx.environment.dag` | [Bayesian networks](../environments/bayesian_networks/environment_api.md) | For `num_variables < 6` |
| [`IsingEnvironment`][gfnx.environment.ising.IsingEnvironment] | `gfnx.environment.ising` | [Ising model](../environments/ising_model/environment_api.md) | No |
| [`PhyloTreeEnvironment`][gfnx.environment.phylogenetic_tree.PhyloTreeEnvironment] | `gfnx.environment.phylogenetic_tree` | [Phylogenetic trees](../environments/phylogenetic_trees/environment_api.md) | No |

The generic sequence environments that the sequence tasks build on
(autoregressive, prepend/append, non-autoregressive) are documented in
[`gfnx.environment.sequence`][gfnx.environment.sequence].
