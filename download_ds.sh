#!/usr/bin/env bash
set -euo pipefail

# Pinned to a commit of tristandeleu/gfn-maxent-rl so downloads are reproducible.
commit="e8808a4328233277afc728548b0df86f49b0917d"
base_url="https://raw.githubusercontent.com/tristandeleu/gfn-maxent-rl/${commit}/gfn_maxent_rl/envs/phylo_gfn/datasets"
out_dir="datasets"

mkdir -p "$out_dir"

for f in DS{1..8}.json; do
  echo "Downloading $f"
  curl -fsSL "$base_url/$f" -o "$out_dir/$f"
done
