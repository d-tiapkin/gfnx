#!/usr/bin/env bash
set -euo pipefail

base_url="https://raw.githubusercontent.com/tristandeleu/gfn-maxent-rl/master/gfn_maxent_rl/envs/phylo_gfn/datasets"
out_dir="datasets"

mkdir -p "$out_dir"

for f in DS{1..8}.json; do
  echo "Downloading $f"
  curl -fsSL "$base_url/$f" -o "$out_dir/$f"
done
