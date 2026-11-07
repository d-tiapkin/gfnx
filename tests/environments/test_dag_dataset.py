import csv

import numpy as np
import pytest

from gfnx.utils.dag import generate_dataset


@pytest.mark.parametrize("data_seed", range(5))
def test_generate_dataset_columns_follow_adjacency(tmp_path, data_seed):
    """Column `j` of the returned data must be node `j` of the saved `adjacency.npy`.

    Under the linear-Gaussian model, regressing each column on the parents that
    `adjacency.npy` gives it leaves only the observation noise. If the columns were in
    another order (e.g. topological), some child would be scored against the wrong
    parents and its residual would be much larger than `obs_scale`.
    """
    obs_scale = 0.1
    data = generate_dataset(
        data_seed=data_seed,
        num_variables=5,
        num_edges_per_node=1,
        num_train_samples=5000,
        loc_edges=2.0,  # keep edge weights away from 0 so every edge matters
        scale_edges=0.5,
        obs_scale=obs_scale,
        folder=tmp_path,
    )
    adjacency = np.load(tmp_path / "train_data" / "adjacency.npy")
    assert adjacency.sum() > 0

    for j in range(adjacency.shape[0]):
        parents = np.flatnonzero(adjacency[:, j])
        design = data[:, parents]
        coef = np.linalg.lstsq(design, data[:, j])[0] if parents.size else np.zeros(0)
        residual = data[:, j] - design @ coef
        np.testing.assert_allclose(residual.std(), obs_scale, rtol=0.1)

    # The CSV header names the columns; it must be in the same (graph node) order.
    with open(tmp_path / "train_data" / "train_data.csv", newline="") as f:
        header = next(csv.reader(f))
    assert header == sorted(header)  # nodes are named A, B, C, ... in index order
