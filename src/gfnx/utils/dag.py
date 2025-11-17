import csv
import json
import math
import pickle
import string
from itertools import chain, count, islice, permutations, product
from pathlib import Path
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np


def load_dag_samples(samples_path: Path) -> Dict[str, Any]:
    """
    Loads samples and extracts basic information needed for DAG environment setup.

    Args:
        samples_path: Path to the file containing the samples.

    Returns:
        Dictionary with initialization arguments:
        - samples: The loaded samples array
        - num_variables: Number of variables in the samples
    """
    with open(samples_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    data = jnp.array(rows[1:], dtype=jnp.float32)

    return {"samples": data, "num_variables": data.shape[1]}


def construct_all_dags(num_variables):
    """
    Return all possible adjacency matrices for DAGs with a given number of variables
    Shape: (num_graphs, num_variables, num_variables)
    """
    # Generate all the DAGs over num_variables nodes
    shape = (num_variables, num_variables)
    repeat = num_variables * (num_variables - 1) // 2

    # Generate all the possible binary codes
    codes = list(product([0, 1], repeat=repeat))
    codes = np.asarray(codes)

    # Get upper-triangular indices
    x, y = np.triu_indices(num_variables, k=1)

    # Fill the upper-triangular matrices
    trius = np.zeros((len(codes),) + shape, dtype=np.int_)
    trius[:, x, y] = codes

    # Apply permutation, and remove duplicates
    compressed_dags = set()
    for perm in permutations(range(num_variables)):
        permuted = trius[:, :, perm][:, perm, :]
        permuted = permuted.reshape(-1, num_variables**2)
        permuted = np.packbits(permuted, axis=1)
        compressed_dags.update(map(tuple, permuted))

    compressed_dags = sorted(list(compressed_dags))
    compressed_dags = np.array(compressed_dags)
    adjacencies = np.unpackbits(
        compressed_dags, axis=1, count=num_variables**2
    )
    return jnp.array(adjacencies.reshape(-1, num_variables, num_variables))


def get_transitive_closure(adjacency_matrix: chex.Array) -> chex.Array:
    """
    Compute the transitive closure of a DAG.
    It shows which nodes are reachable from any other node.
    Uses powers of the adjacency matrix:
    cl_A = A | A^2 | A^3 | ... | A^n, n = num_variables.
    """

    def power_iteration(
        i: int, carry: Tuple[chex.Array, chex.Array]
    ) -> Tuple[chex.Array, chex.Array]:
        curr_matrix, curr_closure = carry
        next_matrix = curr_matrix @ adjacency_matrix
        next_closure = curr_closure | next_matrix
        return next_matrix, next_closure

    num_variables = adjacency_matrix.shape[0]
    transitive_closure = jax.lax.fori_loop(
        0, num_variables, power_iteration, (adjacency_matrix, adjacency_matrix)
    )[1]
    return transitive_closure


def get_markov_blanket(adjacency_matrix: chex.Array) -> chex.Array:
    """
    Compute the Markov blanket of a DAG.
    blanket = parents + children + spouses.
    blanket[i, j] = 1 if i is in blanket of j.
    """

    def get_single_blanket(adjacency_matrix: chex.Array, i: int) -> chex.Array:
        """
        Compute the blanket of a DAG.
        """
        parents = adjacency_matrix[:, i]
        children = adjacency_matrix[i, :]
        spouses = jnp.sum(adjacency_matrix * children[None,], axis=1).astype(
            jnp.bool
        )
        blanket = parents | children | spouses
        return blanket

    num_variables = adjacency_matrix.shape[0]
    blanket = jax.vmap(get_single_blanket, in_axes=(None, 0))(
        adjacency_matrix, jnp.arange(num_variables)
    )
    # Remove self-loops by masking out the diagonal
    return blanket * jnp.logical_not(jnp.eye(num_variables, dtype=jnp.bool))


def sample_erdos_renyi_graph(
    num_variables,
    p=None,
    num_edges_per_node=None,
    nodes=None,
    create_using=nx.DiGraph,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    if p is None:
        if num_edges_per_node is None:
            raise ValueError("One of p or num_edges must be specified.")
        p = 2.0 * num_edges_per_node / (num_variables - 1)

    if nodes is None:
        uppercase = string.ascii_uppercase
        iterator = chain.from_iterable(
            product(uppercase, repeat=r) for r in count(1)
        )
        nodes = [
            "".join(letters) for letters in islice(iterator, num_variables)
        ]

    adjacency = rng.binomial(1, p=p, size=(num_variables, num_variables))
    adjacency = np.tril(adjacency, k=-1)  # Only keep the lower triangular part

    # Permute the rows and columns
    perm = rng.permutation(num_variables)
    adjacency = adjacency[perm, :]
    adjacency = adjacency[:, perm]

    graph = nx.from_numpy_array(adjacency, create_using=create_using)
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)

    return graph


def sample_linear_gaussian(
    graph, loc_edges=0.0, scale_edges=1.0, obs_scale=math.sqrt(0.1), rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    graph.graph["type"] = "linear-gaussian"
    attrs = {}
    for node in graph.nodes:
        parents = list(graph.predecessors(node))
        theta = rng.normal(loc_edges, scale_edges, size=(len(parents),))
        attrs[node] = {
            "parents": parents,
            "cpd": theta,
            "bias": 0.0,
            "obs_scale": obs_scale,
        }
    nx.set_node_attributes(graph, attrs)
    return graph


def sample_from_linear_gaussian(graph, num_samples, rng=None):
    """Sample from a linear-Gaussian model using ancestral sampling."""
    if rng is None:
        rng = np.random.default_rng()

    if graph.graph.get("type", "") != "linear-gaussian":
        raise ValueError(
            "The graph is not a Linear Gaussian Bayesian Network."
        )

    nodes = list(nx.topological_sort(graph))
    node_index = {node: idx for idx, node in enumerate(nodes)}
    samples = np.zeros((num_samples, len(nodes)))

    for node in nodes:
        idx = node_index[node]
        attrs = graph.nodes[node]
        if attrs["parents"]:
            parent_indices = [
                node_index[parent] for parent in attrs["parents"]
            ]
            values = samples[:, parent_indices]
            mean = attrs["bias"] + np.dot(values, attrs["cpd"])
            samples[:, idx] = rng.normal(mean, attrs["obs_scale"])
        else:
            samples[:, idx] = rng.normal(
                attrs["bias"], attrs["obs_scale"], size=(num_samples,)
            )
    return nodes, samples


def write_csv(file_path, header, data):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


# TODO: (agarkovv) Review the need of test data
def generate_dataset(
    folder,
    train_seed=0,
    test_seed=1,
    num_variables=5,
    num_edges_per_node=1,
    num_train_samples=100,
    num_test_samples=100,
    loc_edges=0.0,
    scale_edges=1.0,
    obs_scale=math.sqrt(0.1),
):
    folder = Path(folder)
    train_rng = np.random.default_rng(seed=train_seed)
    test_rng = np.random.default_rng(seed=test_seed)

    # Generate DAG
    graph = sample_erdos_renyi_graph(
        num_variables=num_variables,
        num_edges_per_node=num_edges_per_node,
        rng=train_rng,
    )
    graph = sample_linear_gaussian(
        graph,
        rng=train_rng,
        loc_edges=loc_edges,
        scale_edges=scale_edges,
        obs_scale=obs_scale,
    )

    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("The graph is not acyclic.")

    # Sample train/test data
    train_header, train_data = sample_from_linear_gaussian(
        graph, num_samples=num_train_samples, rng=train_rng
    )
    test_header, test_data = sample_from_linear_gaussian(
        graph, num_samples=num_test_samples, rng=test_rng
    )

    # Create output folder
    folder.mkdir(exist_ok=True)

    # Save graph object
    with open(folder / "graph.pkl", "wb") as f:
        pickle.dump(graph, f)

    # Save adjacency
    adjacency = nx.to_numpy_array(graph, weight=None)
    with open(folder / "adjacency.npy", "wb") as f:
        np.save(f, adjacency)

    # Save data
    write_csv(folder / "train_data.csv", train_header, train_data)
    write_csv(folder / "test_data.csv", test_header, test_data)

    # Save metadata
    metadata = {
        "model_type": graph.graph.get("type", "unknown"),
        "num_variables": num_variables,
        "num_edges_per_node": num_edges_per_node,
        "num_train_samples": num_train_samples,
        "num_test_samples": num_test_samples,
        "train_seed": train_seed,
        "test_seed": test_seed,
        "loc_edges": loc_edges,
        "scale_edges": scale_edges,
        "obs_scale": obs_scale,
        "nodes": [],
    }

    for node in train_header:
        attrs = graph.nodes[node]
        metadata["nodes"].append({
            "name": node,
            "parents": attrs["parents"],
            "cpd": attrs["cpd"].tolist(),
            "bias": attrs["bias"],
            "obs_scale": attrs["obs_scale"],
        })

    with open(folder / "graph_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
