import csv
import json
import math
import pickle
import string
from itertools import chain, count, islice, permutations, product
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np


def uint8bits_to_int32(bits: chex.Array) -> chex.Array:
    """
    Convert an array of uint8 bits to an array of int32 bits.
    NOTE: The max number of byte-chunks is 4,
        e.g.: [255, 255, 255, 255] -> 4 bytes -> 32 bits
    """
    chex.assert_axis_dimension_lt(bits, 0, 5)  # 4 bytes is the maximum for int32
    powers = jnp.arange(bits.shape[0] - 1, -1, -1)
    return jnp.sum(bits * 256**powers)


def get_all_adjacencies_flat_bits(num_variables: int) -> chex.Array:
    """Compute sorted packed bit representations of all DAGs."""
    # Construct all possible adjacency matrices. Shape: (num_graphs, N, N)
    all_adjacencies = construct_all_dags(num_variables)
    # Reshape the adjacency matrices to a flat array
    all_adjacencies_flat = all_adjacencies.reshape(-1, num_variables**2)
    # Pack the bits of the adjacency matrices
    all_adjacencies_flat_bits = jnp.packbits(all_adjacencies_flat, axis=1)
    all_adjacencies_flat_bits = jax.vmap(uint8bits_to_int32)(all_adjacencies_flat_bits)
    return jnp.sort(all_adjacencies_flat_bits)


def adj_to_index(adj: chex.Array, all_adjacencies_flat_bits: chex.Array) -> chex.Array:
    """Convert adjacency matrix to state index given precomputed bit representations."""
    adj_flat = adj.reshape(-1)
    adj_bits = jnp.packbits(adj_flat)
    adj_bits = uint8bits_to_int32(adj_bits)
    return jnp.searchsorted(all_adjacencies_flat_bits, adj_bits)


# TODO: (agarkovv) Make jit compatible
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

    compressed_dags = sorted(compressed_dags)
    compressed_dags = np.array(compressed_dags)
    adjacencies = np.unpackbits(compressed_dags, axis=1, count=num_variables**2)
    return jnp.array(adjacencies.reshape(-1, num_variables, num_variables))


def get_transitive_closure(adjacency_matrix: chex.Array) -> chex.Array:
    """
    Compute the transitive closure of a DAG.
    It shows which nodes are reachable from any other node.
    Uses powers of the adjacency matrix:
    cl_A = A | A^2 | A^3 | ... | A^n, n = num_variables.
    """

    def power_iteration(
        i: int, carry: tuple[chex.Array, chex.Array]
    ) -> tuple[chex.Array, chex.Array]:
        curr_matrix, curr_closure = carry
        next_matrix = curr_matrix @ adjacency_matrix
        next_closure = curr_closure | next_matrix
        return next_matrix, next_closure

    num_variables = adjacency_matrix.shape[0]
    return jax.lax.fori_loop(
        0, num_variables, power_iteration, (adjacency_matrix, adjacency_matrix)
    )[1]


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
        spouses = jnp.sum(adjacency_matrix * children[None,], axis=1).astype(jnp.bool)
        return parents | children | spouses

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
        iterator = chain.from_iterable(product(uppercase, repeat=r) for r in count(1))
        nodes = ["".join(letters) for letters in islice(iterator, num_variables)]

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
    graph, loc_edges=0.0, scale_edges=1.0, obs_scale=None, rng=None
):
    if obs_scale is None:
        obs_scale = math.sqrt(0.1)
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
        raise ValueError("The graph is not a Linear Gaussian Bayesian Network.")

    nodes = list(nx.topological_sort(graph))
    node_index = {node: idx for idx, node in enumerate(nodes)}
    samples = np.zeros((num_samples, len(nodes)))

    for node in nodes:
        idx = node_index[node]
        attrs = graph.nodes[node]
        if attrs["parents"]:
            parent_indices = [node_index[parent] for parent in attrs["parents"]]
            values = samples[:, parent_indices]
            mean = attrs["bias"] + np.dot(values, attrs["cpd"])
            samples[:, idx] = rng.normal(mean, attrs["obs_scale"])
        else:
            samples[:, idx] = rng.normal(attrs["bias"], attrs["obs_scale"], size=(num_samples,))
    return nodes, samples


def generate_dataset(
    data_seed=0,
    num_variables=5,
    num_edges_per_node=1,
    num_train_samples=100,
    loc_edges=0.0,
    scale_edges=1.0,
    obs_scale: float | None = None,
    folder: Path | None = None,
):
    data_rng = np.random.default_rng(seed=data_seed)
    if obs_scale is None:
        obs_scale = math.sqrt(0.1)

    # Generate DAG
    graph = sample_erdos_renyi_graph(
        num_variables=num_variables,
        num_edges_per_node=num_edges_per_node,
        rng=data_rng,
    )
    graph = sample_linear_gaussian(
        graph,
        rng=data_rng,
        loc_edges=loc_edges,
        scale_edges=scale_edges,
        obs_scale=obs_scale,
    )

    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("The graph is not acyclic.")

    # Sample train data
    node_names, train_data = sample_from_linear_gaussian(
        graph, num_samples=num_train_samples, rng=data_rng
    )
    adjacency = nx.to_numpy_array(graph, weight=None)

    if folder is not None:
        # Create output folder
        folder = Path(folder) / "train_data"
        folder.mkdir(exist_ok=True)
        # Save graph object
        with open(folder / "graph.pkl", "wb") as f:
            pickle.dump(graph, f)
        # Save adjacency
        with open(folder / "adjacency.npy", "wb") as f:
            np.save(f, adjacency)
        # Save data
        with open(folder / "train_data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(node_names)
            writer.writerows(train_data)

        # Save metadata
        metadata = {
            "model_type": graph.graph.get("type", "unknown"),
            "num_variables": num_variables,
            "num_edges_per_node": num_edges_per_node,
            "num_train_samples": num_train_samples,
            "data_seed": data_seed,
            "loc_edges": loc_edges,
            "scale_edges": scale_edges,
            "obs_scale": obs_scale,
            "nodes": [],
        }

        for node in node_names:
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

    return train_data
