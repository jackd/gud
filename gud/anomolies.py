from contextlib import contextmanager
from typing import Tuple

import numpy as np
import scipy.sparse as sp


@contextmanager
def random_state_context(seed: int):
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


def with_cliques(
    adjacency: sp.csr_matrix, clique_size: int, num_cliques: int = 1
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Get adjacency matrix of dataset with cliques.

    A clique is defined as a set of nodes where each node is a neighbor of every other
    node.

    Args:
        adjacency: adjacency matrix to start with.
        clique_size: size of each clique.
        num_cliques: number of cliques to add.

    Returns:
        augmented_adjacency: adjacency with cliques added.
        cliques: [num_cliques, clique_size] int32 array of indices of clique nodes.
    """
    num_nodes = adjacency.shape[0]
    adjacency = adjacency.tolil()
    dtype = adjacency.dtype
    rows = adjacency.rows
    data = adjacency.data
    cliques = np.empty((num_cliques, clique_size), dtype=np.int32)
    for i in range(num_cliques):
        clique = np.random.choice(num_nodes, clique_size, replace=False)
        clique.sort()
        cliques[i] = clique
        for c in clique:
            row = set(rows[c])
            contains_c = c in row
            row.update(clique)
            if not contains_c:
                row.remove(c)
            rows[c] = sorted(row)
            data[c] = np.ones((len(row),), dtype=dtype)
    return adjacency.tocsr(), cliques


def with_attribute_anomolies(
    node_attrs: sp.csr_matrix, num_candidates: int, num_anomolies: int = 1
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Get attribute matrix with some rows replaced with others.

    For each anomoly, we replace the attributes with those of the node with attributes
    furthest away from the original w.r.t. Euclidean norm from `num_candidates`
    candidates of the original.

    Args:
        node_attrs: [num_nodes, num_attrs] sparse attributes.
        num_candidates: number of candidates per anomoly.
        num_anomolies: number of anomolies to overwrite.

    Returns:
        augmented_node_attrs: node attributes with anomolous node attributes replaced.
        mapping: [num_anomolies, 2] int32 array, where
        `augmented_node_attrs[mapping[i, 1]] == node_attrs[mapping[i, 0]]`
    """
    num_nodes = node_attrs.shape[0]
    node_attrs_lil = node_attrs.tolil()
    anomolies = np.random.choice(num_nodes, num_anomolies, replace=False)
    anomolies.sort()
    mapping = np.empty((num_anomolies, 2), dtype=np.int32)
    for i, a in enumerate(anomolies):
        candidates = np.random.choice(num_nodes, num_candidates, replace=False)
        norms = np.linalg.norm(
            node_attrs[a].todense() - node_attrs[candidates].todense(), axis=-1
        )
        max_norm = np.argmax(norms)
        replacement = candidates[max_norm]
        node_attrs_lil[a] = node_attrs[replacement]
        mapping[i] = a, replacement
    return node_attrs_lil.tocsr(), mapping
