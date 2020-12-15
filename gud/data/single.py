"""Single graphs."""
import functools
import os
from typing import Any, Mapping, NamedTuple, Optional

import numpy as np
import scipy.sparse as sp
import wget

from gud.data.path import data_dir
from gud.data.registry import Registry

single_graphs = Registry("single_graphs")


class GraphData(NamedTuple):
    adjacency: sp.csr_matrix
    node_attrs: Optional[sp.csr_matrix]
    node_labels: np.ndarray  # np.int32
    metadata: Optional[Mapping[str, Any]]


def _download(url):
    filename = os.path.split(url)[-1]
    single_graphs_dir = data_dir("single_graphs")
    os.makedirs(single_graphs_dir, exist_ok=True)
    path = os.path.join(single_graphs_dir, filename)

    if not os.path.exists(path):
        print(f"Downloading {name} data to {path}")
        dl_path = wget.download(url, out=path)
        assert dl_path == path
    return path


def _download_el_graph_data(name):
    """Download data from `EdisonLeeeee/GraphData`."""
    url = f"https://github.com/EdisonLeeeee/GraphData/raw/master/datasets/{name}.npz"
    path = _download(url)
    data = np.load(path, allow_pickle=True)
    adj_matrix = data["adj_matrix"].item()
    node_attrs = data["node_attr"].item() if "node_attr" in data else None
    node_labels = data["node_label"]
    metadata = data["metadata"].item() if "metadata" in data else None
    return GraphData(adj_matrix, node_attrs, node_labels, metadata)


def _download_klic_graph_data(name):
    """Download data from `klicperajo/ppnp`."""
    url = f"https://github.com/klicperajo/ppnp/raw/master/ppnp/data/{name}.npz"
    path = _download(url)
    data = np.load(path)

    def load_csr(mat_name):
        return sp.csr_matrix(
            (
                data[f"{mat_name}_data"],
                data[f"{mat_name}_indices"],
                data[f"{mat_name}_indptr"],
            ),
            shape=data[f"{mat_name}_shape"],
        )

    adj_matrix = load_csr("adj")
    node_attrs = load_csr("attr")
    node_labels = data["labels"].astype(np.int32)
    metadata = None
    return GraphData(adj_matrix, node_attrs, node_labels, metadata)


for name in (
    "acm",
    "amazon_cs",
    "amazon_photo",
    "blogcatalog",
    "citeseer",
    "citeseer_full",
    "coauthor_cs",
    "coauthor_phy",
    "cora",
    "cora_full",
    "cora_ml",
    "dblp",
    "flickr",
    "karate_club",
    "polblogs",
    "pubmed",
    "uai",
):

    single_graphs.register(name)(functools.partial(_download_el_graph_data, name=name))


for name in ("ms_academic",):
    single_graphs.register(name)(
        functools.partial(_download_klic_graph_data, name=name)
    )

if __name__ == "__main__":
    for name in sorted(single_graphs):
        data = single_graphs[name]
        assert isinstance(data.adjacency, sp.csr_matrix)
        assert data.node_attrs is None or isinstance(data.node_attrs, sp.csr_matrix)
        assert isinstance(data.node_labels, np.ndarray)
        assert data.node_labels.dtype == np.int32
        assert data.metadata is None or isinstance(data.metadata, dict)
