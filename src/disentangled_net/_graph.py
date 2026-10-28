"""Graph input: a networkx graph or a scipy.sparse adjacency, normalised to indexed arrays."""
from collections.abc import Mapping

import networkx as nx
import numpy as np
import scipy.sparse as sp


class IndexedGraph:
    """Undirected weighted graph on nodes 0..n-1. Self-loops are dropped."""

    def __init__(self, nodes, adjacency):
        self.nodes = tuple(nodes)
        self.n = len(self.nodes)
        self.adjacency = adjacency                    # csr_array, symmetric, zero diagonal
        self.total_weight = float(adjacency.sum()) / 2
        self.degrees = np.asarray(adjacency.sum(axis=1)).ravel()
        self.source = None                            # the networkx graph, when given

    def edges(self):
        """Upper-triangle edges as arrays (i, j, weight) with i < j."""
        upper = sp.triu(self.adjacency, k=1, format="coo")
        return upper.row.astype(np.intp), upper.col.astype(np.intp), upper.data.astype(float)


def as_indexed(G, weight="weight"):
    """Normalise a networkx graph or a square symmetric sparse matrix."""
    if isinstance(G, IndexedGraph):
        return G
    if sp.issparse(G):
        A = sp.csr_array(G, dtype=float)
        if A.shape[0] != A.shape[1]:
            raise ValueError("adjacency matrix must be square")
        if A.nnz and abs(A - A.T).max() > 1e-12 * max(1., abs(A).max()):
            raise ValueError("adjacency matrix must be symmetric")
        nodes, source = range(A.shape[0]), None
    elif isinstance(G, nx.Graph):
        if G.is_directed() or G.is_multigraph():
            raise TypeError("use a simple undirected graph")
        nodes, source = tuple(G), G
        A = (sp.csr_array(nx.to_scipy_sparse_array(G, nodelist=nodes, weight=weight, dtype=float))
             if nodes else sp.csr_array((0, 0), dtype=float))
    else:
        raise TypeError("G must be a networkx graph or a scipy.sparse adjacency matrix")
    A = A.tolil()
    A.setdiag(0.)
    A = sp.csr_array(A)
    A.eliminate_zeros()
    if A.nnz and (not np.isfinite(A.data).all() or (A.data < 0).any()):
        raise ValueError("edge weights must be finite and nonnegative")
    graph = IndexedGraph(nodes, A)
    graph.source = source
    return graph


def node_values(graph, values, name, dtype=float):
    """Per-node values aligned with graph.nodes: attribute name, mapping or array."""
    if isinstance(values, str):
        if graph.source is None:
            raise ValueError(f"{name}: give an array aligned with the matrix rows for sparse input")
        try:
            values = [graph.source.nodes[u][values] for u in graph.nodes]
        except KeyError as error:
            raise ValueError(f"{name}: node attribute {values!r} missing on some nodes") from error
    elif isinstance(values, Mapping):
        try:
            values = [values[u] for u in graph.nodes]
        except KeyError as error:
            raise ValueError(f"{name}: value missing for node {error.args[0]!r}") from error
    array = np.asarray(values) if dtype is None else np.asarray(values, dtype=dtype)
    if len(array) != graph.n:
        raise ValueError(f"{name}: expected {graph.n} values, got {len(array)}")
    return array
