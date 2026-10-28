"""Communities relative to a null model."""
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ._geometry import LandmarkGeometry
from ._graph import IndexedGraph, as_indexed
from ._louvain.backends import (DENSE_LIMIT, BlockBackend, MatrixBackend, SamplingBackend, SparseBackend, SparseBlockBackend,
                                SumBackend, TiledBackend)
from ._louvain.engine import optimize
from .approximations import Landmarks, Sampling, as_approximation
from .fitting import _attribute_null, fit_on_geometry
from .nulls import BlockNull, CustomNull, NullModel, SpatialNull, SumNull


BLOCK_DENSE_LIMIT = 64 * 2**20  # one n x b block cache; small caches are faster dense


@dataclass
class CommunityInfo:
    """Returned by ``disentangled_communities(..., return_info=True)``."""
    quality: float              # Q under the null actually optimised (an estimate with sampling)
    n_communities: int
    approximation: str          # 'exact', 'landmarks', 'sampling'
    backend: str                # how the null was evaluated
    levels: list                # per cycle and level: nodes, communities, passes, moves, gain
    seconds: float
    details: dict = field(default_factory=dict)


def disentangled_communities(G, null=None, *, attribute=None, attribute_type="auto", metric="euclidean",
                             gamma=1.0, epsilon=None, approximation=None, standardize=False,
                             weight="weight", init=None, random_state=None, n_iterations=2,
                             max_passes=None, max_levels=None, tolerance=1e-7, return_info=False):
    """Communities maximising Q = (1/m) sum_{u<v, same community} (A_uv - gamma P_uv - epsilon).

    Parameters
    ----------
    G : networkx graph or scipy.sparse adjacency matrix.
    null : a null model (see :class:`~disentangled_net.NullModel`).
        Alternatively give an attribute name here or with ``attribute=``.
    attribute : optional node attribute name, mapping or array.
        Fit the null and select exact/compressed/landmark evaluation automatically;
        do not pass ``approximation`` or ``standardize`` in this simple interface.
    attribute_type, metric : as in :func:`~disentangled_net.fit_null`.
    gamma : multiplicative resolution: two groups merge only if observed > gamma x expected
        between them. Larger values give more, smaller communities.
    epsilon : additive resolution, in expected weight per pair of nodes: two groups merge only
        if their excess density (O - gamma E) / (n_a n_b) exceeds epsilon. Being a density
        per pair of nodes, its scale depends on the size of the units. Default: graph density
        with an attribute or fit_null result; zero with other explicit nulls (existing behaviour).
    approximation : None (exact), 'landmarks' or :class:`Landmarks` (spatial nulls),
        'sampling' or :class:`Sampling` (any null).
    standardize : optimise the standardized residual (A - P) / sqrt(P (1 - P)); needs a
        probability null, exact evaluation.
    init : initial partition {node: community}.
    n_iterations : full multilevel cycles; each restarts from the original nodes.
    max_passes, max_levels, tolerance : bounds on the search.
    return_info : also return a :class:`CommunityInfo`.

    Returns
    -------
    {node: community}, communities numbered by decreasing size; with return_info=True,
    (partition, CommunityInfo).
    """
    start = time.perf_counter()
    graph = as_indexed(G, weight)
    automatic = attribute is not None or isinstance(null, str)
    if automatic and (approximation is not None or standardize):
        raise ValueError("attribute mode selects evaluation automatically; use fit_null first for advanced options")
    null = _attribute_null(graph, null, attribute, attribute_type, metric)
    if epsilon is None:
        epsilon = null.diagnostics.get("default_epsilon", 0.)
    if not np.isfinite([gamma, epsilon]).all() or gamma < 0 or epsilon < 0:
        raise ValueError("gamma and epsilon must be finite and nonnegative")
    graph, factory, details = _prepare(graph, null, approximation, standardize, weight, random_state)
    if automatic or null.diagnostics.get("automatic"):
        details["null_diagnostics"] = dict(null.diagnostics)
        details["epsilon"] = float(epsilon)
    rng = np.random.default_rng(random_state)
    spec = as_approximation(approximation)
    sampling = isinstance(spec, Sampling)
    labels, quality, levels = optimize(
        graph, factory, gamma=gamma, epsilon=epsilon,
        init=None if init is None else np.array([init[u] for u in graph.nodes]), rng=rng,
        n_iterations=n_iterations, tolerance=tolerance,
        max_passes=spec.max_passes if sampling and max_passes is None else max_passes,
        max_levels=spec.max_levels if sampling and max_levels is None else max_levels,
        max_candidates=spec.max_candidates if sampling else None)
    partition = _by_size(graph.nodes, labels)
    if not return_info:
        return partition
    info = CommunityInfo(quality=float(quality), n_communities=int(labels.max()) + 1 if len(labels) else 0,
                         approximation=null.diagnostics.get("approximation", "exact") if spec is None else type(spec).__name__.lower(),
                         backend=details.pop("backend"), levels=levels,
                         seconds=time.perf_counter() - start, details=details)
    return partition, info


def _by_size(nodes, labels):
    sizes = np.bincount(labels)
    rank = np.empty(len(sizes), np.intp)
    rank[np.argsort(-sizes, kind="stable")] = np.arange(len(sizes))
    return dict(zip(nodes, rank[labels].tolist()))


def _aligned(null, graph):
    if not isinstance(null, NullModel):
        raise TypeError("null must be a null model; wrap matrices and functions with custom_null")
    if null.nodes == graph.nodes:
        return null
    if set(null.nodes) != set(graph.nodes):
        raise ValueError("the null model and the graph have different nodes "
                         "(with a networkx graph, build the null with nodes=list(G))")
    return null.reindexed(graph.nodes)


def _prepare(G, null, approximation, standardize, weight, random_state):
    """Indexed graph, a factory of level-0 backends, and details for CommunityInfo."""
    graph = as_indexed(G, weight)
    null = _aligned(null, graph)
    spec = as_approximation(approximation)
    if standardize:
        if spec is not None and not isinstance(spec, Sampling):
            raise ValueError("standardize needs exact evaluation or sampling")
        graph, null = _standardized(graph, null)
    if isinstance(spec, Sampling):
        rng = np.random.default_rng(None if random_state is None else random_state + 1)
        return graph, lambda: SamplingBackend(null, rng, spec.sample_size, spec.background_weight), \
            dict(backend="sampling")
    if isinstance(spec, Landmarks):
        backend, details = _landmark_backend(null, graph, spec)
        return graph, lambda: _fresh(backend), dict(details, backend=backend.name)
    backend = _exact_backend(null)
    return graph, lambda: _fresh(backend), dict(backend=backend.name)


def _fresh(backend):
    """Backends are rebuilt for every cycle; the level-0 one is cheap to copy."""
    if isinstance(backend, SumBackend):
        return SumBackend([_fresh(p) for p in backend.parts])
    if isinstance(backend, SparseBlockBackend):
        return SparseBlockBackend(backend.B, cells=backend.cells, theta=backend.theta)
    if isinstance(backend, BlockBackend):
        return BlockBackend(backend.B, cells=backend.cells, theta=backend.theta)
    if isinstance(backend, MatrixBackend):
        return MatrixBackend(backend.P)
    if isinstance(backend, SparseBackend):
        return SparseBackend(backend.C)
    return TiledBackend(backend.null)


def _exact_backend(null):
    n = null.n
    dense_fits = 2 * n * n * 8 <= DENSE_LIMIT
    if isinstance(null, BlockNull):
        B = null.block_matrix
        if sp.issparse(B):
            if n * B.shape[0] * 8 > BLOCK_DENSE_LIMIT and B.nnz * 5 < B.shape[0] ** 2:
                return SparseBlockBackend(B, cells=null.labels, theta=null.masses)
            B = B.toarray()
        return BlockBackend(B, cells=null.labels, theta=null.masses)
    if isinstance(null, CustomNull) and null.matrix is not None:
        return MatrixBackend(null.matrix)
    if dense_fits:
        return MatrixBackend(null.to_matrix())
    if isinstance(null, SumNull):
        return SumBackend([_exact_backend(p) for p in null.parts])
    return TiledBackend(null)


def _landmark_backend(null, graph, spec):
    if isinstance(null, SumNull):
        parts, details = [], {}
        for p in null.parts:
            if isinstance(p, SpatialNull):
                backend, details = _landmark_backend(p, graph, spec)
            else:
                backend = _exact_backend(p)
            parts.append(backend)
        return SumBackend(parts), details
    if not isinstance(null, SpatialNull) or null.link != "log":
        raise ValueError("landmarks apply to spatial nulls with the Poisson (log) link")
    if null.groups is not None:
        raise ValueError("spatial nulls with within-group factors require exact evaluation")
    geometry = LandmarkGeometry(null.positions, null.metric, spec.rank, spec.placement, spec.exact_within,
                                weights=null.masses, exact_neighbors=spec.exact_neighbors)
    details = dict(rank=geometry.rank, near_pairs_per_node=2 * len(geometry.near[0]) / max(graph.n, 1))
    if spec.refit and null.fit_settings is not None:
        masses, deterrence, fit = fit_on_geometry(graph, geometry, null.fit_settings)
        details["refit"] = fit
    else:
        masses, deterrence = null.masses, null.deterrence
        details["refit"] = None
    B = deterrence(geometry.rep_distance)
    backend = BlockBackend(B, cells=geometry.cells, theta=masses)
    I, J, d = geometry.near
    if len(I):
        cells = geometry.cells
        values = masses[I] * masses[J] * (deterrence(d) - B[cells[I], cells[J]])
        C = sp.coo_array((np.r_[values, values], (np.r_[I, J], np.r_[J, I])), shape=(graph.n, graph.n))
        backend = SumBackend([backend, SparseBackend(C.tocsr())])
    return backend, details


class _StandardizedNull(NullModel):
    """P / sqrt(P (1 - P)) of a probability null."""
    kind = "expected_weight"

    def __init__(self, base, clip=1e-9):
        super().__init__(base.nodes)
        self.base, self.clip = base, clip

    def _transform(self, P):
        P = np.clip(P, self.clip, 1 - self.clip)
        return np.sqrt(P / (1 - P))

    def block(self, rows, cols):
        values = self._transform(self.base.block(rows, cols))
        values[rows[:, None] == cols[None, :]] = 0.
        return values

    def pairs(self, I, J):
        return np.where(I == J, 0., self._transform(self.base.pairs(I, J)))


def _standardized(graph, null):
    if null.kind != "probability":
        raise ValueError("standardize needs a probability null (e.g. likelihood='bernoulli')")
    I, J, w = graph.edges()
    P = np.clip(null.pairs(I, J), 1e-9, 1 - 1e-9)
    w = w / np.sqrt(P * (1 - P))
    A = sp.coo_array((np.r_[w, w], (np.r_[I, J], np.r_[J, I])), shape=(graph.n, graph.n)).tocsr()
    return IndexedGraph(graph.nodes, A), _StandardizedNull(null)


# ----------------------------------------------------------------------------- reading results
def partition_quality(G, partition, null, *, gamma=1.0, epsilon=None, standardize=False, weight="weight"):
    """Q of a partition under the exact null (same definition as disentangled_communities).

    Costs n_c^2 pair evaluations per community of n_c nodes (O(n_c + b^2) for block nulls).
    With epsilon=None, use the fitted automatic model's density default, or zero otherwise.
    """
    graph = as_indexed(G, weight)
    null = _aligned(null, graph)
    if epsilon is None:
        epsilon = null.diagnostics.get("default_epsilon", 0.)
    if graph.total_weight <= 0:
        return 0.
    if standardize:
        graph, null = _standardized(graph, null)
    labels = np.unique([partition[u] for u in graph.nodes], return_inverse=True)[1]
    I, J, w = graph.edges()
    observed = float(w[labels[I] == labels[J]].sum())
    expected, pairs = 0., 0.
    for c in range(labels.max() + 1):
        members = np.flatnonzero(labels == c)
        pairs += len(members) * (len(members) - 1) / 2
        expected += _internal(null, members)
    return (observed - gamma * expected - epsilon * pairs) / graph.total_weight


def _internal(null, members):
    """sum_{u<v in members} P_uv."""
    if isinstance(null, BlockNull):
        mass = np.bincount(null.labels[members], weights=null.masses[members], minlength=null.block_matrix.shape[0])
        own = null.masses[members] ** 2 * np.asarray(null.block_matrix[null.labels[members], null.labels[members]]).ravel()
        return float(mass @ null.block_matrix @ mass - own.sum()) / 2
    if isinstance(null, SumNull):
        return sum(_internal(p, members) for p in null.parts)
    total = 0.
    for s in range(0, len(members), 2048):
        rows = members[s:s + 2048]
        total += float(null.block(rows, members).sum())
    return total / 2


def excess_density(G, null, a, b, *, weight="weight"):
    """Observed O, expected E and excess density (O - E) / (n_a n_b) between two disjoint node sets.

    A merge of a and b increases Q exactly when the excess density exceeds epsilon (gamma=1).
    """
    graph = as_indexed(G, weight)
    null = _aligned(null, graph)
    index = {u: i for i, u in enumerate(graph.nodes)}
    ia = np.array([index[u] for u in a], dtype=np.intp)
    ib = np.array([index[u] for u in b], dtype=np.intp)
    observed = float(graph.adjacency[ia][:, ib].sum())
    expected = 0.
    for s in range(0, len(ia), 2048):
        expected += float(null.block(ia[s:s + 2048], ib).sum())
    pairs = len(ia) * len(ib)
    return dict(observed=observed, expected=expected, pairs=pairs,
                density=(observed - expected) / pairs if pairs else float("nan"))


def epsilon_profile(G, null, epsilons, *, gamma=1.0, seeds=range(10), approximation=None, weight="weight"):
    """Run the search for each epsilon and seed; one row per epsilon.

    Rows: epsilon, n_communities (mean over seeds), quality (best), agreement (mean
    adjusted Rand index between seeds). Choose epsilon in the middle of a stable plateau.
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("provide at least one seed")
    graph, factory, _ = _prepare(G, null, approximation, False, weight, None)
    spec = as_approximation(approximation)
    sampling = isinstance(spec, Sampling)
    rows = []
    for eps in epsilons:
        runs = []
        for seed in seeds:
            labels, quality, _ = optimize(graph, factory, gamma=gamma, epsilon=eps, init=None,
                                          rng=np.random.default_rng(seed), n_iterations=2,
                                          max_passes=spec.max_passes if sampling else None,
                                          max_levels=spec.max_levels if sampling else None,
                                          tolerance=1e-7, max_candidates=spec.max_candidates if sampling else None)
            runs.append((quality, labels))
        agreement = [adjusted_rand(runs[i][1], runs[j][1]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
        rows.append(dict(epsilon=float(eps), n_communities=float(np.mean([l.max() + 1 for _, l in runs])),
                         quality=float(max(q for q, _ in runs)),
                         agreement=float(np.mean(agreement)) if agreement else float("nan")))
    return rows


def adjusted_rand(x, y):
    """Adjusted Rand index between two labelings."""
    if len(x) != len(y):
        raise ValueError("labelings must have the same length")
    if len(x) < 2:
        return 1.
    _, x = np.unique(x, return_inverse=True)
    _, y = np.unique(y, return_inverse=True)
    table = sp.coo_array((np.ones(len(x)), (x, y))).tocsr()
    comb = lambda v: (v * (v - 1) / 2).sum()
    index = comb(table.data)
    a, b = comb(np.asarray(table.sum(axis=1)).ravel()), comb(np.asarray(table.sum(axis=0)).ravel())
    expected = a * b / comb(np.array([len(x)], dtype=float))
    top = (a + b) / 2 - expected
    return 1. if top == 0 else float((index - expected) / top)
