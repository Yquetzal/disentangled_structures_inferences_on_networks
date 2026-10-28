"""Null models: the expected weight P_uv of every pair of nodes.

Every null exposes ``block(rows, cols)``: the |rows| x |cols| matrix of expected weights
between node indices, with zero for a node paired with itself. Everything else (calls on
node ids, expected degrees, dense copies, sums of nulls) is built on it.
"""
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

from ._geometry import TILE_ENTRIES, bin_index, distance_matrix, pair_distances


# ----------------------------------------------------------------------------- deterrence
@dataclass(frozen=True)
class Binned:
    """Piecewise-constant deterrence: f(d) = values[b] for edges[b] <= d < edges[b+1].

    Distances beyond the last edge use the last value.
    """
    edges: np.ndarray
    values: np.ndarray

    def __call__(self, d):
        return np.asarray(self.values)[bin_index(np.asarray(d), np.asarray(self.edges))]


@dataclass(frozen=True)
class Exponential:
    """f(d) = scale * exp(-beta * d)."""
    beta: float
    scale: float = 1.0

    def __call__(self, d):
        return self.scale * np.exp(-self.beta * np.asarray(d))


# ----------------------------------------------------------------------------- base class
class NullModel:
    """Common interface. ``kind`` is 'expected_weight' (values may exceed 1) or
    'probability' (values in [0, 1]). ``diagnostics`` holds fit information."""

    kind = "expected_weight"

    def __init__(self, nodes, diagnostics=None):
        self.nodes = tuple(nodes)
        self.diagnostics = dict(diagnostics or {})
        self._index = {u: i for i, u in enumerate(self.nodes)}
        if len(self._index) != len(self.nodes):
            raise ValueError("nodes must be unique")

    @property
    def n(self):
        return len(self.nodes)

    def block(self, rows, cols):
        raise NotImplementedError

    def pairs(self, I, J):
        """Values of aligned index pairs (I[k], J[k])."""
        return np.array([self.block(np.array([i]), np.array([j]))[0, 0] for i, j in zip(I, J)])

    def __call__(self, u, v):
        return float(self.block(np.array([self._index[u]]), np.array([self._index[v]]))[0, 0])

    def expected_degrees(self):
        """Row sums excluding the diagonal, aligned with ``nodes``."""
        out = np.empty(self.n)
        everything = np.arange(self.n)
        for rows in _row_tiles(self.n):
            out[rows] = self.block(rows, everything).sum(axis=1)
        return out

    def to_matrix(self, *, max_bytes=None):
        """Dense n x n copy; raises if it would exceed ``max_bytes``."""
        if max_bytes is not None and self.n * self.n * 8 > max_bytes:
            raise MemoryError(f"a dense {self.n} x {self.n} matrix exceeds max_bytes={max_bytes}")
        everything = np.arange(self.n)
        return self.block(everything, everything)

    def __add__(self, other):
        if not isinstance(other, NullModel):
            return NotImplemented
        return SumNull(self, other)

    def reindexed(self, nodes):
        """The same null with its nodes listed in the given order."""
        order = np.array([self._index[u] for u in nodes], dtype=np.intp)
        return self._reorder(order, tuple(nodes))

    def _reorder(self, order, nodes):
        raise NotImplementedError


def _row_tiles(n, width=None):
    step = max(1, TILE_ENTRIES // max(width or n, 1))
    for start in range(0, n, step):
        yield np.arange(start, min(start + step, n))


def _zero_self(values, rows, cols):
    """Zero the entries where a row index equals a column index."""
    same = rows[:, None] == cols[None, :]
    if same.any():
        values[same] = 0.
    return values


# ----------------------------------------------------------------------------- spatial
class SpatialNull(NullModel):
    """Node masses and a distance deterrence f.

    link='log'   : P_uv = theta_u theta_v f(d_uv)                         (expected weights)
    link='logit' : P_uv = sigmoid(log theta_u + log theta_v + log f(d_uv))  (probabilities)

    metric: 'euclidean' on vector positions, 'haversine' on (lat, lon) in degrees (km),
    or a callable metric(X, Y) returning the distances between the rows of X and Y.
    Usually built by :func:`fit_spatial_null`; ``fit_settings`` then records how, so that
    landmark approximations can be re-fitted on their own geometry.
    """

    def __init__(self, positions, deterrence, masses=None, *, metric="euclidean", link="log",
                 nodes=None, fit_settings=None, diagnostics=None, groups=None, group_factors=None):
        X = np.asarray(positions, dtype=float)
        X = X[:, None] if X.ndim == 1 else X
        super().__init__(range(len(X)) if nodes is None else nodes, diagnostics)
        if len(X) != self.n:
            raise ValueError("positions and nodes have different lengths")
        if link not in ("log", "logit"):
            raise ValueError("link must be 'log' or 'logit'")
        if metric == "haversine" and X.shape[1] != 2:
            raise ValueError("haversine positions must be (lat, lon) pairs in degrees")
        self.positions, self.deterrence, self.metric, self.link = X, deterrence, metric, link
        self.masses = np.ones(self.n) if masses is None else np.asarray(masses, dtype=float)
        if len(self.masses) != self.n or (self.masses < 0).any() or not np.isfinite(self.masses).all():
            raise ValueError("masses must be finite, nonnegative and one per node")
        self.fit_settings = fit_settings
        self.kind = "probability" if link == "logit" else "expected_weight"
        self.groups = None
        if groups is not None:
            if link != "log" or len(groups) != self.n:
                raise ValueError("groups require the log link and one label per node")
            self.groups = np.asarray(groups)
            _, self.group_codes = np.unique(self.groups, return_inverse=True)
            self.group_factors = np.asarray(group_factors, dtype=float)
            if (self.group_factors.shape != (len(np.unique(self.groups)),)
                    or not np.isfinite(self.group_factors).all() or (self.group_factors < 0).any()):
                raise ValueError("group_factors must be finite nonnegative values in sorted group order")

    def block(self, rows, cols):
        f = self.deterrence(distance_matrix(self.positions[rows], self.positions[cols], self.metric))
        mr, mc = self.masses[rows][:, None], self.masses[cols][None, :]
        if self.link == "log":
            values = mr * mc * f
            if self.groups is not None:
                values *= np.where(self.group_codes[rows, None] == self.group_codes[cols],
                                   self.group_factors[self.group_codes[rows, None]], 1.)
        else:
            with np.errstate(divide="ignore"):
                values = expit(np.log(mr) + np.log(mc) + np.log(f))
        return _zero_self(np.asarray(values, dtype=float), rows, cols)

    def pairs(self, I, J):
        f = self.deterrence(pair_distances(self.positions, I, J, self.metric))
        if self.link == "log":
            values = self.masses[I] * self.masses[J] * f
            if self.groups is not None:
                values *= np.where(self.group_codes[I] == self.group_codes[J],
                                   self.group_factors[self.group_codes[I]], 1.)
        else:
            with np.errstate(divide="ignore"):
                values = expit(np.log(self.masses[I]) + np.log(self.masses[J]) + np.log(f))
        return np.where(I == J, 0., values)

    def _reorder(self, order, nodes):
        return SpatialNull(self.positions[order], self.deterrence, self.masses[order], metric=self.metric,
                           link=self.link, nodes=nodes, fit_settings=self.fit_settings,
                           diagnostics=self.diagnostics, groups=None if self.groups is None else self.groups[order],
                           group_factors=None if self.groups is None else self.group_factors[
                               np.unique(self.group_codes[order])])


# ----------------------------------------------------------------------------- blocks
class BlockNull(NullModel):
    """P_uv = theta_u theta_v B[block_u, block_v]: SBM, degree-corrected SBM, configuration.

    labels are integers indexing the rows of ``block_matrix`` (or any labels with
    ``block_order`` giving the label of each row). masses=None means theta = 1.
    """

    def __init__(self, labels, block_matrix, masses=None, *, nodes=None, block_order=None, diagnostics=None):
        labels = np.asarray(labels)
        super().__init__(range(len(labels)) if nodes is None else nodes, diagnostics)
        B = sp.csr_array(block_matrix, dtype=float) if sp.issparse(block_matrix) else np.asarray(block_matrix, dtype=float)
        symmetric = ((B - B.T).nnz == 0) if sp.issparse(B) else np.allclose(B, B.T)
        if B.ndim != 2 or B.shape[0] != B.shape[1] or not symmetric:
            raise ValueError("block_matrix must be square and symmetric")
        data = B.data if sp.issparse(B) else B
        if not np.isfinite(data).all() or (data < 0).any():
            raise ValueError("block_matrix must be finite and nonnegative")
        if block_order is not None:
            position = {b: k for k, b in enumerate(block_order)}
            labels = np.array([position[b] for b in labels], dtype=np.intp)
        labels = labels.astype(np.intp)
        if len(labels) != self.n or labels.min(initial=0) < 0 or labels.max(initial=-1) >= B.shape[0]:
            raise ValueError("labels must index the rows of block_matrix")
        self.labels, self.block_matrix = labels, B
        self.masses = np.ones(self.n) if masses is None else np.asarray(masses, dtype=float)
        if self.masses.shape != (self.n,) or not np.isfinite(self.masses).all() or (self.masses < 0).any():
            raise ValueError("masses must be finite, nonnegative and one per node")

    def block(self, rows, cols):
        rates = self.block_matrix[np.ix_(self.labels[rows], self.labels[cols])]
        rates = rates.toarray() if sp.issparse(rates) else rates
        values = (self.masses[rows][:, None] * self.masses[cols][None, :]
                  * rates)
        return _zero_self(values, rows, cols)

    def pairs(self, I, J):
        values = self.masses[I] * self.masses[J] * np.asarray(self.block_matrix[self.labels[I], self.labels[J]]).ravel()
        return np.where(I == J, 0., values)

    def expected_degrees(self):
        mass = np.bincount(self.labels, weights=self.masses, minlength=self.block_matrix.shape[0])
        diagonal = self.block_matrix.diagonal()
        return self.masses * (self.block_matrix @ mass)[self.labels] - self.masses ** 2 * diagonal[self.labels]

    def _reorder(self, order, nodes):
        return BlockNull(self.labels[order], self.block_matrix, self.masses[order], nodes=nodes,
                         diagnostics=self.diagnostics)


# ----------------------------------------------------------------------------- custom
class CustomNull(NullModel):
    """Any null: a dense matrix or a function f(u, v) of node ids. See :func:`custom_null`."""

    def __init__(self, values, nodes, *, kind="expected_weight", vectorized=False, cache_size=65_536,
                 diagnostics=None):
        super().__init__(nodes, diagnostics)
        if kind not in ("expected_weight", "probability"):
            raise ValueError("kind must be 'expected_weight' or 'probability'")
        self.kind, self.vectorized = kind, vectorized
        if callable(values):
            self.matrix, self.function = None, values
            self._cache, self._cache_size = OrderedDict(), int(cache_size)
        else:
            M = np.array(values, dtype=float)
            if M.shape != (self.n, self.n):
                raise ValueError(f"matrix must be {self.n} x {self.n}")
            if not np.isfinite(M).all() or (M < 0).any() or not np.allclose(M, M.T):
                raise ValueError("matrix must be finite, nonnegative and symmetric")
            np.fill_diagonal(M, 0.)
            self.matrix, self.function = M, None
        if kind == "probability" and self.matrix is not None and (self.matrix > 1).any():
            raise ValueError("probabilities must not exceed 1")

    def block(self, rows, cols):
        if self.matrix is not None:
            return self.matrix[np.ix_(rows, cols)].copy()
        if self.vectorized:
            R, C = np.meshgrid(rows, cols, indexing="ij")
            values = np.asarray(self.function([self.nodes[i] for i in R.ravel()],
                                              [self.nodes[j] for j in C.ravel()]), dtype=float)
            return _zero_self(values.reshape(R.shape), rows, cols)
        out = np.empty((len(rows), len(cols)))
        for a, i in enumerate(rows):
            for b, j in enumerate(cols):
                out[a, b] = 0. if i == j else self._pair(int(i), int(j))
        return out

    def pairs(self, I, J):
        if self.matrix is not None:
            return self.matrix[I, J]
        if self.vectorized:
            values = np.asarray(self.function([self.nodes[i] for i in I], [self.nodes[j] for j in J]), dtype=float)
            return np.where(np.asarray(I) == np.asarray(J), 0., values)
        return np.array([0. if i == j else self._pair(int(i), int(j)) for i, j in zip(I, J)])

    def _pair(self, i, j):
        key = (i, j) if i < j else (j, i)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        value = float(self.function(self.nodes[key[0]], self.nodes[key[1]]))
        if self._cache_size:
            self._cache[key] = value
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return value

    def _reorder(self, order, nodes):
        if self.matrix is not None:
            return CustomNull(self.matrix[np.ix_(order, order)], nodes, kind=self.kind, diagnostics=self.diagnostics)
        return CustomNull(self.function, nodes, kind=self.kind, vectorized=self.vectorized,
                          cache_size=self._cache_size, diagnostics=self.diagnostics)


class SumNull(NullModel):
    """Sum of nulls, built by ``null_a + null_b``."""

    def __init__(self, *parts):
        flat = []
        for part in parts:
            flat.extend(part.parts if isinstance(part, SumNull) else [part])
        if not flat:
            raise ValueError("provide at least one null model")
        super().__init__(flat[0].nodes)
        if any(set(p.nodes) != set(self.nodes) for p in flat):
            raise ValueError("summed nulls must have the same nodes")
        self.parts = tuple(p if p.nodes == self.nodes else p.reindexed(self.nodes) for p in flat)

    def block(self, rows, cols):
        return sum(part.block(rows, cols) for part in self.parts)

    def pairs(self, I, J):
        return sum(part.pairs(I, J) for part in self.parts)

    def _reorder(self, order, nodes):
        return SumNull(*(part._reorder(order, nodes) for part in self.parts))


def custom_null(values, nodes=None, *, kind="expected_weight", vectorized=False, cache_size=65_536,
                validation_samples=64):
    """Wrap any null model.

    Parameters
    ----------
    values : array or callable
        An n x n matrix (rows in ``nodes`` order), or a function f(u, v) -> float of node
        ids. A matrix is evaluated with the fast dense engine; a function on demand
        (generic, slowest).
    nodes : sequence, optional
        Node ids in matrix order (default 0..n-1); required for a function.
    kind : {'expected_weight', 'probability'}
        Poisson reading (values may exceed 1) or Bernoulli reading.
    vectorized : bool
        The function takes two aligned sequences of node ids and returns an array.
    cache_size : int
        Number of function values kept in memory.
    validation_samples : int
        Symmetry and nonnegativity of a function are spot-checked on this many pairs.
    """
    if callable(values) and nodes is None:
        raise ValueError("nodes are required with a function")
    nodes = range(len(values)) if nodes is None else nodes
    null = CustomNull(values, nodes, kind=kind, vectorized=vectorized, cache_size=cache_size)
    if null.function is not None and validation_samples and null.n > 1:
        rng = np.random.default_rng(0)
        I = rng.integers(0, null.n, validation_samples)
        J = rng.integers(0, null.n, validation_samples)
        I, J = I[I != J], J[I != J]
        # Evaluate the original callable: the scalar evaluation cache canonicalises pairs,
        # so checking through block() would hide an asymmetric function.
        if vectorized:
            forward = np.asarray(values([null.nodes[i] for i in I], [null.nodes[j] for j in J]))
            backward = np.asarray(values([null.nodes[j] for j in J], [null.nodes[i] for i in I]))
        else:
            forward = np.array([values(null.nodes[i], null.nodes[j]) for i, j in zip(I, J)])
            backward = np.array([values(null.nodes[j], null.nodes[i]) for i, j in zip(I, J)])
        if not np.isfinite(forward).all() or (forward < 0).any() or not np.allclose(forward, backward):
            raise ValueError("the function must be finite, nonnegative and symmetric")
        if kind == "probability" and (forward > 1).any():
            raise ValueError("probabilities must not exceed 1")
    return null
