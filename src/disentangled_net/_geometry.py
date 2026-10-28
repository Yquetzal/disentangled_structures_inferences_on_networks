"""Distances, and the two pair geometries used to fit and evaluate spatial nulls.

ExactGeometry    : every pair at its true distance, visited in row tiles (O(n) memory).
LandmarkGeometry : pairs at the distance between their cells' representatives, except the
                   near pairs (closer than ``exact_within``, or among each node's
                   ``exact_neighbors`` nearest nodes), kept exact (hybrid).
Both expose the same sums, so fitting code does not depend on the geometry.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

EARTH_RADIUS_KM = 6371.0088
TILE_ENTRIES = 4_000_000          # pairs per tile
CACHE_BYTES = 512 * 2**20         # exact distances are cached below this size


def distance_matrix(X, Y, metric):
    """|X| x |Y| distances. 'euclidean', 'haversine' ((lat, lon) degrees -> km) or callable."""
    if callable(metric):
        return np.asarray(metric(X, Y), dtype=float)
    if metric == "euclidean":
        return cdist(X, Y)
    if metric == "haversine":
        lat1, lon1 = np.radians(X[:, 0])[:, None], np.radians(X[:, 1])[:, None]
        lat2, lon2 = np.radians(Y[:, 0])[None, :], np.radians(Y[:, 1])[None, :]
        h = (np.sin((lat2 - lat1) / 2) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
        return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0., 1.)))
    raise ValueError("metric must be 'euclidean', 'haversine' or a callable")


def pair_distances(X, I, J, metric):
    """Distances of aligned pairs (I[k], J[k])."""
    if callable(metric):
        return np.array([metric(X[i:i + 1], X[j:j + 1])[0, 0] for i, j in zip(I, J)], dtype=float)
    a, b = X[I], X[J]
    if metric == "euclidean":
        return np.linalg.norm(a - b, axis=1)
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (a[:, 0], a[:, 1], b[:, 0], b[:, 1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0., 1.)))


def euclidean_embedding(X, metric):
    """Coordinates in which straight-line distance orders pairs like the metric."""
    if metric == "euclidean":
        return np.asarray(X, dtype=float)
    if metric == "haversine":
        lat, lon = np.radians(X[:, 0]), np.radians(X[:, 1])
        return EARTH_RADIUS_KM * np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    raise ValueError("landmarks need metric='euclidean' or 'haversine'")


def _from_embedding(C, metric):
    if metric == "euclidean":
        return C
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    return np.c_[np.degrees(np.arcsin(np.clip(C[:, 2], -1, 1))), np.degrees(np.arctan2(C[:, 1], C[:, 0]))]


def _chord(radius, metric):
    if metric == "euclidean":
        return radius
    return 2 * EARTH_RADIUS_KM * np.sin(min(radius / (2 * EARTH_RADIUS_KM), np.pi / 2))


def bin_index(d, edges):
    """Bin of each distance; distances beyond the last edge fall into the last bin."""
    return np.searchsorted(edges[1:-1], d, side="right")


class ExactGeometry:
    def __init__(self, X, metric):
        self.X, self.metric, self.n = X, metric, len(X)
        self.tile = max(1, TILE_ENTRIES // max(self.n, 1))
        self._cache = {} if self.n * self.n * 8 <= CACHE_BYTES else None

    def tiles(self):
        """Yield (rows, distances rows x all). Self distances are at (k, rows[k])."""
        for start in range(0, self.n, self.tile):
            rows = np.arange(start, min(start + self.tile, self.n))
            if self._cache is not None and start in self._cache:
                yield rows, self._cache[start]
                continue
            D = distance_matrix(self.X[rows], self.X, self.metric)
            if self._cache is not None:
                self._cache[start] = D
            yield rows, D

    def edge_distances(self, I, J):
        return pair_distances(self.X, I, J, self.metric)

    def row_sums(self, w, g):
        """sum_{j != i} w_j g(d_ij) for every i."""
        out = np.empty(self.n)
        for rows, D in self.tiles():
            G = g(D)
            G[np.arange(len(rows)), rows] = 0.
            out[rows] = G @ w
        return out

    def pair_sum(self, w, g):
        """sum_{i<j} w_i w_j g(d_ij)."""
        return float(w @ self.row_sums(w, g)) / 2

    def bin_sums(self, w, edges):
        """Per distance bin: sum_{i<j in bin} w_i w_j."""
        nb = len(edges) - 1
        out = np.zeros(nb)
        for rows, D in self.tiles():
            W = w[rows, None] * w[None, :]
            W[np.arange(len(rows)), rows] = 0.
            out += np.bincount(bin_index(D, edges).ravel(), weights=W.ravel(), minlength=nb)
        return out / 2


class LandmarkGeometry:
    """Representative distances between cells, plus exact near pairs (hybrid).

    cells : node -> representative; rep_distance : r x r; near : (I, J, d) with I < J.
    """

    @classmethod
    def repeated_positions(cls, X, metric, representatives, cells):
        """Exact compression: one cell per distinct position, without clustering."""
        obj = cls.__new__(cls)
        obj.X, obj.metric, obj.n = X, metric, len(X)
        obj.representatives, obj.cells = representatives, cells.astype(np.intp)
        obj.rank = len(representatives)
        obj.rep_distance = distance_matrix(representatives, representatives, metric)
        np.fill_diagonal(obj.rep_distance, 0.)
        obj.exact_within, obj.exact_neighbors = 0., 0
        obj._near_keys = np.zeros(0, np.int64)
        obj.near = (np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros(0))
        return obj

    def __init__(self, X, metric, rank, placement="centroids", exact_within=None, weights=None,
                 exact_neighbors=None):
        self.X, self.metric, self.n = X, metric, len(X)
        E = euclidean_embedding(X, metric)
        r = min(int(rank), self.n)
        if r < 1:
            raise ValueError("rank must be at least 1")
        weights = np.ones(self.n) if weights is None or not np.any(weights > 0) else np.asarray(weights, float)
        centers = E[_farthest_first(E, r)]
        if placement == "centroids" and r < self.n:
            centers = _lloyd(E, centers, weights)
        elif placement not in ("centroids", "farthest"):
            raise ValueError("placement must be 'centroids' or 'farthest'")
        self.cells = cKDTree(centers).query(E)[1].astype(np.intp)
        self.representatives = _from_embedding(centers, metric)
        self.rep_distance = distance_matrix(self.representatives, self.representatives, metric)
        np.fill_diagonal(self.rep_distance, 0.)
        self.rank = r
        self.exact_within = float(exact_within) if exact_within else 0.
        self.exact_neighbors = int(exact_neighbors) if exact_neighbors else 0
        keys = []
        tree = cKDTree(E)
        if exact_within:
            pairs = tree.query_pairs(_chord(float(exact_within), metric), output_type="ndarray")
            keys.append(pairs[:, 0].astype(np.int64) * self.n + pairs[:, 1])
        if exact_neighbors:
            k = min(self.exact_neighbors + 1, self.n)
            neighbors = tree.query(E, k=k)[1][:, 1:]
            I = np.repeat(np.arange(self.n), k - 1)
            J = neighbors.ravel()
            keys.append(np.minimum(I, J).astype(np.int64) * self.n + np.maximum(I, J))
        self._near_keys = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
        I, J = (self._near_keys // self.n).astype(np.intp), (self._near_keys % self.n).astype(np.intp)
        d = pair_distances(X, I, J, metric) if len(I) else np.zeros(0)
        if exact_within and not exact_neighbors:
            keep = d < exact_within
            I, J, d, self._near_keys = I[keep], J[keep], d[keep], self._near_keys[keep]
        self.near = (I, J, d)

    def edge_distances(self, I, J):
        d = self.rep_distance[self.cells[I], self.cells[J]]
        if len(self._near_keys):
            keys = np.minimum(I, J).astype(np.int64) * self.n + np.maximum(I, J)
            near = np.isin(keys, self._near_keys)
            d = np.where(near, pair_distances(self.X, I, J, self.metric), d)
        return d

    def _cell_sums(self, w):
        r = self.rank
        return (np.bincount(self.cells, weights=w, minlength=r),
                np.bincount(self.cells, weights=w * w, minlength=r))

    def row_sums(self, w, g):
        W, _ = self._cell_sums(w)
        G = g(self.rep_distance)
        out = (G @ W)[self.cells] - w * G[self.cells, self.cells]
        I, J, d = self.near
        if len(I):
            delta = g(d) - G[self.cells[I], self.cells[J]]
            out += np.bincount(I, weights=delta * w[J], minlength=self.n)
            out += np.bincount(J, weights=delta * w[I], minlength=self.n)
        return out

    def pair_sum(self, w, g):
        return float(w @ self.row_sums(w, g)) / 2

    def bin_sums(self, w, edges):
        nb = len(edges) - 1
        W, W2 = self._cell_sums(w)
        B = bin_index(self.rep_distance, edges)
        out = np.bincount(B.ravel(), weights=np.outer(W, W).ravel(), minlength=nb)
        out -= np.bincount(np.diag(B), weights=W2, minlength=nb)
        out /= 2
        I, J, d = self.near
        if len(I):
            ww = w[I] * w[J]
            out += np.bincount(bin_index(d, edges), weights=ww, minlength=nb)
            out -= np.bincount(B[self.cells[I], self.cells[J]], weights=ww, minlength=nb)
        return out


def _farthest_first(E, r):
    chosen = np.empty(r, np.intp)
    closest = np.full(len(E), np.inf)
    i = 0
    for k in range(r):
        chosen[k] = i
        closest = np.minimum(closest, np.sum((E - E[i]) ** 2, axis=1))
        closest[chosen[:k + 1]] = -1.
        i = int(np.argmax(closest))
    return chosen


def _lloyd(E, centers, weights, iterations=10):
    """Mass-weighted k-means: representatives follow where the mass is."""
    for _ in range(iterations):
        labels = cKDTree(centers).query(E)[1]
        mass = np.bincount(labels, weights=weights, minlength=len(centers))
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, E * weights[:, None])
        updated = centers.copy()
        ok = mass > 0
        updated[ok] = sums[ok] / mass[ok, None]
        if np.allclose(updated, centers, rtol=0, atol=1e-9):
            break
        centers = updated
    return centers
