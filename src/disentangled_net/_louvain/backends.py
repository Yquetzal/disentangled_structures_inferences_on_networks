"""Ways to compute sum_{j in c, j != i} P_ij during the search.

Every backend works on the current level's (super-)nodes and offers:
    start(labels)          labels is the engine's array, mutated in place by the engine
    sums(i, cands, a)      expected weight from node i to each candidate community
                           (i currently belongs to community a; i itself is excluded)
    move(i, a, b)          node i leaves community a for b
    coarsen(comp, K)       backend for the next level, where node i becomes super-node comp[i]
"""
import numpy as np
import scipy.sparse as sp

DENSE_LIMIT = 2 ** 30      # bytes of dense working memory allowed for exact evaluation


def onehot(comp, K):
    n = len(comp)
    return sp.csr_array((np.ones(n), (np.arange(n), comp)), shape=(n, K))


class MatrixBackend:
    """Dense P and T[:, c] = sum_{j in c} P[:, j]. Memory 2 n^2 floats; O(1) sums, O(n) moves."""
    exact = True
    name = "dense"

    def __init__(self, P):
        self.P = np.asarray(P, dtype=float, order="C")        # shared, never modified

    def start(self, labels):
        n = len(self.P)
        if np.array_equal(labels, np.arange(n)):
            self.T = np.array(self.P, order="F")
        else:
            self.T = np.zeros((n, n), order="F")
            np.add.at(self.T.T, labels, self.P)
        self.labels = labels

    def sums(self, i, cands, a):
        values = self.T[i, cands]
        values[cands == a] -= self.P[i, i]
        return values

    def move(self, i, a, b):
        row = self.P[i]
        self.T[:, a] -= row
        self.T[:, b] += row

    def coarsen(self, comp, K):
        old = np.empty(K, np.intp)
        old[comp] = self.labels
        coarse = np.zeros((K, K))
        np.add.at(coarse, comp, self.T[:, old])
        self.P = self.T = None
        return MatrixBackend(coarse)


class BlockBackend:
    """P_ij = M_i B M_j^T with M_i the mass of node i in each block (cell).

    Stores F[c] = sum_{j in c} M_j B. At the first level M is one-hot, so a sum costs O(1);
    afterwards O(r). Memory O(n r).
    """
    exact = True
    name = "blocks"

    def __init__(self, B, cells=None, theta=None, M=None):
        self.B = np.asarray(B, dtype=float)
        self.cells, self.theta, self.M = cells, theta, M
        if M is None:
            self.R = theta[:, None] * self.B[cells]
            self.self_term = theta * self.R[np.arange(len(cells)), cells]
        else:
            self.R = M @ self.B
            self.self_term = np.einsum("ij,ij->i", M, self.R)

    def start(self, labels):
        self.F = np.zeros_like(self.R)
        np.add.at(self.F, labels, self.R)

    def sums(self, i, cands, a):
        if self.M is None:
            values = self.theta[i] * self.F[cands, self.cells[i]]
        else:
            values = self.F[cands] @ self.M[i]
        values[cands == a] -= self.self_term[i]
        return values

    def move(self, i, a, b):
        self.F[a] -= self.R[i]
        self.F[b] += self.R[i]

    def coarsen(self, comp, K):
        M = np.zeros((K, self.B.shape[0]))
        if self.M is None:
            np.add.at(M, (comp, self.cells), self.theta)
        else:
            np.add.at(M, comp, self.M)
        return BlockBackend(self.B, M=M)


class SparseBlockBackend(BlockBackend):
    """The same block sums, kept sparse when an attribute has many categories."""
    name = "sparse_blocks"

    def __init__(self, B, cells=None, theta=None, M=None):
        self.B = sp.csr_array(B)
        self.cells, self.theta = cells, theta
        if M is None:
            n = len(cells)
            M = sp.csr_array((theta, (np.arange(n), cells)), shape=(n, B.shape[0]))
        self.M = sp.csr_array(M)
        self.R = self.M @ self.B
        self.self_term = np.asarray(self.M.multiply(self.R).sum(axis=1)).ravel()

    def start(self, labels):
        self.F = (onehot(labels, len(labels)).T @ self.R).tolil()

    def sums(self, i, cands, a):
        values = (self.F[cands].tocsr() @ self.M[i:i + 1].T).toarray().ravel()
        values[cands == a] -= self.self_term[i]
        return values

    def move(self, i, a, b):
        lo, hi = self.R.indptr[i:i + 2]
        cols, values = self.R.indices[lo:hi], self.R.data[lo:hi]
        self.F[a:a + 1, cols] = self.F[a:a + 1, cols].toarray() - values[None, :]
        self.F[b:b + 1, cols] = self.F[b:b + 1, cols].toarray() + values[None, :]

    def coarsen(self, comp, K):
        return SparseBlockBackend(self.B, M=onehot(comp, K).T @ self.M)


class SparseBackend:
    """A sparse symmetric correction C (e.g. exact near pairs minus their cell value)."""
    exact = True
    name = "sparse"

    def __init__(self, C):
        C = sp.csr_array(C)
        C = C - sp.diags(C.diagonal(), format="csr")
        C.eliminate_zeros()
        self.C = C

    def start(self, labels):
        self.labels = labels

    def sums(self, i, cands, a):
        lo, hi = self.C.indptr[i], self.C.indptr[i + 1]
        out = np.zeros(len(cands))
        if hi > lo:
            order = np.argsort(cands)
            sorted_cands = cands[order]
            cs = self.labels[self.C.indices[lo:hi]]
            pos = np.minimum(np.searchsorted(sorted_cands, cs), len(cands) - 1)
            hit = sorted_cands[pos] == cs
            out[order] = np.bincount(pos[hit], weights=self.C.data[lo:hi][hit], minlength=len(cands))
        return out

    def move(self, i, a, b):
        pass

    def coarsen(self, comp, K):
        Z = onehot(comp, K)
        return SparseBackend(Z.T @ self.C @ Z)


class SumBackend:
    def __init__(self, parts):
        self.parts = list(parts)
        self.exact = all(p.exact for p in self.parts)
        self.name = "+".join(p.name for p in self.parts)

    def start(self, labels):
        for p in self.parts:
            p.start(labels)

    def sums(self, i, cands, a):
        return sum(p.sums(i, cands, a) for p in self.parts)

    def move(self, i, a, b):
        for p in self.parts:
            p.move(i, a, b)

    def coarsen(self, comp, K):
        return SumBackend([p.coarsen(comp, K) for p in self.parts])


class _Members:
    """Original nodes of each super-node, and super-nodes of each community."""

    def __init__(self, members):
        self.members = members

    def start(self, labels):
        self.groups = [set() for _ in range(len(labels))]
        for i, c in enumerate(labels):
            self.groups[c].add(i)

    def community(self, c, without):
        parts = [self.members[j] for j in self.groups[c] if j != without]
        return np.concatenate(parts) if parts else np.zeros(0, np.intp)

    def move(self, i, a, b):
        self.groups[a].discard(i)
        self.groups[b].add(i)

    def merged(self, comp, K):
        merged = [[] for _ in range(K)]
        for i, c in enumerate(comp):
            merged[c].append(self.members[i])
        return [np.concatenate(m) for m in merged]


def _tiled_sum(null, rows, cols, width=2048):
    total = 0.
    for s in range(0, len(rows), width):
        for t in range(0, len(cols), width):
            total += float(null.block(rows[s:s + width], cols[t:t + width]).sum())
    return total


class TiledBackend(_Members):
    """Exact sums evaluated from the null on original pairs, in bounded tiles.

    Memory O(n + tile^2), time up to O(n^2) per sweep. Once a coarser level fits in
    memory, the aggregated null is computed once and the dense backend takes over.
    """
    exact = True
    name = "tiled"

    def __init__(self, null, members=None):
        super().__init__(members if members is not None else [np.array([i]) for i in range(null.n)])
        self.null = null

    def sums(self, i, cands, a):
        mine = self.members[i]
        return np.array([_tiled_sum(self.null, mine, self.community(c, without=i)) for c in cands])

    def coarsen(self, comp, K):
        merged = self.merged(comp, K)
        if 2 * K * K * 8 > DENSE_LIMIT:
            return TiledBackend(self.null, merged)
        original = np.empty(self.null.n, np.intp)
        for c, m in enumerate(merged):
            original[m] = c
        Z = onehot(original, K)
        coarse = np.zeros((K, K))
        everything = np.arange(self.null.n)
        for s in range(0, self.null.n, 1024):
            rows = everything[s:s + 1024]
            np.add.at(coarse, original[rows], self.null.block(rows, everything) @ Z)
        return MatrixBackend(coarse)


class SamplingBackend(_Members):
    """Unbiased estimates of each sum from uniformly drawn original pairs."""
    exact = False
    name = "sampling"

    def __init__(self, null, rng, sample_size, background_weight, members=None):
        super().__init__(members if members is not None else [np.array([i]) for i in range(null.n)])
        self.null, self.rng = null, rng
        self.sample_size, self.background_weight = int(sample_size), float(background_weight)

    def start(self, labels):
        super().start(labels)
        self.background = {}

    def sums(self, i, cands, a):
        mine = self.members[i]
        out = np.empty(len(cands))
        for k, c in enumerate(cands):
            other = self.community(c, without=i)
            pairs = len(mine) * len(other)
            if pairs <= self.sample_size:
                out[k] = _tiled_sum(self.null, mine, other) if pairs else 0.
                continue
            u = mine[self.rng.integers(0, len(mine), self.sample_size)]
            v = other[self.rng.integers(0, len(other), self.sample_size)]
            mean = float(self.null.pairs(u, v).mean())
            estimate = pairs * mean
            if self.background_weight and c in self.background:
                estimate = (1 - self.background_weight) * estimate + self.background_weight * pairs * self.background[c]
            self.background[c] = mean if c not in self.background else (self.background[c] + mean) / 2
            out[k] = estimate
        return out

    def coarsen(self, comp, K):
        return SamplingBackend(self.null, self.rng, self.sample_size, self.background_weight, self.merged(comp, K))
