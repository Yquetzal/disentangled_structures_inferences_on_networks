"""Multilevel Louvain search for Q = (1/m) sum_{u<v same} (A_uv - gamma P_uv - epsilon).

The null enters only through a backend (see backends.py). epsilon is applied here, from
community sizes counted in original nodes, so it costs O(1) per candidate.
"""
import numpy as np
import scipy.sparse as sp

from .backends import onehot


def optimize(graph, make_backend, *, gamma, epsilon, init, rng, n_iterations, max_passes,
             max_levels, tolerance, max_candidates=None):
    """Returns (labels on original nodes, quality, per-level records)."""
    n, m = graph.n, graph.total_weight
    labels = np.arange(n) if init is None else np.unique(init, return_inverse=True)[1].astype(np.intp)
    if m <= 0:
        return np.arange(n), 0., []
    backend = make_backend()
    quality = 0. if init is None else _score(graph.adjacency, np.ones(n), labels, backend, gamma, epsilon) / m
    records = []
    for cycle in range(n_iterations):
        backend = make_backend() if cycle else backend
        A, sizes, current, membership = graph.adjacency, np.ones(n), labels.copy(), np.arange(n)
        level = 0
        while True:
            gain, passes, moves = _one_level(A, sizes, current, backend, gamma, epsilon, m, rng,
                                             max_passes, tolerance, max_candidates)
            quality += gain / m
            comp, K = _compact(current)
            membership = comp[membership]
            records.append(dict(cycle=cycle, level=level, nodes=len(current), communities=K,
                                passes=passes, moves=moves, gain=gain / m))
            level += 1
            if K == len(current) or (level > 1 and moves == 0) or (max_levels and level >= max_levels):
                break
            Z = onehot(comp, K)
            A = sp.csr_array(Z.T @ A @ Z)
            A = A - sp.diags(A.diagonal(), format="csr")
            A.eliminate_zeros()
            sizes = np.bincount(comp, weights=sizes, minlength=K)
            backend = backend.coarsen(comp, K)
            current = np.arange(K)
        labels = membership
    return labels, quality, records


def _compact(labels):
    _, comp = np.unique(labels, return_inverse=True)
    return comp.astype(np.intp), int(comp.max()) + 1


def _score(A, sizes, labels, backend, gamma, epsilon):
    """Q * m of a partition, from the backend (used for a given initial partition)."""
    backend.start(labels)
    comm = np.bincount(labels, weights=sizes, minlength=len(labels))
    total = 0.
    for i in range(len(labels)):
        a = labels[i]
        js, ws = A.indices[A.indptr[i]:A.indptr[i + 1]], A.data[A.indptr[i]:A.indptr[i + 1]]
        observed = ws[labels[js] == a].sum()
        expected = backend.sums(i, np.array([a]), a)[0]
        total += observed - gamma * expected - epsilon * sizes[i] * (comm[a] - sizes[i])
    return total / 2


def _one_level(A, sizes, labels, backend, gamma, epsilon, m, rng, max_passes, tolerance, max_candidates):
    n = len(labels)
    count = np.bincount(labels, minlength=n)                 # super-nodes per community
    comm = np.bincount(labels, weights=sizes, minlength=n)   # original nodes per community
    free = [c for c in range(n) if count[c] == 0]
    backend.start(labels)
    indptr, indices, data = A.indptr, A.indices, A.data
    total, passes, moves_total, seen = 0., 0, 0, set()
    while max_passes is None or passes < max_passes:
        moves, gain_pass = 0, 0.
        for i in rng.permutation(n):
            a, si = labels[i], sizes[i]
            js = indices[indptr[i]:indptr[i + 1]]
            if len(js):
                cands, inverse = np.unique(labels[js], return_inverse=True)
                observed = np.bincount(inverse, weights=data[indptr[i]:indptr[i + 1]])
            else:
                cands, observed = np.zeros(0, np.intp), np.zeros(0)
            here = np.searchsorted(cands, a)
            if here == len(cands) or cands[here] != a:
                cands, observed = np.insert(cands, here, a), np.insert(observed, here, 0.)
            if max_candidates is not None and len(cands) > max_candidates + 1:
                keep = np.argsort(-np.where(cands == a, np.inf, observed))[:max_candidates + 1]
                cands, observed = cands[keep], observed[keep]
            score = (observed - gamma * backend.sums(i, cands, a)
                     - epsilon * si * (comm[cands] - np.where(cands == a, si, 0.)))
            stay = score[cands == a][0]
            best = int(np.argmax(score))
            target, best_score = cands[best], score[best]
            if count[a] > 1 and best_score < 0.:
                target, best_score = free[-1], 0.            # leave for an empty community
            if target == a or best_score <= stay + 1e-12 * max(1., abs(stay), abs(best_score)):
                continue
            backend.move(i, a, target)
            labels[i] = target
            count[a] -= 1
            comm[a] -= si
            if count[target] == 0:
                free.pop() if free[-1] == target else free.remove(target)
            count[target] += 1
            comm[target] += si
            if count[a] == 0:
                free.append(a)
            moves += 1
            gain_pass += best_score - stay
        passes += 1
        total += gain_pass
        moves_total += moves
        if moves == 0:
            break
        if backend.exact:
            if gain_pass / m <= tolerance:
                break
        else:
            signature = hash(labels.tobytes())
            if signature in seen:
                break
            seen.add(signature)
    return total, passes, moves_total
