"""Node embeddings of the residual R = A - gamma P - epsilon (what the null does not explain).

'spectral' never forms R: it only multiplies by A and P, exactly or through landmarks. 'sine'
never forms it either when the null has a block structure (a block null, or a spatial null
through landmarks): pairs are then drawn from the edges and from the blocks. The other
methods use the dense residual: O(n^2) memory. 'sine' and 'contrastive' need PyTorch
(``pip install disentangled-net[embeddings]``).
"""
from contextlib import nullcontext
from dataclasses import dataclass, fields

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh

from ._graph import as_indexed
from ._louvain.backends import BlockBackend, MatrixBackend, SparseBackend, SumBackend
from .approximations import Landmarks, as_approximation
from .communities import _aligned, _exact_backend, _landmark_backend
from .fitting import _attribute_null


@dataclass(frozen=True)
class Sine:
    """Signed pairwise embedding: sampled pairs attract (R > 0) or repel (R < 0).

    ``sampling='proportional'`` (default) draws the pairs of each node in proportion to abs(R),
    an unbiased estimate of the residual-weighted loss normalized per source node. The null's
    values, not only its signs, shape the embedding. ``sampling='softmax'`` is the original rule,
    softmax(abs(R) / temperature).

    ``engine='auto'`` draws the same pairs without forming R when the null has a block
    structure (O(m + n) memory, O(n x samples) per epoch); ``'dense'`` always forms R.

    ``init='spectral'`` (default) starts from the Laplacian eigenmaps of the residual
    (``Spectral(normalized=True)``), standardised to a spread of ``init_scale``: the pairwise
    loss only acts locally, and from a random start it cannot unfold a large geometry.
    ``init='random'`` starts from small Gaussian coordinates.

    Attracted pairs (R > 0) have the energy d^power / power. ``repulsion='hinge'`` (default)
    gives repelled pairs (R < 0) the energy max(0, margin - d): no force beyond the margin.
    ``repulsion='log'`` gives them -ln d, a force 1 / d at every distance (Noack's LinLog for
    power = 1 in its distance dependence), and weights each node's pairs by its
    residual mass sum_v abs(R_uv), so that the sampled loss estimates the energy
    sum_{R>0} abs(R) d^power / power - sum_{R<0} abs(R) ln d. This signed-residual energy differs
    from LinLog energies attracting along all observed edges and repelling along all null pairs. The
    step size then decays (cosine) to lr / 50. Proportional sampling only. The long-range
    forces take longer to settle than the hinge: use about ``epochs=1000``.
    """
    epochs: int = 100
    lr: float = 0.1
    temperature: float = 0.5
    samples_per_node: int = 15
    margin: float = 2.0
    sampling: str = "proportional"
    engine: str = "auto"
    init: str = "spectral"
    init_scale: float = 1.0
    repulsion: str = "hinge"
    power: float = 1.0

    def __repr__(self):
        # repulsion and power are shown only when changed, so that the text of earlier settings
        # (used as a cache key by experiment scripts) stays the same
        shown = [f for f in fields(self) if f.name not in ("repulsion", "power") or getattr(self, f.name) != f.default]
        return f"Sine({', '.join(f'{f.name}={getattr(self, f.name)!r}' for f in shown)})"


@dataclass(frozen=True)
class DeepWalk:
    """Deterministic DeepWalk-style PPMI factorisation of max(R, 0)."""
    window_size: int = 5


@dataclass(frozen=True)
class Spectral:
    """Leading positive eigenvectors of R, scaled by sqrt(eigenvalue).

    ``normalized=True``: Laplacian eigenmaps of the residual. Leading eigenvectors of
    D^-1/2 R D^-1/2 (D: observed degrees), mapped back by D^-1/2. Excluding self-pairs introduces
    a diagonal correction relative to ordinary Laplacian eigenmaps with the configuration null.
    """
    normalized: bool = False


@dataclass(frozen=True)
class Contrastive:
    """Balanced metric learning: positive residuals attract, negative ones repel."""
    epochs: int = 1000
    lr: float = 0.03
    batch_size: int = 4096
    margin: float = 2.0
    n_restarts: int = 3


_METHODS = {"sine": Sine, "deepwalk": DeepWalk, "spectral": Spectral, "contrastive": Contrastive}


def disentangled_embeddings(G, null=None, *, attribute=None, attribute_type="auto", metric="euclidean",
                            dim=64, method=None, gamma=1.0, epsilon=0.0, approximation=None,
                            weight="weight", random_state=None):
    """Embeddings of the residual R = A - gamma P - epsilon: {node: vector of length dim}.

    Parameters
    ----------
    G : networkx graph or scipy.sparse adjacency matrix.
    null : a null model aligned with the nodes of G.
        Alternatively give an attribute name here or with ``attribute=``.
    attribute : optional node attribute name, mapping or array.
        Fit a compatible null and choose its evaluation automatically.
    attribute_type, metric : as in :func:`~disentangled_net.fit_null`.
    dim : int
        Number of coordinates.
    method : str or settings object
        'sine', 'deepwalk', 'spectral', 'contrastive', or Sine(...), DeepWalk(...),
        Spectral(), Contrastive(...). Default with an attribute or fit_null result:
        normalized spectral, with no PyTorch dependency. Other explicit nulls: 'sine' (unchanged).
    gamma, epsilon : float
        As in :func:`disentangled_communities`.
    approximation : None or Landmarks
        'spectral' and 'sine'. None is exact; Landmarks(...) evaluates a spatial null on
        representative positions: O(m + n r) per product with the residual (spectral), pairs
        drawn from the cells (sine, without exact near field).
    weight : str
        Edge attribute holding weights.
    random_state : int, optional
    """
    automatic = attribute is not None or isinstance(null, str)
    if automatic and approximation is not None:
        raise ValueError("attribute mode selects evaluation automatically; use fit_null first for advanced options")
    graph = as_indexed(G, weight)
    null = _attribute_null(graph, null, attribute, attribute_type, metric)
    if method is None:
        method = Spectral(normalized=True) if null.diagnostics.get("automatic") else Sine()
    settings = _METHODS[method]() if isinstance(method, str) and method in _METHODS else method
    if not isinstance(settings, tuple(_METHODS.values())):
        raise ValueError(f"method must be one of {sorted(_METHODS)} or a settings object")
    if int(dim) != dim or int(dim) < 1:
        raise ValueError("dim must be a positive integer")
    if not np.isfinite([gamma, epsilon]).all() or gamma < 0 or epsilon < 0:
        raise ValueError("gamma and epsilon must be finite and nonnegative")
    null = _aligned(null, graph)
    if graph.n < 2 or (null.diagnostics.get("fully_conditioned") and gamma == 1 and epsilon == 0):
        return {u: np.zeros(dim) for u in graph.nodes}
    spec = as_approximation(approximation)
    if isinstance(settings, Spectral):
        if spec is not None and not isinstance(spec, Landmarks):
            raise ValueError("embeddings support exact evaluation or Landmarks only")
        backend = _landmark_backend(null, graph, spec)[0] if spec is not None else _exact_backend(null)
        Z = _spectral(graph.adjacency, _product(backend, graph.n), gamma, epsilon, dim, settings.normalized)
        return {u: Z[i] for i, u in enumerate(graph.nodes)}
    start = None
    if isinstance(settings, Sine):
        if settings.init not in ("spectral", "random"):
            raise ValueError("init must be 'spectral' or 'random'")
        if settings.repulsion not in ("hinge", "log"):
            raise ValueError("repulsion must be 'hinge' or 'log'")
        if not settings.power > 0:
            raise ValueError("power must be positive")
        if settings.repulsion == "log" and settings.sampling != "proportional":
            raise ValueError("repulsion='log' needs sampling='proportional'")
        if settings.engine not in ("auto", "dense"):
            raise ValueError("engine must be 'auto' or 'dense'")
        if settings.sampling not in ("proportional", "softmax"):
            raise ValueError("sampling must be 'proportional' or 'softmax'")
        if spec is not None and not isinstance(spec, Landmarks):
            raise ValueError("embeddings support exact evaluation or Landmarks only")
        backend = _landmark_backend(null, graph, spec)[0] if spec is not None else _exact_backend(null)
        if settings.init == "spectral":
            start = _standardised(_spectral(graph.adjacency, _product(backend, graph.n), gamma, epsilon, dim, True),
                                  settings.init_scale, random_state)
    if isinstance(settings, Sine) and settings.sampling == "proportional" and settings.engine == "auto":
        if isinstance(backend, BlockBackend):
            Z = _sine_sampled(graph.adjacency, backend, gamma, epsilon, dim, settings, random_state, start)
            return {u: Z[i] for i, u in enumerate(graph.nodes)}
        if spec is not None:
            raise ValueError("sine with landmarks needs them without exact near field (no exact_within/exact_neighbors)")
    elif spec is not None:
        raise ValueError("approximations apply to method='spectral', or 'sine' with proportional sampling")
    R = graph.adjacency.toarray() - gamma * null.to_matrix() - epsilon
    R = (R + R.T) / 2
    np.fill_diagonal(R, 0.)
    if isinstance(settings, DeepWalk):
        Z = _deepwalk(R, dim, settings.window_size)
    elif isinstance(settings, Contrastive):
        Z = _contrastive(R, dim, settings, random_state)
    else:
        Z = _sine(R, dim, settings, random_state, start)
    return {u: Z[i] for i, u in enumerate(graph.nodes)}


def _standardised(Z, scale, random_state):
    """Centred coordinates with standard deviation ``scale``; a degenerate column (fewer
    positive eigenvalues than dimensions) is filled with small noise."""
    Z = Z - Z.mean(axis=0)
    sd = Z.std(axis=0)
    flat = sd <= 1e-12
    if flat.any():
        Z[:, flat] = np.random.default_rng(random_state).normal(size=(len(Z), int(flat.sum()))) * 1e-3
        sd[flat] = Z[:, flat].std(axis=0)
    return Z / sd * scale


def _eigen(matrix, dim, absolute=False):
    n = len(matrix)
    out = np.zeros((n, dim))
    if n < 2 or not np.any(matrix):
        return out
    k = min(dim, n)
    if n > 64 and k < n - 1:
        values, vectors = eigsh(matrix, k=k, which="LM" if absolute else "LA",
                                v0=np.random.default_rng(0).normal(size=n), tol=1e-10)
    else:
        values, vectors = eigh(matrix, check_finite=False)
    strength = np.abs(values) if absolute else values
    order = np.argsort(-strength, kind="stable")[:k]
    values, vectors, strength = values[order], vectors[:, order], strength[order]
    keep = strength > np.finfo(float).eps * n * max(1., float(np.abs(values).max()))
    vectors, strength = vectors[:, keep], strength[keep]
    if vectors.shape[1]:
        signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(vectors.shape[1])])
        out[:, :len(strength)] = vectors * signs * np.sqrt(strength)
    return out


def _product(backend, n):
    """x -> P x for a level-0 backend, P without self pairs, never forming P when avoidable."""
    if isinstance(backend, SumBackend):
        parts = [_product(p, n) for p in backend.parts]
        return lambda x: sum(f(x) for f in parts)
    if isinstance(backend, MatrixBackend):
        P, diagonal = backend.P, np.diag(backend.P)[:, None]
        return lambda x: P @ x - diagonal * x
    if isinstance(backend, BlockBackend):
        theta, B = backend.theta[:, None], backend.B
        S = sp.csr_array((np.ones(n), (np.arange(n), backend.cells)), shape=(n, B.shape[0]))
        self_term = backend.self_term[:, None]
        return lambda x: theta * (S @ (B @ (S.T @ (theta * x)))) - self_term * x
    if isinstance(backend, SparseBackend):
        return lambda x: backend.C @ x
    null, everyone = backend.null, np.arange(n)                # tiled: exact, O(n) memory

    def tiled(x):
        out = np.empty_like(x)
        for start in range(0, n, 1024):
            rows = everyone[start:start + 1024]
            out[rows] = null.block(rows, everyone) @ x
        return out
    return tiled


def _spectral(A, product, gamma, epsilon, dim, normalized=False):
    """Leading positive eigenvectors of R = A - gamma P - epsilon (off-diagonal), scaled by
    sqrt(eigenvalue): the continuous relaxation of the communities' objective. ``normalized``:
    eigenvectors of D^-1/2 R D^-1/2 mapped back by D^-1/2 (Laplacian eigenmaps of R)."""
    n = A.shape[0]
    A = sp.csr_array(A, dtype=float)
    degree = np.asarray(A.sum(axis=1)).ravel()
    # ARPACK cannot start on the zero operator. Nulls are nonnegative, so zero
    # row sums certify P = 0 without forming a dense matrix.
    if not A.nnz and (normalized or (epsilon == 0 and
                     (gamma == 0 or not np.any(product(np.ones((n, 1))))))):
        return np.zeros((n, dim))
    inverse = np.where(degree > 0, 1 / np.sqrt(np.maximum(degree, 1e-300)), 0.)[:, None] if normalized else 1.

    def residual(x):
        x = inverse * np.asarray(x, dtype=float).reshape(n, -1)
        out = A @ x - gamma * product(x)
        if epsilon:
            out -= epsilon * (x.sum(axis=0, keepdims=True) - x)
        return inverse * out
    if normalized:
        return _leading(residual, n, dim) * inverse
    if n <= 256:
        return _eigen(residual(np.eye(n)), dim)
    operator = LinearOperator((n, n), matvec=residual, matmat=residual, dtype=float)
    out = np.zeros((n, dim))
    k = min(dim, n - 2)
    values, vectors = eigsh(operator, k=k, which="LA", v0=np.random.default_rng(0).normal(size=n), tol=1e-8)
    order = np.argsort(-values, kind="stable")
    values, vectors = values[order], vectors[:, order]
    keep = values > np.finfo(float).eps * n * max(1., float(np.abs(values).max()))
    vectors, values = vectors[:, keep], values[keep]
    if len(values):
        signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(len(values))])
        out[:, :len(values)] = vectors * signs * np.sqrt(values)
    return out


def _leading(operator, n, dim):
    """Eigenvectors of the ``dim`` largest positive eigenvalues of a symmetric operator (unscaled)."""
    out = np.zeros((n, dim))
    if n <= 256:
        values, vectors = eigh(operator(np.eye(n)))
    else:
        op = LinearOperator((n, n), matvec=operator, matmat=operator, dtype=float)
        values, vectors = eigsh(op, k=min(dim, n - 2), which="LA", v0=np.random.default_rng(0).normal(size=n), tol=1e-8)
    order = np.argsort(-values, kind="stable")[:dim]
    values, vectors = values[order], vectors[:, order]
    keep = values > np.finfo(float).eps * n * max(1., float(np.abs(values).max()))
    vectors = vectors[:, keep]
    if vectors.shape[1]:
        signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(vectors.shape[1])])
        out[:, :vectors.shape[1]] = vectors * signs
    return out


def _deepwalk(R, dim, window_size):
    W = np.maximum(R, 0.)
    Z = np.zeros((len(R), dim))
    if not np.any(W):
        return Z
    W /= W.max()
    degree = W.sum(axis=1)
    T = np.divide(W, degree[:, None], out=np.zeros_like(W), where=degree[:, None] > 0)
    sparse_T = csr_matrix(T) if np.count_nonzero(T) < T.size / 4 else None
    power, cooc = T.copy(), T.copy()
    for _ in range(1, window_size):
        power = power @ T if sparse_T is None else (sparse_T.T @ power.T).T
        cooc += power
    cooc = (cooc + cooc.T) / 2
    np.fill_diagonal(cooc, 0.)
    total, mass = cooc.sum(), cooc.sum(axis=1)
    if total == 0:
        return Z
    rows, cols = np.nonzero(cooc)
    ppmi = np.zeros_like(cooc)
    ppmi[rows, cols] = np.maximum(np.log(cooc[rows, cols]) + np.log(total) - np.log(mass[rows]) - np.log(mass[cols]), 0.)
    Z = _eigen(ppmi, dim, absolute=True)
    Z[degree == 0] = 0.
    return Z


def _sine(R, dim, s, random_state, start=None):
    import torch
    import torch.nn.functional as F
    context = torch.random.fork_rng() if random_state is not None else nullcontext()
    with context:
        if random_state is not None:
            torch.manual_seed(random_state)
        n = len(R)
        if n < 2:
            return np.zeros((n, dim))
        Rt = torch.from_numpy(R).float()
        magnitude = Rt.abs()
        valid = magnitude > 1e-6
        if s.sampling == "proportional":
            probs = magnitude
        elif s.sampling == "softmax":
            probs = torch.where(valid, F.softmax(torch.where(valid, magnitude / s.temperature, -torch.inf), dim=1),
                                torch.zeros(()))
            probs = torch.nan_to_num(probs)
        else:
            raise ValueError("sampling must be 'proportional' or 'softmax'")
        empty = probs.sum(dim=1) <= 0                            # no residual: any other node
        if empty.all():
            return np.zeros((n, dim))
        mass = probs.sum(dim=1) if s.sampling == "proportional" else torch.ones(n)
        mass[empty] = 0.
        probs[empty] = 1.
        probs.fill_diagonal_(0.)
        probs = probs / probs.sum(dim=1, keepdim=True)
        rows = torch.arange(n).repeat_interleave(s.samples_per_node)
        weight = (mass / mass.mean().clamp_min(1e-300))[rows]
        initial = torch.randn(n, dim) * 0.1
        Z = torch.nn.Parameter(initial if start is None else torch.tensor(start, dtype=torch.float32))
        optimizer, schedule = _optimizer(Z, s)
        for _ in range(s.epochs):
            cols = torch.multinomial(probs, s.samples_per_node, replacement=True).reshape(-1)
            positive = (Rt[rows, cols] >= 0).float()
            loss = _pair_loss(Z, rows, cols, positive, weight, s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if schedule is not None:
                schedule.step()
        result = Z.detach().numpy().astype(float)
        result[empty.numpy()] = 0.
        return result


_SOFTENING = 1e-2      # ln d is evaluated as ln sqrt(d^2 + _SOFTENING^2)


def _optimizer(Z, s):
    """Adam at lr; with the log repulsion, a cosine decay to lr / 50 over the epochs."""
    import torch
    optimizer = torch.optim.Adam([Z], lr=s.lr)
    if s.repulsion != "log":
        return optimizer, None
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(s.epochs, 1), eta_min=s.lr / 50)


def _pair_loss(Z, source, partner, positive, weight, s):
    """Mean loss of drawn pairs: attraction d^power / power when positive, else max(0, margin - d)
    (hinge) or -ln d weighted by the source's residual mass (log)."""
    import torch
    if s.repulsion == "hinge":
        distance = torch.norm(Z[source] - Z[partner], dim=1)
        attraction = distance if s.power == 1 else distance ** s.power / s.power
        return ((weight > 0) * (positive * attraction + (1 - positive) * torch.clamp(s.margin - distance, min=0.))).mean()
    distance = torch.sqrt(((Z[source] - Z[partner]) ** 2).sum(dim=1) + _SOFTENING ** 2)
    return (weight * (positive * distance ** s.power / s.power - (1 - positive) * torch.log(distance))).mean()


class _BlockPairs:
    """Draws, for each source node i, a partner j with probability proportional to abs(R_ij),
    R = A - gamma P - epsilon and P_ij = theta_i theta_j B[c_i, c_j], without forming R.

    Edges are drawn within the row of A (weights abs(R_ij)); non-edges from the blocks (a cell
    with weight B[c_i, c] Theta_c, then a node with weight theta_j) or uniformly (epsilon), and
    redrawn when they hit i or a neighbour: an exact draw from abs(R_i.) restricted to non-edges.
    """

    def __init__(self, A, backend, gamma, epsilon, rng):
        A = sp.csr_array(A, dtype=float)
        A = A - sp.diags(A.diagonal(), format="csr")
        A.eliminate_zeros()
        n = A.shape[0]
        self.n, self.rng, self.indptr, self.cols = n, rng, A.indptr, A.indices
        self.B, self.cells, self.theta = backend.B, np.asarray(backend.cells), np.asarray(backend.theta, dtype=float)
        degree = np.diff(A.indptr)
        source = np.repeat(np.arange(n), degree)
        P_edge = self.theta[source] * self.theta[self.cols] * np.asarray(self.B[self.cells[source], self.cells[self.cols]]).ravel()
        residual = A.data - gamma * P_edge - epsilon
        self.edge_sign = np.where(residual >= 0, 1., -1.)
        weight = np.abs(residual)
        self.edge_cum = np.cumsum(weight)
        self.edge_base = np.r_[0., self.edge_cum][A.indptr[:-1]]
        self.edge_mass = np.bincount(source, weights=weight, minlength=n)
        r = self.B.shape[0]
        self.cell_mass = np.bincount(self.cells, weights=self.theta, minlength=r)        # Theta_c
        row_P = self.theta * (self.B @ self.cell_mass)[self.cells] - np.asarray(backend.self_term)
        edge_P = np.bincount(source, weights=P_edge, minlength=n)
        remaining = row_P - edge_P
        remaining[np.abs(remaining) <= 64 * np.finfo(float).eps * np.maximum(np.abs(row_P), edge_P)] = 0.
        block_mass = gamma * np.maximum(remaining, 0.)
        uniform_mass = epsilon * np.maximum(n - 1 - degree, 0)
        self.block_share = np.divide(block_mass, block_mass + uniform_mass, out=np.ones(n),
                                     where=block_mass + uniform_mass > 0)
        total = self.edge_mass + block_mass + uniform_mass
        self.mass = total                                                                 # sum_j |R_ij|
        self.edge_share = np.divide(self.edge_mass, total, out=np.zeros(n), where=total > 0)
        self.empty = total <= 0
        cell_weights = sp.csr_array(self.B).multiply(self.cell_mass[None, :]).tocsr()
        cell_weights.eliminate_zeros()
        self.cell_targets = cell_weights.indices
        self.cell_cum = np.cumsum(cell_weights.data)
        self.cell_base = np.r_[0., self.cell_cum][cell_weights.indptr[:-1]]
        self.cell_total = np.asarray(cell_weights.sum(axis=1)).ravel()
        self.order = np.argsort(self.cells, kind="stable")                               # node draws, within a cell
        self.node_cum = np.cumsum(self.theta[self.order])
        self.node_base = np.r_[0., self.node_cum][np.searchsorted(self.cells[self.order], np.arange(r))]
        self.keys = np.sort(source.astype(np.int64) * n + self.cols)

    def _non_edges(self, source):
        rng, out, todo = self.rng, np.empty(len(source), np.int64), np.arange(len(source))
        # Choose the component once. Redrawing it after rejection would favour the
        # component whose proposals hit fewer edges, changing the target distribution.
        from_block = rng.random(len(source)) < self.block_share[source]
        while len(todo):
            i = source[todo]
            j = rng.integers(0, self.n, len(todo))                                   # uniform part (epsilon)
            k = np.flatnonzero(from_block[todo])                                 # block part: cell, then node
            if len(k):
                ci = self.cells[i[k]]
                target = self.cell_base[ci] + rng.random(len(k)) * self.cell_total[ci]
                index = np.searchsorted(self.cell_cum, target, side="right")
                cell = self.cell_targets[np.minimum(index, len(self.cell_targets) - 1)]
                target = self.node_base[cell] + rng.random(len(k)) * self.cell_mass[cell]
                j[k] = self.order[np.minimum(np.searchsorted(self.node_cum, target, side="right"), self.n - 1)]
            key = i.astype(np.int64) * self.n + j
            hit = np.searchsorted(self.keys, key)
            bad = j == i
            if len(self.keys):
                bad |= (hit < len(self.keys)) & (self.keys[np.minimum(hit, len(self.keys) - 1)] == key)
            out[todo[~bad]] = j[~bad]
            todo = todo[bad]
        return out

    def draw(self, samples):
        source = np.repeat(np.arange(self.n), samples)
        partner, sign = np.empty(len(source), np.int64), np.empty(len(source))
        on_edge = (self.rng.random(len(source)) < self.edge_share[source]) & ~self.empty[source]
        e = np.flatnonzero(on_edge)
        if len(e):
            i = source[e]
            target = self.edge_base[i] + self.rng.random(len(e)) * self.edge_mass[i]
            index = np.clip(np.searchsorted(self.edge_cum, target, side="right"), self.indptr[i], self.indptr[i + 1] - 1)
            partner[e], sign[e] = self.cols[index], self.edge_sign[index]
        rest = np.flatnonzero(~on_edge & ~self.empty[source])
        partner[rest], sign[rest] = self._non_edges(source[rest]), -1.
        none = np.flatnonzero(self.empty[source])                                    # no residual: any other node
        if len(none):
            shift = self.rng.integers(1, self.n, len(none))
            partner[none], sign[none] = (source[none] + shift) % self.n, 1.
        return source, partner, sign


def _sine_sampled(A, backend, gamma, epsilon, dim, s, random_state, start=None):
    """SiNE with pairs drawn by _BlockPairs (same distribution as the dense proportional rule)."""
    import torch
    n = A.shape[0]
    if n < 2:
        return np.zeros((n, dim))
    pairs = _BlockPairs(A, backend, gamma, epsilon, np.random.default_rng(random_state))
    if pairs.empty.all():
        return np.zeros((n, dim))
    context = torch.random.fork_rng() if random_state is not None else nullcontext()
    with context:
        if random_state is not None:
            torch.manual_seed(random_state)
        initial = torch.randn(n, dim) * 0.1
        Z = torch.nn.Parameter(initial if start is None else torch.tensor(start, dtype=torch.float32))
        optimizer, schedule = _optimizer(Z, s)
        node_weight = pairs.mass / max(pairs.mass.mean(), 1e-300)
        for _ in range(s.epochs):
            source, partner, sign = pairs.draw(s.samples_per_node)
            weight = torch.from_numpy(node_weight[source]).float()
            source, partner = torch.from_numpy(source), torch.from_numpy(partner)
            positive = torch.from_numpy((sign + 1) / 2).float()
            loss = _pair_loss(Z, source, partner, positive, weight, s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if schedule is not None:
                schedule.step()
        result = Z.detach().numpy().astype(float)
        result[pairs.empty] = 0.
        return result


def _contrastive(R, dim, s, random_state):
    import torch
    out = np.zeros((len(R), dim))
    active = np.flatnonzero(np.any(R != 0, axis=1))
    local = R[np.ix_(active, active)]
    left, right = np.triu_indices(len(active), 1)
    residual = local[left, right]
    positive, negative = np.flatnonzero(residual > 0), np.flatnonzero(residual < 0)
    if not len(positive) or not len(negative):
        return out
    if s.batch_size < 2 or s.batch_size % 2:
        raise ValueError("batch_size must be even and at least 2")

    def cdf(indices):
        p = np.abs(residual[indices])
        c = np.cumsum(p / p.sum())
        c[-1] = 1.
        return c

    pos_cdf, neg_cdf = cdf(positive), cdf(negative)
    rng = np.random.default_rng(random_state)
    generator = torch.Generator().manual_seed(int(rng.integers(0, 2 ** 63 - 1)))
    left_t, right_t = torch.from_numpy(left), torch.from_numpy(right)

    def normalize(raw):
        centered = raw - raw.mean(dim=0, keepdim=True)
        return centered / centered.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)

    def objective(z):
        pos_p, neg_p = np.diff(np.r_[0, pos_cdf]), np.diff(np.r_[0, neg_cdf])
        dp = np.linalg.norm(z[left[positive]] - z[right[positive]], axis=1)
        dn = np.linalg.norm(z[left[negative]] - z[right[negative]], axis=1)
        return float(pos_p @ dp ** 2 + neg_p @ np.maximum(s.margin - dn, 0.) ** 2)

    best = np.inf
    for _ in range(s.n_restarts):
        raw = torch.nn.Parameter(torch.randn(len(active), dim, generator=generator))
        optimizer = torch.optim.Adam([raw], lr=s.lr)
        for _ in range(s.epochs):
            pos = positive[np.searchsorted(pos_cdf, rng.random(s.batch_size // 2), side="right")]
            neg = negative[np.searchsorted(neg_cdf, rng.random(s.batch_size // 2), side="right")]
            z = normalize(raw)
            dp = torch.linalg.vector_norm(z[left_t[pos]] - z[right_t[pos]], dim=1)
            dn = torch.linalg.vector_norm(z[left_t[neg]] - z[right_t[neg]], dim=1)
            loss = (dp ** 2).mean() + (torch.clamp(s.margin - dn, min=0.) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        fitted = normalize(raw).detach().numpy().astype(float)
        value = objective(fitted)
        if value < best:
            best, out[active] = value, fitted
    return out
