"""Synthetic benchmarks: a community layer hidden under a spatial layer.

The observed graph is the union S | B of a signal layer S (planted communities or any
block model) and a spatial layer B, whose edge probability p(d) decreases with distance.
A benchmark also returns its oracle null P_uv = q + (1 - q) p(d_uv), where q is the
overall density of the signal layer: the expected graph when only the spatial layer and
the amount of signal are known, not the communities.

Both layers can be degree-corrected by node masses theta: a pair then has
theta_u theta_v independent chances of being linked (Poisson contacts).
"""
import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.special import expit

from ._geometry import distance_matrix
from .nulls import NullModel, SpatialNull, custom_null


# ----------------------------------------------------------------------------- spatial probabilities
@dataclass(frozen=True)
class ExponentialNoise:
    """p(d) = amplitude * exp(-d / length)."""
    amplitude: float
    length: float

    def __call__(self, d):
        return self.amplitude * np.exp(-np.asarray(d) / self.length)


@dataclass(frozen=True)
class LogisticNoise:
    """p(d) = sigmoid(intercept - beta * d / length). beta = 0 gives uniform noise."""
    intercept: float
    beta: float
    length: float = 1.0

    def __call__(self, d):
        return expit(self.intercept - self.beta * np.asarray(d) / self.length)


@dataclass(frozen=True)
class ContactNoise:
    """Spatial contacts plus uniform ones; a pair is linked if it has at least one contact.

    Contacts are Poisson with intensity
    lambda(d) = amplitude * ((1 - share) + share * exp(-d / length) / mean_kernel),
    and p(d) = 1 - exp(-lambda(d)). ``share`` is the spatial part of the noise (0: uniform
    noise, 1: purely spatial at range ``length``); ``mean_kernel`` makes both parts carry the
    same mean intensity. The probability stays smooth for any amount of noise: its slope is
    at most 1 / (e * length), so no saturated disk with a sharp edge can form.
    """
    amplitude: float
    share: float
    length: float
    mean_kernel: float = 1.0

    def __call__(self, d):
        spatial = np.exp(-np.asarray(d) / self.length) / self.mean_kernel
        return -np.expm1(-self.amplitude * ((1 - self.share) + self.share * spatial))


@dataclass(frozen=True)
class UnionNull:
    """Deterrence of the oracle null: q + (1 - q) p(d)."""
    q: float
    noise: object

    def __call__(self, d):
        return self.q + (1 - self.q) * self.noise(d)


@dataclass(frozen=True)
class DegreeCorrectedUnion:
    """Oracle of a degree-corrected benchmark, as a probability of pairs of nodes:
    P_uv = 1 - exp(-q theta_u theta_v) (1 - p(d_uv)) ** (theta_u theta_v).
    Not a product theta_u theta_v f(d): use it with the exact engine only."""
    q: float
    noise: object
    positions: np.ndarray
    degrees: np.ndarray

    def __call__(self, I, J):
        I, J = np.asarray(I), np.asarray(J)
        w = self.degrees[I] * self.degrees[J]
        p = self.noise(np.linalg.norm(self.positions[I] - self.positions[J], axis=-1))
        return np.where(I == J, 0., 1 - np.exp(-self.q * w) * (1 - _rounds(p, w)))


@dataclass
class Benchmark:
    """A generated graph with its planted communities and oracle null."""
    adjacency: sp.csr_array
    labels: np.ndarray
    positions: np.ndarray
    null: NullModel
    info: dict = field(default_factory=dict)

    def to_networkx(self):
        return nx.from_scipy_sparse_array(self.adjacency)


# ----------------------------------------------------------------------------- layers
def planted_communities(n, *, communities=None, exponent=0.6, seed=None):
    """Signal layer: connected random-regular communities, no edge between them.

    By default K = round(sqrt(n) / 2) communities. Sizes vary around n / K following a
    stratified truncated power law (density x^-2 on [1/2, 2], normalised). A community of
    s nodes has the even-compatible degree closest to (s - 1) ** exponent.
    Returns (adjacency, labels).
    """
    rng = np.random.default_rng(seed)
    k = communities or max(1, round(math.sqrt(n) / 2))
    u = (np.arange(k) + rng.random(k)) / k
    target = n / (2 - 1.5 * u)
    target *= n / target.sum()
    sizes = np.floor(target).astype(int)
    sizes[np.argsort(-(target - sizes), kind="stable")[:n - sizes.sum()]] += 1
    rng.shuffle(sizes)
    if sizes.min() < 3:
        raise ValueError("communities must have at least 3 nodes; use fewer communities")
    labels = np.repeat(np.arange(k), sizes)
    rows, cols, offset = [], [], 0
    for s in sizes:
        degree = min((d for d in range(1, s) if (d * s) % 2 == 0), key=lambda d: (abs(d - (s - 1) ** exponent), d))
        for _ in range(100):
            g = nx.random_regular_graph(degree, int(s), seed=int(rng.integers(2 ** 31)))
            if nx.is_connected(g):
                break
        else:
            raise RuntimeError("could not draw a connected community")
        e = np.array(g.edges(), dtype=np.int64) + offset
        rows.append(e[:, 0])
        cols.append(e[:, 1])
        offset += s
    return _symmetric(np.concatenate(rows), np.concatenate(cols), n), labels


def block_model(labels, block_matrix, *, degrees=None, seed=None):
    """Signal layer from a stochastic block model.

    Without ``degrees``: Bernoulli, P(u ~ v) = B[label_u, label_v].
    With ``degrees`` theta (degree-corrected): u and v share Poisson(theta_u theta_v B) contacts
    and are linked if they have at least one, P(u ~ v) = 1 - exp(-theta_u theta_v B[label_u, label_v]).
    B then holds intensities, which may exceed 1. Exact sampling in O(n + contacts).
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    B = np.asarray(block_matrix, dtype=float)
    members = [np.flatnonzero(labels == b) for b in range(len(B))]
    if degrees is not None:
        return _contacts(members, B, np.asarray(degrees, dtype=float), rng, len(labels))
    rows, cols = [], []
    for r in range(len(B)):
        for s in range(r, len(B)):
            a, b = members[r], members[s]
            pairs = len(a) * (len(a) - 1) // 2 if r == s else len(a) * len(b)
            k = rng.binomial(pairs, B[r, s]) if pairs else 0
            if not k:
                continue
            chosen = rng.choice(pairs, k, replace=False)
            if r == s:
                i, j = _triangle(chosen)
                rows.append(a[i]), cols.append(a[j])
            else:
                rows.append(a[chosen // len(b)]), cols.append(b[chosen % len(b)])
    if not rows:
        return sp.csr_array((len(labels), len(labels)))
    return _symmetric(np.concatenate(rows), np.concatenate(cols), len(labels))


def _contacts(members, B, theta, rng, n):
    """Degree-corrected block model: Poisson contacts, endpoints drawn with probability
    proportional to theta inside each block, repeated contacts merged."""
    mass = [theta[a].sum() for a in members]
    rows, cols = [], []
    for r in range(len(B)):
        for s in range(r, len(B)):
            a, b = members[r], members[s]
            if not (mass[r] > 0 and mass[s] > 0 and B[r, s] > 0):
                continue
            k = rng.poisson(B[r, s] * mass[r] * mass[s] / (2 if r == s else 1))
            u = rng.choice(a, k, p=theta[a] / mass[r])
            v = rng.choice(b, k, p=theta[b] / mass[s])
            keep = u != v                                     # each unordered pair: theta_u theta_v B contacts
            rows.append(u[keep]), cols.append(v[keep])
    return _symmetric(rows, cols, n)


def _rounds(p, weight):
    """1 - (1 - p) ** weight: probability of at least one success in ``weight`` rounds."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(weight > 0, -np.expm1(weight * np.log1p(-p)), 0.)


def spatial_layer(positions, probability, *, degrees=None, seed=None, cell=None):
    """Spatial layer: each pair is linked independently with probability p(d).

    Parameters
    ----------
    positions : array of shape (n, dim)
    probability : callable
        Nonincreasing function of the Euclidean distance, values in [0, 1].
    degrees : array, optional
        Node masses theta (degree correction). A pair then has theta_u theta_v independent
        chances: P(u ~ v) = 1 - (1 - p(d)) ** (theta_u theta_v). For contact noise,
        p = 1 - exp(-lambda(d)), this is 1 - exp(-theta_u theta_v lambda(d)), a gravity model.
    seed : int, optional
    cell : float, optional
        Side of the cells of the rejection sampler (default: about 16 nodes per cell).

    Small graphs draw one uniform mark per pair in a fixed order, so layers drawn with the
    same seed are nested when p increases. Large graphs use exact rejection sampling over
    a grid of cells.
    """
    X = np.asarray(positions, dtype=float)
    n = len(X)
    theta = None if degrees is None else np.asarray(degrees, dtype=float)
    rng = np.random.default_rng(seed)
    if n * (n - 1) / 2 <= 2e7:
        rows, cols = [], []
        for start in range(0, n - 1, 512):
            i = np.arange(start, min(start + 512, n - 1))
            P = probability(distance_matrix(X[i], X, "euclidean"))
            if theta is not None:
                P = _rounds(P, theta[i][:, None] * theta[None, :])
            marks = rng.random(P.shape)
            hit = (marks < P) & (np.arange(n)[None, :] > i[:, None])
            a, b = np.nonzero(hit)
            rows.append(i[a]), cols.append(b)
        return _symmetric(rows, cols, n)
    return _rejection(X, probability, rng, cell, theta)


def _rejection(X, probability, rng, cell, theta=None):
    lo = X.min(axis=0)
    side = cell or math.sqrt(np.prod(np.ptp(X, axis=0) + 1e-12) * 16 / len(X))
    grid = np.floor((X - lo) / side).astype(np.int64)
    width = grid[:, 1].max() + 1
    ids = grid[:, 0] * width + grid[:, 1]
    order = np.argsort(ids, kind="stable")
    occupied, counts = np.unique(ids, return_counts=True)
    groups = np.split(order, np.cumsum(counts)[:-1])
    xy = np.c_[occupied // width, occupied % width]
    top = np.ones(len(groups)) if theta is None else np.array([theta[g].max() for g in groups])
    linked = (lambda a, b, p: p) if theta is None else (lambda a, b, p: _rounds(p, theta[a] * theta[b]))
    rows, cols = [], []

    def accept(a, b, bound):
        d = np.linalg.norm(X[a] - X[b], axis=1)
        keep = rng.random(len(a)) * bound < linked(a, b, probability(d))
        rows.append(a[keep]), cols.append(b[keep])

    for c, members in enumerate(groups):                      # pairs inside a cell
        pairs = len(members) * (len(members) - 1) // 2
        bound = float(probability(0.)) if theta is None else float(_rounds(probability(0.), top[c] ** 2))
        k = rng.binomial(pairs, bound) if pairs else 0
        if k:
            i, j = _triangle(rng.choice(pairs, k, replace=False))
            accept(members[i], members[j], bound)
    left, right = np.triu_indices(len(groups), 1)              # pairs across cells
    gap = np.maximum(np.abs(xy[left] - xy[right]) - 1, 0) * side
    bounds = probability(np.linalg.norm(gap, axis=1))
    if theta is not None:
        bounds = _rounds(bounds, top[left] * top[right])
    trials = rng.binomial(counts[left] * counts[right], bounds)
    for t in np.flatnonzero(trials):
        a, b = groups[left[t]], groups[right[t]]
        chosen = rng.choice(len(a) * len(b), trials[t], replace=False)
        accept(a[chosen // len(b)], b[chosen % len(b)], float(bounds[t]))
    return _symmetric(rows, cols, len(X))


def _triangle(k):
    """Index k of the pairs (j, i), j < i, listed row by row -> (j, i)."""
    i = np.floor((np.sqrt(8 * np.asarray(k, dtype=float) + 1) + 1) / 2).astype(np.int64)
    i -= (i * (i - 1) // 2 > k)
    i += ((i + 1) * i // 2 <= k)
    return k - i * (i - 1) // 2, i


def _symmetric(rows, cols, n):
    if isinstance(rows, list):
        rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    A = sp.coo_array((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.
    return sp.csr_array(A)


# ----------------------------------------------------------------------------- calibration
def mean_proximity(length, order=192):
    """E[exp(-|X - Y| / length)] for X, Y independent and uniform in the unit square."""
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = (x + 1) / 2, w / 2
    d = np.hypot(x[:, None], x[None, :])
    return float(4 * np.sum(w[:, None] * w[None, :] * (1 - x[:, None]) * (1 - x[None, :]) * np.exp(-d / length)))


def _square_density(d):
    """Density of the distance between two independent uniform points of the unit square."""
    d = np.asarray(d, dtype=float)
    near = 2 * d * (d * d - 4 * d + np.pi)
    far = 2 * d * (4 * np.sqrt(np.maximum(d * d - 1, 0)) - (d * d + 2 - np.pi)
                   - 4 * np.arccos(np.minimum(1, 1 / np.maximum(d, 1e-300))))
    return np.where(d <= 1, near, np.where(d <= np.sqrt(2), far, 0.))


def _square_mean(g, scale):
    """E[g(|X - Y|)] for X, Y independent and uniform in the unit square; g varies at ``scale``."""
    from scipy.integrate import quad
    f = lambda d: g(d) * _square_density(d)
    points = [scale * x for x in (1, 5, 20) if scale * x < 1] or None
    return quad(f, 0, 1, points=points, limit=200)[0] + quad(f, 1, np.sqrt(2), limit=200)[0]


def calibrate(positions, noise_of, target, low, high, *, degrees=None, pair_weights=None):
    """Parameter x in [low, high] such that the expected number of edges of noise_of(x),
    sum_{u<v} w_uv P_uv, equals target. P_uv = p(d_uv), or 1 - (1 - p(d_uv)) ** (theta_u theta_v)
    with ``degrees`` theta (see :func:`spatial_layer`); p must increase with x.
    pair_weights(rows, cols) returns the weights of a block of pairs (default 1), e.g. the
    probability that a pair is not already a signal edge."""
    X = np.asarray(positions, dtype=float)
    n = len(X)
    everyone = np.arange(n)
    theta = None if degrees is None else np.asarray(degrees, dtype=float)

    def excess(x):
        p = noise_of(x)
        total = 0.
        for start in range(0, n - 1, 512):
            rows = np.arange(start, min(start + 512, n - 1))
            P = p(distance_matrix(X[rows], X, "euclidean"))
            if theta is not None:
                P = _rounds(P, theta[rows][:, None] * theta[None, :])
            w = (everyone[None, :] > rows[:, None]).astype(float)
            if pair_weights is not None:
                w *= pair_weights(rows, everyone)
            total += float((P * w).sum())
        return total - target
    return float(brentq(excess, low, high, xtol=1e-12, rtol=1e-12))


def calibrate_logistic(positions, beta, target, *, length=1.0, pair_weights=None):
    """Intercept such that sum_{u<v} w_uv sigmoid(intercept - beta d_uv / length) = target.

    pair_weights(rows, cols) returns the weights of a block of pairs (default 1), e.g. the
    probability that a pair is not already a signal edge. Safeguarded Newton iterations;
    pairs are visited in tiles, so memory stays O(n).
    """
    X = np.asarray(positions, dtype=float)
    n = len(X)
    everyone = np.arange(n)

    def excess(t):
        value = slope = 0.
        for start in range(0, n - 1, 512):
            rows = np.arange(start, min(start + 512, n - 1))
            p = expit(t - beta * distance_matrix(X[rows], X, "euclidean") / length)
            w = (everyone[None, :] > rows[:, None]).astype(float)
            if pair_weights is not None:
                w *= pair_weights(rows, everyone)
            value += float((p * w).sum())
            slope += float((p * (1 - p) * w).sum())
        return value - target, slope

    low, high = -60., 60. + beta * float(np.linalg.norm(np.ptp(X, axis=0))) / length
    t = 0.
    for _ in range(200):
        f, df = excess(t)
        if abs(f) <= 1e-9 * max(target, 1.):
            return float(t)
        low, high = (low, t) if f > 0 else (t, high)
        newton = t - f / df if df > 0 else None
        t = newton if newton is not None and low < newton < high else (low + high) / 2
    raise RuntimeError("calibration did not converge")


# ----------------------------------------------------------------------------- benchmarks
def mixed_benchmark(signal, labels, positions, noise, *, degrees=None, seed=None, q=None, cell=None):
    """Union of a signal layer and a spatial layer drawn with probability ``noise``.

    q defaults to the observed signal density; the oracle null is q + (1 - q) noise(d).
    With ``degrees`` theta, the spatial layer is degree-corrected (see :func:`~disentangled_net.synthetic.spatial_layer`)
    and q is a signal intensity, by default (signal edges) / sum_{u<v} theta_u theta_v. The
    oracle is then :class:`~disentangled_net.synthetic.DegreeCorrectedUnion`, which is not a product theta_u theta_v f(d):
    approximations such as landmarks need a fitted null (:func:`fit_spatial_null`).
    """
    n = signal.shape[0]
    B = spatial_layer(positions, noise, degrees=degrees, seed=seed, cell=cell)
    A = signal + B
    A.data[:] = 1.
    if degrees is None:
        q = float(signal.sum() / (n * (n - 1))) if q is None else q
        null = SpatialNull(positions, UnionNull(q, noise), metric="euclidean")
    else:
        theta = np.asarray(degrees, dtype=float)
        q = float(signal.sum() / (theta.sum() ** 2 - (theta ** 2).sum())) if q is None else q
        oracle = DegreeCorrectedUnion(q, noise, np.asarray(positions, dtype=float), theta)
        null = custom_null(oracle, range(n), kind="probability", vectorized=True)
    m_signal, m_noise, m_all = signal.sum() / 2, B.sum() / 2, A.sum() / 2
    info = dict(n=n, communities=int(len(np.unique(labels))), signal_edges=int(m_signal), noise_edges=int(m_noise),
                observed_edges=int(m_all), q=q, added_fraction=float((m_all - m_signal) / m_all) if m_all else 0.)
    return Benchmark(sp.csr_array(A), np.asarray(labels), np.asarray(positions), null, info)


def spatial_benchmark(n, *, noise="contact", rho=3.0, share=1.0, beta=8.0, communities=None, exponent=0.6, seed=None):
    """The benchmark of the paper: planted communities under spatial noise in the unit square.

    Signal: :func:`~disentangled_net.synthetic.planted_communities`. Positions: independent and uniform in the unit
    square. Noise range: length l_n such that a node's weighted number of close nodes,
    (n - 1) E[exp(-d / l_n)], equals the mean community size minus one.

    Parameters
    ----------
    n : int
        Number of nodes.
    noise : {'contact', 'exponential', 'logistic'}
        'contact' (default): :class:`~disentangled_net.synthetic.ContactNoise` at range l_n,
        as in the article. 'exponential': p(d) = a exp(-d / l_n), needs a < 1.
        'logistic': p(d) = sigmoid(eta - beta d / l_n). With large rho and beta it saturates into a disk of
        probability 1 with a sharp edge.
    rho : float
        Expected number of noise edges divided by the number of signal edges. Above 2e7 pairs,
        contact noise is calibrated on the distance law of uniform positions rather than on the
        drawn positions (the exact sum costs O(n^2)); the difference is negligible at that size.
    share : float
        Contact noise only: spatial part of the noise, from 0 (uniform) to 1 (purely spatial).
    beta : float
        Logistic noise only: how much the noise prefers close pairs (0: uniform).
    communities, exponent : number of communities (default round(sqrt(n) / 2)) and the exponent
        of their internal degree, about (size - 1) ** exponent (see :func:`~disentangled_net.synthetic.planted_communities`).
        Fewer, larger and sparser communities hide better under the spatial noise.
    seed : int
        The same seed gives the same signal and positions for every rho and beta. Below
        2e7 pairs, noise marks are also shared, so layers are nested when probabilities increase.
    """
    rng = np.random.default_rng(seed)
    signal, labels = planted_communities(n, communities=communities, exponent=exponent, seed=int(rng.integers(2 ** 31)))
    positions = rng.random((n, 2))
    noise_seed = int(rng.integers(2 ** 31))
    m_signal = signal.sum() / 2
    mean_size = n / len(np.unique(labels))
    length = brentq(lambda l: (n - 1) * mean_proximity(l) - (mean_size - 1), 1e-6, 10., xtol=1e-14)
    pairs = n * (n - 1) / 2
    if rho == 0:
        p = ExponentialNoise(0., length)
    elif noise == "contact":
        mean = mean_proximity(length)
        shape = lambda x: ContactNoise(np.exp(x), share, length, mean)
        if pairs <= 2e7:
            log_amplitude = calibrate(positions, shape, rho * m_signal, -30., 30.)
        else:                   # the exact sum costs O(n^2): use the distance law of uniform positions
            log_amplitude = brentq(lambda x: pairs * _square_mean(shape(x), length) - rho * m_signal, -30., 30.,
                                   xtol=1e-12)
        p = ContactNoise(np.exp(log_amplitude), share, length, mean)
    elif noise == "exponential":
        a = rho * (2 * m_signal / n) / (mean_size - 1)
        if not 0 < a < 1:
            raise ValueError(f"amplitude {a:.3g} is not a probability: lower rho")
        p = ExponentialNoise(a, length)
    elif noise == "logistic":
        p = LogisticNoise(calibrate_logistic(positions, beta, rho * m_signal, length=length), beta, length)
    else:
        raise ValueError("noise must be 'contact', 'exponential' or 'logistic'")
    bench = mixed_benchmark(signal, labels, positions, p, seed=noise_seed, q=2 * m_signal / n / (n - 1), cell=length)
    sizes = np.bincount(labels)
    degrees = np.bincount(labels, weights=np.asarray(signal.sum(axis=1)).ravel()) / sizes
    bench.info.update(rho=rho, beta=beta if noise == "logistic" else None, share=share if noise == "contact" else None,
                      noise=noise, length=length,
                      community_density=(degrees / (sizes - 1)).tolist())
    return bench
