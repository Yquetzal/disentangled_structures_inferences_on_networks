"""Fitting null models to a graph."""
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from ._geometry import ExactGeometry, LandmarkGeometry, bin_index, pair_distances
from ._graph import as_indexed, node_values
from .approximations import Landmarks
from .nulls import Binned, BlockNull, Exponential, SpatialNull


AUTO_EXACT_LIMIT = 2000
AUTO_LANDMARKS = 128
AUTO_DISTINCT_LIMIT = 256


def fit_null(G, attribute, *, attribute_type="auto", metric="euclidean", weight="weight"):
    """Fit a ready-to-use null for an attribute, choosing compatible computations automatically.

    Parameters
    ----------
    G : networkx graph or scipy.sparse adjacency matrix.
    attribute : node attribute name, mapping, or values aligned with graph nodes.
        Numbers/vectors are continuous; strings and booleans are categorical.
    attribute_type : 'auto', 'continuous' or 'categorical'.
        Use 'categorical' for integer category codes; their meaning cannot be inferred.
    metric : 'euclidean' (default), 'haversine' for (latitude, longitude) in degrees,
        or a distance-matrix callable. Only used for continuous attributes.
    weight : edge weight attribute (default 'weight').

    Continuous values use a degree-corrected binned Poisson null with 20 equal-link bins.
    Up to 2000 nodes, fitting is exact. Repeated positions with at most 256 distinct values
    are compressed exactly. Larger Euclidean/haversine inputs use 128 landmarks, fitted
    and returned as a block null: reuse it directly in communities and embeddings without
    choosing an approximation again. A custom metric uses exact evaluation.
    Categorical values use the degree-corrected SBM. Inspect ``null.diagnostics`` for
    the choice and convergence. Advanced fitting options remain in fit_spatial_null.
    """
    graph = as_indexed(G, weight)
    values = node_values(graph, attribute, "attribute", dtype=None)
    if attribute_type not in ("auto", "continuous", "categorical"):
        raise ValueError("attribute_type must be 'auto', 'continuous' or 'categorical'")
    if attribute_type == "auto":
        attribute_type = "continuous" if values.dtype.kind in "iuf" else "categorical"
    metadata = dict(automatic=True, attribute_type=attribute_type,
                    default_epsilon=2 * graph.total_weight / (graph.n * (graph.n - 1)) if graph.n > 1 else 0.)
    if attribute_type == "categorical":
        if values.ndim != 1 or any(v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)) for v in values):
            raise ValueError("categorical attributes need one non-missing label per node")
        null = fit_sbm_null(graph, values)
        null.diagnostics.update(metadata, approximation="exact")
        null.diagnostics["fully_conditioned"] = len(null.diagnostics["blocks"]) == graph.n
        return null
    try:
        X = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("continuous attributes must contain finite numbers or vectors") from error
    X = X[:, None] if X.ndim == 1 else X
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("continuous attributes must contain finite numbers or vectors")
    if metric == "haversine" and X.shape[1] != 2:
        raise ValueError("haversine needs (latitude, longitude) pairs in degrees")
    if not callable(metric) and metric not in ("euclidean", "haversine"):
        raise ValueError("metric must be 'euclidean', 'haversine' or a distance-matrix callable")
    if graph.total_weight <= 0:
        return BlockNull(np.zeros(graph.n, np.intp), [[0.]], nodes=graph.nodes,
                         diagnostics=dict(metadata, approximation="exact", converged=True))
    representatives, cells = np.unique(X, axis=0, return_inverse=True)
    if len(representatives) == 1:
        null = fit_configuration_null(graph)
        null.diagnostics.update(metadata, approximation="exact", constant_attribute=True, converged=True)
        return null
    compressed = len(representatives) < graph.n and len(representatives) <= AUTO_DISTINCT_LIMIT
    approximate = graph.n > AUTO_EXACT_LIMIT and not compressed and metric in ("euclidean", "haversine")
    if compressed:
        geometry = LandmarkGeometry.repeated_positions(X, metric, representatives, cells)
    elif approximate:
        geometry = LandmarkGeometry(X, metric, AUTO_LANDMARKS, weights=graph.degrees)
    else:
        geometry = ExactGeometry(X, metric)
    settings = dict(deterrence="binned", likelihood="poisson", degree_correction=True,
                    tol=1e-3, max_iter=1000, metric=metric,
                    edges=_bin_edges(X, metric, 20, "edges", graph))
    masses, det, diagnostics = fit_on_geometry(graph, geometry, settings)
    _warn_unconverged(diagnostics)
    diagnostics.update(metadata,
                       approximation="landmarks" if approximate else "exact",
                       compressed=compressed)
    if isinstance(geometry, LandmarkGeometry):
        diagnostics["rank"] = geometry.rank
        return BlockNull(geometry.cells, det(geometry.rep_distance), masses, nodes=graph.nodes,
                         diagnostics=diagnostics)
    return SpatialNull(X, det, masses, metric=metric, nodes=graph.nodes, fit_settings=settings,
                       diagnostics=diagnostics)


def _attribute_null(graph, null, attribute, attribute_type, metric):
    """Resolve the one-call attribute interface without changing the explicit-null API."""
    if attribute is not None:
        if null is not None:
            raise ValueError("provide either null or attribute, not both")
        return fit_null(graph, attribute, attribute_type=attribute_type, metric=metric)
    if isinstance(null, str):
        return fit_null(graph, null, attribute_type=attribute_type, metric=metric)
    if null is None:
        raise ValueError("provide a node attribute with attribute='name', or a fitted null model")
    return null


def fit_spatial_null(G, positions="pos", *, metric="euclidean", deterrence="binned", bins=10,
                     binning="quantile", degree_correction=True, likelihood="poisson", landmarks=None,
                     groups=None, group_effect="individual", weight="weight", tol=1e-3, max_iter=500):
    """Fit a spatial null from node positions.

    Parameters
    ----------
    G : networkx graph or scipy.sparse adjacency matrix.
    positions : node attribute name, {node: position} or array aligned with the nodes.
    metric : 'euclidean', 'haversine' ((lat, lon) in degrees, distances in km) or a callable
        metric(X, Y) returning the matrix of distances between the rows of X and Y.
    deterrence : 'binned' (one free value per distance bin) or 'exponential' (f = c exp(-beta d)).
    bins : number of bins, or explicit bin edges starting at 0 (metric units).
    binning : with a number of bins: 'quantile' (equal pair counts), 'edges' (equal observed
        edge counts, as in the article; weighted quantiles for weighted graphs), 'linear' or 'log'.
    degree_correction : one mass per node, fitted so that expected degrees match the
        observed ones. False: no masses.
    likelihood : 'poisson' (expected weights, any nonnegative weights) or 'bernoulli'
        (probabilities, binary graphs, logit link).
    landmarks : None (exact distances, O(n^2) time per iteration, O(n) memory), an int r or
        a :class:`Landmarks`: fit on the approximate geometry, O(n (r + k)) per iteration.
        Poisson likelihood only. The returned null still uses exact distances unless an
        approximation is requested when partitioning.
    tol, max_iter : convergence of the fit (relative degree error tol in L1, 20 tol at most).
    groups : optional node attribute, mapping or array of categorical labels.
        Also fit within-group factors: P_uv = theta_u theta_v f(d_uv) b_c inside group c,
        unchanged across groups. Binned Poisson only, exact evaluation, O(n^2) memory.
    group_effect : 'individual' (one factor per group) or 'shared' (one within-group factor).

    Moments matched at convergence: degrees (with degree correction), and per-bin edge
    masses (binned) or total mass and mean distance (exponential).
    """
    graph = as_indexed(G, weight)
    if graph.total_weight <= 0:
        raise ValueError("the graph has no edges")
    if not np.isfinite(tol) or tol <= 0 or int(max_iter) != max_iter or max_iter < 0:
        raise ValueError("tol must be positive and max_iter a nonnegative integer")
    X = node_values(graph, positions, "positions")
    X = X[:, None] if X.ndim == 1 else X
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("positions must be finite scalars or vectors")
    if likelihood not in ("poisson", "bernoulli"):
        raise ValueError("likelihood must be 'poisson' or 'bernoulli'")
    if deterrence not in ("binned", "exponential"):
        raise ValueError("deterrence must be 'binned' or 'exponential'")
    settings = dict(deterrence=deterrence, likelihood=likelihood, degree_correction=bool(degree_correction),
                    tol=tol, max_iter=max_iter, metric=metric,
                    edges=_bin_edges(X, metric, bins, binning, graph) if deterrence == "binned" else None)
    geometry = ExactGeometry(X, metric)
    if groups is not None:
        if landmarks is not None or likelihood != "poisson" or deterrence != "binned":
            raise ValueError("groups require exact evaluation and a binned Poisson null")
        labels = node_values(graph, groups, "groups", dtype=None)
        masses, det, factors, diagnostics = _fit_group_binned(graph, geometry, settings, labels, group_effect)
        _warn_unconverged(diagnostics)
        return SpatialNull(X, det, masses, metric=metric, nodes=graph.nodes, groups=labels,
                           group_factors=factors, diagnostics=diagnostics)
    if landmarks is not None:
        if likelihood != "poisson":
            raise ValueError("landmarks are available for the Poisson likelihood only")
        spec = Landmarks(rank=landmarks) if isinstance(landmarks, int) else landmarks
        geometry = LandmarkGeometry(X, metric, spec.rank, spec.placement, spec.exact_within, graph.degrees,
                                    spec.exact_neighbors)
    masses, det, diagnostics = fit_on_geometry(graph, geometry, settings)
    _warn_unconverged(diagnostics)
    return SpatialNull(X, det, masses, metric=metric, link="log" if likelihood == "poisson" else "logit",
                       nodes=graph.nodes, fit_settings=settings, diagnostics=diagnostics)


def fit_on_geometry(graph, geometry, settings):
    """Masses and deterrence for the given pair geometry (exact or landmarks)."""
    if settings["likelihood"] == "bernoulli":
        if not isinstance(geometry, ExactGeometry):
            raise ValueError("the Bernoulli likelihood needs exact distances")
        result = _fit_bernoulli(graph, geometry, settings)
    elif settings["deterrence"] == "binned":
        result = _fit_poisson_binned(graph, geometry, settings)
    else:
        result = _fit_poisson_exponential(graph, geometry, settings)
    result[2]["geometry"] = ("exact" if isinstance(geometry, ExactGeometry) else
                             f"landmarks (r={geometry.rank}, exact_within={geometry.exact_within or None}, "
                             f"exact_neighbors={geometry.exact_neighbors or None})")
    return result


def _warn_unconverged(diagnostics):
    if not diagnostics["converged"]:
        warnings.warn("null-model fit did not converge; inspect null.diagnostics or increase max_iter",
                      RuntimeWarning, stacklevel=3)


def _bin_edges(X, metric, bins, binning, graph=None):
    if np.ndim(bins):
        edges = np.asarray(bins, dtype=float)
        if (edges.ndim != 1 or len(edges) < 2 or edges[0] != 0
                or np.isnan(edges).any() or np.any(np.diff(edges) <= 0)):
            raise ValueError("bin edges must start at 0 and increase strictly")
        return edges
    if int(bins) != bins or int(bins) < 1:
        raise ValueError("bins must be a positive integer or a sequence of edges")
    if binning == "edges":
        I, J, weights = graph.edges()
        d = pair_distances(X, I, J, metric)
        levels = np.linspace(0, 1, int(bins) + 1)[1:-1]
        if np.all(weights == weights[0]):
            cuts = np.quantile(d, levels)
        else:
            order = np.argsort(d)
            cdf = (np.cumsum(weights[order]) - weights[order] / 2) / weights.sum()
            cuts = np.interp(levels, cdf, d[order])
        return np.unique(np.r_[0., cuts, np.inf])
    rng = np.random.default_rng(0)
    n = len(X)
    I, J = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
    keep = I != J
    I, J = I[keep], J[keep]
    if callable(metric):
        I, J = I[:2000], J[:2000]
    d = pair_distances(X, I, J, metric)
    if binning == "quantile":
        edges = np.quantile(d, np.linspace(0, 1, int(bins) + 1))
    elif binning == "linear":
        edges = np.linspace(0, d.max(), int(bins) + 1)
    elif binning == "log":
        positive = d[d > 0]
        edges = (np.concatenate([[0.], np.geomspace(positive.min(), d.max(), int(bins))])
                 if len(positive) else np.array([0., np.inf]))
    else:
        raise ValueError("binning must be 'quantile', 'edges', 'linear' or 'log'")
    edges[0] = 0.
    edges = np.unique(edges)
    return edges if len(edges) > 1 else np.array([0., np.inf])


def _degree_errors(expected, k, active):
    err = np.abs(expected - k)
    l1 = float(err.sum() / max(k.sum(), 1e-300))
    worst = float((err[active] / k[active]).max()) if active.any() else 0.
    return l1, worst


def _initial_masses(k, degree_correction):
    active = k > 0
    if not degree_correction:
        return np.ones(len(k)), active
    w = np.where(active, k, 0.)
    return w / w[active].mean(), active


def _fit_poisson_binned(graph, geometry, s):
    I, J, wt = graph.edges()
    k, edges = graph.degrees, s["edges"]
    nb = len(edges) - 1
    observed = np.bincount(bin_index(geometry.edge_distances(I, J), edges), weights=wt, minlength=nb)
    w, active = _initial_masses(k, s["degree_correction"])
    converged, it = not s["degree_correction"], 0
    for it in range(s["max_iter"] + 1):
        exposure = geometry.bin_sums(w, edges)
        f = np.divide(observed, exposure, out=np.zeros(nb), where=exposure > 0)
        expected = w * geometry.row_sums(w, lambda D: f[bin_index(D, edges)])
        if not s["degree_correction"]:
            break
        l1, worst = _degree_errors(expected, k, active)
        if l1 <= s["tol"] and worst <= 20 * s["tol"]:
            converged = True
            break
        if it == s["max_iter"]:
            break
        w[active] *= np.sqrt(k[active] / np.maximum(expected[active], 1e-300))
        w /= w[active].mean()
    l1, worst = _degree_errors(expected, k, active)
    diagnostics = dict(iterations=it, converged=converged, degree_error_l1=l1, degree_error_max=worst,
                       observed_by_bin=observed, expected_by_bin=f * exposure,
                       total_observed=float(wt.sum()), total_expected=float(expected.sum() / 2))
    return w, Binned(edges, f), diagnostics


def _fit_poisson_exponential(graph, geometry, s):
    I, J, wt = graph.edges()
    k, total = graph.degrees, float(wt.sum())
    d_edges = geometry.edge_distances(I, J)
    w, active = _initial_masses(k, s["degree_correction"])
    mean_d = float(wt @ d_edges / total)
    beta, upper = 1. / max(mean_d, 1e-12), 100. / max(mean_d, 1e-12)
    converged, it, previous = False, 0, np.inf

    def profile(b):                           # negative profile log-likelihood
        return b * (wt @ d_edges) + total * np.log(geometry.pair_sum(w, lambda D: np.exp(-b * D)))

    for it in range(s["max_iter"] + 1):
        beta = float(minimize_scalar(profile, bounds=(0., upper), method="bounded",
                                     options=dict(xatol=1e-6 * max(beta, 1e-12))).x)
        g = lambda D, b=beta: np.exp(-b * D)
        scale = total / geometry.pair_sum(w, g)
        expected = scale * w * geometry.row_sums(w, g)
        l1, worst = _degree_errors(expected, k, active)
        stable = abs(beta - previous) <= s["tol"] * max(beta, 1e-12)
        if (not s["degree_correction"] or (l1 <= s["tol"] and worst <= 20 * s["tol"])) and stable:
            converged = True
            break
        if it == s["max_iter"]:
            break
        previous = beta
        if s["degree_correction"]:
            w[active] *= np.sqrt(k[active] / np.maximum(expected[active], 1e-300))
            w /= w[active].mean()
    diagnostics = dict(iterations=it, converged=converged, degree_error_l1=l1, degree_error_max=worst,
                       beta=beta, scale=scale, total_observed=total, total_expected=float(expected.sum() / 2))
    return w, Exponential(beta, scale), diagnostics


def _fit_bernoulli(graph, geometry, s):
    """logit P_ij = a_i + a_j + phi(d_ij), by damped Newton steps on the moment equations."""
    I, J, wt = graph.edges()
    if not np.allclose(wt, 1.):
        raise ValueError("the Bernoulli likelihood needs a binary graph")
    n, k = graph.n, graph.degrees
    active = k > 0
    binned = s["deterrence"] == "binned"
    d_edges = geometry.edge_distances(I, J)
    if binned:
        edges = s["edges"]
        nb = len(edges) - 1
        target = np.bincount(bin_index(d_edges, edges), minlength=nb).astype(float)
        phi = np.zeros(nb)
    else:
        target = np.array([len(I), d_edges.sum()])
        phi = np.zeros(2)                     # log-scale, beta
    a = np.where(active, 0.5 * logit(np.clip(k / max(n - 1, 1), 1e-6, 0.5)), -40.)
    if not s["degree_correction"]:
        a[:] = 0.
    converged, it = False, 0
    for it in range(s["max_iter"] + 1):
        rowP, rowV = np.zeros(n), np.zeros(n)
        mom, hess = np.zeros_like(phi), np.zeros((len(phi), len(phi)))
        for rows, D in geometry.tiles():
            if binned:
                B = bin_index(D, edges)
                x = a[rows, None] + a[None, :] + phi[B]
            else:
                x = a[rows, None] + a[None, :] + phi[0] - phi[1] * D
            P = expit(x)
            P[np.arange(len(rows)), rows] = 0.
            V = P * (1 - P)
            rowP[rows], rowV[rows] = P.sum(1), V.sum(1)
            if binned:
                mom += np.bincount(B.ravel(), weights=P.ravel(), minlength=nb) / 2
                hess[np.diag_indices(nb)] += np.bincount(B.ravel(), weights=V.ravel(), minlength=nb) / 2
            else:
                mom += np.array([P.sum(), (P * D).sum()]) / 2
                hess += np.array([[V.sum(), -(V * D).sum()], [-(V * D).sum(), (V * D * D).sum()]]) / 2
        l1, worst = _degree_errors(rowP, k, active)
        moment_err = float(np.max(np.abs(mom - target) / np.maximum(np.abs(target), 1.)))
        degrees_ok = not s["degree_correction"] or (l1 <= s["tol"] and worst <= 20 * s["tol"])
        if degrees_ok and moment_err <= s["tol"]:
            converged = True
            break
        if it == s["max_iter"]:
            break
        # Alternate the two blocks of parameters. Each pair enters two degrees, so the
        # per-node Newton step on a is halved.
        if s["degree_correction"] and it % 2 == 0:
            a[active] += 0.5 * np.clip((k - rowP)[active] / (rowV[active] + 1e-12), -4, 4)
        else:
            grad = target - mom if binned else np.array([target[0] - mom[0], -(target[1] - mom[1])])
            step = grad / (np.diag(hess) + 1e-12) if binned else np.linalg.solve(hess + 1e-12 * np.eye(2), grad)
            phi += np.clip(step, -4, 4)
    masses = np.exp(a)
    det = Binned(edges, np.exp(phi)) if binned else Exponential(phi[1], float(np.exp(phi[0])))
    diagnostics = dict(iterations=it, converged=converged, degree_error_l1=l1, degree_error_max=worst,
                       moment_error=moment_err, total_observed=float(len(I)), total_expected=float(rowP.sum() / 2))
    return masses, det, diagnostics


def _fit_group_binned(graph, geometry, s, groups, effect):
    """Joint Poisson fit of distance, degrees and within-group counts (moderate graphs)."""
    if effect not in ("individual", "shared"):
        raise ValueError("group_effect must be 'individual' or 'shared'")
    names, codes = np.unique(groups, return_inverse=True)
    same = codes[:, None] == codes[None, :]
    category = np.broadcast_to(codes[:, None], same.shape) if effect == "individual" else np.zeros(same.shape, int)
    ng = len(names) if effect == "individual" else 1
    n, edges = graph.n, s["edges"]
    D = np.empty((n, n))
    for rows, distances in geometry.tiles():
        D[rows] = distances
    B = bin_index(D, edges)
    I, J, wt = graph.edges()
    observed = np.bincount(B[I, J], weights=wt, minlength=len(edges) - 1)
    inside = same[I, J]
    target = np.bincount(category[I, J][inside], weights=wt[inside], minlength=ng)
    w, active = _initial_masses(graph.degrees, s["degree_correction"])
    factors, converged = np.ones(ng), False
    for it in range(s["max_iter"] + 1):
        exposure = w[:, None] * w[None, :] * np.where(same, factors[category], 1.)
        np.fill_diagonal(exposure, 0.)
        by_bin = np.bincount(B.ravel(), weights=exposure.ravel(), minlength=len(observed)) / 2
        f = np.divide(observed, by_bin, out=np.zeros_like(observed), where=by_bin > 0)
        P = exposure * f[B]
        expected_group = np.bincount(category[same], weights=P[same], minlength=ng) / 2
        ratio = np.divide(target, expected_group, out=np.ones(ng), where=expected_group > 0)
        factors *= ratio
        P *= np.where(same, ratio[category], 1.)
        expected = P.sum(axis=1)
        expected_bins = np.bincount(B.ravel(), weights=P.ravel(), minlength=len(observed)) / 2
        l1, worst = _degree_errors(expected, graph.degrees, active)
        moment_error = float(np.max(np.abs(expected_bins - observed) / np.maximum(observed, 1.)))
        if (not s["degree_correction"] or (l1 <= s["tol"] and worst <= 20 * s["tol"])) and moment_error <= s["tol"]:
            converged = True
            break
        if it == s["max_iter"]:
            break
        if s["degree_correction"]:
            w[active] *= np.sqrt(graph.degrees[active] / np.maximum(expected[active], 1e-300))
            w /= w[active].mean()
    diagnostics = dict(iterations=it, converged=converged, geometry="exact", group_effect=effect,
                       groups=list(names), group_factors=factors.copy(), observed_within_groups=target,
                       expected_within_groups=expected_group * ratio, moment_error=moment_error,
                       degree_error_l1=l1, degree_error_max=worst, observed_by_bin=observed,
                       expected_by_bin=expected_bins, total_observed=graph.total_weight,
                       total_expected=float(expected.sum() / 2))
    return w, Binned(edges, f), factors if effect == "individual" else np.repeat(factors, len(names)), diagnostics


def fit_sbm_null(G, labels, *, degree_corrected=True, weight="weight"):
    """Stochastic block model null for given block labels.

    degree_corrected=True : P_uv = theta_u theta_v omega_rs with theta_u = k_u / kappa_r.
    degree_corrected=False: P_uv = omega_rs.
    The expected weight between every pair of blocks, and inside every block, equals the
    observed one. Cost O(n + m + b^2).
    """
    graph = as_indexed(G, weight)
    blocks, z = np.unique(node_values(graph, labels, "labels", dtype=None), return_inverse=True)
    z = z.astype(np.intp)
    b = len(blocks)
    I, J, wt = graph.edges()
    if b > AUTO_DISTINCT_LIMIT:
        E = sp.coo_array((wt, (z[I], z[J])), shape=(b, b)).tocsr()
        E = E + E.T
        if degree_corrected:
            kappa = np.bincount(z, weights=graph.degrees, minlength=b)
            theta = np.divide(graph.degrees, kappa[z], out=np.zeros(graph.n), where=kappa[z] > 0)
            within = 1 - np.bincount(z, weights=theta ** 2, minlength=b)
            diagonal = E.diagonal()
            corrected = np.divide(diagonal, within, out=np.zeros(b), where=within > 0)
            omega = E + sp.diags(corrected - diagonal, format="csr")
            omega.eliminate_zeros()
        else:
            theta = np.ones(graph.n)
            size = np.bincount(z, minlength=b).astype(float)
            e = E.tocoo()
            pairs = size[e.row] * (size[e.col] - (e.row == e.col))
            omega = sp.coo_array((np.divide(e.data, pairs, out=np.zeros_like(e.data), where=pairs > 0),
                                  (e.row, e.col)), shape=(b, b)).tocsr()
        return BlockNull(z, omega, theta, nodes=graph.nodes,
                         diagnostics=dict(blocks=list(blocks), degree_corrected=bool(degree_corrected),
                                          fully_conditioned=b == graph.n))
    E = np.zeros((b, b))
    np.add.at(E, (z[I], z[J]), wt)
    E = E + E.T                                # E[r, s]: weight between r and s; E[r, r] = 2 x internal
    if degree_corrected:
        kappa = np.bincount(z, weights=graph.degrees, minlength=b)
        theta = np.divide(graph.degrees, kappa[z], out=np.zeros(graph.n), where=kappa[z] > 0)
        within = 1 - np.bincount(z, weights=theta ** 2, minlength=b)
        omega = E.copy()
        omega[np.diag_indices(b)] = np.divide(np.diag(E), within, out=np.zeros(b), where=within > 0)
    else:
        theta = np.ones(graph.n)
        size = np.bincount(z, minlength=b).astype(float)
        pairs = np.outer(size, size)
        pairs[np.diag_indices(b)] = size * (size - 1)
        omega = np.divide(E, pairs, out=np.zeros((b, b)), where=pairs > 0)
    return BlockNull(z, omega, theta, nodes=graph.nodes,
                     diagnostics=dict(blocks=list(blocks), degree_corrected=bool(degree_corrected),
                                      fully_conditioned=b == graph.n))


def fit_configuration_null(G, *, weight="weight"):
    """Configuration (Chung-Lu) null P_uv = k_u k_v / 2m.

    With gamma=1 and epsilon=0, community detection maximises standard modularity.
    """
    graph = as_indexed(G, weight)
    m2 = 2 * graph.total_weight
    if m2 <= 0:
        raise ValueError("the graph has no edges")
    return BlockNull(np.zeros(graph.n, np.intp), [[1. / m2]], graph.degrees, nodes=graph.nodes,
                     diagnostics=dict(total_weight=graph.total_weight))
