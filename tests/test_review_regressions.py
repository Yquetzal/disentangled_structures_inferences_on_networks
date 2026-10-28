"""Scientific and input regressions found during the article/library review."""
import importlib.util

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

import disentangled_net as dn


def test_edge_quantiles_follow_observed_links():
    G = nx.path_graph(41)
    X = np.cumsum(np.r_[0., np.arange(1, 41)])[:, None]
    null = dn.fit_spatial_null(G, X, bins=4, binning="edges", degree_correction=False)
    assert np.allclose(null.diagnostics["observed_by_bin"], 10)
    assert np.isinf(null.deterrence.edges[-1])


@pytest.mark.parametrize("effect", ["shared", "individual"])
def test_joint_spatial_group_fit_matches_all_moments(effect):
    rng = np.random.default_rng(4)
    X = rng.random((90, 2))
    groups = np.repeat(["a", "b", "c"], 30)
    base = dn.SpatialNull(X, dn.Exponential(3., .12), groups=groups,
                          group_factors=[2., 4., 6.])
    rates = base.to_matrix()
    A = rng.poisson(rates)
    A = np.triu(A, 1)
    A = sp.csr_array(A + A.T)
    null = dn.fit_spatial_null(A, X, bins=4, binning="edges", groups=groups,
                               group_effect=effect, tol=1e-4, max_iter=1500)
    d = null.diagnostics
    assert d["converged"]
    assert np.allclose(null.expected_degrees(), np.asarray(A.sum(axis=1)).ravel(), rtol=.002)
    assert np.allclose(d["observed_by_bin"], d["expected_by_bin"], rtol=1e-4)
    P = null.to_matrix()
    observed, expected = [], []
    for label in np.unique(groups):
        idx = np.flatnonzero(groups == label)
        observed.append(A[idx][:, idx].sum() / 2)
        expected.append(P[np.ix_(idx, idx)].sum() / 2)
    if effect == "shared":
        assert sum(expected) == pytest.approx(sum(observed))
    else:
        assert np.allclose(expected, observed)
    I, J = np.triu_indices(len(X), 1)
    assert np.allclose(null.pairs(I, J), P[I, J])
    order = rng.permutation(len(X))
    assert np.allclose(null.reindexed(order).to_matrix(), P[np.ix_(order, order)])
    p, info = dn.disentangled_communities(A, null, random_state=0, return_info=True)
    assert info.quality == pytest.approx(dn.partition_quality(A, p, null))
    with pytest.raises(ValueError, match="exact evaluation"):
        dn.disentangled_communities(A, null, approximation=dn.Landmarks(5))


def test_non_edge_sampler_matches_residual_mixture_with_epsilon():
    from disentangled_net.communities import _exact_backend
    from disentangled_net.embeddings import _BlockPairs
    G = nx.empty_graph(10)
    G.add_edges_from((0, i) for i in range(1, 7))
    theta = np.r_[1., np.full(6, 5.), .1, 1., 3.]
    null = dn.BlockNull(np.zeros(10, int), [[1.]], theta)
    pairs = _BlockPairs(nx.to_scipy_sparse_array(G), _exact_backend(null), 1., 2., np.random.default_rng(0))
    draws = pairs._non_edges(np.zeros(40000, int))
    frequency = np.bincount(draws, minlength=10) / len(draws)
    expected = theta[7:] + 2.
    expected /= expected.sum()
    assert np.all(frequency[:7] == 0)
    assert np.allclose(frequency[7:], expected, atol=.009)


@pytest.mark.parametrize("vectorized", [False, True])
def test_positive_asymmetric_callable_is_rejected(vectorized):
    with pytest.raises(ValueError, match="symmetric"):
        dn.custom_null(lambda u, v: 1. + np.asarray(u), range(10), vectorized=vectorized)


@pytest.mark.parametrize("binning", ["quantile", "linear", "log", "edges"])
def test_coincident_positions_have_one_valid_bin(binning):
    G = nx.complete_graph(5)
    null = dn.fit_spatial_null(G, np.zeros(5), bins=3, binning=binning)
    assert null.diagnostics["converged"]
    assert np.allclose(null.expected_degrees(), 4)


def test_exhausted_fit_reports_actual_returned_model():
    G = nx.barabasi_albert_graph(30, 2, seed=0)
    for likelihood in ["poisson", "bernoulli"]:
        with pytest.warns(RuntimeWarning, match="did not converge"):
            null = dn.fit_spatial_null(G, np.arange(30), bins=4, likelihood=likelihood, max_iter=0)
        assert null.to_matrix().sum() / 2 == pytest.approx(null.diagnostics["total_expected"])


def test_empty_and_singleton_inputs():
    for n in (0, 1):
        G = nx.empty_graph(n)
        null = dn.custom_null(np.zeros((n, n)))
        p = dn.disentangled_communities(G, null)
        assert len(p) == n and dn.partition_quality(G, p, null) == 0.
        for method in ("spectral", "sine"):
            z = dn.disentangled_embeddings(G, null, dim=2, method=method)
            assert all(np.array_equal(v, np.zeros(2)) for v in z.values())
    assert dn.adjusted_rand([], []) == 1.
    assert dn.adjusted_rand([0], [1]) == 1.
    with pytest.raises(ValueError, match="no edges"):
        dn.fit_spatial_null(nx.empty_graph(3), np.arange(3))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
@pytest.mark.parametrize("engine", ["auto", "dense"])
def test_sine_zero_residual_has_no_inferred_coordinates(engine):
    G = nx.complete_graph(5)
    null = dn.BlockNull(np.zeros(5, int), [[1.]])
    z = dn.disentangled_embeddings(G, null, dim=2, method=dn.Sine(epochs=2, engine=engine), random_state=0)
    assert np.array_equal(np.array(list(z.values())), np.zeros((5, 2)))


def test_non_edge_sampler_handles_graph_without_edges():
    from disentangled_net.communities import _exact_backend
    from disentangled_net.embeddings import _BlockPairs
    null = dn.BlockNull([0, 0, 0], [[.1]])
    pairs = _BlockPairs(sp.csr_array((3, 3)), _exact_backend(null), 1., 0., np.random.default_rng(0))
    source, partner, sign = pairs.draw(10)
    assert np.all(source != partner) and np.all(sign == -1)


@pytest.mark.parametrize("method", ["spectral", dn.Spectral(normalized=True)])
def test_large_empty_spectral_input_does_not_fail_in_arpack(method):
    A = sp.csr_array((300, 300))
    null = dn.BlockNull(np.zeros(300, int), [[0.]])
    z = dn.disentangled_embeddings(A, null, dim=2, method=method)
    assert np.array_equal(np.array(list(z.values())), np.zeros((300, 2)))
