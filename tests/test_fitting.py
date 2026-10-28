import networkx as nx
import numpy as np
import pytest

import disentangled_net as dn
from disentangled_net._geometry import bin_index


@pytest.fixture(scope="module")
def spatial_graph():
    """Random geometric-like graph with distance decay, 150 nodes."""
    rng = np.random.default_rng(1)
    X = rng.random((150, 2))
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    P = 0.6 * np.exp(-6 * D)
    A = np.triu(rng.random(P.shape) < P, 1)
    G = nx.from_numpy_array((A | A.T).astype(float))
    nx.set_node_attributes(G, {u: X[u] for u in G}, "pos")
    return G, X


def test_poisson_binned_matches_degrees_and_bin_masses(spatial_graph):
    G, X = spatial_graph
    null = dn.fit_spatial_null(G, "pos", bins=6, tol=1e-4)
    k = np.array([d for _, d in G.degree()], float)
    assert null.diagnostics["converged"]
    assert np.allclose(null.expected_degrees(), k, rtol=2e-3, atol=1e-6)
    d = null.diagnostics
    assert np.allclose(d["expected_by_bin"], d["observed_by_bin"])


def test_without_degree_correction_bins_still_match(spatial_graph):
    G, _ = spatial_graph
    null = dn.fit_spatial_null(G, "pos", bins=[0, .1, .3, 2], degree_correction=False)
    assert np.allclose(null.masses, 1)
    edges = null.deterrence.edges
    M = null.to_matrix()
    I, J = np.triu_indices(len(M), 1)
    D = np.linalg.norm(null.positions[I] - null.positions[J], axis=1)
    expected = np.bincount(bin_index(D, edges), weights=M[I, J], minlength=3)
    assert np.allclose(expected, null.diagnostics["observed_by_bin"])


def test_exponential_recovers_decay(spatial_graph):
    G, _ = spatial_graph
    null = dn.fit_spatial_null(G, "pos", deterrence="exponential", degree_correction=False)
    assert null.diagnostics["total_expected"] == pytest.approx(G.number_of_edges(), rel=1e-6)
    assert null.deterrence.beta == pytest.approx(6, rel=0.25)


def test_bernoulli_probabilities_match_degrees(spatial_graph):
    G, _ = spatial_graph
    null = dn.fit_spatial_null(G, "pos", bins=5, likelihood="bernoulli", tol=1e-4)
    k = np.array([d for _, d in G.degree()], float)
    assert null.kind == "probability" and null.diagnostics["converged"]
    assert (null.to_matrix() <= 1).all()
    assert np.allclose(null.expected_degrees(), k, rtol=5e-3, atol=1e-3)
    with pytest.raises(ValueError):
        H = G.copy()
        nx.set_edge_attributes(H, 2., "weight")
        dn.fit_spatial_null(H, "pos", likelihood="bernoulli")


def test_landmarks_at_full_rank_equal_exact_fit(spatial_graph):
    G, _ = spatial_graph
    exact = dn.fit_spatial_null(G, "pos", bins=6)
    full = dn.fit_spatial_null(G, "pos", bins=6, landmarks=dn.Landmarks(rank=G.number_of_nodes()))
    assert np.allclose(full.masses, exact.masses) and np.allclose(full.deterrence.values, exact.deterrence.values)
    hybrid = dn.fit_spatial_null(G, "pos", bins=6, landmarks=dn.Landmarks(rank=4, exact_within=10.))
    assert np.allclose(hybrid.deterrence.values, exact.deterrence.values)
    knn = dn.fit_spatial_null(G, "pos", bins=6, landmarks=dn.Landmarks(rank=4, exact_neighbors=149))
    assert np.allclose(knn.deterrence.values, exact.deterrence.values)


def test_sbm_and_configuration():
    G = nx.planted_partition_graph(3, 20, .5, .05, seed=0)
    labels = {u: u // 20 for u in G}
    for dc in (True, False):
        null = dn.fit_sbm_null(G, labels, degree_corrected=dc)
        M = null.to_matrix()
        z = np.array([labels[u] for u in G])
        A = nx.to_numpy_array(G)
        for r in range(3):
            for s in range(3):
                block = np.ix_(z == r, z == s)
                assert M[block].sum() == pytest.approx(A[block].sum())
    conf = dn.fit_configuration_null(G)
    k = A.sum(1)
    P = np.outer(k, k) / k.sum()
    np.fill_diagonal(P, 0)
    assert np.allclose(conf.to_matrix(), P)
