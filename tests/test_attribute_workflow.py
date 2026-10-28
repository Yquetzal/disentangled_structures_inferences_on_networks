"""Ordinary user workflows: choose an attribute, leave numerical choices to the library."""
import importlib.util
import sys

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

import disentangled_net as dn
from disentangled_net import fitting


@pytest.fixture
def graph():
    G = nx.planted_partition_graph(3, 30, .35, .03, seed=3)
    rng = np.random.default_rng(4)
    nx.set_node_attributes(G, {u: float(x) for u, x in zip(G, rng.random(len(G)))}, "age")
    nx.set_node_attributes(G, {u: ["FR", "BE", "DE"][u % 3] for u in G}, "country")
    nx.set_node_attributes(G, {u: rng.random(2) for u in G}, "pos")
    return G


@pytest.mark.parametrize("attribute", ["age", "country", "pos"])
def test_one_call_and_fit_once_have_identical_defaults(graph, attribute):
    before = nx.to_numpy_array(graph)
    p, info = dn.disentangled_communities(graph, attribute=attribute, random_state=0, return_info=True)
    null = dn.fit_null(graph, attribute)
    assert p == dn.disentangled_communities(graph, null, random_state=0)
    assert p == dn.disentangled_communities(graph, attribute, random_state=0)
    assert info.quality == pytest.approx(dn.partition_quality(graph, p, null))
    assert info.details["epsilon"] == pytest.approx(graph.size(weight="weight") / (len(graph) * (len(graph) - 1) / 2))
    assert np.array_equal(nx.to_numpy_array(graph), before)


def test_continuous_exact_fit_matches_explicit_article_settings(graph):
    auto = dn.fit_null(graph, "pos")
    explicit = dn.fit_spatial_null(graph, "pos", bins=20, binning="edges", max_iter=1000)
    assert isinstance(auto, dn.SpatialNull)
    assert auto.diagnostics["approximation"] == "exact"
    assert np.allclose(auto.to_matrix(), explicit.to_matrix())


def test_repeated_values_are_compressed_exactly(graph):
    ages = np.tile([20., 35., 60.], len(graph) // 3)
    auto = dn.fit_null(graph, ages)
    exact = dn.fit_spatial_null(graph, ages, bins=20, binning="edges", max_iter=1000)
    assert isinstance(auto, dn.BlockNull)
    assert auto.diagnostics["compressed"] and auto.diagnostics["rank"] == 3
    assert auto.diagnostics["approximation"] == "exact"
    assert np.allclose(auto.to_matrix(), exact.to_matrix(), rtol=.005, atol=1e-5)


def test_automatic_landmarks_are_fitted_once_and_reused(monkeypatch):
    G = nx.watts_strogatz_graph(320, 12, .2, seed=0)
    X = np.random.default_rng(0).random((len(G), 2))
    monkeypatch.setattr(fitting, "AUTO_EXACT_LIMIT", 100)
    monkeypatch.setattr(fitting, "AUTO_LANDMARKS", 12)
    null = dn.fit_null(G, X)
    assert isinstance(null, dn.BlockNull) and null.diagnostics["rank"] == 12
    assert null.diagnostics["approximation"] == "landmarks"

    def no_dense(*args, **kwargs):
        raise AssertionError("the automatic path must not materialize the null")
    monkeypatch.setattr(dn.NullModel, "to_matrix", no_dense)
    p, info = dn.disentangled_communities(G, null, random_state=0, return_info=True)
    assert info.approximation == "landmarks" and info.backend == "blocks"
    assert info.quality == pytest.approx(dn.partition_quality(G, p, null))
    z = dn.disentangled_embeddings(G, null, dim=2)
    assert np.isfinite(np.array(list(z.values()))).all()


def test_categorical_integer_codes_can_be_explicit(graph):
    codes = np.arange(len(graph)) % 3
    null = dn.fit_null(graph, codes, attribute_type="categorical")
    expected = dn.fit_sbm_null(graph, codes)
    assert np.allclose(null.to_matrix(), expected.to_matrix())
    assert dn.fit_null(graph, np.asarray(codes, dtype=bool)).diagnostics["attribute_type"] == "categorical"


def test_sparse_input_mapping_and_arbitrary_node_ids(graph):
    H = nx.relabel_nodes(graph, {u: f"person-{u}" for u in graph})
    values = nx.get_node_attributes(H, "country")
    p = dn.disentangled_communities(H, attribute=values, random_state=0)
    A = nx.to_scipy_sparse_array(H)
    q = dn.disentangled_communities(A, attribute=list(values.values()), random_state=0)
    assert list(p.values()) == list(q.values())


def test_default_embeddings_work_without_torch(graph, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    one = dn.disentangled_embeddings(graph, attribute="country", dim=2)
    two = dn.disentangled_embeddings(graph, dn.fit_null(graph, "country"), dim=2)
    assert np.allclose(list(one.values()), list(two.values()))
    assert np.isfinite(np.array(list(one.values()))).all()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
def test_sine_also_accepts_the_automatic_large_graph_null(graph, monkeypatch):
    monkeypatch.setattr(fitting, "AUTO_EXACT_LIMIT", 20)
    monkeypatch.setattr(fitting, "AUTO_LANDMARKS", 8)
    z = dn.disentangled_embeddings(graph, attribute="pos", dim=2,
                                   method=dn.Sine(epochs=2), random_state=0)
    assert np.isfinite(np.array(list(z.values()))).all()


@pytest.mark.parametrize("n", [0, 1, 5])
def test_no_edges_and_constant_attributes(n):
    G = nx.empty_graph(n)
    nx.set_node_attributes(G, 42., "age")
    assert len(dn.disentangled_communities(G, attribute="age")) == n
    z = dn.disentangled_embeddings(G, attribute="age", dim=2)
    assert all(np.array_equal(v, [0., 0.]) for v in z.values())


def test_constant_attribute_uses_configuration_model(graph):
    null = dn.fit_null(graph, np.ones(len(graph)))
    assert null.diagnostics["constant_attribute"]
    assert np.allclose(null.to_matrix(), dn.fit_configuration_null(graph).to_matrix())


def test_geographic_coordinates_and_custom_metric(graph):
    X = np.random.default_rng(4).random((len(graph), 2)) * [90., 180.]
    null = dn.fit_null(graph, X, metric="haversine")
    assert null.diagnostics["converged"]
    metric = lambda X, Y: np.abs(X[:, None, 0] - Y[None, :, 0])
    null = dn.fit_null(graph, np.arange(len(graph)), metric=metric)
    assert null.diagnostics["approximation"] == "exact"


def test_missing_attributes_and_conflicting_inputs_are_explained(graph):
    with pytest.raises(ValueError, match="missing"):
        dn.disentangled_communities(graph, attribute="unknown")
    with pytest.raises(ValueError, match="either null or attribute"):
        dn.disentangled_communities(graph, dn.fit_null(graph, "age"), attribute="country")
    with pytest.raises(ValueError, match="finite"):
        dn.fit_null(graph, np.full(len(graph), np.nan))


@pytest.mark.parametrize("dc", [True, False])
def test_many_categories_sparse_backend_matches_dense(dc, monkeypatch):
    from disentangled_net import communities
    monkeypatch.setattr(communities, "BLOCK_DENSE_LIMIT", 0)
    G = nx.watts_strogatz_graph(600, 6, .2, seed=1)
    null = dn.fit_sbm_null(G, np.arange(600) % 300, degree_corrected=dc)
    assert sp.issparse(null.block_matrix)
    dense = dn.BlockNull(null.labels, null.block_matrix.toarray(), null.masses)
    assert np.allclose(null.to_matrix(), dense.to_matrix())
    assert np.allclose(null.expected_degrees(), dense.expected_degrees())
    p, info = dn.disentangled_communities(G, null, random_state=0, return_info=True)
    assert info.backend == "sparse_blocks"
    assert p == dn.disentangled_communities(G, dense, random_state=0)
    assert info.quality == pytest.approx(dn.partition_quality(G, p, null))
    assert any(level["moves"] for level in info.levels)
    z = dn.disentangled_embeddings(G, null, method="spectral", dim=2)
    assert np.isfinite(np.array(list(z.values()))).all()


def test_unique_categorical_ids_do_not_allocate_quadratic_matrices():
    G = nx.cycle_graph(3000)
    values = [f"id-{i}" for i in G]
    null = dn.fit_null(G, values)
    assert sp.issparse(null.block_matrix) and null.block_matrix.nnz == 2 * G.number_of_edges()
    p, info = dn.disentangled_communities(G, null, return_info=True)
    assert len(set(p.values())) == len(G) and info.backend == "sparse_blocks"
    z = dn.disentangled_embeddings(G, null, dim=2)
    assert np.array_equal(list(z.values()), np.zeros((len(G), 2)))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
def test_many_categories_also_support_sampled_sine(monkeypatch):
    from disentangled_net import communities
    monkeypatch.setattr(communities, "BLOCK_DENSE_LIMIT", 0)
    G = nx.watts_strogatz_graph(600, 6, .2, seed=1)
    null = dn.fit_null(G, [f"town-{i % 300}" for i in G])
    z = dn.disentangled_embeddings(G, null, method=dn.Sine(epochs=2, init="random"), dim=2, random_state=0)
    assert np.isfinite(np.array(list(z.values()))).all()
