import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

import disentangled_net as dn
import disentangled_net.communities as communities
import disentangled_net._louvain.backends as backends


@pytest.fixture(scope="module")
def planted():
    G = nx.planted_partition_graph(4, 25, .4, .02, seed=3)
    rng = np.random.default_rng(0)
    nx.set_node_attributes(G, {u: rng.random(2) for u in G}, "pos")
    return G, np.array([u // 25 for u in G])


def labels(G, partition):
    return np.array([partition[u] for u in G])


def test_planted_partition_and_quality(planted):
    G, truth = planted
    null = dn.fit_configuration_null(G)
    p, info = dn.disentangled_communities(G, null, random_state=0, return_info=True)
    assert dn.adjusted_rand(labels(G, p), truth) == 1
    assert info.quality == pytest.approx(dn.partition_quality(G, p, null))
    p2, info2 = dn.disentangled_communities(G, null, init=p, random_state=1, return_info=True)
    assert info2.quality == pytest.approx(dn.partition_quality(G, p2, null))


def test_exact_backends_agree(planted, monkeypatch):
    G, _ = planted
    spatial = dn.fit_spatial_null(G, "pos", bins=5)
    block = dn.fit_sbm_null(G, {u: u % 3 for u in G})
    for null in (spatial, block):
        reference = dn.disentangled_communities(G, dn.custom_null(null.to_matrix(), list(G)), random_state=4)
        assert dn.disentangled_communities(G, null, random_state=4) == reference
    monkeypatch.setattr(backends, "DENSE_LIMIT", 0)
    monkeypatch.setattr(communities, "DENSE_LIMIT", 0)
    p, info = dn.disentangled_communities(G, spatial, random_state=4, return_info=True)
    assert info.backend == "tiled"
    assert p == dn.disentangled_communities(G, dn.custom_null(spatial.to_matrix(), list(G)), random_state=4)


def test_epsilon_is_an_excess_density_threshold():
    G = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    G.add_edges_from([(0, 5), (1, 6), (2, 7)])
    null = dn.custom_null(np.full((10, 10), .05))
    density = dn.excess_density(G, null, range(5), range(5, 10))["density"]
    assert density == pytest.approx((3 - 25 * .05) / 25)
    merged = dn.disentangled_communities(G, null, epsilon=density * .9, random_state=0)
    split = dn.disentangled_communities(G, null, epsilon=density * 1.1, random_state=0)
    assert len(set(merged.values())) == 1 and len(set(split.values())) == 2


def test_gamma_follows_the_standard_convention(planted):
    G, _ = planted
    null = dn.fit_configuration_null(G)
    sizes = [len(set(dn.disentangled_communities(G, null, gamma=g, random_state=0).values())) for g in (.2, 1, 5)]
    assert sizes[0] <= sizes[1] <= sizes[2] and sizes[0] < sizes[2]


def test_sparse_input_matches_networkx(planted):
    G, _ = planted
    null = dn.fit_configuration_null(G)
    A = sp.csr_array(nx.to_scipy_sparse_array(G, nodelist=sorted(G)))
    assert dn.disentangled_communities(A, dn.fit_configuration_null(A), random_state=2) == \
        dn.disentangled_communities(G, null, random_state=2)


def test_landmarks(planted):
    G, _ = planted
    null = dn.fit_spatial_null(G, "pos", bins=5)
    exact = dn.disentangled_communities(G, null, random_state=5)
    full = dn.Landmarks(rank=G.number_of_nodes())
    assert dn.disentangled_communities(G, null, approximation=full, random_state=5) == exact
    everything_near = dn.Landmarks(rank=3, exact_within=10.)
    p, info = dn.disentangled_communities(G, null, approximation=everything_near, random_state=5, return_info=True)
    assert p == exact and info.backend == "blocks+sparse"
    every_neighbor = dn.Landmarks(rank=3, exact_neighbors=G.number_of_nodes() - 1)
    assert dn.disentangled_communities(G, null, approximation=every_neighbor, random_state=5) == exact
    p, info = dn.disentangled_communities(G, null, approximation=dn.Landmarks(rank=8, exact_neighbors=5), return_info=True)
    assert 5 <= info.details["near_pairs_per_node"] <= 10
    p, info = dn.disentangled_communities(G, null, approximation="landmarks", return_info=True)
    assert info.approximation == "landmarks" and info.details["rank"] == 100
    with pytest.raises(ValueError):
        dn.disentangled_communities(G, dn.fit_configuration_null(G), approximation="landmarks")


def test_sampling(planted):
    G, truth = planted
    p, info = dn.disentangled_communities(G, dn.fit_configuration_null(G), approximation=dn.Sampling(sample_size=64),
                                          random_state=0, return_info=True)
    assert info.approximation == "sampling" and dn.adjusted_rand(labels(G, p), truth) > .9


def test_profile_and_standardize(planted):
    G, _ = planted
    rows = dn.epsilon_profile(G, dn.fit_configuration_null(G), [0, .01, .1], seeds=range(3))
    assert [r["epsilon"] for r in rows] == [0, .01, .1]
    assert rows[0]["n_communities"] <= rows[-1]["n_communities"]
    bernoulli = dn.fit_spatial_null(G, "pos", bins=3, likelihood="bernoulli")
    p = dn.disentangled_communities(G, bernoulli, standardize=True, random_state=0)
    assert len(p) == G.number_of_nodes()
