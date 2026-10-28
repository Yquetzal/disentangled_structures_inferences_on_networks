import importlib.util

import networkx as nx
import numpy as np
import pytest

import disentangled_net as dn


@pytest.fixture(scope="module")
def graph():
    G = nx.planted_partition_graph(2, 15, .6, .05, seed=0)
    return G, dn.fit_configuration_null(G)


@pytest.mark.parametrize("method", ["spectral", "deepwalk", dn.DeepWalk(window_size=2)])
def test_matrix_methods(graph, method):
    G, null = graph
    Z = dn.disentangled_embeddings(G, null, dim=3, method=method)
    assert set(Z) == set(G) and all(z.shape == (3,) for z in Z.values())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
@pytest.mark.parametrize("method", [dn.Sine(epochs=5), dn.Sine(epochs=5, sampling="softmax"),
                                    dn.Sine(epochs=5, repulsion="log"), dn.Sine(epochs=5, repulsion="log", power=2),
                                    dn.Contrastive(epochs=5, n_restarts=1, batch_size=64)])
def test_torch_methods(graph, method):
    G, null = graph
    Z = dn.disentangled_embeddings(G, null, dim=2, method=method, random_state=0)
    again = dn.disentangled_embeddings(G, null, dim=2, method=method, random_state=0)
    assert all(np.allclose(Z[u], again[u]) for u in G)


def test_unknown_method(graph):
    with pytest.raises(ValueError):
        dn.disentangled_embeddings(*graph, method="word2vec")


def _span_overlap(X, Y):
    """Mean squared cosine of the principal angles between the column spans (1 = same span)."""
    return float(np.mean(np.linalg.svd(np.linalg.qr(X)[0].T @ np.linalg.qr(Y)[0], compute_uv=False) ** 2))


def test_spectral_is_the_residual_spectrum():
    b = dn.spatial_benchmark(600, rho=2, seed=3)
    P = b.null.to_matrix()
    for gamma, epsilon in [(1., 0.), (.8, .002)]:
        R = b.adjacency.toarray() - gamma * P - epsilon
        np.fill_diagonal(R, 0.)
        values, vectors = np.linalg.eigh(R)
        Z = dn.disentangled_embeddings(b.adjacency, b.null, dim=6, method="spectral", gamma=gamma, epsilon=epsilon)
        Z = np.array([Z[i] for i in range(600)])
        assert np.allclose(np.sort((Z ** 2).sum(axis=0))[::-1], values[::-1][:6], rtol=1e-6)
        assert _span_overlap(Z, vectors[:, -6:]) > 1 - 1e-8


def test_spectral_products():
    from disentangled_net._louvain.backends import TiledBackend
    from disentangled_net.embeddings import _product
    b = dn.spatial_benchmark(300, rho=2, seed=4)
    x = np.random.default_rng(0).normal(size=(300, 3))
    P = b.null.to_matrix()
    assert np.allclose(_product(TiledBackend(b.null), 300)(x), P @ x)
    block = dn.fit_sbm_null(b.adjacency, b.labels)
    assert np.allclose(dn.disentangled_embeddings(b.adjacency, block, dim=4, method="spectral")[0] ** 2,
                       dn.disentangled_embeddings(b.adjacency, dn.custom_null(block.to_matrix()), dim=4,
                                                  method="spectral")[0] ** 2, atol=1e-6)


def test_spectral_landmarks():
    b = dn.spatial_benchmark(1500, rho=2, seed=5)
    k = b.info["communities"]
    exact = dn.disentangled_embeddings(b.adjacency, b.null, dim=k, method="spectral")
    approx = dn.disentangled_embeddings(b.adjacency, b.null, dim=k, method="spectral",
                                        approximation=dn.Landmarks(128))
    as_array = lambda Z: np.array([Z[i] for i in range(1500)])
    assert _span_overlap(as_array(exact), as_array(approx)) > .9
    with pytest.raises(ValueError):
        dn.disentangled_embeddings(b.adjacency, b.null, method="deepwalk", approximation=dn.Landmarks(64))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
def test_sine_pairs_follow_the_residual():
    from disentangled_net.communities import _exact_backend
    from disentangled_net.embeddings import _BlockPairs
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 200)
    b = dn.spatial_benchmark(200, rho=2, seed=1)
    null = dn.fit_sbm_null(b.adjacency, labels)
    for gamma, epsilon in [(1., 0.), (.7, .01)]:
        R = b.adjacency.toarray() - gamma * null.to_matrix() - epsilon
        np.fill_diagonal(R, 0.)
        pairs = _BlockPairs(b.adjacency, _exact_backend(null), gamma, epsilon, np.random.default_rng(1))
        counts = np.zeros_like(R)
        for _ in range(300):
            source, partner, sign = pairs.draw(20)
            assert np.all(sign == np.where(R[source, partner] >= 0, 1, -1)) and np.all(source != partner)
            np.add.at(counts, (source, partner), 1)
        expected = np.abs(R) / np.abs(R).sum(axis=1, keepdims=True) * 300 * 20
        assert np.corrcoef(counts.ravel(), expected.ravel())[0, 1] > .98
    Z = dn.disentangled_embeddings(b.adjacency, null, dim=2, method=dn.Sine(epochs=5), random_state=0)
    assert len(Z) == 200
    with pytest.raises(ValueError):
        dn.disentangled_embeddings(b.adjacency, b.null, method="sine", approximation=dn.Landmarks(32, exact_neighbors=5))


def test_residual_laplacian_eigenmaps():
    """With the configuration null, Laplacian eigenmaps of the residual are Laplacian eigenmaps."""
    from scipy.linalg import eigh
    G = nx.planted_partition_graph(4, 80, .2, .02, seed=3)
    A = nx.to_numpy_array(G, nodelist=range(320))
    d = A.sum(1)
    values, vectors = eigh(np.eye(320) - A / np.sqrt(np.outer(d, d)))
    le = vectors[:, 1:4] / np.sqrt(d)[:, None]
    Z = dn.disentangled_embeddings(G, dn.fit_configuration_null(G), dim=3, method=dn.Spectral(normalized=True))
    assert _span_overlap(np.array([Z[i] for i in range(320)]), le) > .99


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
def test_sine_spectral_start():
    b = dn.spatial_benchmark(400, rho=1, seed=2)
    null = dn.fit_sbm_null(b.adjacency, b.labels)
    for method in [dn.Sine(epochs=0), dn.Sine(epochs=0, engine="dense")]:
        Z = dn.disentangled_embeddings(b.adjacency, null, dim=2, method=method, random_state=0)
        start = dn.disentangled_embeddings(b.adjacency, null, dim=2, method=dn.Spectral(normalized=True))
        Z, start = np.array([Z[i] for i in range(400)]), np.array([start[i] for i in range(400)])
        assert np.allclose(Z.std(axis=0), 1, atol=1e-3) and _span_overlap(Z - Z.mean(0), start - start.mean(0)) > .999
    Z = dn.disentangled_embeddings(b.adjacency, null, dim=2, method=dn.Sine(epochs=0, init="random"), random_state=0)
    assert np.array([Z[i] for i in range(400)]).std() < .2


def test_sine_settings_are_checked(graph):
    for method in [dn.Sine(repulsion="cubic"), dn.Sine(power=0), dn.Sine(repulsion="log", sampling="softmax")]:
        with pytest.raises(ValueError):
            dn.disentangled_embeddings(*graph, dim=2, method=method)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
@pytest.mark.parametrize("power", [1., 2.])
def test_sine_log_loss_estimates_the_energy(power):
    """With repulsion='log', the mean loss of the drawn pairs, each weighted by its source's
    residual mass, estimates sum_{R>0} |R| d^p / p - sum_{R<0} |R| ln d over sum |R|."""
    import torch
    from disentangled_net.communities import _exact_backend
    from disentangled_net.embeddings import _SOFTENING, _BlockPairs, _pair_loss
    b = dn.spatial_benchmark(200, rho=2, seed=1)
    null = dn.fit_sbm_null(b.adjacency, np.random.default_rng(0).integers(0, 3, 200))
    s = dn.Sine(repulsion="log", power=power)
    R = b.adjacency.toarray() - null.to_matrix() - .01
    np.fill_diagonal(R, 0.)
    Z = np.random.default_rng(2).normal(scale=2., size=(200, 2))
    d = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1) + _SOFTENING ** 2)
    terms = np.abs(R) * np.where(R >= 0, d ** power / power, -np.log(d))
    np.fill_diagonal(terms, 0.)
    energy, scale = terms.sum() / np.abs(R).sum(), np.abs(terms).sum() / np.abs(R).sum()
    pairs = _BlockPairs(b.adjacency, _exact_backend(null), 1., .01, np.random.default_rng(3))
    node_weight = pairs.mass / pairs.mass.mean()
    assert np.allclose(pairs.mass, np.abs(R).sum(axis=1))
    Zt, losses = torch.tensor(Z, dtype=torch.float32), []
    for _ in range(400):
        source, partner, sign = pairs.draw(20)
        losses.append(float(_pair_loss(Zt, torch.from_numpy(source), torch.from_numpy(partner),
                                       torch.from_numpy((sign + 1) / 2).float(),
                                       torch.from_numpy(node_weight[source]).float(), s)))
    assert abs(np.mean(losses) - energy) < .01 * scale


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
@pytest.mark.parametrize("engine", ["auto", "dense"])
def test_sine_log_distances_follow_the_excess(engine):
    """LinLog relative to the configuration null: groups with more links than expected end up
    closer (A-B strongly tied, B-C weakly, A-C hardly)."""
    import networkx as nx
    sizes = [40, 40, 40]
    G = nx.stochastic_block_model(sizes, [[.5, .12, .01], [.12, .5, .04], [.01, .04, .5]], seed=4)
    Z = dn.disentangled_embeddings(G, dn.fit_configuration_null(G), dim=2, random_state=0,
                                   method=dn.Sine(repulsion="log", epochs=300, engine=engine))
    Z = np.array([Z[i] for i in range(120)])
    centre = [Z[40 * k:40 * (k + 1)].mean(axis=0) for k in range(3)]
    gap = lambda a, b: np.linalg.norm(centre[a] - centre[b])
    spread = np.mean([np.linalg.norm(Z[40 * k:40 * (k + 1)] - centre[k], axis=1).mean() for k in range(3)])
    assert spread < gap(0, 1) < gap(1, 2) < gap(0, 2)


def test_sine_text_is_stable():
    """Settings written before repulsion and power existed keep the same text (cache keys)."""
    assert repr(dn.Sine()) == ("Sine(epochs=100, lr=0.1, temperature=0.5, samples_per_node=15, margin=2.0, "
                               "sampling='proportional', engine='auto', init='spectral', init_scale=1.0)")
    assert repr(dn.Sine(repulsion="log", power=2.)).endswith("init_scale=1.0, repulsion='log', power=2.0)")
    assert dn.Sine(repulsion="log") != dn.Sine()
