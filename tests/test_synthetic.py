import numpy as np
import pytest
import scipy.sparse as sp

import disentangled_net as dn
from disentangled_net import synthetic as syn


def test_planted_communities():
    A, labels = syn.planted_communities(600, seed=1)
    assert len(np.unique(labels)) == round(np.sqrt(600) / 2) and A.shape == (600, 600)
    rows, cols = sp.triu(A, 1).nonzero()
    assert np.all(labels[rows] == labels[cols])                     # no edge between communities
    degrees = np.asarray(A.sum(1)).ravel()
    for c in np.unique(labels):
        d = degrees[labels == c]
        assert d.min() == d.max()                                   # regular inside a community
        assert abs(d[0] - ((labels == c).sum() - 1) ** 0.6) <= 2


def test_triangle_decoding():
    k = np.arange(100_000)
    j, i = syn._triangle(k)
    assert np.all(i * (i - 1) // 2 + j == k) and np.all(j < i)


def test_block_model_rates():
    labels = np.repeat([0, 1], 300)
    B = np.array([[.05, .01], [.01, .08]])
    A = syn.block_model(labels, B, seed=0)
    within0 = A[:300][:, :300].sum() / 2
    assert within0 == pytest.approx(.05 * 300 * 299 / 2, rel=.1)
    assert A[:300][:, 300:].sum() == pytest.approx(.01 * 300 * 300, rel=.15)


@pytest.mark.parametrize("noise", ["contact", "exponential", "logistic"])
def test_spatial_benchmark(noise):
    b = dn.spatial_benchmark(1024, noise=noise, rho=3, beta=8, seed=41)
    n, info = 1024, b.info
    mean_size = n / info["communities"]
    assert (n - 1) * syn.mean_proximity(info["length"]) == pytest.approx(mean_size - 1, rel=1e-9)
    assert info["noise_edges"] == pytest.approx(3 * info["signal_edges"], rel=.05)
    P = b.null.to_matrix()
    iu = np.triu_indices(n, 1)
    assert P[iu].sum() == pytest.approx(info["q"] * len(iu[0]) + (1 - info["q"]) * 3 * info["signal_edges"], rel=.03)


@pytest.mark.parametrize("rho,share", [(3, 1.), (32, 1.), (32, .5), (8, 0.)])
def test_contact_noise_is_smooth_and_calibrated(rho, share):
    b = dn.spatial_benchmark(512, rho=rho, share=share, seed=3)
    noise, ell = b.null.deterrence.noise, b.info["length"]
    d = np.linspace(0, 1.5, 3000)
    assert np.abs(np.diff(noise(d)) / np.diff(d)).max() * ell <= 1 / np.e + 1e-6
    D = np.linalg.norm(b.positions[:, None] - b.positions[None], axis=2)[np.triu_indices(512, 1)]
    assert noise(D).sum() == pytest.approx(rho * b.info["signal_edges"], rel=1e-6)


def test_logistic_calibration_and_pairing():
    small = dn.spatial_benchmark(512, noise="logistic", rho=3, beta=4, seed=7)
    large = dn.spatial_benchmark(512, noise="logistic", rho=8, beta=4, seed=7)
    assert ((dn.spatial_benchmark(512, rho=3, seed=7).adjacency - dn.spatial_benchmark(512, rho=8, seed=7).adjacency) > 0).nnz == 0
    noise = small.null.deterrence.noise
    D = np.linalg.norm(small.positions[:, None] - small.positions[None], axis=2)
    iu = np.triu_indices(512, 1)
    assert noise(D[iu]).sum() == pytest.approx(3 * small.info["signal_edges"], rel=1e-8)
    assert ((small.adjacency - large.adjacency) > 0).nnz == 0       # same marks: nested noise layers


def test_rejection_sampler_is_unbiased():
    rng = np.random.default_rng(0)
    X = rng.random((700, 2))
    p = syn.ExponentialNoise(.4, .08)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    expected = p(D[np.triu_indices(700, 1)]).sum()
    counts = [syn._rejection(X, p, np.random.default_rng(s), .08).sum() / 2 for s in range(20)]
    assert np.mean(counts) == pytest.approx(expected, rel=.02)


def test_empty_layers():
    X = np.random.default_rng(0).random((300, 2))
    zero = syn.ExponentialNoise(0., .1)
    assert syn.spatial_layer(X, zero, seed=0).nnz == 0
    assert syn._rejection(X, zero, np.random.default_rng(0), .1).nnz == 0
    assert syn.block_model(np.zeros(5, int), [[0.]], seed=0).nnz == 0


def test_recovery_on_the_benchmark():
    b = dn.spatial_benchmark(1024, rho=3, seed=41)
    p = dn.disentangled_communities(b.adjacency, b.null, approximation=dn.Landmarks(128), random_state=0)
    assert dn.adjusted_rand(b.labels, [p[i] for i in range(1024)]) > .9


def test_degree_corrected_block_model():
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 200)
    theta = rng.pareto(2.5, 400) + 1
    B = np.array([[.02, .004], [.004, .03]])
    P = 1 - np.exp(-np.outer(theta, theta) * B[np.ix_(labels, labels)])
    np.fill_diagonal(P, 0)
    draws = [syn.block_model(labels, B, degrees=theta, seed=s) for s in range(30)]
    mean = sum(draws).toarray() / len(draws)
    assert mean.sum() / 2 == pytest.approx(P.sum() / 2, rel=.01)
    degrees = mean.sum(1)
    assert np.corrcoef(degrees, P.sum(1))[0, 1] > .99                 # hubs stay hubs
    assert all(d.diagonal().sum() == 0 and d.max() == 1 for d in draws)


@pytest.mark.parametrize("dense", [True, False])
def test_degree_corrected_spatial_layer(dense):
    rng = np.random.default_rng(1)
    X = rng.random((600, 2))
    theta = rng.pareto(2.5, 600) + 1
    p = syn.ContactNoise(.02, 1., .1)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    P = 1 - np.exp(-np.outer(theta, theta) * -np.log1p(-p(D)))
    np.fill_diagonal(P, 0)
    draw = (lambda s: syn.spatial_layer(X, p, degrees=theta, seed=s)) if dense else \
        (lambda s: syn._rejection(X, p, np.random.default_rng(s), .1, theta))
    mean = sum(draw(s) for s in range(30)).toarray() / 30
    assert mean.sum() / 2 == pytest.approx(P.sum() / 2, rel=.03)            # ~3 standard errors
    assert np.corrcoef(mean.sum(1), P.sum(1))[0, 1] > .98
    target = P.sum() / 4
    x = syn.calibrate(X, lambda x: syn.ContactNoise(np.exp(x), 1., .1), target, -20., 5., degrees=theta)
    assert x == pytest.approx(np.log(.02) + np.log(.5), abs=.25)


def test_degree_corrected_oracle():
    rng = np.random.default_rng(2)
    X, theta, labels = rng.random((300, 2)), rng.pareto(2.5, 300) + 1, np.repeat([0, 1, 2], 100)
    signal = syn.block_model(labels, np.eye(3) * .05, degrees=theta, seed=0)
    noise = syn.ContactNoise(.01, 1., .2)
    b = dn.mixed_benchmark(signal, labels, X, noise, degrees=theta, seed=1, q=.01)
    P = b.null.to_matrix()
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    W = np.outer(theta, theta)
    expected = 1 - np.exp(-.01 * W - .01 * W * np.exp(-D / .2))
    np.fill_diagonal(expected, 0)
    assert np.allclose(P, expected)
    partition = dn.disentangled_communities(b.adjacency, b.null, random_state=0)
    assert dn.adjusted_rand(labels, [partition[i] for i in range(300)]) > .5


def test_large_contact_calibration():
    X = np.random.default_rng(5).random((3000, 2))
    D = np.linalg.norm(X[:, None] - X[None], axis=2)[np.triu_indices(3000, 1)]
    for ell in (.3, .03):
        g = syn.ContactNoise(.2, 1., ell, syn.mean_proximity(ell))
        assert syn._square_mean(g, ell) == pytest.approx(g(D).mean(), rel=.03)
    import time
    start = time.perf_counter()
    b = dn.spatial_benchmark(8000, rho=3, seed=0)                    # 3.2e7 pairs: calibrated on the law
    assert time.perf_counter() - start < 60
    assert b.info["noise_edges"] == pytest.approx(3 * b.info["signal_edges"], rel=.03)


def test_spatial_benchmark_community_options():
    default = dn.spatial_benchmark(600, rho=2, seed=3)
    same = dn.spatial_benchmark(600, rho=2, communities=None, exponent=0.6, seed=3)
    assert (default.adjacency != same.adjacency).nnz == 0
    sparse = dn.spatial_benchmark(600, rho=2, communities=4, exponent=0.4, seed=3)
    assert sparse.info["communities"] == 4
    signal_degree = 2 * sparse.info["signal_edges"] / 600
    assert abs(signal_degree - 149 ** 0.4) < 1.5
