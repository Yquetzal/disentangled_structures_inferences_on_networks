import networkx as nx
import numpy as np
import pytest

import disentangled_net as dn
from disentangled_net._geometry import distance_matrix


def _spatial(n=12, metric="euclidean", link="log"):
    rng = np.random.default_rng(0)
    X = rng.random((n, 2)) * (60 if metric == "haversine" else 1)
    det = dn.Binned(np.array([0., .3, .6, 1e9]), np.array([.5, .2, .05])) if metric == "euclidean" else dn.Exponential(.001, .3)
    return dn.SpatialNull(X, det, rng.random(n) + .5, metric=metric, link=link)


@pytest.mark.parametrize("null", [
    _spatial(), _spatial(metric="haversine"), _spatial(link="logit"),
    dn.BlockNull([0, 0, 1, 1, 2, 2], [[1., .2, .1], [.2, 2., .3], [.1, .3, .5]], [1, 2, 1, 1, 3, 1]),
    dn.custom_null(np.ones((5, 5))),
    dn.custom_null(lambda u, v: 1. / (1 + abs(u - v)), range(6)),
    dn.custom_null(lambda U, V: 1. / (1 + np.abs(np.subtract(U, V))), range(6), vectorized=True),
])
def test_block_is_symmetric_nonnegative_with_zero_diagonal(null):
    M = null.to_matrix()
    assert np.allclose(M, M.T) and (M >= 0).all() and np.all(np.diag(M) == 0)
    I, J = np.triu_indices(null.n, 1)
    assert np.allclose(null.pairs(I, J), M[I, J])
    assert np.allclose(null.expected_degrees(), M.sum(1))
    assert null(null.nodes[0], null.nodes[1]) == pytest.approx(M[0, 1])


def test_sum_and_reindex():
    a, b = _spatial(6), dn.custom_null(np.full((6, 6), .1))
    s = a + b
    assert np.allclose(s.to_matrix(), a.to_matrix() + b.to_matrix())
    order = [3, 1, 5, 0, 2, 4]
    r = s.reindexed(order)
    assert all(r(u, v) == pytest.approx(s(u, v)) for u in order for v in order)


def test_probability_values_and_validation():
    assert (_spatial(link="logit").to_matrix() <= 1).all()
    with pytest.raises(ValueError):
        dn.custom_null(lambda u, v: float(u - v), range(5))           # negative and asymmetric
    with pytest.raises(ValueError):
        dn.custom_null(np.array([[0., 1.], [2., 0.]]))
    with pytest.raises(MemoryError):
        _spatial().to_matrix(max_bytes=8)


def test_haversine_distance():
    paris, london = np.array([[48.8566, 2.3522]]), np.array([[51.5074, -0.1278]])
    assert distance_matrix(paris, london, "haversine")[0, 0] == pytest.approx(343.6, abs=1)
