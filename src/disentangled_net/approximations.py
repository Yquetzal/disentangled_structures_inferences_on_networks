"""Speed/precision settings, passed as ``approximation=`` (or ``landmarks=`` when fitting)."""
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class Landmarks:
    """Evaluate a spatial null on r representative positions.

    Each node joins the cell of its nearest representative; the distance between two nodes
    becomes the distance between their representatives (zero inside a cell). The null is
    then a degree-corrected SBM whose r blocks are geographic cells, so the expected weight
    between a node and a community costs O(1), with memory O(n r).

    Parameters
    ----------
    rank : int
        Number of representatives r.
    placement : {'centroids', 'farthest'}
        'centroids': mass-weighted k-means, more cells where the mass is.
        'farthest': farthest-first, cells spread evenly over space.
    exact_within : float, optional
        Hybrid mode. Pairs closer than this distance (metric units, km for 'haversine')
        keep their exact distance, stored as a sparse list of about k pairs per node:
        memory O(n (r + k)). Cells err mostly at short range, where the deterrence varies
        fastest. The number of near pairs grows with the local density of nodes.
    exact_neighbors : int, optional
        Hybrid mode with a bounded budget: each node keeps exact distances to its k
        nearest nodes (pairs symmetrised), whatever the density. Memory O(n (r + k)).
        With both options, the near pairs are the union.
    refit : bool
        Re-fit masses and deterrence on the approximate geometry, with the settings of
        :func:`fit_spatial_null` and the graph being partitioned. Without it the exact
        parameters are used on approximate distances, which is much less accurate.
        Ignored for nulls not built by fit_spatial_null.
    """
    rank: int = 128
    placement: Literal["centroids", "farthest"] = "centroids"
    exact_within: Optional[float] = None
    exact_neighbors: Optional[int] = None
    refit: bool = True


@dataclass(frozen=True)
class Sampling:
    """Approximate the expected weight between a node and a community by sampling pairs.

    Works with any null, as a last resort: when a few pairs carry most of the expected
    weight (spatial nulls), uniform draws miss them; prefer :class:`Landmarks` there.
    Each sum over the pairs between a node and a community is estimated from
    ``sample_size`` uniform draws (exact when there are fewer pairs).

    Parameters
    ----------
    sample_size : int
        Draws per (node, candidate community).
    background_weight : float
        0 gives unbiased estimates; w > 0 mixes in the running mean of past draws for the
        same community (biased, less noisy).
    max_candidates : int, optional
        Only test the strongest observed destinations (None = all).
    max_passes, max_levels : int
        Bounds on the search, which may otherwise oscillate.
    """
    sample_size: int = 16
    background_weight: float = 0.0
    max_candidates: Optional[int] = None
    max_passes: int = 20
    max_levels: int = 16

    @classmethod
    def fast(cls):
        """Aggressive preset: 2 draws, 2 candidates, 3 passes, one level."""
        return cls(sample_size=2, max_candidates=2, max_passes=3, max_levels=1)


def as_approximation(value):
    if value is None or isinstance(value, (Landmarks, Sampling)):
        return value
    if value == "landmarks":
        return Landmarks()
    if value == "sampling":
        return Sampling()
    if isinstance(value, int) and not isinstance(value, bool):
        return Landmarks(rank=value)
    raise ValueError("approximation must be None, 'landmarks', 'sampling', Landmarks(...) or Sampling(...)")
