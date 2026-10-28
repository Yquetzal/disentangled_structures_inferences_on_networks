"""Communities and embeddings relative to a null model.

    partition = disentangled_communities(G, attribute="age")
"""
from .approximations import Landmarks, Sampling
from .communities import (CommunityInfo, adjusted_rand, disentangled_communities, epsilon_profile,
                          excess_density, partition_quality)
from .embeddings import Contrastive, DeepWalk, Sine, Spectral, disentangled_embeddings
from .fitting import fit_configuration_null, fit_null, fit_sbm_null, fit_spatial_null
from .nulls import Binned, BlockNull, CustomNull, Exponential, NullModel, SpatialNull, SumNull, custom_null
from . import synthetic
from .synthetic import Benchmark, mixed_benchmark, spatial_benchmark

__version__ = "2.0.0"

__all__ = [
    "fit_null", "fit_spatial_null", "fit_sbm_null", "fit_configuration_null", "custom_null",
    "NullModel", "SpatialNull", "BlockNull", "CustomNull", "SumNull", "Binned", "Exponential",
    "disentangled_communities", "partition_quality", "epsilon_profile", "excess_density", "adjusted_rand",
    "CommunityInfo", "Landmarks", "Sampling",
    "disentangled_embeddings", "Sine", "DeepWalk", "Spectral", "Contrastive",
    "synthetic", "spatial_benchmark", "mixed_benchmark", "Benchmark",
]
