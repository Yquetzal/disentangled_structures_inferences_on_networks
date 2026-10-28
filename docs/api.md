# API reference

```{eval-rst}
.. currentmodule:: disentangled_net
```

## Null models

```{eval-rst}
.. autofunction:: disentangled_net.fit_spatial_null
.. autofunction:: disentangled_net.fit_null
.. autofunction:: disentangled_net.fit_sbm_null
.. autofunction:: disentangled_net.fit_configuration_null
.. autofunction:: disentangled_net.custom_null
.. autoclass:: disentangled_net.NullModel
   :members: expected_degrees, to_matrix, reindexed
.. autoclass:: disentangled_net.SpatialNull
.. autoclass:: disentangled_net.BlockNull
.. autoclass:: disentangled_net.Binned
.. autoclass:: disentangled_net.Exponential
```

## Communities

```{eval-rst}
.. autofunction:: disentangled_net.disentangled_communities
.. autoclass:: disentangled_net.Landmarks
.. autoclass:: disentangled_net.Sampling
   :members: fast
.. autoclass:: disentangled_net.CommunityInfo
.. autofunction:: disentangled_net.partition_quality
.. autofunction:: disentangled_net.epsilon_profile
.. autofunction:: disentangled_net.excess_density
.. autofunction:: disentangled_net.adjusted_rand
```

## Embeddings

```{eval-rst}
.. autofunction:: disentangled_net.disentangled_embeddings
.. autoclass:: disentangled_net.Sine
.. autoclass:: disentangled_net.Contrastive
.. autoclass:: disentangled_net.Spectral
.. autoclass:: disentangled_net.DeepWalk
```

## Synthetic benchmarks

```{eval-rst}
.. autofunction:: disentangled_net.spatial_benchmark
.. autofunction:: disentangled_net.mixed_benchmark
.. autoclass:: disentangled_net.Benchmark
.. currentmodule:: disentangled_net.synthetic
.. autofunction:: disentangled_net.synthetic.planted_communities
.. autofunction:: disentangled_net.synthetic.block_model
.. autofunction:: disentangled_net.synthetic.spatial_layer
.. autofunction:: disentangled_net.synthetic.calibrate
.. autofunction:: disentangled_net.synthetic.calibrate_logistic
.. autoclass:: disentangled_net.synthetic.ContactNoise
.. autoclass:: disentangled_net.synthetic.ExponentialNoise
.. autoclass:: disentangled_net.synthetic.LogisticNoise
.. autoclass:: disentangled_net.synthetic.DegreeCorrectedUnion
```
