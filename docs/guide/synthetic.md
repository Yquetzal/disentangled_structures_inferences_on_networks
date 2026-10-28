# Synthetic benchmarks

```python
bench = dn.spatial_benchmark(1024, rho=8, share=0.5, seed=0)
partition = dn.disentangled_communities(bench.adjacency, bench.null)
```

A benchmark hides communities under spatial noise and knows the answer. It returns a {class}`~disentangled_net.Benchmark`:

| field | content |
|---|---|
| `adjacency` | the observed graph (sparse, binary) |
| `labels` | the planted communities |
| `positions` | node positions |
| `null` | the oracle null $P_{ij} = q + (1-q)\,p(d_{ij})$ |
| `info` | edge counts, noise parameters |

$q$ is the signal density and $p$ the noise probability. The oracle knows the noise and the amount of signal, but not the communities.

## The benchmark of the article

`dn.spatial_benchmark(n, noise=..., rho=..., beta=..., seed=...)` builds it as follows.

- **Signal:** $K = \mathrm{round}(\sqrt n / 2)$ planted communities with varying sizes. Each is a connected random regular graph of degree $\approx (s-1)^{0.6}$, with no edge between communities.
- **Positions:** uniform in the unit square, independent of the communities.
- **Noise range:** $\ell_n$ is set so that a node's weighted number of close nodes, $(n-1)\,\mathbb{E}[e^{-d/\ell_n}]$, equals the mean community size minus one. The noise neighbourhood thus grows with the communities.
- **Noise:** three shapes.
  - `noise="contact"` (default): every pair receives Poisson contacts of intensity $\lambda(d) = a\,[(1-\kappa) + \kappa\,e^{-d/\ell_n}/M]$ and is linked if it has at least one, so $p(d) = 1 - e^{-\lambda(d)}$. The parameter `share` $=\kappa$ is the spatial part of the noise (0 is uniform). The probability stays smooth for any amount of noise ($|p'| \le 1/(e\,\ell_n)$): no saturated disk with a sharp edge.
  - `noise="exponential"`: $p(d) = a\,e^{-d/\ell_n}$. It needs $a < 1$.
  - `noise="logistic"`: $p(d) = \sigma(\eta - \beta d/\ell_n)$. With large `rho` and `beta` it saturates into a disk of probability 1 with a sharp edge; prefer `"contact"`.
- **Amount of noise:** the expected number of noise edges is `rho` times the number of signal edges. For contact noise above 20 million pairs, calibration uses the distance distribution of the unit square rather than the realized positions, so the target is approximate.
- **Pairing:** the same seed shares the signal and positions across settings. Below 20 million pairs it also shares the noise marks; layers are nested when their probabilities increase pointwise. Large-graph rejection draws do not guarantee nesting.
- **Sampling:** the pair probabilities are sampled exactly for every size. Large graphs use rejection over a grid of cells, without a distance cutoff. The current implementation materializes cell pairs: memory O(n + m + c²) for c occupied cells, rather than strictly linear memory.

## Building blocks

To build other benchmarks, `dn.synthetic` exposes:
- the layers: `planted_communities`, `block_model` and `spatial_layer`;
- the noise shapes: `ContactNoise`, `ExponentialNoise` and `LogisticNoise`;
- `calibrate` and `calibrate_logistic`, which set the amount of noise.

`dn.mixed_benchmark` unites any signal layer with a spatial layer and returns the oracle null:

```python
signal = dn.synthetic.block_model(labels, B, seed=0)
noise = dn.synthetic.LogisticNoise(dn.synthetic.calibrate_logistic(X, beta=2, target=5000), beta=2)
bench = dn.mixed_benchmark(signal, labels, X, noise, seed=1)
```

## Degree correction

Real graphs have hubs. Both layers accept node masses $\theta$ (e.g. the degrees of a real graph):

```python
signal = dn.synthetic.block_model(labels, W, degrees=k, seed=0)
noise = dn.synthetic.ContactNoise(a, share=1., length=0.5)
bench = dn.mixed_benchmark(signal, labels, X, noise, degrees=k, seed=1)
```

A pair then has $\theta_u\theta_v$ independent chances of being linked:
- in the block model, $P_{uv} = 1 - e^{-\theta_u\theta_v W_{b_u b_v}}$, where $W$ holds intensities;
- in the spatial layer, $P_{uv} = 1 - (1 - p(d_{uv}))^{\theta_u\theta_v}$. With contact noise, this is $1 - e^{-\theta_u\theta_v \lambda(d_{uv})}$, a gravity model. Its slope stays below $1/(e\,\ell)$ for any masses: hubs saturate gradually, without a sharp edge.

Give both layers the same masses: hubs are then hubs in both, and the observed degrees keep the shape of $\theta$.

The oracle is then the exact probability without communities, {class}`~disentangled_net.synthetic.DegreeCorrectedUnion`. It is not a product $\theta_u\theta_v f(d)$, so it only runs with the exact engine. For landmarks, fit a null on the graph instead: `dn.fit_spatial_null(bench.adjacency, bench.positions)`.
