# Null models

For ordinary use, start with {doc}`quickstart`: `dn.fit_null(G, "age")` chooses
the model and its computations automatically, and can be passed directly to
communities and embeddings. The options below are for explicit scientific settings.

A null model gives the expected weight $P_{uv}$ of every pair of nodes. All nulls share one interface: `null(u, v)`, `null.expected_degrees()`, `null.to_matrix()`, `null.diagnostics`, and `null_a + null_b` to add two of them.

| function | model | typical use |
|---|---|---|
| `fit_spatial_null` | $P_{uv} = \theta_u\theta_v f(d_{uv})$ | distance shapes the links |
| `fit_sbm_null` | $P_{uv} = \theta_u\theta_v\,\omega_{b_u b_v}$ | known groups shape the links |
| `fit_configuration_null` | $P_{uv} = k_u k_v / 2m$ | only degrees (standard modularity) |
| `custom_null` | any matrix or function | anything else |

(spatial-null)=
## Spatial null

```python
null = dn.fit_spatial_null(G, "pos", metric="haversine", bins=[0, 25, 50, 100, 200, 500, 1000, 5000, 20000])
```

To use the article's settings, pass `bins=20, binning="edges"`: bins contain equal numbers of observed links (equal total observed weight for weighted graphs). Tied distances can reduce the number of bins. Fit on the training graph only for link prediction.

The masses $\theta_u$ and the deterrence $f$ are fitted by maximum likelihood. At convergence, the expected degree of every node equals its observed degree. The expected weight in every distance bin also equals the observed one.

| option | values | default |
|---|---|---|
| `positions` | node attribute name, `{node: position}`, or array aligned with the nodes | `"pos"` |
| `metric` | `"euclidean"`, `"haversine"` ((lat, lon) in degrees, km), or `metric(X, Y)` returning a distance matrix | `"euclidean"` |
| `deterrence` | `"binned"`: one free value per distance bin; `"exponential"`: $f(d) = c\,e^{-\beta d}$ | `"binned"` |
| `bins`, `binning` | number of bins with `"quantile"` (pairs), `"edges"` (observed links), `"linear"` or `"log"` spacing, or explicit edges starting at 0 | `10`, `"quantile"` |
| `degree_correction` | fit one mass per node; `False`: distance only | `True` |
| `likelihood` | `"poisson"`: expected weights, any weights; `"bernoulli"`: probabilities, logit link, binary graphs | `"poisson"` |
| `landmarks` | fit on an approximate geometry (below) | `None` |

**Cost.** Exact fitting visits every pair once per iteration: O(n²) time, but only O(n) memory. Below about 8 000 nodes, distances are cached.

**Large graphs.** `landmarks=dn.Landmarks(rank=512, exact_within=1000)` (or `exact_neighbors=50`) fits on the approximate geometry described in {ref}`landmarks`, at a cost of O(n (r + k)) per iteration. The returned null keeps the fitted parameters but still uses exact distances, unless an approximation is requested when partitioning.

`null.diagnostics` reports convergence, degree errors and observed versus expected weight per bin.
An unconverged fit emits a warning; the returned model and diagnostics describe the last evaluated parameters.

### Distance and known groups together

The additional country-aware experiment fits distance and within-country links jointly:

```python
null = dn.fit_spatial_null(G, "pos", metric="haversine", bins=20, binning="edges",
                           groups="country", group_effect="individual", max_iter=1000)
```

Within country $c$, $P_{uv}=\theta_u\theta_v f(d_{uv})b_c$; across countries, the factor is 1. Use `group_effect="shared"` for a single factor for all within-country pairs. The fit matches degrees, distance-bin totals and within-country totals (their sum in shared mode). This is a multiplicative joint fit; adding two separately fitted nulls with `+` describes a different model. This option currently supports binned Poisson fitting and exact evaluation, and uses O(n²) memory; landmarks are rejected explicitly.

(block-nulls)=
## Block and configuration nulls

```python
null = dn.fit_sbm_null(G, "party")                  # degree-corrected by default
null = dn.fit_configuration_null(G)
```

The SBM null reproduces the observed weight between every pair of blocks and inside every block. The configuration null with $\gamma = 1$, $\varepsilon = 0$ gives standard modularity.
The degree-corrected SBM sets $\theta_u=k_u/\kappa_{b_u}$ and corrects within-block rates to exclude self-pairs. It preserves block totals, but does not in general match every individual degree exactly.

For a continuous attribute with few distinct values, such as integer ages, `fit_spatial_null` also works. A landmark at every distinct value gives the exact geometry; use `landmarks=dn.Landmarks(rank=number_of_distinct_values)` when fitting and the same `approximation` when finding communities. Merely supplying the attribute does not automatically compress repeated positions.

## Any null

```python
null = dn.custom_null(P, nodes=list(G))                       # dense matrix
null = dn.custom_null(lambda u, v: ..., nodes=list(G))       # function of node ids
null = dn.fit_spatial_null(G, "pos") + dn.fit_configuration_null(G)   # sum
```

A matrix uses the fast dense engine. A function is evaluated on demand; pass `vectorized=True` if it accepts two aligned sequences of nodes. Values must be nonnegative and symmetric. They may exceed 1 (`kind="expected_weight"`, the default) or be probabilities (`kind="probability"`).
