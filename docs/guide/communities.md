# Communities

The simplest call is `dn.disentangled_communities(G, attribute="age")`; it
handles the null, resolution and computation automatically ({doc}`quickstart`).
The following options describe explicit control with a supplied null model.

```python
partition = dn.disentangled_communities(G, null, epsilon=0.005, random_state=0)
partition, info = dn.disentangled_communities(G, null, return_info=True)
```

The search is a multilevel Louvain. It maximises

$$Q = \frac{1}{m}\sum_{u<v,\ \text{same community}} \left(A_{uv} - \gamma P_{uv} - \varepsilon\right),$$

where $m$ is the total edge weight. Communities are numbered by decreasing size. With `return_info=True` you also get the quality, the number of communities and the backend used.

## Two resolutions: γ and ε

Moving from one partition to another changes $Q$ by local amounts, so each resolution reads as a merge rule. Two groups $a$ and $b$, with observed weight $O$ and expected weight $E$ between them, are merged only if:

| parameter | merge rule | meaning |
|---|---|---|
| `gamma` (default 1) | $O > \gamma E$ | a threshold on the observed/expected **ratio** |
| `epsilon` (default 0) | $\dfrac{O - \gamma E}{\lvert a\rvert\,\lvert b\rvert} > \varepsilon$ | a threshold on the **excess density** |

`gamma` follows the usual convention (networkx, igraph, leidenalg): larger values give more, smaller communities.

`epsilon` is the minimal excess of links per pair of nodes that justifies a merge. With $\varepsilon = 0$, any positive excess suffices, however diluted. Two large groups can then be glued by a handful of links that are each plausible on their own: a typical failure with spatial nulls, where far-apart pairs have tiny expectations. The objective with ε is a Constant Potts Model of resolution ε on the residual $A - P$. In a partition where no merge improves the objective, no two communities have an excess density above ε between them; the bounded heuristic search is not guaranteed to reach such a partition.

ε is a density per *pair of nodes*, so its scale depends on the size of the units. In the article, use the graph density for applications (including link prediction), and zero when evaluating planted-community recovery alone. For a simple unweighted graph:

```python
epsilon = 2 * G.number_of_edges() / (len(G) * (len(G) - 1))
partition = dn.disentangled_communities(G, null, epsilon=epsilon, random_state=0)
```

For weighted graphs the analogous scale is total edge weight divided by the number of unordered pairs. A stability profile is an optional diagnostic, not the article's selection rule:

```python
rows = dn.epsilon_profile(G, null, [0, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2], seeds=range(10))
```

Each row gives the number of communities, the best quality and the agreement between seeds (adjusted Rand index). Take ε in the middle of a stable plateau. To check a particular merge, `dn.excess_density(G, null, nodes_a, nodes_b)` returns $O$, $E$ and the density that ε is compared with.

(approximations)=
## Exact or approximate

By default the sums $\sum_{v \in c} P_{uv}$ are exact. The engine picks the fastest exact method for the null:

| null | method | memory |
|---|---|---|
| block nulls (SBM, configuration) | block sums | O(n · blocks) |
| small graphs (any null, n ≲ 8 000) | dense matrix | 2 n² floats |
| larger graphs (spatial, custom function) | tiles evaluated on demand; dense once the coarsened graph fits | O(n) |

On large graphs, exact sums on a spatial null cost O(n²) time per sweep. Two approximations trade precision for speed. The result reports which one was used.

(landmarks)=
### Landmarks: the spatial null as a block model

```python
dn.disentangled_communities(G, null, approximation=dn.Landmarks(rank=256))
```

```{figure} ../_static/landmarks.svg
:width: 100%

The same node (orange) under the three evaluations. Left, exact: true distance to every node. Middle, landmarks: nodes are grouped into cells around representatives (purple), and a pair takes the distance between its representatives. Right, hybrid: pairs closer than R (dashed circle) keep their true distance.
```

Each node joins the cell of its nearest representative, and the distance between two nodes is replaced by the distance between their representatives. It is zero inside a cell. Then

$$\hat P_{uv} = \theta_u \theta_v\, f(D_{\text{cell}(u),\,\text{cell}(v)}),$$

which is a **degree-corrected block model whose blocks are geographic cells** (not communities). The expected weight between a node and a community then only needs the community's mass in each cell: O(1) per evaluation, memory O(n r).

| option | role |
|---|---|
| `rank` | number of representatives r |
| `placement` | `"centroids"` (default): mass-weighted k-means, so cells follow the mass. `"farthest"`: cells spread evenly over space, empty areas included |
| `refit` | re-fit masses and deterrence on the cell geometry, with the settings of `fit_spatial_null` (default `True`). Without it the exact parameters meet approximate distances, which is much less accurate |
| `exact_within`, `exact_neighbors` | hybrid mode, below |

### Hybrid: exact at short range

```python
dn.disentangled_communities(G, null, approximation=dn.Landmarks(rank=128, exact_within=1000))
```

Cells err mostly at short range. Inside a cell, distances become zero; between neighbouring cells, the error can reach the cell size. Yet short distances are where the deterrence varies fastest, and where most of the expected weight between a node and its own community lies.

The hybrid keeps the true distance for near pairs. They are stored as a sparse correction of about k pairs per node and handled like extra edges, so memory is O(n (r + k)). Near pairs are defined in one of two ways:

- **`exact_within=R`:** pairs closer than R, in metric units (km for haversine). k then grows with the local density of nodes.
- **`exact_neighbors=k`:** each node's k nearest nodes. The budget is fixed, whatever the density, which suits graphs with very uneven densities or large sizes.

With both options, the near pairs are the union of the two sets.

On the article's Facebook Social Connectedness Index (3 168 regions, 20 equal-link distance bins, ε = density; ten runs):

| evaluation | quality lost vs exact |
|---|---|
| landmarks, r = 128 | 2.0 % |
| landmarks, r = 512 | 0.08 % |
| hybrid, r = 128, `exact_within=1000` km | 0.43 % |
| hybrid, r = 128, `exact_neighbors=50` | 0.77 % |

Raising r alone converges slowly; adding the near field is much more effective.

The hybrid helps when the deterrence varies fastest at short range, as it usually does. If the null changes abruptly at a larger distance, landmarks misplace pairs around that distance, and only a larger rank or exact evaluation helps.

### Sampling: any null

```python
dn.disentangled_communities(G, null, approximation=dn.Sampling(sample_size=32))
```

Each sum over |node| × |community| pairs is estimated from `sample_size` uniform draws. It is exact when there are fewer pairs than that. The estimates are unbiased (`background_weight=0`), but gains are noisy, so passes and levels are bounded (`max_passes=20`, `max_levels=16`). `dn.Sampling.fast()` is an aggressive preset. The reported quality is an estimate; use `partition_quality` for the exact value.

Use sampling only when no structure can be exploited. When a few pairs carry most of the expected weight, as with spatial nulls, uniform draws miss them. On the SCI example above, `Sampling(sample_size=64)` loses 18 % of quality and merges unrelated regions. Prefer landmarks for spatial nulls.

## Other options

| option | role |
|---|---|
| `init` | initial partition `{node: community}` |
| `n_iterations` | full multilevel cycles (default 2), each restarting from the original nodes with the current partition |
| `standardize` | optimise $(A - P)/\sqrt{P(1-P)}$; needs a probability null (`likelihood="bernoulli"`) |
| `random_state`, `max_passes`, `max_levels`, `tolerance` | search control |
