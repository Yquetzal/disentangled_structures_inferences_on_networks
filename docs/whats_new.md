# What's new in version 2

Version 2 is a rewrite, not compatible with 0.1.

**An attribute is enough.** `disentangled_communities(G, attribute="age")` and
`disentangled_embeddings(G, attribute="age")` fit the null and choose compatible
computations. `fit_null` lets you fit once and reuse those choices. Repeated values
are compressed exactly; large spatial attributes become fitted landmark block models;
many categories use sparse block sums. See {doc}`guide/quickstart`.

**One workflow.** First fit a null (`fit_spatial_null`, `fit_sbm_null`, `fit_configuration_null`, `custom_null`). Then call `disentangled_communities` or `disentangled_embeddings`. The old `spatial_*`, `custom_*` and `best_partition` entry points are gone. Functions no longer write into the graph, and no longer pick a resolution with a size heuristic.

**Additive resolution ε.** Merges now require a minimal excess density ($\varepsilon$), which prevents communities glued by a few diluted long-range links. `epsilon_profile` and `excess_density` help choose and check it. See {doc}`guide/communities`.

**Standard γ.** `gamma` multiplies the null, as in networkx and igraph. Larger values give smaller communities. The former `resolution` worked the other way round.

**Scalable spatial nulls.**
- Fitting needs O(n) memory instead of O(n²).
- Fitting and partitioning can both use **landmarks** (the spatial null as a block model over geographic cells).
- The **hybrid** keeps exact distances at short range.
- See {ref}`landmarks`.

**Automatic exact engine.** The fastest exact evaluation is chosen from the null: block sums, dense matrix, or tiles. **Sampling** approximates any other null, as a last resort.

**Fitting options.**
- Great-circle distances (`metric="haversine"`).
- Explicit bin edges.
- `binning="edges"` for the article's equal-link bins, and `groups=` for the additional distance/within-country experiments (exact binned Poisson fit).
- A Bernoulli likelihood with the same deterrences (`likelihood="bernoulli"`), replacing the former logistic model.
- Exponential deterrence fitted by maximum likelihood with exact degree matching.

**Inputs.** `G` can be a networkx graph or a `scipy.sparse` adjacency matrix.

**Embeddings.** `disentangled_embeddings` takes `gamma`, `epsilon` and, for the spectral method, `approximation=dn.Landmarks(...)`: it never forms the n × n residual and scales like the communities. SiNE draws its pairs in proportion to the residual, without forming it for block-structured nulls, and starts from the Laplacian eigenmaps of the residual (`Spectral(normalized=True)`). `dn.Sine(repulsion="log", power=a)` replaces its hinge repulsion by the LinLog / ForceAtlas2 energy relative to the null, for layouts and residuals made of many groups. See {doc}`guide/embeddings`.

**Synthetic benchmarks.** `spatial_benchmark` builds the planted-communities-under-spatial-noise graphs of the article, with their oracle null; `dn.synthetic` provides the building blocks, with optional degree correction of both layers. See {doc}`guide/synthetic`.
