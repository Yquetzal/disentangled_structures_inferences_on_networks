# disentangled-net

Communities and embeddings relative to a **null model**: find the structure of a network that distance, known groups or degrees do not explain.

This checkout is **version 2.0.0**, with a new API incompatible with 0.1. Until
2.0.0 is published, install it from this directory:

```bash
python -m pip install .
# Add PyTorch for SiNE and contrastive embeddings:
python -m pip install ".[embeddings]"
```

```python
import disentangled_net as dn
import networkx as nx

G = nx.karate_club_graph()
partition = dn.disentangled_communities(G, attribute="club", random_state=0)
embedding = dn.disentangled_embeddings(G, attribute="club", dim=2)
```

Replace `"club"` with the node attribute you want to account for: `"age"`,
`"country"`, `"pos"`, etc. **You do not need to choose a null model or landmarks.**
Numbers/vectors are treated as continuous, strings/booleans as categories.
Use `attribute_type="categorical"` for numeric category codes. For geographic
`(latitude, longitude)` coordinates, add `metric="haversine"`.

The automatic path uses exact computations on small graphs, exact compression
of repeated values, or landmarks on larger continuous inputs. Fitting and inference
use the **same** prepared model. Communities use ε equal to graph density;
default automatic embeddings are normalized spectral and need no PyTorch.
See the [quick start](https://github.com/Yquetzal/disentangled_structures_inferences_on_networks/blob/v2/docs/guide/quickstart.md).

To fit once and reuse the same choices for both outputs:

```python
null = dn.fit_null(G, "club")
partition = dn.disentangled_communities(G, null, random_state=0)
embedding = dn.disentangled_embeddings(G, null, dim=2)
print(null.diagnostics)
```

The explicit `fit_spatial_null` / `fit_sbm_null` interface remains available for
advanced settings and reproducing experiments. Its existing defaults are unchanged.

- Null models: spatial (binned or exponential deterrence, Poisson or Bernoulli), stochastic block model, configuration model, or any matrix or function.
- Communities maximise $\sum_{u<v \text{ same}} (A_{uv} - \gamma P_{uv} - \varepsilon)$. The additive resolution ε sets a minimal excess density for merges.
- Large spatial graphs: landmarks turn the spatial null into a block model over geographic cells; the hybrid keeps exact distances at short range.
- Input: networkx graphs or `scipy.sparse` adjacency matrices.
- Joint distance and within-group effects: `fit_spatial_null(..., groups="country")`.

Read the [documentation](https://github.com/Yquetzal/disentangled_structures_inferences_on_networks/tree/v2/docs),
[migration notes](https://github.com/Yquetzal/disentangled_structures_inferences_on_networks/blob/v2/docs/whats_new.md)
and [publication guide](https://github.com/Yquetzal/disentangled_structures_inferences_on_networks/blob/v2/docs/releasing.md). Build the HTML documentation locally
with `python -m pip install ".[docs]"` followed by
`python -m sphinx -n -W -b html docs docs/_build/html`.

Run tests with `python -m pip install ".[test,embeddings]"` and
`python -m pytest -q`. PyTorch is optional for the base package.

Authors: Guilhem Dupuy and Rémy Cazabet. Released under the [MIT license](https://github.com/Yquetzal/disentangled_structures_inferences_on_networks/blob/v2/LICENSE).
