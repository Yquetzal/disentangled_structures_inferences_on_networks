# disentangled-net

**Discover network communities and embeddings beyond the effects of known node attributes.**

Find communities hidden behind a known spatial or social structure. Run a
Louvain-based method while accounting for geographic proximity, similarities in
age or income, or a categorical attribute such as country or organisation.

Give the library the node attribute you want to account for: a **continuous
attribute** such as age or geographic coordinates, or a **categorical attribute**
such as language, country or organisation. It models connectivity based on that
attribute and node degrees, then finds communities or embeddings from the links
that are stronger or weaker than expected. The model and computational settings
are selected automatically, with explicit controls available for research.

The methods are described in the accompanying manuscript by Guilhem Dupuy and
Rémy Cazabet, [*Beyond the Dominant Meso-Scale Structure: Scalable Mixed Models
for Interpretable Network Analysis*](https://yquetzal.github.io/disentangled_structures_inferences_on_networks/paper.html).
This is a temporary reference; the public manuscript link will be added when available.

## Get started

Install from GitHub while the package release is being prepared:

```bash
python -m pip install "git+https://github.com/Yquetzal/disentangled_structures_inferences_on_networks.git@v2"
```

```python
import networkx as nx
import disentangled_net as dn

G = nx.karate_club_graph()  # includes the categorical node attribute "club"
partition = dn.disentangled_communities(G, attribute="club", random_state=0)
embedding = dn.disentangled_embeddings(G, attribute="club", dim=2)
```

`partition` maps each node to a community; `embedding` maps each node to its
coordinates. Replace `"club"` with your own attribute, such as `"age"`, `"country"`
or `"pos"`. Use `attribute_type="categorical"` for numeric category codes, and
`metric="haversine"` for geographic `(latitude, longitude)` coordinates in degrees.
See the [quick start](https://yquetzal.github.io/disentangled_structures_inferences_on_networks/guide/quickstart.html)
for complete examples and automatic model choices.

The default methods work without PyTorch. For SiNE and contrastive embeddings,
install the optional dependencies with
`python -m pip install "disentangled-net[embeddings] @ git+https://github.com/Yquetzal/disentangled_structures_inferences_on_networks.git@v2"`.

## More control

Use `fit_null` to fit once and reuse the model for both communities and embeddings:

```python
null = dn.fit_null(G, "club")
partition = dn.disentangled_communities(G, null, random_state=0)
embedding = dn.disentangled_embeddings(G, null, dim=2)
```

The [documentation](https://yquetzal.github.io/disentangled_structures_inferences_on_networks/)
covers explicit spatial and categorical models, degree correction, custom
matrices/functions, large-graph approximations and synthetic benchmarks.
Inputs can be networkx graphs or `scipy.sparse` adjacency matrices.

For development, install this checkout with `python -m pip install ".[test,docs,embeddings]"`.
Run tests with `python -m pytest -q` and build the documentation with
`python -m sphinx -n -W -b html docs docs/_build/html`.

Authors: Guilhem Dupuy and Rémy Cazabet. Released under the [MIT license](https://github.com/Yquetzal/disentangled_structures_inferences_on_networks/blob/v2/LICENSE).
