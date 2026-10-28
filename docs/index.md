# disentangled-net

**Discover network communities and embeddings beyond the effects of known node attributes.**

Links in a network can reflect several overlapping patterns. In a network of
social ties between regions, for example, geographic proximity may dominate the
communities you detect or the network layout you draw. How can you explore the
connections that geography alone does not explain?

`disentangled-net` lets you specify the attribute you want to account for. It
models how that attribute and node degrees shape connectivity, then uses the
links that are stronger or weaker than expected to find:

- **Communities:** groups of nodes with more connections than expected under
  that model.
- **Embeddings:** coordinates for each node, for visualising or analysing
  network structure beyond the modelled effect.

The attribute can be **continuous**, such as age or geographic coordinates, or
**categorical**, such as language, country or organisation. The library selects
the model and computational settings automatically; you can also specify them
explicitly for research or custom applications.

The methods are described in the accompanying {doc}`paper` by Guilhem Dupuy
and Rémy Cazabet (manuscript; public link forthcoming).

## Get started

Install from GitHub while the package release is being prepared:

```bash
python -m pip install "git+https://github.com/Yquetzal/disentangled_structures_inferences_on_networks.git@v2"
```

Give a graph and the name of a node attribute. This runnable example accounts
for the categorical `"club"` attribute in the Karate Club network:

```python
import networkx as nx
import disentangled_net as dn

G = nx.karate_club_graph()
partition = dn.disentangled_communities(G, attribute="club", random_state=0)
embedding = dn.disentangled_embeddings(G, attribute="club", dim=2)
```

`partition` maps each node to a community; `embedding` maps each node to its
coordinates. Replace `"club"` with your own attribute. The {doc}`guide/quickstart`
explains numeric attributes, geographic coordinates and categorical codes.

## Explore the documentation

- {doc}`guide/quickstart`: use your own graph and attribute, and reuse a fitted model.
- {doc}`guide/nulls`: choose or customise the model of expected connectivity.
- {doc}`guide/communities` and {doc}`guide/embeddings`: control the analysis and interpret its outputs.
- {doc}`guide/synthetic`: generate networks with known structure to test the methods.
- {doc}`api`: full function and parameter reference.

```{toctree}
:hidden:
guide/quickstart
paper
guide/nulls
guide/communities
guide/embeddings
guide/synthetic
api
releasing
```
