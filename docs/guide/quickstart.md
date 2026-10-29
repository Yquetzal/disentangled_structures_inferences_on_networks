# Give the attribute

The ordinary workflow needs a graph and a known node attribute:

```python
import networkx as nx
import disentangled_net as dn

G = nx.karate_club_graph()  # includes the categorical attribute "club"
partition = dn.disentangled_communities(G, attribute="club", random_state=0)
Z = dn.disentangled_embeddings(G, attribute="club", dim=2)
```

Both results are dictionaries indexed by the original node IDs. The graph is
unchanged. The library fits what the attribute explains, then looks for remaining
structure; this does not guarantee statistical independence from the attribute.

## Your own attributes

What the attribute explains is a *null model*: for every pair of nodes $u, v$, the
number of links $P_{uv}$ expected from the attribute and the degrees alone. The
library fits it on your graph, then looks for the structure that it does not
explain. The last column gives the model fitted for each kind of attribute; it is
also the `null` that the functions accept instead of an attribute
(see {ref}`fit-once` and {doc}`nulls`).

| Attribute | Call on your graph | Interpretation | Fitted null model |
|---|---|---|---|
| Age, income, a numeric score | `dn.disentangled_communities(G, attribute="age")` | Effect of distance between numeric values | {ref}`Spatial <spatial-null>`: $\theta_u\theta_v f(\lvert x_u - x_v\rvert)$ |
| Categorical attributes: country, language, organisation | `dn.disentangled_communities(G, attribute="country")` | Connectivity between categories | {ref}`Block model <block-nulls>`: $\theta_u\theta_v\,\omega_{b_u b_v}$ |
| Geographic coordinates | `dn.disentangled_communities(G, attribute="pos", metric="haversine")` | `pos` contains `(latitude, longitude)` in degrees | {ref}`Spatial <spatial-null>`: $\theta_u\theta_v f(d_{uv})$, $d_{uv}$ in km |
| Integer codes for categories | `dn.disentangled_communities(G, attribute="group_id", attribute_type="categorical")` | Codes are labels, not ordered quantities | {ref}`Block model <block-nulls>`: $\theta_u\theta_v\,\omega_{b_u b_v}$ |

$\theta_u$ accounts for the degree of node $u$; $f$ is a function of the
distance, with one value per distance bin; $\omega_{ab}$ is the density of
links between categories $a$ and $b$. All are fitted on the graph.

The same arguments work for `disentangled_embeddings`. Automatic type detection
treats numbers/vectors as continuous and strings/booleans as categorical. Numeric
codes and geographic units cannot be inferred reliably: these are the two
semantic choices you may need to specify. All nodes must have a finite value or
a non-missing category.

Instead of an attribute name, you can pass `{node: value}` or an array in graph
node order. For a sparse adjacency matrix, pass values aligned with its rows:

```python
partition = dn.disentangled_communities(A, attribute=ages, random_state=0)
```

## What happens automatically

- **Categories:** a degree-corrected block model, evaluated exactly. Many
  categories use sparse block matrices and sparse community sums.
- **Repeated continuous values:** up to 256 distinct positions are compressed
  exactly, useful for ages or other discrete measurements. A constant attribute
  reduces to the configuration null.
- **Other continuous values, up to 2,000 nodes:** exact binned Poisson fitting,
  with 20 equal-link distance bins and degree correction.
- **Larger Euclidean/geographic inputs:** 128 representative positions, with the
  model fitted on that geometry and returned as a block model. The subsequent
  algorithms reuse it directly. This trades some spatial precision for speed;
  it does not fall back to evaluating every pair at its original distance.
- **Community resolution:** ε is the graph's total edge weight divided by its
  number of unordered node pairs, the application rule used in the article.
  Set `epsilon=0` explicitly for planted-community recovery experiments.
- **Embedding:** normalized spectral by default, using the same model. This
  works with the base installation. To use the article's trained pairwise
  method, install `disentangled-net[embeddings]` and pass `method="sine"`.

There is no separate approximation to choose at fitting and inference. Exact
compression keeps the original model; landmarks define an approximate one.
Custom distance functions retain exact evaluation and can therefore be slow
on large graphs. The automatic path currently accounts for one attribute at a time.

(fit-once)=
## Fit once and reuse

```python
null = dn.fit_null(G, "club")
partition, info = dn.disentangled_communities(G, null, random_state=0, return_info=True)
Z = dn.disentangled_embeddings(G, null, dim=2)

print(null.diagnostics)       # fitted model, convergence, exact/compressed/landmarks
print(info.approximation)     # exact or landmarks
print(info.details["epsilon"])
quality = dn.partition_quality(G, partition, null)
```

The one-call and fit-once paths share their defaults, including ε and the default
embedding method. Reuse the null only with the graph on which you fitted it, or
intentionally as a fixed reference. Fit on training edges only for link prediction.

No edges or a single node are supported. A categorical attribute with a different
label for every node conditions on the entire adjacency: there is no residual
information, so embeddings are zero and communities are singletons.

## Explicit scientific settings

For manual likelihoods, bins, hybrid corrections, multiple effects or a custom
matrix/function, use {doc}`nulls`, {doc}`communities` and {doc}`embeddings`.
With an explicitly fitted null model, the defaults are ε = 0 and SiNE embeddings.
The automatic model's diagnostics describe its own defaults, so it stays
consistent when passed between functions. On small continuous inputs,
`fit_spatial_null(G, ..., bins=20, binning="edges")` gives the corresponding
explicit fit.
