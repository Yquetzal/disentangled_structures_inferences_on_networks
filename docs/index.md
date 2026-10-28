# disentangled-net

Find the structure of a network that a **null model** does not explain.

Many networks are shaped by a known factor: distance, existing groups, degrees. Classic community detection rediscovers that factor. Give `disentangled-net` the known attribute: it fits the expected links, then finds communities or embeddings in what remains.

```bash
pip install "disentangled-net>=2"            # communities, once v2 is published
pip install "disentangled-net[embeddings]>=2" # + PyTorch embedding methods
```

This documentation describes version 2. Until it is published, install the v2
checkout with `python -m pip install .` or `python -m pip install ".[embeddings]"`.
The published 0.1 release has a different API.

## Give the attribute

```python
import disentangled_net as dn

partition = dn.disentangled_communities(G, attribute="age")
embedding = dn.disentangled_embeddings(G, attribute="age", dim=2)
```

`G` is a networkx graph with an `"age"` node attribute. Strings and booleans give a categorical null; numbers and vectors give a continuous one. The default path chooses compatible computations automatically. See the {doc}`guide/quickstart` for countries, geographic positions, sparse input and reusing a fitted model. It needs no knowledge of landmarks or sampling.

## Advanced use

1. **Build a null model**: fit one to the graph ({doc}`guide/nulls`): spatial, block model, configuration model. You can also supply any matrix or function.
2. **Find communities relative to it** ({doc}`guide/communities`), maximising

   $$Q = \frac{1}{m}\sum_{u<v,\ \text{same community}} \left(A_{uv} - \gamma P_{uv} - \varepsilon\right).$$

   The attribute workflow handles the computational choice. Explicit approximations are available for custom models and controlled experiments.
3. **Read and reuse** the result: `partition_quality`, `excess_density`, `epsilon_profile`, `disentangled_embeddings` ({doc}`guide/embeddings`).

To test a method on graphs where the answer is known, see {doc}`guide/synthetic`.

New in version 2: see {doc}`whats_new`.

```{toctree}
:hidden:
guide/quickstart
guide/nulls
guide/communities
guide/embeddings
guide/synthetic
whats_new
api
releasing
```
