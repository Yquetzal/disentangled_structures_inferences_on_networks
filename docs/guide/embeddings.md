# Embeddings

The automatic attribute workflow is `dn.disentangled_embeddings(G, attribute="age", dim=2)`:
it fits a compatible model and defaults to normalized spectral, without PyTorch
({doc}`quickstart`). The explicit-null interface below retains SiNE as its default;
the same is true when you explicitly pass `method="sine"` in the attribute workflow.

```python
Z = dn.disentangled_embeddings(G, null, dim=16, method="spectral")   # {node: vector}
```

Embeddings of the residual $R = A - \gamma P - \varepsilon$: nodes are close when they are linked *more* than the null predicts. `gamma` and `epsilon` mean the same as for {doc}`communities`.

| method | idea | needs |
|---|---|---|
| `"sine"` (default) | sampled pairs attract when $R > 0$ and repel when $R < 0$ | PyTorch |
| `"spectral"` | leading positive eigenvectors of $R$, scaled by $\sqrt{\lambda}$ | — |
| `dn.Spectral(normalized=True)` | Laplacian eigenmaps of the residual (below) | — |
| `"contrastive"` | balanced metric learning on positive and negative residuals | PyTorch |
| `"deepwalk"` | DeepWalk-style PPMI factorisation of $\max(R, 0)$ | — |

To tune a method, pass a settings object instead of its name: `method=dn.Sine(epochs=300, lr=0.05)`, `dn.Contrastive(...)`, `dn.DeepWalk(window_size=3)`.

**Laplacian eigenmaps of the residual.** `dn.Spectral(normalized=True)` takes the leading positive eigenvectors of $D^{-1/2} R\, D^{-1/2}$, with $D$ the observed degrees, and maps them back by $D^{-1/2}$. With the full configuration matrix, subtracting the null removes the trivial degree mode. The library excludes self-pairs, so its operator differs from that construction by a diagonal correction; it need not give exactly the same vectors as ordinary Laplacian eigenmaps. With another null, what that null explains (known groups, distance) is removed as well.

**SiNE start.** By default (`init="spectral"`), SiNE starts from the Laplacian eigenmaps of the residual, standardised to a spread of `init_scale`, and refines them with its pairwise loss; `init="random"` starts from small Gaussian coordinates.

**SiNE sampling.** Each node draws its pairs in proportion to $|R|$. With the default hinge loss, every source node receives the same sampling budget: the expected loss is $\frac{1}{n}\sum_u\sum_v |R_{uv}|\,\ell_{uv}/\sum_w|R_{uw}|$, with zero-information rows contributing zero. This is a residual-weighted loss normalized separately for each source, rather than a global sum weighted by $|R|$. When the null reproduces the degrees, attraction and repulsion balance for every node. `dn.Sine(sampling="softmax")` restores the original rule, $\mathrm{softmax}(|R| / 0.5)$.

**SiNE repulsion: hinge or LinLog.** Attracted pairs ($R > 0$) have the energy $d^a / a$ (`power=a`, default 1). By default (`repulsion="hinge"`), repelled pairs ($R < 0$) have the energy $\max(0, m - d)$: no force beyond the margin $m$, so nothing arranges groups that are already apart, and the global layout comes entirely from the start. With `repulsion="log"`, they have the energy $-\ln d$, a force $1/d$ at every distance, and the sampled loss estimates

$$E(X) = \sum_{R_{uv} > 0} |R_{uv}|\, \frac{d_{uv}^a}{a} \;-\; \sum_{R_{uv} < 0} |R_{uv}|\, \ln d_{uv} .$$

This is a signed-residual energy inspired by LinLog; its attraction and repulsion use the positive and negative parts of $R$. It differs from energies that attract along every observed edge and repel along every expected pair. For two compact groups considered in isolation, the unsoftened energy balances at $d^a=N/P$, with $P$ their total positive residual and $N$ the magnitude of their negative residual (when both are positive). Other groups also exert forces, so distances do not directly estimate connection probabilities. The long-range forces take longer to settle: use about `epochs=1000` (the step size decays to `lr / 50`).

```python
Z = dn.disentangled_embeddings(G, null, dim=2, method=dn.Sine(repulsion="log", epochs=1000))
```

Use it for 2D layouts (graph drawing) and whenever the residual is made of many discrete groups, which the hinge cannot arrange in few dimensions. On random graphs drawn from the Facebook SCI, relative to a spatial null, it improves hidden-link prediction over the hinge in 2D (average precision +0.011 at mean degree 114); in 16 dimensions both are equivalent.

## Large graphs

The spectral embedding is the continuous relaxation of the communities' objective. It never forms $R$: it only multiplies vectors by $A$ and by $P$.

```python
Z = dn.disentangled_embeddings(G, null, dim=64, method="spectral", approximation=dn.Landmarks(128))
```

- **Exact** (default): $P$ is a dense matrix below about 8,000 nodes, then evaluated in tiles; O(n) memory, O(n²) time per product.
- **Landmarks:** the approximation of {ref}`landmarks`, hybrids included; O(m + n r) per product. 65,000 nodes take a few seconds.

**SiNE** never forms $R$ either when the null has a block structure: a block null (`fit_sbm_null`), or a spatial null through landmarks (`approximation=dn.Landmarks(...)`, without exact near field). Each node then draws its partners from its edges and from the blocks, with exactly the probabilities of the dense rule: O(m + n) memory, O(n × samples) per epoch. `dn.Sine(engine="dense")` forces the dense residual.

The other methods use the dense residual: O(n²) memory.

Zero residual rows receive zero coordinates in SiNE. Spectral and DeepWalk pad unused dimensions with zeros. Contrastive returns zeros when there are not both positive and negative constraints. These zeros mean no inferred position; exclude them when interpreting distances.
