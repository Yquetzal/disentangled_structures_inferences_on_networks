# disentangled-net
A Python library designed for disentangled structure inferences on networks (communities and embeddings). It provides advanced topological tools to purge networks of spatial (given nodes position) or custom null model effects, allowing the extraction of unbiased communities and node embeddings.

> **Citation Note:** If you use this code or repository for your research paper, please cite: *[Insert Citation Here Upon Publication]*.

## Installation
To install the official release via PyPI:
```bash
pip install disentangled-net
```

To install in development mode from the cloned repository root:
```Bash
pip install -e .
```

## Repository Structure & Core Modules
The library is modularly designed around five key components inside the `src/` directory:

- `disentangled_louvain_communities.py`: Exposes the public API for community detection. It includes two callable functions: `infer_spatial_disentangled_louvain_communities` to extract communities using node coordinates, and `infer_custom_disentangled_louvain_communities` to infer communities using a user-provided probability matrix of size $N \times N$ representing a custom Null Model.

- `disentangled_embeddings.py`: Implements a custom PyTorch-based signed network embedding algorithm. It optimizes a continuous node representation space based on network residual intensities, leveraging either spatial positions or a custom $N \times N$ external Null Model matrix. Available functions : `infer_spatial_disentangled_embeddings` and `infer_custom_disentangled_embeddings`.

- `SpatialNullModelInference.py`: Handles the maximum likelihood estimation (MLE) of the Degree-Corrected Spatial Null Model. It infers coordinate-based $\alpha$ and $\beta$ parameters to generate the background spatial probability matrix.

- `MetaLouvain.py`: Contains the core modularity optimization algorithms tailored for the inference of Null-Model disentangled network partitions.

- `utils.py`: Provides mathematical and structural validation utilities, such as checking partition robustness across different resolution scales.

## Quick Start
```Python
import networkx as nx
import numpy as np
from disentangled_net import infer_spatial_disentangled_louvain_communities, infer_spatial_disentangled_embeddings

# 1. Load a NetworkX Graph with a spatial position attribute
G = nx.erdos_renyi_graph(n=50, p=0.1)

# Example : a random 2D position in [0;1] is assigned to every node
for node in G.nodes:
    G.nodes[node]['GT_pos'] = np.random.rand(2)

# 2. Extract spatially disentangled communities (modifies G in place)
G = infer_spatial_disentangled_louvain_communities(
    G=G, 
    pos_attr="GT_pos", 
    attr_name="spatially_disentangled_louvain_id"
)

# 3. Generate spatially disentangled continuous embeddings (modifies G in place)
G = infer_spatial_disentangled_embeddings(
    G=G, 
    pos_attr="GT_pos", 
    attr_name= "spatially_disentangled_embedding",
    embedding_dim=64
)

# Nodes now contain both disentangled attributes
print(G.nodes[0])
```

If you want to infer disentangled communities and embeddings from something else than node positions, you can do so by using your own NullModel matrix, of size N_nodes * N_nodes :

```Python
import networkx as nx
import numpy as np
from disentangled_net import infer_custom_disentangled_louvain_communities, infer_custom_disentangled_embeddings

G = nx.erdos_renyi_graph(n=50, p=0.1)
nodes_list = list(G.nodes())

# Example: a dummy 50x50 probability matrix
customNullModel = np.full((50, 50), 0.05)

# 2. Extract custom disentangled communities (modifies G in place)
G = infer_custom_disentangled_louvain_communities(
    G=G, 
    null_model_matrix=customNullModel, 
    ordered_nodes=nodes_list,
    attr_name="custom_disentangled_louvain_id"
)

# 3. Generate custom disentangled continuous embeddings (modifies G in place)
G = infer_custom_disentangled_embeddings(
    G=G, 
    null_model_matrix=customNullModel, 
    ordered_nodes=nodes_list,
    attr_name="custom_disentangled_embedding",
    embedding_dim=64
)

# Nodes now contain both custom disentangled attributes
print(G.nodes[0])
```

## Authors
DUPUY Guilhem - guilhempds@gmail.com
CAZABET Rémy - remy.cazabet@gmail.com