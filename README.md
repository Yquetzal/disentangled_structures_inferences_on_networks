# decorrnet
A Python library designed for decorrelated structure inferences on networks (communities and embeddings). It provides advanced topological tools to purge networks of spatial (given nodes position) or custom null model effects, allowing the extraction of unbiased communities and node embeddings.

> **Citation Note:** If you use this code or repository for your research paper, please cite: *[Insert Citation Here Upon Publication]*.

## Installation
To install the official release via PyPI:
```bash
pip install decorrnet
```

To install in development mode from the cloned repository root:
```Bash
pip install -e .
```

## Repository Structure & Core Modules
The library is modularly designed around five key components inside the `src/` directory:

- `communities.py`: Exposes the public API for community detection. It includes two callable functions: `infer_spatial_decorrelated_communities` to extract communities using node coordinates, and `infer_custom_decorrelated_communities` to infer communities using a user-provided probability matrix of size $N \times N$ representing a custom Null Model.

- `embeddings.py`: Implements a custom PyTorch-based Signed Network Embedding (SiNE) algorithm. It optimizes a continuous node representation space based on network residual intensities, leveraging either spatial positions or a custom $N \times N$ external Null Model matrix.

- `SpatialNullModelInference.py`: Handles the maximum likelihood estimation (MLE) of the Degree-Corrected Spatial Null Model. It infers coordinate-based $\alpha$ and $\beta$ parameters to generate the background spatial probability matrix.

- `MetaLouvain.py`: Contains the core modularity optimization algorithms tailored for the inference of Null-Model decorrelated network partitions.

- `utils.py`: Provides mathematical and structural validation utilities, such as checking partition robustness across different resolution scales.

## Quick Start
```Python
import networkx as nx
from decorrnet import infer_spatial_decorrelated_communities, infer_spatial_decorrelated_embedding

# 1. Load a NetworkX Graph with a spatial position attribute
G = nx.erdos_renyi_graph(n=50, p=0.1)

# 2. Extract spatially decorrelated communities (modifies G in place)
G = infer_spatial_decorrelated_communities(
    G=G, 
    pos_attr="GT_pos", 
    attr_name="spatially_decorrelated_louvain_id"
)

# 3. Generate spatially decorrelated continuous embeddings (modifies G in place)
G = infer_spatial_decorrelated_embedding(
    G=G, 
    pos_attr="GT_pos", 
    attr_name= "spatially_decorrelated_embedding",
    embedding_dim=64
)

# Nodes now contain both decorrelated attributes
print(G.nodes[0])
```

If you want to infer decorrelated communities and embeddings from something else thant node positions, you can do so by using your own NullModel matrix, of size N_nodes * N_nodes :

```Python
import networkx as nx
from decorrnet import infer_custom_decorrelated_communities, infer_custom_decorrelated_embedding

G = nx.erdos_renyi_graph(n=50, p=0.1)
nodes_list = list(G.nodes())

# Example: a dummy 50x50 probability matrix
customNullModel = np.full((50, 50), 0.05)

# 2. Extract custom decorrelated communities (modifies G in place)
G = infer_custom_decorrelated_communities(
    G=G, 
    null_model_matrix=customNullModel, 
    ordered_nodes=nodes_list,
    attr_name="custom_decorrelated_louvain_id"
)

# 3. Generate custom decorrelated continuous embeddings (modifies G in place)
G = infer_custom_decorrelated_embedding(
    G=G, 
    null_model_matrix=customNullModel, 
    attr_name="custom_decorrelated_embedding",
    embedding_dim=64
)

# Nodes now contain both custom decorrelated attributes
print(G.nodes[0])
```

## Authors
DUPUY Guilhem - guilhempds@gmail.com
CAZABET Rémy - remy.cazabet@gmail.com