import inspect
import numpy as np
import networkx as nx
from typing import Callable, Optional, Dict
from .utils import is_partition_robust

# Internal package imports
from .MetaLouvain import standardized_residual_best_partition
from .SpatialNullModelInference import get_gravity_null_model_manual_iterative 


def _find_best_partition(
    G: nx.Graph, 
    partition_func: Callable, 
    K_min: int = 3, 
    min_edge_ratio: float = 0.01, 
    resolutions: Optional[list] = None, 
    **kwargs
) -> dict:
    """
    Bidirectional resolution scanner (Internal function).
    Explores community granularity to find a stable and robust partition topology.
    """
    null_model = kwargs.get('null_model', None)
    sig = inspect.signature(partition_func)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    if null_model is not None and 'null_model' not in filtered_kwargs:
        filtered_kwargs['null_model'] = null_model

    if resolutions is None:
        resolutions = [1.0]
        res_up = 1.0
        res_down = 1.0
        for _ in range(5):
            res_up *= 1.2
            res_down /= 1.2
            resolutions.append(round(res_up, 2))
            resolutions.append(round(res_down, 2))

    best_overall_partition = None
    best_res = 1.0
    
    for res in resolutions:
        if res <= 0:
            continue
            
        communities_raw = partition_func(G, resolution=res, **filtered_kwargs)
        
        if isinstance(communities_raw, dict):
            partition_dict = communities_raw.copy()
        else:
            partition_dict = {}
            for i, community in enumerate(communities_raw):
                for node in community:
                    partition_dict[node] = i

        num_commus = len(set(partition_dict.values()))
        print(f"[Log] Resolution evaluation - inferred {num_commus} communities at res = {res:.2f}")

        if best_overall_partition is None:
            best_overall_partition = partition_dict.copy()
            best_res = res

        if is_partition_robust(G, partition_dict, K_min=K_min, min_edge_ratio=min_edge_ratio):
            print(f"[Info] Robust community structure successfully identified at res = {res:.2f}")
            return partition_dict

    print(f"[Warning] No resolution level satisfied K_min={K_min} criteria.")
    print(f"[Warning] Falling back to default partition layout (res = {best_res:.2f})")
    return best_overall_partition


def infer_spatial_disentangled_louvain_communities(
    G: nx.Graph, 
    pos_attr: str = "GT_pos", 
    attr_name: str = "spatially_disentangled_louvain_id",
    K_min: int = 3, 
    min_edge_ratio: float = 0.01
) -> nx.Graph:
    """
    Generates a spatial gravity null model from node positions, infers disentangled 
    communities using standardized residuals, and returns the enriched NetworkX Graph.
    """
    P, nodes = get_gravity_null_model_manual_iterative(G, pos_attr)
    P_symmetric = (P + P.T) / 2
    
    # Asymmetry check logging
    asymmetry_sum = np.sum(np.abs(P - P_symmetric))
    max_diff = np.max(np.abs(P - P_symmetric))
    print("--- Spatial Null Model Asymmetry Analysis ---")
    print(f"Absolute difference sum (|B_before - B_after|): {asymmetry_sum:.2e}")
    print(f"Max point-wise discrepancy: {max_diff:.2e}")
    print("---------------------------------------------")

    mapping = {node: i for i, node in enumerate(nodes)}
    
    def matrix_null_model(u, v):
        return P_symmetric[mapping[u], mapping[v]]

    partition = _find_best_partition(
        G, 
        standardized_residual_best_partition, 
        K_min=K_min, 
        min_edge_ratio=min_edge_ratio, 
        null_model=matrix_null_model
    )
    
    print("--- Community Partition Diagnostics ---")
    print(f"Total assigned nodes: {len(partition)}")
    print(f"Total distinct communities found: {len(set(partition.values()))}")
    print("---------------------------------------")

    nx.set_node_attributes(G, partition, attr_name)
    return G


def infer_custom_disentangled_louvain_communities(
    G: nx.Graph, 
    null_model_matrix: np.ndarray, 
    ordered_nodes: list, 
    attr_name: str = "disentangled_louvain_id",
    K_min: int = 3, 
    min_edge_ratio: float = 0.01
) -> nx.Graph:
    """
    Infers disentangled communities using a user-provided custom null model probability matrix, 
    and returns the enriched NetworkX Graph.
    """
    P_symmetric = (null_model_matrix + null_model_matrix.T) / 2
    mapping = {node: i for i, node in enumerate(ordered_nodes)}
    
    def matrix_null_model(u, v):
        return P_symmetric[mapping[u], mapping[v]]
        
    partition = _find_best_partition(
        G, 
        standardized_residual_best_partition, 
        K_min=K_min, 
        min_edge_ratio=min_edge_ratio, 
        null_model=matrix_null_model
    )
    
    print("--- Community Partition Diagnostics ---")
    print(f"Total assigned nodes: {len(partition)}")
    print(f"Total distinct communities found: {len(set(partition.values()))}")
    print("---------------------------------------")

    nx.set_node_attributes(G, partition, attr_name)
    return G