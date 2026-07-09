import networkx as nx

def is_partition_robust(G: nx.Graph, partition_dict: dict, K_min: int = 3, min_edge_ratio: float = 0.01) -> bool:
    """
    Vérifie si la partition contient au moins K_min communautés significatives
    en termes de proportion de liens internes.
    """
    community_edge_counts = {comm_id: 0 for comm_id in set(partition_dict.values())}
    total_edges = G.number_of_edges()
    
    if total_edges == 0:
        return False
        
    min_edges = total_edges * min_edge_ratio
    
    for u, v in G.edges():
        if u in partition_dict and v in partition_dict and partition_dict[u] == partition_dict[v]:
            community_edge_counts[partition_dict[u]] += 1
            
    robust_commus = [count for count in community_edge_counts.values() if count >= min_edges]
    return len(robust_commus) >= K_min