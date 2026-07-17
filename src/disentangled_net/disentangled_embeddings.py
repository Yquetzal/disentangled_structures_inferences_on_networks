import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Tuple

# Internal package import
from .SpatialNullModelInference import get_gravity_null_model_manual_iterative


class PairwiseSignLoss(nn.Module):
    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin

    def forward(self, z: torch.Tensor, pairs: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        if pairs.shape[0] == 0:
            return z.sum() * 0.0
            
        anchors = z[pairs[:, 0]]
        neighbors = z[pairs[:, 1]]
        distances = torch.norm(anchors - neighbors, p=2, dim=1)
        
        loss_pos = distances
        loss_neg = torch.clamp(self.margin - distances, min=0.0)
        
        # Transform signs from {-1, 1} to {0, 1}
        is_pos = (signs + 1) / 2.0
        total_loss = is_pos * loss_pos + (1.0 - is_pos) * loss_neg
        return total_loss.mean()


def _generate_pairwise_samples(
    R_tensor: torch.Tensor, 
    num_samples_per_node: int = 15, 
    temperature: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sign-aware pairwise samples based on residual matrix intensities.
    """
    N = R_tensor.shape[0]
    device = R_tensor.device
    
    abs_residues = torch.abs(R_tensor)
    abs_residues.diagonal().fill_(0.0)
    
    all_pairs = []
    all_signs = []
    
    for i in range(N):
        row = abs_residues[i]
        valid_mask = row > 1e-6
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # Isolated node handling: uniform probability distribution over remaining nodes
            probs = torch.ones(N, device=device)
            probs[i] = 0.0
            probs /= probs.sum()
            eff_num_samples = min(num_samples_per_node, N - 1)
        else:
            # Scale scores using temperature before applying Softmax
            sub_scores = row[valid_indices] / temperature
            sub_probs = F.softmax(sub_scores, dim=0)
            probs = torch.zeros(N, device=device)
            probs[valid_indices] = sub_probs
            eff_num_samples = min(num_samples_per_node, len(valid_indices))
            
        if eff_num_samples > 0:
            # Sampling without replacement via multinomial sampling
            sampled_nodes = torch.multinomial(probs, num_samples=eff_num_samples, replacement=False)
            src = torch.full((eff_num_samples,), i, dtype=torch.long, device=device)
            pairs = torch.stack([src, sampled_nodes], dim=1)
            sampled_signs = torch.where(R_tensor[i, sampled_nodes] >= 0, 1.0, -1.0)
            
            all_pairs.append(pairs)
            all_signs.append(sampled_signs)
            
    if not all_pairs:
        return torch.empty((0, 2), dtype=torch.long, device=device), torch.empty((0,), device=device)
    return torch.cat(all_pairs, dim=0), torch.cat(all_signs, dim=0)


def train_custom_signed_embeddings(
    R_matrix: np.ndarray, 
    embedding_dim: int = 64, 
    epochs: int = 100, 
    lr: float = 0.01, 
    temperature: float = 0.5
) -> np.ndarray:
    """
    Optimizes a signed representation space given an arbitrary residual matrix.
    """
    N = R_matrix.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    R_tensor = torch.from_numpy(R_matrix).float().to(device)
    node_embeddings = torch.nn.Parameter(torch.randn(N, embedding_dim, device=device) * 0.1)
    
    optimizer = torch.optim.Adam([node_embeddings], lr=lr)
    loss_fn = PairwiseSignLoss(margin=2.0)
    
    print(f"[Embedding Engine] Starting training on device: {device}")
    for epoch in range(epochs):
        pairs, signs = _generate_pairwise_samples(R_tensor, num_samples_per_node=15, temperature=temperature)
        optimizer.zero_grad()
        loss = loss_fn(node_embeddings, pairs, signs)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f" -> Epoch {epoch:03d}/{epochs:03d} | Batch Loss: {loss.item():.4f}")
            
    return node_embeddings.detach().cpu().numpy()


def infer_spatial_disentangled_embeddings(
    G: nx.Graph, 
    pos_attr: str = "GT_pos", 
    attr_name: str = "spatially_disentangled_embedding",
    embedding_dim: int = 64, 
    epochs: int = 100, 
    lr: float = 0.1, 
    temperature: float = 0.5
) -> nx.Graph:
    """
    Computes node embeddings over the network matrix purged of its spatial gravity 
    null model components, storing results back into the NetworkX Graph.
    """
    start_time = time.time()
    print("[Embedding Process] Generating spatial gravity null model matrix...")
    
    A = nx.to_numpy_array(G)
    nodes = list(G.nodes())
    
    P, _ = get_gravity_null_model_manual_iterative(G, pos_attr)
    P_symmetric = (P + P.T) / 2
    R_matrix = A - P_symmetric
    
    # Asymmetry diagnosis logging
    asymmetry_sum = np.sum(np.abs(P - P_symmetric))
    max_diff = np.max(np.abs(P - P_symmetric))
    print("--- Spatial Null Model Asymmetry Analysis ---")
    print(f"Absolute difference sum (|B_before - B_after|): {asymmetry_sum:.2e}")
    print(f"Max point-wise discrepancy: {max_diff:.2e}")
    print("----------------------------------------------------")
    
    embedding_matrix = train_custom_signed_embeddings(
        R_matrix=R_matrix, 
        embedding_dim=embedding_dim, 
        epochs=epochs, 
        lr=lr, 
        temperature=temperature
    )
    
    # Build dictionary and assign mapping back to node attributes
    embeddings_dict = {node_id: embedding_matrix[i].tolist() for i, node_id in enumerate(nodes)}
    nx.set_node_attributes(G, embeddings_dict, attr_name)
    
    duration = time.time() - start_time
    print(f"[Success] disentangled embeddings complete in {duration:.2f}s. Added '{attr_name}' to nodes.")
    return G


def infer_custom_disentangled_embeddings(
    G: nx.Graph, 
    null_model_matrix: np.ndarray, 
    ordered_nodes: list, 
    attr_name: str = "disentangled_embedding",
    embedding_dim: int = 64, 
    epochs: int = 100, 
    lr: float = 0.1, 
    temperature: float = 0.5
) -> nx.Graph:
    """
    Computes node embeddings using an externally provided custom null model 
    probability matrix, storing results back into the NetworkX Graph.
    """
    start_time = time.time()
    
    A = nx.to_numpy_array(G)
    nodes = list(G.nodes())
    
    P_symmetric = (null_model_matrix + null_model_matrix.T) / 2
    R_matrix = A - P_symmetric
    
    embedding_matrix = train_custom_signed_embeddings(
        R_matrix=R_matrix, 
        embedding_dim=embedding_dim, 
        epochs=epochs, 
        lr=lr, 
        temperature=temperature
    )
    
    # Map results to node layout attributes
    embeddings_dict = {node_id: embedding_matrix[i].tolist() for i, node_id in enumerate(ordered_nodes)}
    nx.set_node_attributes(G, embeddings_dict, attr_name)
    
    duration = time.time() - start_time
    print(f"[Success] Custom disentangled embeddings complete in {duration:.2f}s. Added '{attr_name}' to nodes.")
    return G