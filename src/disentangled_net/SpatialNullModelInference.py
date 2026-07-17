import networkx as nx
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Tuple, List

def get_gravity_null_model_manual_iterative(
    G: nx.Graph, 
    pos_attr: str = 'pos', 
    tol: float = 0.01, 
    max_iter: int = 1000
) -> Tuple[np.ndarray, List]:
    """
    Maximum likelihood inference of an undirected geometric ERGM / Gravity Null Model
    using coordinate descent and scalar optimization.
    
    Parameters:
    -----------
    G : nx.Graph
        The input NetworkX graph.
    pos_attr : str
        The node attribute key containing spatial positions.
    tol : float
        Convergence tolerance for the Mean Absolute Error (MAE) of degrees.
    max_iter : int
        Maximum number of optimization iterations.
        
    Returns:
    --------
    current_P : np.ndarray
        Symmetric probability matrix of the inferred null model.
    nodes : list
        Ordered list of node IDs corresponding to the matrix indices.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    adj = nx.to_numpy_array(G)
    degrees = np.sum(adj, axis=1)
    
    # Compute pairwise distance matrix (N, N)
    pos_array = np.array([G.nodes[u][pos_attr] for u in nodes])
    if len(pos_array.shape) == 1 or (len(pos_array.shape) == 2 and pos_array.shape[1] == 1):
        # 1D case: direct absolute difference
        dist_matrix = np.abs(pos_array[:, np.newaxis] - pos_array[np.newaxis, :])
    else:
        # Multidimensional case: Euclidean norm
        dist_matrix = np.linalg.norm(pos_array[:, np.newaxis] - pos_array[np.newaxis, :], axis=2)
    
    # Parameter initialization
    alphas = np.zeros(n)
    beta = 1.0
    
    def total_log_likelihood_beta(b: float, current_alphas: np.ndarray) -> float:
        """Negative log-likelihood function for beta optimization."""
        theta = current_alphas[:, np.newaxis] + current_alphas[np.newaxis, :] - b * dist_matrix
        log_q = np.logaddexp(0, theta)
        # Upper triangle processing (undirected network, no self-loops)
        ll = np.sum(np.triu(adj * theta - log_q, k=1))
        return -ll

    for iteration in range(max_iter):
        
        # 1. Sequential alpha update (vectorized coordinate descent)
        for _ in range(5):
            theta = alphas[:, np.newaxis] + alphas[np.newaxis, :] - beta * dist_matrix
            theta = np.clip(theta, -50, 50)  # Numerical stability guard
            
            P = 1 / (1 + np.exp(-theta))
            np.fill_diagonal(P, 0)
            
            f_x = np.sum(P, axis=1) - degrees
            f_prime = np.sum(P * (1 - P), axis=1) + 1e-5  # Hessian approximation
            
            step = f_x / f_prime
            alphas -= 0.5 * np.clip(step, -2.0, 2.0)
            
        # 2. Beta parameters update
        res_beta = minimize_scalar(
            total_log_likelihood_beta, 
            args=(alphas,), 
            bounds=(0, 20), 
            method='bounded'
        )
        beta = res_beta.x
        
        # Convergence evaluation
        theta = alphas[:, np.newaxis] + alphas[np.newaxis, :] - beta * dist_matrix
        current_P = 1 / (1 + np.exp(-theta))
        np.fill_diagonal(current_P, 0)
        
        predicted_degrees = np.sum(current_P, axis=1)
        mae_degrees = np.mean(np.abs(predicted_degrees - degrees))
        
        if mae_degrees < tol:
            break

    print(f"[Null Model] Convergence reached at iter {iteration}. Degree MAE = {mae_degrees:.4f} | Beta = {beta:.4f}")
    
    return current_P, nodes