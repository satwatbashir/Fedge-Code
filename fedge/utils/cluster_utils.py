"""Utility functions for dynamic server clustering using symmetric KL divergence.

This module is intentionally self-contained so it can be imported by the
cloud-level strategy without pulling in any Flower-specific dependencies.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
# Removed sklearn.cluster import - using pure similarity clustering

from fedge.task import Net, get_cifar10_test_loader, set_weights  # type: ignore

# -----------------------------------------------------------------------------
# Configuration: Load batch sizes from pyproject.toml (strict, no fallbacks)
# -----------------------------------------------------------------------------
try:
    import toml
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _CFG = toml.load(_PROJECT_ROOT / "pyproject.toml")
    _BATCH_CFG = _CFG["tool"]["flwr"]["cluster"]["batch_sizes"]
    LOGIT_BATCH_DEFAULT = int(_BATCH_CFG["logit_batch"])
    FEATURE_BATCH_DEFAULT = int(_BATCH_CFG["feature_batch"])
except Exception as e:  # pragma: no cover
    raise KeyError(
        "Missing required [tool.flwr.cluster.batch_sizes] configuration in pyproject.toml"
    ) from e

# ----------------------------------------------------------------------------
# 1.  Public reference set loader
# ----------------------------------------------------------------------------

_REF_CACHE: dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

def _hash_dataset(ds) -> str:
    """Return SHA256 hash of labels to detect accidental dataset drift."""
    h = hashlib.sha256()
    # hashing only labels is enough for identity & cheap
    for _, lbl in ds:
        h.update(int(lbl).to_bytes(2, byteorder="little", signed=False))
    return h.hexdigest()



# ----------------------------------------------------------------------------
# 2.  Symmetric KL divergence helpers
# ----------------------------------------------------------------------------

def _safe_softmax(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.softmax(logits, dim=-1).clamp_min_(eps)


def sym_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """Stable symmetric KL on two 1-D probability vectors."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    kl_pq = (p * (np.log(p) - np.log(q))).sum()
    kl_qp = (q * (np.log(q) - np.log(p))).sum()
    return float(0.5 * (kl_pq + kl_qp))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """Jensen-Shannon divergence between two probability distributions."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return float(0.5 * (sym_kl(p, m, eps) + sym_kl(q, m, eps)))

# ----------------------------------------------------------------------------
# 3.  Distance matrix and clustering
# ----------------------------------------------------------------------------

def extract_logits(model: Net, imgs: torch.Tensor, batch_size: int | None = None, device: str = "cpu") -> np.ndarray:
    """Run *model* on *imgs* and return the mean logits as a NumPy vector."""
    model.eval()
    model.to(device)
    logits_accum: List[torch.Tensor] = []
    with torch.no_grad():
        bs = batch_size or LOGIT_BATCH_DEFAULT
        for i in range(0, len(imgs), bs):
            batch = imgs[i : i + bs].to(device)
            logits_accum.append(model(batch).cpu())
    logits = torch.cat(logits_accum, dim=0).mean(dim=0)  # (num_classes,)
    return logits.numpy(force=True) if hasattr(logits, "numpy") else logits.numpy()

def extract_probs(model: Net, imgs: torch.Tensor, batch_size: int | None = None, device: str = "cpu") -> np.ndarray:
    """Run *model* on *imgs* and return the mean probabilities as a NumPy vector."""
    model.eval()
    model.to(device)
    probs_accum: List[torch.Tensor] = []
    with torch.no_grad():
        bs = batch_size or LOGIT_BATCH_DEFAULT
        for i in range(0, len(imgs), bs):
            batch = imgs[i : i + bs].to(device)
            probs_accum.append(torch.softmax(model(batch).cpu(), dim=-1))
    probs = torch.cat(probs_accum, dim=0).mean(dim=0)  # (num_classes,)
    return probs.numpy(force=True) if hasattr(probs, "numpy") else probs.numpy()



def distance_matrix(
    probs_list: List[np.ndarray],
    metric: str = "sym_kl",
    eps: float = 1e-6,
) -> np.ndarray:
    """Return symmetric |S|×|S| matrix for the requested metric.

    Parameters
    ----------
    metric : {"sym_kl", "js", "cosine"}
        Distance metric to use.
    """
    metric_fns = {
        "sym_kl": lambda a, b: sym_kl(a, b, eps),
        "js": lambda a, b: js_divergence(a, b, eps),
        "cosine": lambda a, b: 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)),
    }
    if metric not in metric_fns:
        raise ValueError(f"Unknown distance metric: {metric}")
    dist_fn = metric_fns[metric]

    S = len(probs_list)
    D = np.zeros((S, S), dtype=np.float32)
    for i in range(S):
        for j in range(i + 1, S):
            d = dist_fn(probs_list[i], probs_list[j])
            D[i, j] = D[j, i] = d
    return D


def connected_components_from_adj(A: np.ndarray) -> np.ndarray:
    """Find connected components from adjacency matrix using simple BFS.
    
    Args:
        A: Binary adjacency matrix (n x n)
        
    Returns:
        labels: Array of cluster labels for each node
    """
    n = len(A)
    labels = np.full(n, -1, dtype=int)
    current_label = 0
    
    for start in range(n):
        if labels[start] != -1:
            continue
            
        # BFS to find all connected nodes
        queue = [start]
        labels[start] = current_label
        
        while queue:
            node = queue.pop(0)
            for neighbor in range(n):
                if A[node, neighbor] and labels[neighbor] == -1:
                    labels[neighbor] = current_label
                    queue.append(neighbor)
                    
        current_label += 1
        
    return labels


def flatten_last_layer(weights: List[np.ndarray]) -> np.ndarray:
    """Extract and flatten the last layer weights (Linear layer: weight only, no bias).
    
    This function is more flexible and can handle different model architectures by
    finding the last 2D weight matrix in the model, which is typically the classifier head.
    
    Args:
        weights: List of weight arrays from a model
        
    Returns:
        Flattened last layer weights as 1D array
    """
    # Find the last 2D weight matrix (classifier head)
    fc_w_idx = None
    
    # First try to find by known CIFAR-10 shape (10, X) for faster matching
    for idx, arr in enumerate(weights):
        if arr.ndim == 2 and arr.shape[0] == 10:  # CIFAR-10 head.weight
            fc_w_idx = idx
            break
    
    # If not found, use a more general approach: find the last 2D array
    if fc_w_idx is None:
        for idx in range(len(weights)-1, -1, -1):  # Search from the end
            if weights[idx].ndim == 2:
                fc_w_idx = idx
                break
    
    if fc_w_idx is None:
        raise ValueError("Failed to locate any 2D weight matrix in the model")
    
    # Return only the weight, no bias
    return weights[fc_w_idx].ravel()


# Remove metadata_based_clustering - not needed for dynamic weight-based clustering


def cifar10_weight_clustering(
    server_weights_list: List[List[np.ndarray]], 
    global_weights: List[np.ndarray],
    reference_imgs: torch.Tensor,
    round_num: int,
    tau: float,
    stability_history: List[dict] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    CIFAR-10 dynamic weight-based clustering using absolute cosine similarity.
    
    Simple rule: merge servers if abs(cosine_similarity(last_layer_i, last_layer_j)) >= tau
    This handles sign ambiguity in the last layer weights.
    
    Args:
        server_weights_list: List of weight lists from each server
        global_weights: Global model weights (unused in this implementation)
        reference_imgs: Reference images (unused in this implementation)
        round_num: Current round number for logging
        tau: Similarity threshold (0.0 to 1.0)
        stability_history: Unused in simplified implementation
        
    Returns:
        Tuple of (labels, similarity_matrix, tau)
    """
    print(f"[Round {round_num}] Using absolute cosine similarity clustering (tau={tau:.3f})")
    n_servers = len(server_weights_list)
    
    if n_servers == 1:
        return np.array([0]), np.array([[1.0]]), tau
    
    # 1. Extract and normalize last-layer vectors
    V = np.stack([flatten_last_layer(w) for w in server_weights_list], axis=0)
    
    # Check for NaN/Inf values and log if found
    if not np.isfinite(V).all():
        nan_count = np.count_nonzero(~np.isfinite(V))
        print(f"[WARNING] Found {nan_count} NaN/Inf values in server weights. Sanitizing...")
    
    # Sanitize weird values before normalization
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize vectors
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    
    # 2. Cosine similarity matrix with absolute values to handle sign ambiguity
    S = np.abs(V @ V.T)
    
    # Check for NaN/Inf values in similarity matrix
    if not np.isfinite(S).all():
        nan_count = np.count_nonzero(~np.isfinite(S))
        print(f"[WARNING] Found {nan_count} NaN/Inf values in similarity matrix. Sanitizing...")
    
    # Sanitize similarity matrix too (very defensive)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 3. Threshold and connected components
    np.fill_diagonal(S, 0.0)  # Remove self-connections
    A = (S >= tau)  # Adjacency matrix: 1 if similarity >= tau
    labels = connected_components_from_adj(A)
    
    # Logging
    print(f"[Clustering] Absolute cosine similarity matrix:")
    print(np.round(S, 3))
    print(f"[Clustering] Adjacency matrix (tau={tau:.3f}):")
    print(A.astype(int))
    
    unique_labels = np.unique(labels)
    cluster_sizes = [np.sum(labels == label) for label in unique_labels]
    print(f"[Round {round_num}] Result: {len(unique_labels)} clusters with sizes {cluster_sizes}")
    
    # Restore diagonal for return (caller might expect it)
    np.fill_diagonal(S, 1.0)
    
    return labels, S, tau




def compute_gradient_similarity(weights1: List[np.ndarray], weights2: List[np.ndarray], 
                               global_weights: List[np.ndarray]) -> float:
    """Compute cosine similarity between gradient directions."""
    grad1 = [w1 - gw for w1, gw in zip(weights1, global_weights)]
    grad2 = [w2 - gw for w2, gw in zip(weights2, global_weights)]
    
    # Flatten gradients
    grad1_flat = np.concatenate([g.flatten() for g in grad1])
    grad2_flat = np.concatenate([g.flatten() for g in grad2])
    
    # Cosine similarity
    dot_product = np.dot(grad1_flat, grad2_flat)
    norms = np.linalg.norm(grad1_flat) * np.linalg.norm(grad2_flat)
    
    return dot_product / (norms + 1e-8)


# ----------------------------------------------------------------------------
# 4.  Rebuild model from raw NumPy weight list (helper for cloud server)
# ----------------------------------------------------------------------------

def rebuild_model_from_weights(weights: List[np.ndarray]) -> Net:
    model = Net()
    set_weights(model, weights)
    return model
