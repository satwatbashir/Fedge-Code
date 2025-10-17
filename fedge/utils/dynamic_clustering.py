"""
Dynamic Weight-Based Clustering for Hierarchical Federated Learning

This module implements dynamic clustering based on model weight similarity,
replacing static view-based clustering to allow clusters to evolve as training progresses.

Key Features:
- Weight-based similarity using final layer parameters
- Agglomerative clustering with adaptive thresholds
- Periodic re-clustering every N rounds
- Automatic cluster number determination

Reference: FedClust and related dynamic clustering approaches
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class DynamicWeightClustering:
    """
    Dynamic clustering based on model weight similarity.
    
    Clusters clients/servers based on their final layer weights,
    allowing cluster assignments to evolve during training.
    """
    
    def __init__(self, 
                 clustering_frequency: int,
                 distance_metric: str,
                 linkage: str,
                 min_cluster_size: int):
        """
        Initialize dynamic clustering manager.
        
        Args:
            clustering_frequency: Re-cluster every N rounds
            distance_metric: "cosine", "euclidean", or "manhattan"
            linkage: Linkage criterion for agglomerative clustering
            min_cluster_size: Minimum clients per cluster
        """
        self.clustering_frequency = clustering_frequency
        self.distance_metric = distance_metric
        self.linkage = linkage
        self.min_cluster_size = min_cluster_size
        self.cluster_history = []
        self.last_clustering_round = -1
        
    def should_recluster(self, current_round: int) -> bool:
        """Check if re-clustering should be performed this round."""
        return (current_round % self.clustering_frequency == 0 and 
                current_round != self.last_clustering_round)
    
    def extract_final_layer_weights(self, model: nn.Module) -> np.ndarray:
        """
        Extract final layer weights as feature vector for clustering.
        
        Args:
            model: PyTorch model
            
        Returns:
            Flattened weight vector from final layer(s)
        """
        final_weights = []
        
        # Get the last few layers (classifier layers)
        layers = list(model.named_parameters())
        
        # Take last 2 layers (typically final linear layer + bias)
        for name, param in layers[-2:]:
            if param.requires_grad:
                final_weights.append(param.data.cpu().flatten())
                
        if not final_weights:
            # Hard failure: no fallback to all parameters
            raise RuntimeError("No final layer weights found in model - cannot proceed with clustering")
                    
        # Concatenate all weight vectors
        weight_vector = torch.cat(final_weights).numpy()
        
        logger.debug(f"Extracted weight vector of size {len(weight_vector)}")
        return weight_vector
    
    def compute_distance_matrix(self, weight_vectors: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise distance matrix between weight vectors.
        
        Args:
            weight_vectors: List of weight vectors from different models
            
        Returns:
            Distance matrix (n_models x n_models)
        """
        if len(weight_vectors) < 2:
            return np.array([[0.0]])
            
        # Stack vectors into matrix
        weight_matrix = np.stack(weight_vectors)
        
        # Compute distances
        if self.distance_metric == "cosine":
            distances = cosine_distances(weight_matrix)
        elif self.distance_metric == "euclidean":
            distances = euclidean_distances(weight_matrix)
        elif self.distance_metric == "manhattan":
            distances = np.abs(weight_matrix[:, None] - weight_matrix).sum(axis=2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
        logger.debug(f"Computed {self.distance_metric} distance matrix: {distances.shape}")
        return distances
    
    def determine_optimal_clusters(self, 
                                 distances: np.ndarray, 
                                 max_clusters: Optional[int] = None) -> int:
        """
        Determine optimal number of clusters using elbow method or silhouette analysis.
        
        Args:
            distances: Distance matrix
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            Optimal number of clusters
        """
        n_samples = distances.shape[0]
        
        if max_clusters is None:
            max_clusters = min(n_samples, 6)  # Reasonable upper bound
            
        if n_samples <= 2:
            return 1
            
        # Simple heuristic: use sqrt(n) clusters, bounded
        optimal_k = max(1, min(int(np.sqrt(n_samples)), max_clusters))
        
        logger.debug(f"Determined optimal clusters: {optimal_k} for {n_samples} samples")
        return optimal_k
    
    def perform_clustering(self, 
                         weight_vectors: List[np.ndarray],
                         n_clusters: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Perform agglomerative clustering on weight vectors.
        
        Args:
            weight_vectors: List of model weight vectors
            n_clusters: Number of clusters (auto-determined if None)
            
        Returns:
            Tuple of (cluster_labels, clustering_info)
        """
        if len(weight_vectors) <= 1:
            return np.array([0]), {"n_clusters": 1, "method": "single_sample"}
            
        # Compute distance matrix
        distances = self.compute_distance_matrix(weight_vectors)
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters(distances)
            
        # Perform clustering
        try:
            if self.linkage == "ward" and self.distance_metric != "euclidean":
                # Ward requires euclidean distance
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage="complete",  # Fallback linkage
                    metric=self.distance_metric
                )
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=self.linkage,
                    metric=self.distance_metric
                )
                
            # Fit clustering
            weight_matrix = np.stack(weight_vectors)
            cluster_labels = clustering.fit_predict(weight_matrix)
            
            # Enforce minimum cluster size
            cluster_sizes = np.bincount(cluster_labels)
            small_clusters = np.where(cluster_sizes < self.min_cluster_size)[0]
            
            if len(small_clusters) > 0:
                logger.info(f"Merging {len(small_clusters)} clusters smaller than {self.min_cluster_size}")
                
                # Reassign small clusters to the largest cluster
                largest_cluster = np.argmax(cluster_sizes)
                for small_cluster_id in small_clusters:
                    cluster_labels[cluster_labels == small_cluster_id] = largest_cluster
                
                # Relabel clusters to be consecutive (0, 1, 2, ...)
                unique_labels = np.unique(cluster_labels)
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
                cluster_labels = np.array([label_mapping[label] for label in cluster_labels])
                
                # Update cluster sizes
                cluster_sizes = np.bincount(cluster_labels)
                n_clusters = len(unique_labels)
            
            # Clustering info
            info = {
                "n_clusters": n_clusters,
                "method": "agglomerative",
                "linkage": self.linkage,
                "metric": self.distance_metric,
                "cluster_sizes": cluster_sizes,
                "min_cluster_size_enforced": self.min_cluster_size
            }
            
            logger.info(f"Clustering completed: {n_clusters} clusters, sizes: {info['cluster_sizes']}")
            return cluster_labels, info
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise RuntimeError(f"Dynamic clustering failed and cannot proceed: {e}") from e
    
    def cluster_models(self, 
                      models: List[nn.Module],
                      current_round: int,
                      force_recluster: bool = False) -> Tuple[Dict[int, int], Dict]:
        """
        Cluster models based on weight similarity.
        
        Args:
            models: List of PyTorch models to cluster
            current_round: Current training round
            force_recluster: Force re-clustering even if not scheduled
            
        Returns:
            Tuple of (cluster_assignments, clustering_info)
        """
        # Check if clustering is needed
        if not (self.should_recluster(current_round) or force_recluster):
            # Return previous clustering if available
            if self.cluster_history:
                last_assignment = self.cluster_history[-1]["assignments"]
                return last_assignment, {"method": "cached", "round": current_round}
            
        # Extract weight vectors
        weight_vectors = []
        for i, model in enumerate(models):
            try:
                weights = self.extract_final_layer_weights(model)
                weight_vectors.append(weights)
            except Exception as e:
                logger.error(f"Failed to extract weights from model {i}: {e}")
                # Hard failure: no fallback dummy vectors
                raise RuntimeError(f"Weight extraction failed for model {i} and cannot proceed: {e}") from e
                
        # Perform clustering
        cluster_labels, clustering_info = self.perform_clustering(weight_vectors)
        
        # Convert to assignment dictionary
        cluster_assignments = {i: int(label) for i, label in enumerate(cluster_labels)}
        
        # Store clustering history
        clustering_record = {
            "round": current_round,
            "assignments": cluster_assignments,
            "info": clustering_info,
            "n_models": len(models)
        }
        self.cluster_history.append(clustering_record)
        self.last_clustering_round = current_round
        
        logger.info(f"Round {current_round}: Dynamic clustering assigned {len(models)} models to {clustering_info['n_clusters']} clusters")
        return cluster_assignments, clustering_info
    
    def get_clustering_stability(self, window_size: int = 5) -> float:
        """
        Compute clustering stability over recent rounds.
        
        Args:
            window_size: Number of recent rounds to consider
            
        Returns:
            Stability score (0.0 = completely unstable, 1.0 = perfectly stable)
        """
        if len(self.cluster_history) < 2:
            return 1.0  # Perfectly stable if only one clustering
            
        recent_history = self.cluster_history[-window_size:]
        if len(recent_history) < 2:
            return 1.0
            
        # Compare consecutive clusterings
        stability_scores = []
        for i in range(1, len(recent_history)):
            prev_assignments = recent_history[i-1]["assignments"]
            curr_assignments = recent_history[i]["assignments"]
            
            # Compute agreement (same assignments)
            agreements = 0
            total = len(prev_assignments)
            
            for model_id in prev_assignments:
                if model_id in curr_assignments:
                    if prev_assignments[model_id] == curr_assignments[model_id]:
                        agreements += 1
                        
            stability = agreements / total if total > 0 else 1.0
            stability_scores.append(stability)
            
        return np.mean(stability_scores) if stability_scores else 1.0


def create_dynamic_clustering(config: Dict) -> DynamicWeightClustering:
    """
    Factory function to create dynamic clustering manager from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured DynamicWeightClustering instance
    """
    # Require explicit config values, no defaults
    required_params = ['clustering_frequency', 'distance_metric', 'linkage', 'min_cluster_size']
    for param in required_params:
        if param not in config:
            raise ValueError(f"{param} must be specified in [tool.flwr.cluster] section of pyproject.toml")
    
    return DynamicWeightClustering(
        clustering_frequency=int(config["clustering_frequency"]),
        distance_metric=str(config["distance_metric"]),
        linkage=str(config["linkage"]),
        min_cluster_size=int(config["min_cluster_size"])
    )


def is_dynamic_clustering_enabled(config: Dict) -> bool:
    """Check if dynamic clustering is enabled in configuration."""
    return config.get("dynamic_clustering", False)
