"""
Cloud-level metrics collection for hierarchical federated learning.
Tracks cluster composition, performance, and statistical measures.
"""
import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy import stats
import torch

class CloudMetricsCollector:
    """Collects and stores cloud-level clustering and performance metrics."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.cloud_metrics_dir = self.project_root / "metrics" / "cloud"
        self.cloud_metrics_dir.mkdir(parents=True, exist_ok=True)
        
    def save_cluster_composition(self, global_round: int, cluster_assignments: Dict[str, int]):
        """Save cluster composition for the round."""
        composition_file = self.cloud_metrics_dir / f"cluster_composition_round_{global_round}.json"
        
        # Count servers per cluster
        cluster_counts = {}
        for server_id, cluster_id in cluster_assignments.items():
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        
        composition_data = {
            "global_round": global_round,
            "num_clusters": len(set(cluster_assignments.values())),
            "cluster_assignments": cluster_assignments,
            "servers_per_cluster": cluster_counts,
            "total_servers": len(cluster_assignments)
        }
        
        with open(composition_file, 'w') as f:
            json.dump(composition_data, f, indent=2)
    
    def evaluate_cluster_performance(self, global_round: int, cluster_assignments: Dict[str, int], 
                                   server_models: Dict[int, Any], server_weights: Dict[int, int],
                                   test_loader, device) -> Dict[str, Any]:
        """Evaluate performance of each cluster on relevant server datashards."""
        from fedge.task import Net
        
        cluster_metrics = {}
        unique_clusters = set(cluster_assignments.values())
        
        for cluster_id in unique_clusters:
            # Get servers in this cluster
            cluster_servers = [int(sid) for sid, cid in cluster_assignments.items() if cid == cluster_id]
            
            # Load cluster model
            cluster_model_path = self.project_root / "rounds" / f"round_{global_round}" / "cloud" / f"model_cluster{cluster_id}_g{global_round}.pkl"
            
            if cluster_model_path.exists():
                # Load and evaluate cluster model
                with open(cluster_model_path, 'rb') as f:
                    import pickle
                    cluster_weights = pickle.load(f)
                
                model = Net()
                model.to(device)
                
                # Set model parameters
                params_dict = zip(model.state_dict().keys(), cluster_weights)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                model.load_state_dict(state_dict, strict=True)
                
                # Evaluate on test data (representing cluster's data distribution)
                model.eval()
                cluster_loss = 0.0
                cluster_correct = 0
                cluster_samples = 0
                
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.squeeze().long().to(device)
                        outputs = model(images)
                        loss = torch.nn.functional.cross_entropy(outputs, labels)
                        cluster_loss += loss.item() * images.size(0)
                        
                        _, predicted = torch.max(outputs.data, 1)
                        cluster_correct += (predicted == labels).sum().item()
                        cluster_samples += images.size(0)
                
                if cluster_samples > 0:
                    cluster_accuracy = cluster_correct / cluster_samples
                    cluster_avg_loss = cluster_loss / cluster_samples
                    
                    # Calculate weighted samples for this cluster
                    cluster_total_samples = sum(server_weights.get(sid, 0) for sid in cluster_servers)
                    
                    cluster_metrics[cluster_id] = {
                        "servers": cluster_servers,
                        "num_servers": len(cluster_servers),
                        "test_accuracy": cluster_accuracy,
                        "test_loss": cluster_avg_loss,
                        "total_samples": cluster_total_samples,
                        "test_samples": cluster_samples
                    }
        
        return cluster_metrics
    
    def calculate_cluster_statistics(self, cluster_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistical measures across clusters."""
        if not cluster_metrics:
            return {}
        
        accuracies = [metrics["test_accuracy"] for metrics in cluster_metrics.values()]
        losses = [metrics["test_loss"] for metrics in cluster_metrics.values()]
        
        stats_data = {
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "accuracy_ci_lower": np.percentile(accuracies, 2.5),
            "accuracy_ci_upper": np.percentile(accuracies, 97.5),
            "loss_mean": np.mean(losses),
            "loss_std": np.std(losses),
            "loss_ci_lower": np.percentile(losses, 2.5),
            "loss_ci_upper": np.percentile(losses, 97.5),
            "num_clusters": len(cluster_metrics)
        }
        
        return stats_data
    
    def save_cloud_round_metrics(self, global_round: int, cluster_assignments: Dict[str, int],
                                cluster_metrics: Dict[str, Any], global_accuracy: float, 
                                global_loss: float, communication_cost: int = 0):
        """Save comprehensive cloud metrics for the round."""
        
        # Calculate statistics
        cluster_stats = self.calculate_cluster_statistics(cluster_metrics)
        
        # Calculate generalization gaps (cluster vs global performance)
        cluster_global_gaps = {}
        for cluster_id, metrics in cluster_metrics.items():
            cluster_global_gaps[cluster_id] = {
                "accuracy_gap": metrics["test_accuracy"] - global_accuracy,
                "loss_gap": metrics["test_loss"] - global_loss
            }
        
        # Save to CSV
        cloud_csv = self.cloud_metrics_dir / "cloud_round_metrics.csv"
        fieldnames = [
            "global_round", "num_clusters", "global_accuracy", "global_loss",
            "cluster_accuracy_mean", "cluster_accuracy_std", "cluster_accuracy_ci_lower", "cluster_accuracy_ci_upper",
            "cluster_loss_mean", "cluster_loss_std", "cluster_loss_ci_lower", "cluster_loss_ci_upper",
            "max_cluster_accuracy", "min_cluster_accuracy", "max_cluster_loss", "min_cluster_loss",
            "communication_cost_bytes", "total_servers"
        ]
        
        write_header = not cloud_csv.exists()
        
        with open(cloud_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            
            # Calculate min/max cluster performance
            if cluster_metrics:
                cluster_accuracies = [m["test_accuracy"] for m in cluster_metrics.values()]
                cluster_losses = [m["test_loss"] for m in cluster_metrics.values()]
                max_cluster_acc = max(cluster_accuracies)
                min_cluster_acc = min(cluster_accuracies)
                max_cluster_loss = max(cluster_losses)
                min_cluster_loss = min(cluster_losses)
            else:
                max_cluster_acc = min_cluster_acc = global_accuracy
                max_cluster_loss = min_cluster_loss = global_loss
            
            row_data = {
                "global_round": global_round,
                "num_clusters": len(set(cluster_assignments.values())),
                "global_accuracy": global_accuracy,
                "global_loss": global_loss,
                "cluster_accuracy_mean": cluster_stats.get("accuracy_mean", global_accuracy),
                "cluster_accuracy_std": cluster_stats.get("accuracy_std", 0.0),
                "cluster_accuracy_ci_lower": cluster_stats.get("accuracy_ci_lower", global_accuracy),
                "cluster_accuracy_ci_upper": cluster_stats.get("accuracy_ci_upper", global_accuracy),
                "cluster_loss_mean": cluster_stats.get("loss_mean", global_loss),
                "cluster_loss_std": cluster_stats.get("loss_std", 0.0),
                "cluster_loss_ci_lower": cluster_stats.get("loss_ci_lower", global_loss),
                "cluster_loss_ci_upper": cluster_stats.get("loss_ci_upper", global_loss),
                "max_cluster_accuracy": max_cluster_acc,
                "min_cluster_accuracy": min_cluster_acc,
                "max_cluster_loss": max_cluster_loss,
                "min_cluster_loss": min_cluster_loss,
                "communication_cost_bytes": communication_cost,
                "total_servers": len(cluster_assignments)
            }
            writer.writerow(row_data)
        
        # Save detailed cluster metrics to JSON
        detailed_metrics = {
            "global_round": global_round,
            "cluster_assignments": cluster_assignments,
            "cluster_metrics": cluster_metrics,
            "cluster_statistics": cluster_stats,
            "generalization_gaps": cluster_global_gaps,
            "global_performance": {
                "accuracy": global_accuracy,
                "loss": global_loss
            }
        }
        
        detailed_file = self.cloud_metrics_dir / f"detailed_metrics_round_{global_round}.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2, default=str)

def get_cloud_metrics_collector(project_root: Path) -> CloudMetricsCollector:
    """Get or create a cloud metrics collector instance."""
    return CloudMetricsCollector(project_root)
