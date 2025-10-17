"""
Consolidated metrics storage for hierarchical federated learning.
Handles server, client, and communication metrics with CSV output.
"""
import csv
from pathlib import Path
from typing import Dict, Any, List
import os

class MetricsStorage:
    """Consolidated metrics storage with CSV output."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.metrics_dir = self.project_root / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def save_server_metrics(self, global_round: int, server_id: int, metrics: Dict[str, Any]):
        """Save server-level metrics to CSV."""
        server_csv = self.metrics_dir / f"server_{server_id}_metrics.csv"
        
        # Define fieldnames for server metrics
        fieldnames = [
            "global_round", "server_id", "central_loss", "central_accuracy", 
            "avg_client_loss", "avg_client_accuracy", "num_clients",
            "loss_std", "accuracy_std", "loss_ci_lower", "loss_ci_upper",
            "accuracy_ci_lower", "accuracy_ci_upper", "generalization_gap_loss",
            "generalization_gap_accuracy", "communication_cost_bytes", "computation_time_sec",
            "centralized_on_own_data", "server_data_samples"
        ]
        
        # Write header if file doesn't exist
        write_header = not server_csv.exists()
        
        with open(server_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            
            # Ensure all required fields are present
            row_data = {
                "global_round": global_round,
                "server_id": server_id,
                **{k: metrics.get(k, 0.0) for k in fieldnames[2:]}
            }
            writer.writerow(row_data)
    
    def save_client_metrics(self, global_round: int, server_id: int, client_id: str, metrics: Dict[str, Any]):
        """Save client-level metrics to CSV."""
        client_csv = self.metrics_dir / f"server_{server_id}_client_eval_metrics.csv"
        
        fieldnames = [
            "global_round", "local_round", "client_id", "server_id", "eval_loss", 
            "accuracy", "num_examples", "test_on_own_shard", "shard_size"
        ]
        
        write_header = not client_csv.exists()
        
        with open(client_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            
            row_data = {
                "global_round": global_round,
                "local_round": metrics.get("local_round", 1),
                "client_id": client_id,
                "server_id": server_id,
                "eval_loss": metrics.get("eval_loss", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "num_examples": metrics.get("num_examples", 0),
                "test_on_own_shard": metrics.get("test_on_own_shard", True),
                "shard_size": metrics.get("shard_size", metrics.get("num_examples", 0))
            }
            writer.writerow(row_data)
    
    def save_communication_metrics(self, global_round: int, server_id: int, client_id: str, metrics: Dict[str, Any]):
        """Save communication metrics to CSV."""
        comm_csv = self.metrics_dir / f"server_{server_id}_client_communication_metrics.csv"
        
        fieldnames = [
            "global_round", "client_id", "bytes_up", "bytes_down", 
            "round_time", "compute_time", "bytes_transferred_total"
        ]
        
        write_header = not comm_csv.exists()
        
        with open(comm_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            
            row_data = {
                "global_round": global_round,
                "client_id": client_id,
                "bytes_up": metrics.get("bytes_up", 0),
                "bytes_down": metrics.get("bytes_down", 0),
                "round_time": metrics.get("round_time", 0.0),
                "compute_time": metrics.get("compute_time", 0.0),
                "bytes_transferred_total": metrics.get("bytes_transferred_total", 0)
            }
            writer.writerow(row_data)
    
    def save_global_metrics(self, global_round: int, metrics: Dict[str, Any]):
        """Save global aggregated metrics to CSV."""
        global_csv = self.metrics_dir / "global_metrics.csv"
        
        fieldnames = [
            "global_round", "global_loss", "global_accuracy", "num_servers",
            "total_clients", "convergence_metric"
        ]
        
        write_header = not global_csv.exists()
        
        with open(global_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            
            row_data = {
                "global_round": global_round,
                **{k: metrics.get(k, 0.0) for k in fieldnames[1:]}
            }
            writer.writerow(row_data)

def get_metrics_storage(project_root: Path) -> MetricsStorage:
    """Get or create a metrics storage instance."""
    return MetricsStorage(project_root)
