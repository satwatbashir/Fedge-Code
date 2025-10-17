# cloud_flower.py
"""
Cloud server script for hierarchical federated learning using Flower ≥1.13.
This script starts a Flower ServerApp with communication-cost modifiers,
writes communication metrics to CSV after completion, and handles
dynamic clustering and signal files.
"""

# Suppress warnings
import warnings
import sys
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="flwr")
warnings.simplefilter("ignore")

# Set environment variable to suppress gRPC warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

import time
import json
import pickle
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import toml
import numpy as np

from flwr.server import start_server, ServerConfig
from flwr.common import Metrics, Parameters, NDArrays
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from fedge.task import Net, get_weights, set_weights, test, get_cifar10_test_loader
from fedge.cluster_utils import cifar10_weight_clustering
from fedge.utils.cloud_metrics import get_cloud_metrics_collector

def create_signal_file(signal_path: Path, message: str) -> None:
    """Create a signal file with the given message."""
    try:
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(signal_path, "w") as f:
            f.write(f"{message}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception as e:
        print(f"[Cloud Server] Failed to create signal file {signal_path}: {e}")

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Global evaluation function
def evaluate_global_model(server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Any]) -> Optional[Tuple[float, Dict[str, Any]]]:
    """Evaluate the global model on CIFAR-10 test set once per round."""
    try:
        # Reconstruct global model from parameters
        model = Net()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Set model parameters
        params_dict = zip(model.state_dict().keys(), parameters_ndarrays)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        # Use CIFAR-10 test dataset - evaluate once per round, not per server
        test_loader = get_cifar10_test_loader(batch_size=64)
        
        # Single evaluation on full CIFAR-10 test set
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.squeeze().long().to(device)
                outputs = model(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels, reduction='sum')
                batch_size = labels.size(0)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size
        
        # Calculate global metrics from single evaluation
        if total_samples > 0:
            global_accuracy = total_correct / total_samples
            global_loss = total_loss / total_samples  # Average loss per sample
        else:
            global_accuracy = 0.0
            global_loss = float('inf')
        
        logger.info(f"[Cloud Eval] Global Model: {global_accuracy:.4f} accuracy, {global_loss:.4f} loss, {total_samples} samples")
        
        # Return metrics in the format expected by Flower
        metrics = {
            "accuracy": global_accuracy,
            "loss": global_loss,
            "samples": total_samples,
            "round": server_round
        }
        
        return global_loss, metrics
            
    except Exception as e:
        logger.error(f"[Cloud Eval] Evaluation failed: {e}")
        return None

class CloudFedAvg(FedAvg):
    """FedAvg strategy with cloud-tier clustering and logging."""
    
    def __init__(self, **kwargs):
        # Extract cloud clustering configuration
        cloud_cluster_config = kwargs.pop('cloud_cluster_config', None)
        # Extract optional model reference for advanced algorithms
        model_ref = kwargs.pop('model', None)
        # Extract clustering configuration dict if provided
        cluster_cfg = kwargs.pop('clustering_config', None)
        
        # Pass evaluation function to parent
        kwargs['evaluate_fn'] = evaluate_global_model
        super().__init__(**kwargs)
        
        # Cloud clustering configuration - enable clustering from round 1
        ccfg = cloud_cluster_config or {}
        self.cloud_cluster_config = ccfg
        self._cluster_tau = float(ccfg.get("tau", 0.7))
        self._cloud_cluster_enable = bool(ccfg.get("enable", True))  # Default enabled
        self._start_round = int(ccfg.get("start_round", 1))  # Start from round 1
        self._frequency = max(1, int(ccfg.get("frequency", 1)))  # Every round
        self._model = model_ref  # torch.nn.Module instance
        
        # Cloud clustering state
        self._current_round = 0
        self._cluster_log_data = []
        self._cluster_assignments = {}
        self._reference_set = None
        
        if self._cloud_cluster_enable:
            logger.info(f"[Cloud] Clustering enabled: start_round={self._start_round}, frequency={self._frequency}")
        else:
            logger.info(f"[Cloud] Clustering DISABLED in config: {ccfg}")
    
    def _get_server_id_from_fit_result(self, fit_result) -> Optional[int]:
        """Extract server ID from fit result metrics."""
        try:
            metrics = fit_result.metrics
            # Try multiple property names that the proxy client sends
            for prop_name in ["server_id", "sid", "node_id", "client_id", "partition_id"]:
                if prop_name in metrics:
                    return int(metrics[prop_name])
            return None
        except Exception as exc:
            logger.error(f"[Cloud Server] Failed to get server_id from fit result: {exc}")
            return None
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate fit results; optionally perform cloud-tier clustering; save models."""
        # Use server_round directly (ignore GLOBAL_ROUND env var)
        global_round = server_round
        
        logger.info(f"[Cloud] aggregate_fit called: server_round={server_round}, global_round={global_round}, results={len(results)}, failures={len(failures)}")
        
        if failures:
            logger.warning(f"[Cloud Server] {len(failures)} servers failed during training")
        
        if not results:
            logger.warning("[Cloud Server] No results to aggregate")
            return super().aggregate_fit(server_round, results, failures)
        
        # 1) Standard FedAvg aggregation across all servers
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        # Convert Flower Parameters to list[ndarray] for saving/processing
        agg_list = parameters_to_ndarrays(aggregated_params) if aggregated_params is not None else None
        
        # CRITICAL: Validate aggregated parameters for NaN/inf corruption
        if agg_list is not None:
            has_nan = any(np.isnan(arr).any() for arr in agg_list)
            has_inf = any(np.isinf(arr).any() for arr in agg_list)
            
            if has_nan or has_inf:
                logger.error(f"[Cloud] CRITICAL: Aggregated model contains {'NaN' if has_nan else ''}{'/' if has_nan and has_inf else ''}{'inf' if has_inf else ''} parameters!")
                logger.error(f"[Cloud] FedAvg aggregation corrupted - using fallback weighted average")
                
                # Fallback: Simple weighted average of server models
                total_samples = sum(fit_result.num_examples for _, fit_result in results)
                if total_samples > 0:
                    weighted_sum = None
                    for _, fit_result in results:
                        weight = fit_result.num_examples / total_samples
                        server_params = parameters_to_ndarrays(fit_result.parameters)
                        
                        if weighted_sum is None:
                            weighted_sum = [weight * arr for arr in server_params]
                        else:
                            for i, arr in enumerate(server_params):
                                weighted_sum[i] += weight * arr
                    
                    # Validate fallback parameters
                    if weighted_sum is not None:
                        fallback_has_nan = any(np.isnan(arr).any() for arr in weighted_sum)
                        fallback_has_inf = any(np.isinf(arr).any() for arr in weighted_sum)
                        
                        if not fallback_has_nan and not fallback_has_inf:
                            logger.info(f"[Cloud] Fallback weighted average is clean - using as replacement")
                            agg_list = weighted_sum
                            aggregated_params = ndarrays_to_parameters(weighted_sum)
                        else:
                            logger.error(f"[Cloud] FATAL: Even fallback parameters are corrupted!")
                            return None, {}
                else:
                    logger.error(f"[Cloud] FATAL: No samples to compute fallback!")
                    return None, {}
            else:
                logger.debug(f"[Cloud] Aggregated parameters validated - no NaN/inf detected")

        # Always save the global model using rounds directory structure
        if agg_list is not None:
            # Use rounds directory structure as suggested
            out_dir = Path().resolve() / "rounds" / f"round_{global_round}" / "global"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as model.pkl in the global directory
            model_path = out_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(agg_list, f)
            logger.info(f"[Cloud] Saved aggregated global model to {model_path}")
            
            # Also save in models/ directory for compatibility (single consistent name)
            models_dir = Path().resolve() / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            legacy_path = models_dir / f"model_global_g{global_round}.pkl"
            with open(legacy_path, "wb") as f:
                pickle.dump(agg_list, f)
            logger.info(f"[Cloud] Also saved to models directory: {legacy_path}")

        # 2) Cloud-tier clustering (weight-based) if enabled for this round
        should_cluster = (
            self._cloud_cluster_enable
            and global_round >= self._start_round
            and (global_round - self._start_round) % self._frequency == 0
        )
        
        logger.info(f"[Cloud] Clustering check: enable={self._cloud_cluster_enable}, "
                   f"round={global_round}, start={self._start_round}, freq={self._frequency}, "
                   f"should_cluster={should_cluster}")
        
        if should_cluster:
            logger.info(f"[Cloud] *** CLUSTERING TRIGGERED *** at global round {global_round} "
                        f"(start={self._start_round}, every={self._frequency})")
            
            # Extract server models and weights
            server_models = {}
            server_weights = {}
            
            for client_proxy, fit_result in results:
                server_id = self._get_server_id_from_fit_result(fit_result)
                logger.info(f"[Cloud] Processing result from server_id={server_id}, samples={fit_result.num_examples}")
                if server_id is not None:
                    server_models[server_id] = parameters_to_ndarrays(fit_result.parameters)
                    server_weights[server_id] = fit_result.num_examples
                    logger.info(f"[Cloud] Server {server_id}: {fit_result.num_examples} samples, model params={len(server_models[server_id])}")
                else:
                    logger.warning(f"[Cloud] Failed to extract server_id from fit result")
            
            logger.info(f"[Cloud] Total server models collected: {len(server_models)} (need >=2 for clustering)")
            logger.info(f"[Cloud] Server IDs collected: {list(server_models.keys())}")
            
            if len(server_models) >= 2:
                # Perform clustering using cluster_utils
                from fedge.cluster_utils import cifar10_weight_clustering
                
                try:
                    # Convert server_models dict to list for clustering function
                    server_ids = sorted(server_models.keys())
                    server_weights_list = [server_models[sid] for sid in server_ids]
                    
                    # Read tau from cloud cluster config
                    tau = getattr(self, "_cluster_tau", 0.7)
                    logger.info(f"[Cloud] Using tau = {tau} from config for clustering")
                    
                    # Convert aggregated_params to list if it's Parameters type
                    if aggregated_params is not None:
                        global_weights = parameters_to_ndarrays(aggregated_params)
                    else:
                        global_weights = server_weights_list[0]  # Use first as fallback
                    
                    labels, S, tau_sim = cifar10_weight_clustering(
                        server_weights_list,
                        global_weights,
                        None,  # reference_imgs not needed for weight-based clustering
                        global_round,  # Use global_round consistently
                        tau
                    )
                    cluster_labels_array = labels
                    
                    # Surface pairwise similarities in cloud logs for debugging
                    for i in range(S.shape[0]):
                        logger.info(f"[Cloud] S[{i}]: " + " ".join(f"{v:.3f}" for v in S[i]))
                    logger.info(f"[Cloud] similarity_threshold used: {tau_sim:.3f}")
                    
                    # Convert cluster labels array to dict mapping server_id -> cluster_label
                    cluster_map = {server_ids[i]: int(cluster_labels_array[i]) for i in range(len(server_ids))}
                    unique_labels = set(cluster_map.values())
                    
                    logger.info(f"[Cloud] Clustering results: {len(unique_labels)} clusters")
                    logger.info(f"[Cloud] Cluster assignments: {cluster_map}")
                    
                    # Save clustering metrics
                    metrics_dir = Path().resolve() / "metrics" / "cloud"
                    metrics_dir.mkdir(parents=True, exist_ok=True)
                    cluster_metrics_path = metrics_dir / f"clustering_round_{global_round}.json"
                    with open(cluster_metrics_path, "w") as f:
                        json.dump({
                            "round": global_round,
                            "tau": tau,
                            "num_clusters": len(unique_labels),
                            "assignments": cluster_map,
                            "server_ids": server_ids
                        }, f, indent=2)
                    logger.info(f"[Cloud] Saved clustering metrics to {cluster_metrics_path}")
                    
                    # Save cluster artifacts using rounds directory structure
                    cl_dir = Path().resolve() / "rounds" / f"round_{global_round}" / "cloud"
                    cl_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save cluster mapping in canonical location and mirror to models/
                    cluster_json_path = cl_dir / f"clusters_g{global_round}.json"
                    with open(cluster_json_path, "w", encoding="utf-8") as fp:
                        json.dump({"round": global_round, "assignments": cluster_map}, fp, indent=2)
                    logger.info(f"[Cloud] Saved cluster mapping to {cluster_json_path}")
                    
                    # Save similarity matrix to file for analysis
                    similarity_matrix_path = cl_dir / f"similarity_matrix_g{global_round}.csv"
                    np.savetxt(similarity_matrix_path, S, delimiter=',', fmt='%.6f')
                    logger.info(f"[Cloud] Saved similarity matrix to {similarity_matrix_path}")
                    
                    # Save pairwise similarities as detailed CSV
                    pairwise_csv_path = cl_dir / f"pairwise_similarities_g{global_round}.csv"
                    with open(pairwise_csv_path, 'w', newline='') as f:
                        import csv
                        writer = csv.writer(f)
                        writer.writerow(['server_i', 'server_j', 'similarity', 'above_threshold'])
                        for i in range(len(server_ids)):
                            for j in range(i+1, len(server_ids)):
                                sim_val = S[i, j]
                                above_thresh = sim_val >= tau
                                writer.writerow([server_ids[i], server_ids[j], f'{sim_val:.6f}', above_thresh])
                    logger.info(f"[Cloud] Saved pairwise similarities to {pairwise_csv_path}")
                    
                    # Mirror to models/ directory for compatibility
                    models_dir = Path().resolve() / "models"
                    models_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cluster_json_path, models_dir / f"clusters_g{global_round}.json")
                    
                    # Save cluster-specific models
                    for lab in unique_labels:
                        # Get servers in this cluster
                        cluster_servers = [sid for sid, label in cluster_map.items() if label == lab]
                        
                        # Weighted average within cluster
                        total_weight = sum(server_weights.get(sid, 1) for sid in cluster_servers)
                        sums = None
                        
                        for sid in cluster_servers:
                            if sid in server_models:
                                weight = server_weights.get(sid, 1) / total_weight
                                model_arrays = server_models[sid]
                                
                                if sums is None:
                                    sums = [weight * arr for arr in model_arrays]
                                else:
                                    for i, arr in enumerate(model_arrays):
                                        sums[i] += weight * arr
                        
                        if sums is not None:
                            # Save as plain list for evaluator compatibility
                            cluster_path = cl_dir / f"model_cluster{lab}_g{global_round}.pkl"
                            with open(cluster_path, "wb") as f:
                                pickle.dump(sums, f, protocol=pickle.HIGHEST_PROTOCOL)
                            logger.info(f"[Cloud] Saved cluster {lab} model to {cluster_path}")
                            
                            # Mirror to models/ directory
                            shutil.copy2(cluster_path, models_dir / cluster_path.name)

                    logger.info(f"[Cloud] *** CLUSTERING COMPLETED *** @ global round {global_round}: K={len(unique_labels)} labels={cluster_map}")
                    
                    # Log clustering results for verification
                    for lab in unique_labels:
                        cluster_servers = [sid for sid, label in cluster_map.items() if label == lab]
                        logger.info(f"[Cloud] Cluster {lab}: servers {cluster_servers}")
                    
                except Exception as e:
                    logger.error(f"[Cloud] Clustering failed: {e}, using standard aggregation")
                    import traceback
                    logger.error(f"[Cloud] Clustering error traceback: {traceback.format_exc()}")
            else:
                logger.warning(f"[Cloud] Not enough servers for clustering: {len(server_models)} < 2")
        else:
            logger.info(f"[Cloud] Clustering skipped for round {global_round}")

        logger.info(f"[Cloud] Aggregation completed for global round {global_round}")
        
        # Collect cloud-level metrics if clustering was performed
        if should_cluster and len(server_models) >= 2:
            try:
                cloud_metrics_collector = get_cloud_metrics_collector(Path().resolve())
                
                # Save cluster composition
                cloud_metrics_collector.save_cluster_composition(server_round, cluster_map)
                
                # Evaluate cluster performance
                test_loader = get_cifar10_test_loader()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                cluster_performance = cloud_metrics_collector.evaluate_cluster_performance(
                    server_round, cluster_map, server_models, server_weights, test_loader, device
                )
                
                # Get global performance for comparison
                global_accuracy = metrics.get('accuracy', 0.0) if metrics else 0.0
                global_loss = 0.0  # Will be calculated by evaluate function
                
                # Calculate communication cost (approximate)
                total_comm_cost = sum(fit_result.num_examples * 4594000 for _, fit_result in results)  # Approximate model size
                
                # Save comprehensive cloud metrics
                cloud_metrics_collector.save_cloud_round_metrics(
                    server_round, cluster_map, cluster_performance, 
                    global_accuracy, global_loss, total_comm_cost
                )
                
                logger.info(f"[Cloud] Saved comprehensive cloud metrics for round {server_round}")
                
            except Exception as e:
                logger.error(f"[Cloud] Failed to collect cloud metrics: {e}")
        
        # Create per-round completion signal for orchestrator synchronization
        signal_dir = Path().resolve() / "signals"
        signal_dir.mkdir(exist_ok=True)
        completion_signal = signal_dir / f"cloud_round_{server_round}_completed.signal"
        create_signal_file(completion_signal, f"Cloud round {server_round} completed")
        logger.info(f"[Cloud] Created completion signal: {completion_signal}")
        
        return aggregated_params, metrics
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results and compute agg_acc correctly."""
        try:
            if failures:
                logger.warning(f"[Cloud] Evaluation failures: {len(failures)}")
                return None, {"agg_acc": 0.0}
            
            if not results:
                logger.warning(f"[Cloud] No evaluation results to aggregate")
                return None, {"agg_acc": 0.0}
            
            # Compute weighted average accuracy
            total_examples = 0
            weighted_accuracy_sum = 0.0
            
            for _, eval_res in results:
                num_examples = getattr(eval_res, "num_examples", 0)
                accuracy = (eval_res.metrics or {}).get("accuracy", 0.0)
                
                weighted_accuracy_sum += accuracy * num_examples
                total_examples += num_examples
            
            agg_acc = (weighted_accuracy_sum / total_examples) if total_examples > 0 else 0.0
            
            # Call parent aggregate_evaluate for loss aggregation
            parent_result = super().aggregate_evaluate(server_round, results, failures)
            if parent_result is not None:
                loss, parent_metrics = parent_result
                parent_metrics = parent_metrics or {}
                parent_metrics["agg_acc"] = agg_acc
                return loss, parent_metrics
            else:
                return None, {"agg_acc": agg_acc}
                
        except Exception as e:
            logger.error(f"[Cloud Eval] Evaluation failed: {e}")
            return None, {"agg_acc": 0.0}

def run_server():
    """Run long-running cloud server for all global rounds with dynamic clustering."""
    logger.info("[Cloud Server] Starting long-running cloud aggregation server")
    
    # Create signal directory
    signal_dir = Path().resolve() / "signals"
    signal_dir.mkdir(exist_ok=True)
    
    # Get total number of global rounds from environment
    total_rounds = int(os.getenv('TOTAL_GLOBAL_ROUNDS', '3'))
    logger.info(f"[Cloud Server] Will handle {total_rounds} global rounds")
    
    try:
        # Log environment variables set by orchestrator
        num_servers = os.getenv('NUM_SERVERS', '3')
        logger.info(f"[Cloud Server] Starting for {total_rounds} rounds with {num_servers} servers")
        
        # Load configuration
        config_path = Path("pyproject.toml")
        if not config_path.exists():
            raise FileNotFoundError("pyproject.toml not found")
        
        # Log working directory and models path for path verification
        cwd = Path().resolve()
        logger.info(f"[Cloud Server] Working directory: {cwd}")
        logger.info(f"[Cloud Server] Models directory: {cwd / 'models'}")

        config = toml.load(config_path)
        
        # Extract configuration
        hierarchy_config = config.get("tool", {}).get("flwr", {}).get("hierarchy", {})
        optimizer_config = config.get("tool", {}).get("flwr", {}).get("optimizer", {})
        cluster_config = config.get("tool", {}).get("flwr", {}).get("cluster", {})
        cloud_cluster_cfg = config.get("tool", {}).get("flwr", {}).get("cloud_cluster", {})
        
        # CRITICAL FIX: Load previous round's model or create fresh parameters
        global_round = int(os.getenv('GLOBAL_ROUND', '1'))
        initial_parameters = None
        # Initialize model upfront to avoid undefined variable
        init_model = Net()
        
        # Try to load previous round's global model
        if global_round > 1:
            prev_round = global_round - 1
            candidate_paths = [
                cwd / "rounds" / f"round_{prev_round}" / "global" / "model.pkl",
                cwd / "models" / f"model_global_g{prev_round}.pkl",
                cwd / "models" / f"global_model_round_{prev_round}.pkl"
            ]
            
            for model_path in candidate_paths:
                if model_path.exists():
                    try:
                        with open(model_path, "rb") as f:
                            loaded_data = pickle.load(f)
                            if isinstance(loaded_data, tuple):
                                ndarrays = loaded_data[0]
                            else:
                                ndarrays = loaded_data
                        initial_parameters = ndarrays_to_parameters(ndarrays)
                        # Also set weights on init_model for consistency
                        set_weights(init_model, ndarrays)
                        logger.info(f"[Cloud Server] Loaded trained model from round {prev_round}")
                        break
                    except Exception as e:
                        logger.warning(f"[Cloud Server] Failed to load {model_path}: {e}")
        
        # If no previous model found, create fresh parameters
        if initial_parameters is None:
            initial_parameters = ndarrays_to_parameters(get_weights(init_model))
            logger.info(f"[Cloud Server] Created fresh model parameters for round {global_round}")

        # Get number of servers from environment (set by orchestrator)
        num_servers = int(os.getenv('NUM_SERVERS', '3'))
        
        # Use CloudFedAvg for CIFAR-10 with optional cloud clustering
        strategy = CloudFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_servers,
            min_evaluate_clients=num_servers,
            min_available_clients=num_servers,
            initial_parameters=initial_parameters,
            clustering_config=cluster_config,
            cloud_cluster_config=cloud_cluster_cfg,
            model=init_model,
        )
        # Enforce "all servers must participate"
        strategy.accept_failures = False
        
        # Start server to handle all global rounds
        logger.info(f"[Cloud Server] Starting on port {hierarchy_config.get('cloud_port', 6000)}")
        logger.info(f"[Cloud Server] Expecting {num_servers} proxy clients per round")
        
        # Create start signal file BEFORE starting server
        create_signal_file(signal_dir / "cloud_started.signal", "Cloud server started")
        logger.info("[Cloud Server] Start signal created")
        
        # Start server with proper configuration - handle ALL rounds in one go
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            logger.info(f"[Cloud Server] Starting Flower server for {total_rounds} rounds")
            history = start_server(
                server_address=f"0.0.0.0:{hierarchy_config.get('cloud_port', 6000)}",
                config=ServerConfig(
                    num_rounds=total_rounds,  # Handle ALL rounds in one server instance
                    round_timeout=300.0,  # 5 minutes timeout per round
                ),
                strategy=strategy
            )
        
        logger.info(f"[Cloud Server] All {total_rounds} rounds completed")
        return history
        
    except Exception as e:
        logger.exception(f"[Cloud Server] Unhandled error: {e}")
        create_signal_file(signal_dir / "cloud_error.signal", f"{e!r}")
        raise
    finally:
        # Create final completion signal after all rounds
        create_signal_file(signal_dir / "cloud_all_rounds_completed.signal", "Cloud server completed all rounds")

if __name__ == "__main__":
    run_server()
