# fedge/leaf_server.py
# OPTIMIZED VERSION - Removed redundant evaluations, fixed race conditions

import os
import sys
import signal
import argparse
import time
import pickle
import csv
import gc
import json
import logging
import toml
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from copy import deepcopy
import math

import torch
import numpy as np
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
    FitIns,
)
from flwr.server.strategy import FedAvg
from flwr.server import start_server, ServerConfig
from flwr.common.typing import Metrics, FitRes, EvaluateRes, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedge.task import Net, get_weights, set_weights, load_data, test, get_cifar10_test_loader

# Fix Windows encoding issues
if sys.platform == "win32":
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Suppress Python deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
for name in ("flwr", "ece", "grpc"): 
    logging.getLogger(name).setLevel(logging.ERROR)

# Drop 'DEPRECATED FEATURE' messages
class _DropDeprecated:
    def __init__(self, out): self._out = out
    def write(self, txt):
        if "DEPRECATED FEATURE" not in txt: self._out.write(txt)
    def flush(self): self._out.flush()

sys.stdout = _DropDeprecated(sys.stdout)
sys.stderr = _DropDeprecated(sys.stderr)


class LeafFedAvg(FedAvg):
    """SCAFFOLD strategy for FedGE leaf servers with server-side control variates"""

    def __init__(
        self,
        clients_per_server: int,
        *,
        initial_parameters: Optional[Parameters] = None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        **kwargs
    ):
        # Extract custom parameters before passing to parent
        server_id    = kwargs.pop('server_id', 0)
        project_root = kwargs.pop('project_root', Path.cwd())
        global_round = kwargs.pop('global_round', 0)
        server_lr    = kwargs.pop('server_lr', 1.0)
        global_lr    = kwargs.pop('global_lr', 1.0)

        # Derive minimums from fractions
        min_fit_clients = max(1, int(math.ceil(fraction_fit * clients_per_server)))
        min_available_clients = min_fit_clients
        min_evaluate_clients = 0 if fraction_evaluate == 0.0 else max(1, int(math.ceil(fraction_evaluate * clients_per_server)))

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            accept_failures=True,
            evaluate_fn=self._server_side_evaluate,
        )
        
        self.clients_per_server = clients_per_server
        self.server_id = server_id
        self.project_root = project_root
        self.global_round = global_round
        self.server_lr = server_lr
        self.global_lr = global_lr
        self.server_str = f"Leaf Server {server_id}"
        
        # SCAFFOLD control variates
        self.c_global: NDArrays = []
        self.c_locals: Dict[str, NDArrays] = {}
        self._latest_global_parameters = None
        self.latest_parameters = initial_parameters
        
        # ✅ FIX: Store total examples for model save
        self._last_total_examples = 0
        
        # Get configuration from pyproject.toml
        cfg = toml.load(self.project_root / "pyproject.toml")
        hierarchy_config = cfg["tool"]["flwr"]["hierarchy"]
        self.num_servers = hierarchy_config["num_servers"]
        
        if global_lr == 1.0:
            self.global_lr = hierarchy_config.get("global_lr", 1.0)
        
        self.base_dir = self.project_root
        self.eval_batch_size = hierarchy_config["eval_batch_size"]
        self.cluster_better_delta = hierarchy_config["cluster_better_delta"]  
            
        # Prepare validation loader for server-specific test shard
        try:
            self._valloader_gate = get_cifar10_test_loader(
                batch_size=self.eval_batch_size,
                server_id=self.server_id,
                num_servers=self.num_servers
            )
            shard_size = len(self._valloader_gate.dataset)
            logger.info(f"[{self.server_str}] Loaded server test shard: {shard_size} samples")
        except Exception as e:
            logger.error(f"[{self.server_str}] Error loading server test shard: {e}")
            logger.warning(f"[{self.server_str}] FALLBACK: Using full test set")
            self._valloader_gate = get_cifar10_test_loader(batch_size=self.eval_batch_size)

    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        total_examples = sum([num_examples for num_examples, _ in metrics])
        return {"accuracy": sum(accuracies) / total_examples if total_examples > 0 else 0.0}

    def _server_side_evaluate(self, server_round: int, parameters: Parameters, config: Dict[str, Any]):
        """Enable client evaluation - actual eval happens in aggregate_evaluate"""
        return None

    def aggregate_fit(self, rnd: int, results, failures):
        """SCAFFOLD aggregation with server-side control variates"""
        round_start_time = time.time()
        
        logger.info(f"[{self.server_str}] SCAFFOLD aggregating {len(results)} client fit results for round {rnd}")
        for cid, fit_res in results:
            friendly = fit_res.metrics.get("client_id", str(cid))
            loss = fit_res.metrics.get("train_loss", 0.0)
            n = fit_res.num_examples
            logger.debug(f"  → Client {friendly}: train_loss={loss:.4f}, samples={n}")
        
        if failures:
            logger.warning(f"[{self.server_str}] {len(failures)} client failures in round {rnd}")
        
        # Initialize SCAFFOLD global control variates if needed
        if not self.c_global and self._latest_global_parameters is not None:
            prev_global_nd = parameters_to_ndarrays(self._latest_global_parameters)
            self.c_global = [np.zeros_like(x, dtype=np.float32) for x in prev_global_nd]
            logger.info(f"[{self.server_str}] Initialized SCAFFOLD global control variates")
        
        # SCAFFOLD aggregation
        y_deltas: List[List[np.ndarray]] = []
        weights: List[int] = []
        
        for client, fit_res in results:
            returned_params: Parameters = fit_res.parameters
            nd_list = parameters_to_ndarrays(returned_params)
            num_examples = fit_res.num_examples
            
            # Get client ID
            client_id = fit_res.metrics.get("client_id", str(client))
            
            # Initialize client control variates if needed
            if client_id not in self.c_locals and self.c_global:
                self.c_locals[client_id] = [np.zeros_like(x, dtype=np.float32) for x in self.c_global]
            
            # Extract SCAFFOLD delta if available
            if "scaffold_delta" in fit_res.metrics:
                try:
                    import base64
                    serialized_delta = fit_res.metrics["scaffold_delta"]
                    c_delta = pickle.loads(base64.b64decode(serialized_delta.encode('utf-8')))
                    self.c_locals[client_id] = c_delta
                    logger.debug(f"[{self.server_str}] Updated SCAFFOLD delta for client {client_id}")
                except Exception as e:
                    logger.warning(f"[{self.server_str}] Failed to deserialize SCAFFOLD delta: {e}")
            
            # Compute pseudo-gradient
            if self._latest_global_parameters is not None and self.c_global:
                prev_nd = parameters_to_ndarrays(self._latest_global_parameters)
                c_local = self.c_locals.get(client_id, self.c_global)
                
                y_delta = []
                for w_new, w_prev, c_g, c_l in zip(nd_list, prev_nd, self.c_global, c_local):
                    delta = (w_new - w_prev) + c_g - c_l
                    y_delta.append(delta.astype(np.float32))
                
                y_deltas.append(y_delta)
                weights.append(num_examples)
            else:
                # Fallback: use raw parameters
                y_deltas.append([x.astype(np.float32) for x in nd_list])
                weights.append(num_examples)
        
        # Aggregate pseudo-gradients
        if not y_deltas:
            logger.error(f"[{self.server_str}] No valid client updates to aggregate")
            return None
        
        total_weight = sum(weights)
        aggregated_delta = [np.zeros_like(y_deltas[0][i], dtype=np.float32) for i in range(len(y_deltas[0]))]
        
        for y_delta, w in zip(y_deltas, weights):
            w_frac = w / total_weight
            for i in range(len(aggregated_delta)):
                aggregated_delta[i] += w_frac * y_delta[i]
        
        # Update global control variates (optional: learning rate)
        if self.c_global:
            for i in range(len(self.c_global)):
                self.c_global[i] = self.c_global[i] + (1.0 / len(results)) * aggregated_delta[i]
        
        # Compute new global parameters with server learning rate
        if self._latest_global_parameters is not None:
            prev_nd = parameters_to_ndarrays(self._latest_global_parameters)
            new_nd = [prev + self.server_lr * delta for prev, delta in zip(prev_nd, aggregated_delta)]
        else:
            # First round: use aggregated delta as-is
            new_nd = aggregated_delta
        
        new_global = ndarrays_to_parameters(new_nd)
        
        # ✅ FIX: Track total examples for model save
        self._last_total_examples = total_weight
        
        # Update latest parameters
        self._latest_global_parameters = new_global
        self.latest_parameters = new_global
        
        # Save model after aggregation
        self._save_aggregated_model(new_global, total_weight)
        
        # Compute aggregated accuracy
        agg_acc = None
        if results:
            accs = [fit_res.metrics.get("accuracy", 0.0) for _, fit_res in results]
            nums = [fit_res.num_examples for _, fit_res in results]
            if nums and sum(nums) > 0:
                agg_acc = sum(a * n for a, n in zip(accs, nums)) / sum(nums)
        
        metrics = {"agg_accuracy": float(agg_acc)} if agg_acc is not None else {}
        # ✅ Strict: always include server_id so cloud can cluster without guesswork
        metrics["server_id"] = int(self.server_id)

        round_time = time.time() - round_start_time
        logger.info(f"[{self.server_str}] Aggregation completed in {round_time:.2f}s")
        
        return new_global, metrics


    def _save_aggregated_model(self, parameters: Parameters, total_examples: int):
        """Save aggregated model to disk"""
        try:
            ndarrays_to_save = parameters_to_ndarrays(parameters)
            
            # Primary save location
            from fedge.utils.fs_optimized import get_model_path
            model_path = get_model_path(self.project_root, self.server_id, self.global_round)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(model_path, "wb") as f:
                pickle.dump((ndarrays_to_save, int(total_examples)), f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            logger.info(f"[{self.server_str}] Saved primary model to {model_path}")

            
            # Mirror save to per-round directory
            leaf_dir = self.project_root / "rounds" / f"round_{self.global_round}" / "leaf" / f"server_{self.server_id}"
            leaf_dir.mkdir(parents=True, exist_ok=True)
            leaf_model = leaf_dir / "model.pkl"
            
            with open(leaf_model, "wb") as f2:
                pickle.dump((ndarrays_to_save, int(total_examples)), f2, protocol=pickle.HIGHEST_PROTOCOL)
                f2.flush()
                os.fsync(f2.fileno())
            logger.info(f"[{self.server_str}] Saved per-round model to {leaf_model}")
            
        except Exception as e:
            logger.error(f"[{self.server_str}] Failed to save model: {e}")
            # Create error signal
            try:
                from fedge.utils.fs_optimized import create_error_signal
                create_error_signal(self.project_root, self.global_round, self.server_id, str(e))
            except Exception:
                pass
            raise

    def _write_fit_metrics_csv(self, rnd: int, results: List[Tuple[int, Any]]):
        """Save per-client fit metrics"""
        local_rnd = rnd - 1
        metrics_dir = self.base_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        clients_csv = metrics_dir / f"server_{self.server_id}_client_fit_metrics.csv"
        
        write_header = not clients_csv.exists()
        with open(clients_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["global_round","local_round","client_id","train_loss","num_examples"])
            if write_header:
                writer.writeheader()
            
            for cid, fit_res in results:
                cid_str = fit_res.metrics.get("client_id", str(cid))
                loss = fit_res.metrics.get("train_loss", 0.0)
                n = fit_res.num_examples
                writer.writerow({
                    "global_round": self.global_round,
                    "local_round": local_rnd,
                    "client_id": cid_str,
                    "train_loss": loss,
                    "num_examples": n,
                })
        logger.debug(f"[{self.server_str}] Saved client fit metrics to {clients_csv}")

    def _write_eval_metrics_csv(self, rnd: int, results: List[Tuple[int, Any]]):
        """Save per-client evaluation metrics"""
        local_rnd = rnd - 1
        metrics_dir = self.base_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        clients_eval_csv = metrics_dir / f"server_{self.server_id}_client_eval_metrics.csv"
        
        write_header = not clients_eval_csv.exists()
        with open(clients_eval_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["global_round","local_round","client_id","eval_loss","accuracy","num_examples"])
            if write_header:
                writer.writeheader()
            
            for cid, eval_res in results:
                cid_str = eval_res.metrics.get("client_id", str(cid))
                loss = eval_res.loss
                acc = eval_res.metrics.get("accuracy", 0.0)
                n = eval_res.num_examples
                writer.writerow({
                    "global_round": self.global_round,
                    "local_round": local_rnd,
                    "client_id": cid_str,
                    "eval_loss": loss,
                    "accuracy": acc,
                    "num_examples": n,
                })
        logger.debug(f"[{self.server_str}] Saved client eval metrics to {clients_eval_csv}")

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Configure fit with SCAFFOLD and FedProx parameters"""
        # Get base fit configurations from parent
        fit_cfgs = super().configure_fit(server_round, parameters, client_manager)
        logger.info(f"[{self.server_str}] Selected {len(fit_cfgs)} clients for round {server_round}")
        
        # Load configuration
        try:
            with open(self.project_root / "pyproject.toml", "r", encoding="utf-8") as f:
                config = toml.load(f)
            hierarchy_cfg = config.get("tool", {}).get("flwr", {}).get("hierarchy", {})
        except Exception as e:
            logger.warning(f"Could not load TOML config: {e}, using defaults")
            hierarchy_cfg = {}
        
        # Force disable client-side SCAFFOLD (server-side only)
        scaffold_enabled = False
        
        # Extract hyperparameters
        lr_init_env = os.environ.get("LR_INIT")
        if lr_init_env is None:
            raise ValueError("LR_INIT environment variable required")
        client_lr = float(lr_init_env)
        
        required_params = ['weight_decay', 'clip_norm', 'momentum', 'lr_gamma', 'prox_mu']
        for param in required_params:
            if param not in hierarchy_cfg:
                raise ValueError(f"{param} must be in [tool.flwr.hierarchy]")
        
        weight_decay = float(hierarchy_cfg["weight_decay"])
        clip_norm = float(hierarchy_cfg["clip_norm"])
        momentum = float(hierarchy_cfg["momentum"])
        lr_gamma = float(hierarchy_cfg["lr_gamma"])
        mu_base = float(hierarchy_cfg["prox_mu"])
        
        # Configure all clients with hyperparameters
        for cid, fit_ins in fit_cfgs:
            fit_ins.config["scaffold_enabled"] = scaffold_enabled
            fit_ins.config["learning_rate"] = client_lr
            fit_ins.config["weight_decay"] = weight_decay
            fit_ins.config["momentum"] = momentum
            fit_ins.config["clip_norm"] = clip_norm
            fit_ins.config["lr_gamma"] = lr_gamma
            fit_ins.config["proximal_mu"] = mu_base
            fit_ins.config["global_round"] = self.global_round
        
        return fit_cfgs

    def aggregate_evaluate(
        self, 
        server_round: int, 
        results: List[Tuple[ClientProxy, EvaluateRes]], 
        failures: List[BaseException]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """✅ OPTIMIZED: Single evaluation on server test shard only"""
        round_start = time.time()
        local_rnd = server_round - 1
        
        # Call parent to get weighted average of client evaluations
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is None:
            logger.warning(f"[{self.server_str}] No evaluation results to aggregate")
            return None
        
        agg_loss, agg_metrics = aggregated
        agg_acc = agg_metrics.get("agg_accuracy", 0.0)
        
        # Write per-client eval metrics
        if results:
            self._write_eval_metrics_csv(server_round, results)
        
        # ✅ OPTIMIZED: Single server-side evaluation on server test shard ONLY
        model = Net()
        if hasattr(self, "latest_parameters") and self.latest_parameters is not None:
            nds = parameters_to_ndarrays(self.latest_parameters)
            set_weights(model, nds)
        else:
            logger.warning(f"[{self.server_str}] No parameters for evaluation, using fresh model")
        
        dev = torch.device("cpu")
        
        server_eval_start = time.time()
        batch_size = getattr(self, 'eval_batch_size', 32)
        testloader_server = get_cifar10_test_loader(
            batch_size=batch_size,
            server_id=self.server_id,
            num_servers=self.num_servers
        )
        server_loss, server_acc = test(model, testloader_server, dev)
        server_eval_time = time.time() - server_eval_start
        server_samples = len(testloader_server.dataset)
        
        logger.info(f"[{self.server_str}] Server eval: acc={server_acc:.4f}, loss={server_loss:.4f}, "
                   f"samples={server_samples}, time={server_eval_time:.2f}s")
        
        # Compute statistics from client results
        import numpy as np
        from scipy import stats
        
        client_losses = [res[1].loss for res in results if res[1].loss is not None]
        client_accs = [res[1].metrics.get("accuracy", 0) for res in results 
                      if res[1].metrics.get("accuracy") is not None]
        
        loss_std = np.std(client_losses) if len(client_losses) > 1 else 0.0
        acc_std = np.std(client_accs) if len(client_accs) > 1 else 0.0
        
        loss_ci_lower, loss_ci_upper = None, None
        acc_ci_lower, acc_ci_upper = None, None
        
        if len(client_losses) > 1:
            loss_mean = np.mean(client_losses)
            loss_sem = stats.sem(client_losses)
            loss_ci = stats.t.interval(0.95, len(client_losses)-1, loc=loss_mean, scale=loss_sem)
            loss_ci_lower, loss_ci_upper = loss_ci
            
        if len(client_accs) > 1:
            acc_mean = np.mean(client_accs)
            acc_sem = stats.sem(client_accs)
            acc_ci = stats.t.interval(0.95, len(client_accs)-1, loc=acc_mean, scale=acc_sem)
            acc_ci_lower, acc_ci_upper = acc_ci
        
        # Communication/computation metrics
        total_bytes_up = sum(r[1].metrics.get("bytes_up", 0) for r in results)
        total_bytes_down = sum(r[1].metrics.get("bytes_down_eval", 0) for r in results)
        avg_round_time = np.mean([r[1].metrics.get("round_time", 0) for r in results]) if results else 0
        avg_compute_time = np.mean([r[1].metrics.get("compute_s", 0) for r in results]) if results else 0
        wall_clock_time = time.time() - round_start
        
        # ✅ SIMPLIFIED: Write server metrics (removed local/gap columns)
        metrics_dir = self.base_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        server_csv = metrics_dir / f"server_{self.server_id}_metrics.csv"
        write_header = not server_csv.exists()
        
        with open(server_csv, "a", newline="") as fsv:
            writer = csv.DictWriter(
                fsv,
                fieldnames=[
                    "global_round", "local_round",
                    "agg_loss", "agg_acc",
                    "server_loss", "server_acc", "server_samples", "server_eval_time_s",
                    "loss_std", "acc_std",
                    "loss_ci_lower", "loss_ci_upper",
                    "acc_ci_lower", "acc_ci_upper",
                    "total_bytes_up", "total_bytes_down",
                    "avg_client_round_time", "avg_client_compute_time",
                    "server_wall_clock_time",
                ],
            )
            if write_header:
                writer.writeheader()
            
            writer.writerow({
                "global_round": self.global_round,
                "local_round": local_rnd,
                "agg_loss": agg_loss,
                "agg_acc": agg_acc,
                "server_loss": server_loss,
                "server_acc": server_acc,
                "server_samples": server_samples,
                "server_eval_time_s": server_eval_time,
                "loss_std": loss_std,
                "acc_std": acc_std,
                "loss_ci_lower": loss_ci_lower,
                "loss_ci_upper": loss_ci_upper,
                "acc_ci_lower": acc_ci_upper,
                "acc_ci_upper": acc_ci_upper,
                "total_bytes_up": total_bytes_up,
                "total_bytes_down": total_bytes_down,
                "avg_client_round_time": avg_round_time,
                "avg_client_compute_time": avg_compute_time,
                "server_wall_clock_time": wall_clock_time,
            })
        
        logger.info(f"[{self.server_str}] Server metrics saved to {server_csv}")
        return aggregated


def _write_error_signal_and_exit(server_id: int, error_msg: str, exit_code: int) -> None:
    """Write error signal and exit"""
    try:
        global_round = int(os.getenv('GLOBAL_ROUND', '1'))
        signals_dir = Path().resolve() / "signals" / f"round_{global_round}"
        signals_dir.mkdir(parents=True, exist_ok=True)
        
        error_path = signals_dir / f"server_{server_id}_error.signal"
        error_path.write_text(f"{error_msg}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
        logger.error(f"[Server {server_id}] Error signal written: {error_path}")
    except Exception as e:
        logger.error(f"[Server {server_id}] Failed to write error signal: {e}")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logging.shutdown()
    except Exception:
        pass
    
    os._exit(exit_code)


def handle_signal(sig, frame):
    """Handle termination signals gracefully"""
    server_id = os.environ.get("SERVER_ID", "unknown")
    logger.info(f"[Leaf Server {server_id}] Received signal {sig}, shutting down...")
    sys.exit(0)

def _load_cluster_model_for_server(
    project_root: Path, server_id: int, prev_global_round: int
) -> Optional[List[np.ndarray]]:
    """
    Load this server's cluster model from the previous cloud round.
    """
    if prev_global_round < 1:
        return None

    logger_str = f"Leaf Server {server_id}"
    try:
        rounds_cloud_dir = project_root / "rounds" / f"round_{prev_global_round}" / "cloud"
        models_dir = project_root / "models"

        # Try both common locations
        meta_candidates = [
            models_dir / f"clusters_g{prev_global_round}.json",
            rounds_cloud_dir / f"clusters_g{prev_global_round}.json",
        ]

        assignments = None
        for meta in meta_candidates:
            if meta.exists():
                with open(meta, "r") as f:
                    data = json.load(f)
                # handle {"assignments": {...}} or a plain dict
                assignments = data.get("assignments") if isinstance(data, dict) else None
                if assignments is None and isinstance(data, dict):
                    assignments = data
                break

        if not assignments:
            logger.info(f"[{logger_str}] No cluster assignments JSON for round {prev_global_round}")
            return None

        # keys may be strings
        cid = assignments.get(str(server_id))
        if cid is None:
            cid = assignments.get(server_id)
        if cid is None:
            logger.warning(f"[{logger_str}] Server {server_id} not present in cluster assignments")
            return None

        model_path = rounds_cloud_dir / f"model_cluster{cid}_g{prev_global_round}.pkl"
        if not model_path.exists():
            logger.warning(f"[{logger_str}] Cluster model not found: {model_path}")
            return None

        with open(model_path, "rb") as f:
            cluster_model = pickle.load(f)  # expected List[np.ndarray]
        logger.info(f"[{logger_str}] ✅ Loaded cluster {cid} model from previous round")
        return cluster_model

    except Exception as e:
        logger.error(f"[{logger_str}] Failed to load cluster model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    



def main():
    """Main server entrypoint"""
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_id", type=int, required=True)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--clients_per_server", type=int, default=10)
    parser.add_argument("--server_rounds", type=int, default=1)
    parser.add_argument("--fraction_fit", type=float, default=1.0)
    parser.add_argument("--fraction_evaluate", type=float, default=1.0)
    parser.add_argument("--global_round", type=int, default=1)
    parser.add_argument("--initial_model_path", type=str, default=None)

    args = parser.parse_args()
    
    server_id = args.server_id
    os.environ["SERVER_ID"] = str(server_id)
    os.environ["GLOBAL_ROUND"] = str(args.global_round)
    
    server_str = f"Leaf Server {server_id}"
    logger.info(f"[{server_str}] Starting on port {args.port}")
    logger.info(f"[{server_str}] Global round: {args.global_round}, Server rounds: {args.server_rounds}")
    
    # Determine project root
    script_dir = Path(__file__).resolve().parent
    project_root_local = script_dir.parent
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ✅ CRITICAL FIX: Load cluster model from previous round
    # ═══════════════════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════════════════
    # ✅ Prefer explicit path from orchestrator (net_topo) if provided
    #    Fallback to previous-round cluster model; else fresh model.
    # ═══════════════════════════════════════════════════════════════════════════
    initial_params = None

    # 1) Preferred: exact model path passed by net_topo
    if getattr(args, "initial_model_path", None):
        p = Path(args.initial_model_path)
        if p.is_file():
            try:
                with open(p, "rb") as f:
                    obj = pickle.load(f)
                # cloud cluster models are raw List[np.ndarray]; leaf saves may be (weights, total_examples)
                if isinstance(obj, tuple) and len(obj) == 2:
                    weights = obj[0]
                else:
                    weights = obj
                initial_params = ndarrays_to_parameters(weights)
                logger.info(f"[{server_str}] ✅ Initialized from {p.name}")
            except Exception as e:
                logger.warning(f"[{server_str}] Failed to load initial_model_path {p}: {e}")

    # 2) Fallback: previous-round cluster model
    if initial_params is None and args.global_round > 1:
        logger.info(f"[{server_str}] Round {args.global_round}: trying previous-round cluster model")
        cluster_model = _load_cluster_model_for_server(
            project_root_local, server_id, args.global_round - 1
        )
        if cluster_model is not None:
            initial_params = ndarrays_to_parameters(cluster_model)
            logger.info(f"[{server_str}] ✅✅✅ INITIALIZED WITH CLUSTER MODEL ✅✅✅")

    # 3) Final fallback: fresh model
    if initial_params is None:
        net = Net()
        initial_params = ndarrays_to_parameters(get_weights(net))
        if args.global_round > 1:
            logger.warning(f"[{server_str}] ⚠️ No explicit or cluster model found; using fresh model")
        else:
            logger.info(f"[{server_str}] Round 1: Initialized with fresh global model")

    
    # Create strategy with loaded parameters (cluster or fresh)
    strategy = LeafFedAvg(
        clients_per_server=args.clients_per_server,
        initial_parameters=initial_params,  # ✅ Now uses cluster model if available!
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        server_id=server_id,
        project_root=project_root_local,
        global_round=args.global_round,
    )
    
    try:
        # Start Flower server
        history = start_server(
            server_address=f"0.0.0.0:{args.port}",
            config=ServerConfig(num_rounds=args.server_rounds),
            strategy=strategy,
        )
        
        # ✅ FIX: Ensure model is saved after all rounds complete
        if hasattr(strategy, 'latest_parameters') and strategy.latest_parameters is not None:
            try:
                ndarrays_to_save = parameters_to_ndarrays(strategy.latest_parameters)
                
                # Determine total_examples
                total_examples = getattr(strategy, '_last_total_examples', args.clients_per_server * 1000)
                
                # Primary save
                from fedge.utils.fs_optimized import get_model_path
                model_path = get_model_path(project_root_local, server_id, args.global_round)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(model_path, "wb") as f:
                    pickle.dump((ndarrays_to_save, int(total_examples)), f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.flush()
                    os.fsync(f.fileno())
                logger.info(f"[{server_str}] Final model saved to {model_path}")
                
                # Mirror save
                leaf_dir = project_root_local / "rounds" / f"round_{args.global_round}" / "leaf" / f"server_{server_id}"
                leaf_dir.mkdir(parents=True, exist_ok=True)
                leaf_model = leaf_dir / "model.pkl"
                
                with open(leaf_model, "wb") as f2:
                    pickle.dump((ndarrays_to_save, int(total_examples)), f2, protocol=pickle.HIGHEST_PROTOCOL)
                    f2.flush()
                    os.fsync(f2.fileno())
                logger.info(f"[{server_str}] Final per-round model saved to {leaf_model}")
                
            except Exception as e:
                logger.error(f"[{server_str}] Failed final model save: {e}")
                _write_error_signal_and_exit(server_id, str(e), 2)
        
    except Exception as e:
        import traceback
        logger.error(f"[{server_str}] Server error: {type(e).__name__}: {str(e)}")
        logger.error(f"[{server_str}] Traceback: {traceback.format_exc()}")
        _write_error_signal_and_exit(server_id, f"server failed: {e}", 2)
    
    # ❌ STRICT: Fail if no history
    if 'history' not in locals() or history is None:
        logger.error(f"[{server_str}] CRITICAL: No history - server failed")
        _write_error_signal_and_exit(server_id, "no history", 2)
    
    # Save communication metrics
    try:
        import pandas as pd
        import filelock
        
        mdf = getattr(history, "metrics_distributed_fit", {}) or {}
        
        def _collect(keys):
            for k in keys:
                if k in mdf:
                    return mdf[k]
            return []
        
        up_entries = _collect(["bytes_up", "bytes_written"])
        down_entries = _collect(["bytes_down", "bytes_read"])
        rt_entries = _collect(["round_time"])
        comp_entries = _collect(["compute_s"])
        
        by_round = {}
        for rnd, val in up_entries:
            by_round.setdefault(rnd, {})["bytes_up"] = int(val)
        for rnd, val in down_entries:
            by_round.setdefault(rnd, {})["bytes_down"] = int(val)
        for rnd, val in rt_entries:
            by_round.setdefault(rnd, {})["round_time"] = float(val)
        for rnd, val in comp_entries:
            by_round.setdefault(rnd, {})["compute_s"] = float(val)
        
        rows = [
            {
                "global_round": args.global_round,
                "round": rnd,
                "bytes_up": int(vals.get("bytes_up", 0)),
                "bytes_down": int(vals.get("bytes_down", 0)),
                "round_time": vals.get("round_time", 0.0),
                "compute_s": vals.get("compute_s", 0.0),
            }
            for rnd, vals in sorted(by_round.items())
        ]
        
        if rows:
            df = pd.DataFrame(rows)
            out = Path(os.getenv("RUN_DIR", ".")) / f"edge_comm_{server_id}.csv"
            with filelock.FileLock(out.with_suffix(".lock")):
                mode = "a" if out.exists() else "w"
                df.to_csv(out, index=False, mode=mode, header=not out.exists())
            logger.info(f"[{server_str}] Communication metrics saved to {out}")
    except Exception as csv_err:
        logger.warning(f"[{server_str}] Could not write comm CSV: {csv_err}")
    
    # Create completion signal
    try:
        from fedge.utils.fs_optimized import create_completion_signal
        completion_signal = create_completion_signal(project_root_local, args.global_round, server_id)
        logger.info(f"[{server_str}] Completion signal created: {completion_signal}")
    except Exception as signal_err:
        logger.warning(f"[{server_str}] Could not create completion signal: {signal_err}")
    
    logger.info(f"[{server_str}] Process completed successfully")
    
    # Cleanup
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logging.shutdown()
    except Exception:
        pass
    
    sys.exit(0)


def main_wrapper():
    """Wrapper to ensure graceful exit"""
    try:
        main()
    except Exception as e:
        server_id = int(os.environ.get("SERVER_ID", "0"))
        logger.error(f"[Leaf Server {server_id}] Main function error: {e}")
        _write_error_signal_and_exit(server_id, f"server crashed: {e}", 3)


if __name__ == "__main__":
    gc.collect()
    main_wrapper()

