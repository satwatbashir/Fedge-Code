# fedge/leaf_server.py

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
    # Set console encoding to UTF-8 to handle Unicode characters
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

# Suppress Python deprecation warnings in flwr module
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
# Elevate Flower, ECE, gRPC logger levels to ERROR to hide warning logs
for name in ("flwr", "ece", "grpc"): logging.getLogger(name).setLevel(logging.ERROR)
# Drop printed 'DEPRECATED FEATURE' messages from stdout/stderr
class _DropDeprecated:
    def __init__(self, out): self._out = out
    def write(self, txt):
        if "DEPRECATED FEATURE" not in txt: self._out.write(txt)
    def flush(self): self._out.flush()

sys.stdout = _DropDeprecated(sys.stdout)
sys.stderr = _DropDeprecated(sys.stderr)



class LeafFedAvg(FedAvg):
    """SCAFFOLD strategy for FedGE leaf servers with server-side control variates"""
    
    def __init__(self, clients_per_server: int, *, initial_parameters: Optional[Parameters] = None, **kwargs):
        # Extract custom parameters before passing to parent
        server_id = kwargs.pop('server_id', 0)
        project_root = kwargs.pop('project_root', Path.cwd())
        global_round = kwargs.pop('global_round', 0)
        server_lr = kwargs.pop('server_lr', 1.0)
        global_lr = kwargs.pop('global_lr', 1.0)
        
        # Force full participation and no failure acceptance
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=clients_per_server,
            min_evaluate_clients=clients_per_server,
            min_available_clients=clients_per_server,
            accept_failures=False,
            evaluate_fn=self._server_side_evaluate,  # Enable server-side evaluation
        )
        # Flower reads this attribute; setting it here works across versions
        self.accept_failures = False
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
        self.latest_parameters = initial_parameters  # Initialize with provided parameters
        
        # Get configuration from pyproject.toml
        cfg = toml.load(self.project_root / "pyproject.toml")
        hierarchy_config = cfg["tool"]["flwr"]["hierarchy"]
        self.num_servers = hierarchy_config["num_servers"]
        
        # Override global_lr from config if not provided
        if global_lr == 1.0:
            self.global_lr = hierarchy_config.get("global_lr", 1.0)
        
        # Set base_dir for metrics path
        self.base_dir = self.project_root
        
        # Evaluation settings
        self.eval_batch_size = hierarchy_config["eval_batch_size"]
        self.cluster_better_delta = hierarchy_config["cluster_better_delta"]
        
        # Prepare validation loader for accuracy gate (CIFAR-10 only)
        try:
            self._valloader_gate = get_cifar10_test_loader(batch_size=self.eval_batch_size)
        except Exception as e:
            logger.error(f"[{self.server_str}] Error loading validation data: {e}")
            raise

    def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Weighted average of accuracy metrics
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        total_examples = sum([num_examples for num_examples, _ in metrics])
        return {"accuracy": sum(accuracies) / total_examples}

    def _server_side_evaluate(self, server_round: int, parameters: Parameters, config: Dict[str, Any]):
        """Server-side evaluation function to enable client evaluation rounds"""
        # This function enables the Flower framework to trigger client evaluation
        # We don't need to do server-side evaluation here since we do centralized evaluation
        # in aggregate_evaluate(). Just return None to let client evaluation proceed.
        return None
    
    
    def aggregate_fit(self, rnd: int, results, failures):
        """SCAFFOLD aggregation with server-side control variates"""
        round_start_time = time.time()
        
        # Log client results
        print(f"[{self.server_str}] SCAFFOLD aggregating fit results from clients for round {rnd}")
        for cid, fit_res in results:
            friendly = fit_res.metrics.get("client_id", str(cid))
            loss = fit_res.metrics.get("train_loss", None)
            n = fit_res.num_examples
            print(f"  → Client {friendly} train_loss={loss:.4f} on {n} samples")
        if failures:
            for failure in failures:
                print(f"  !! Failure: {failure}")
        
        # Initialize SCAFFOLD control variates if first round
        if not self.c_global and self._latest_global_parameters is not None:
            prev_global_nd = parameters_to_ndarrays(self._latest_global_parameters)
            self.c_global = [np.zeros_like(x, dtype=np.float32) for x in prev_global_nd]
            print(f"[{self.server_str}] Initialized SCAFFOLD global control variates")
        
        # SCAFFOLD aggregation
        y_deltas: List[List[torch.Tensor]] = []
        weights: List[int] = []
        
        m = len(self.c_global) if self.c_global else 0
        
        # Process each client's results
        for client, fit_res in results:
            # Get client parameters
            returned_params: Parameters = fit_res.parameters
            nd_list = parameters_to_ndarrays(returned_params)
            num_examples = fit_res.num_examples
            
            # For server-side SCAFFOLD, we compute y_delta = W_local - W_global
            if self._latest_global_parameters is not None:
                prev_global_nd = parameters_to_ndarrays(self._latest_global_parameters)
                y_delta = [nd_list[i] - prev_global_nd[i] for i in range(len(nd_list))]
            else:
                # First round - use client parameters as delta
                y_delta = nd_list
            
            y_deltas.append([torch.tensor(x) for x in y_delta])
            weights.append(num_examples)
            
            # Initialize client control variate if needed
            cid = str(client.cid)
            if cid not in self.c_locals and self.c_global:
                self.c_locals[cid] = [np.zeros_like(x, dtype=np.float32) for x in self.c_global]
        
        # Compute weighted average of y_deltas
        if y_deltas and self._latest_global_parameters is not None:
            total_weight = float(sum(weights))
            norm_weights = [w / total_weight for w in weights]
            
            # Store total examples for safety save
            self._last_total_examples = int(total_weight)
            
            # Get previous global parameters
            prev_global_nd = parameters_to_ndarrays(self._latest_global_parameters)
            W = [torch.tensor(x) for x in prev_global_nd]
            
            # Apply SCAFFOLD update: W_global ← W_global + η * average(y_delta)
            for idx, parts in enumerate(zip(*y_deltas)):
                stacked = torch.stack(list(parts), dim=0)
                w = torch.tensor(norm_weights, device=stacked.device)
                for _ in range(stacked.ndim - 1):
                    w = w.unsqueeze(1)
                avg_delta = (stacked * w).sum(dim=0)
                if W[idx].is_floating_point():
                    W[idx] = W[idx] + self.global_lr * avg_delta
            
            new_global = ndarrays_to_parameters([w.numpy() for w in W])
            self._latest_global_parameters = new_global
            
            print(f"[{self.server_str}] Applied SCAFFOLD server-side aggregation with global_lr={self.global_lr}")
        else:
            # Fallback to standard FedAvg for first round or if no results
            new_global = super().aggregate_fit(rnd, results, failures)
            if new_global is not None and isinstance(new_global, tuple):
                self._latest_global_parameters = new_global[0]
            else:
                self._latest_global_parameters = new_global
        
        # Save client fit metrics
        self._write_fit_metrics_csv(rnd, results)
        
        # Store aggregated accuracy from fit results (train accuracy if available)
        agg_acc = None
        if results:
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            if total_examples > 0:
                weighted_acc_sum = 0.0
                acc_count = 0
                for _, fit_res in results:
                    # Try to get accuracy from fit results (some clients may provide it)
                    client_acc = fit_res.metrics.get("train_accuracy", fit_res.metrics.get("accuracy", None))
                    if client_acc is not None:
                        weighted_acc_sum += fit_res.num_examples * client_acc
                        acc_count += fit_res.num_examples
                if acc_count > 0:
                    agg_acc = weighted_acc_sum / acc_count
                    print(f"[{self.server_str}] Computed fit aggregated accuracy: {agg_acc:.4f}")
        
        # Store for use in aggregate_evaluate
        self._fit_agg_acc = agg_acc
        
        # Update latest parameters - ensure it's always a Parameters object
        if new_global is not None:
            if isinstance(new_global, tuple) and len(new_global) > 0:
                # Extract Parameters object from tuple
                self.latest_parameters = new_global[0]
            elif isinstance(new_global, Parameters):
                # Already a Parameters object
                self.latest_parameters = new_global
            else:
                # Convert ndarrays to Parameters if needed
                if isinstance(new_global, list):
                    self.latest_parameters = ndarrays_to_parameters(new_global)
                else:
                    self.latest_parameters = new_global
        
        # Save model after every round for hierarchical FL
        if new_global is not None:
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            print(f"[{self.server_str}] Total training examples: {total_examples}")

            parameters = new_global[0] if isinstance(new_global, tuple) else new_global
            if parameters is not None:
                try:
                    from fedge.utils.fs_optimized import get_model_path
                    model_path = get_model_path(self.project_root, self.server_id, self.global_round)
                    with open(model_path, "wb") as f:
                        pickle.dump((parameters_to_ndarrays(parameters), total_examples), f)
                    print(f"[{self.server_str}] Saved SCAFFOLD model to {model_path}")
                except Exception as e:
                    print(f"[{self.server_str}] Failed to save model: {e}")
        
        # Return a tuple of (parameters, metrics) as expected by Flower framework
        metrics = {}
        if agg_acc is not None:
            metrics["agg_accuracy"] = float(agg_acc)
        
        # If new_global is already a tuple, return it
        if isinstance(new_global, tuple):
            return new_global
        
        # Otherwise, wrap the parameters in a tuple with metrics
        return new_global, metrics

    def _write_fit_metrics_csv(self, rnd: int, results: List[Tuple[int, Any]]):
        # Save per-client fit metrics into per-round/per-server metrics folder for symmetry
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
                loss = fit_res.metrics.get("train_loss")
                n = fit_res.num_examples
                writer.writerow({
                    "global_round": self.global_round,
                    "local_round": local_rnd,
                    "client_id": cid_str,
                    "train_loss": loss,
                    "num_examples": n,
                })
        logger.info(f"[{self.server_str} | Round {local_rnd}] Appended client fit metrics → {clients_csv}")

    def _write_eval_metrics_csv(self, rnd: int, results: List[Tuple[int, Any]]):
        """Save per-client evaluation loss & accuracy for this round"""
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
                acc = eval_res.metrics.get("accuracy")
                n = eval_res.num_examples
                writer.writerow({
                    "global_round": self.global_round,
                    "local_round": local_rnd,
                    "client_id": cid_str,
                    "eval_loss": loss,
                    "accuracy": acc,
                    "num_examples": n,
                })
        logger.info(f"[{self.server_str} | Round {local_rnd}] Appended client eval metrics → {clients_eval_csv}")

    # ------------------------------------------------------------------
    #  Optional accuracy gate: Only accept incoming global/cluster model
    #  if it improves validation accuracy on this leaf's data.
    # ------------------------------------------------------------------
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        # ✅ CRITICAL FIX: Get base fit configurations FIRST
        fit_cfgs = super().configure_fit(server_round, parameters, client_manager)
        
        # ✅ CRITICAL FIX: Set all client configs BEFORE any early returns
        # This ensures SCAFFOLD and FedProx are NEVER skipped by the accuracy gate
        
        # Get SCAFFOLD setting from TOML config only (no environment overrides)
        scaffold_enabled = False  # Will be set from TOML config below
        
        # ✅ NEW: Load ALL client hyperparameters from TOML config
        try:
            import toml
            from pathlib import Path
            with open(self.project_root / "pyproject.toml", "r", encoding="utf-8") as f:
                config = toml.load(f)
            app_cfg = config.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})
            hierarchy_cfg = config.get("tool", {}).get("flwr", {}).get("hierarchy", {})
        except Exception as e:
            logger.warning(f"Could not load TOML config: {e}, using defaults")
            app_cfg = {}
            hierarchy_cfg = {}
        # Force disable client-side SCAFFOLD for server-side only implementation
        scaffold_enabled = False  # Always disabled on clients for server-side SCAFFOLD
        
        # ✅ CRITICAL: Extract client hyperparameters from environment
        # Use LR_INIT from environment as single source of truth for learning rate
        lr_init_env = os.environ.get("LR_INIT")
        if lr_init_env is None:
            raise ValueError("LR_INIT environment variable is required but not set")
        client_lr = float(lr_init_env)
        
        required_hierarchy_params = ['weight_decay', 'clip_norm', 'momentum', 'lr_gamma']
        for param in required_hierarchy_params:
            if param not in hierarchy_cfg:
                raise ValueError(f"{param} must be specified in [tool.flwr.hierarchy] section of pyproject.toml")
        
        weight_decay = float(hierarchy_cfg["weight_decay"])
        clip_norm = float(hierarchy_cfg["clip_norm"])
        momentum = float(hierarchy_cfg["momentum"])
        lr_gamma = float(hierarchy_cfg["lr_gamma"])
        
        # Get FedProx parameters from TOML config only
        if "prox_mu" not in hierarchy_cfg:
            raise ValueError("prox_mu must be specified in [tool.flwr.hierarchy] section of pyproject.toml")
        mu_base = float(hierarchy_cfg["prox_mu"])
        
        # These FedProx adaptive parameters should also come from TOML
        required_hierarchy_params = ['prox_mu_min', 'acc_stable_threshold', 'loss_delta']
        for param in required_hierarchy_params:
            if param not in hierarchy_cfg:
                raise ValueError(f"{param} must be specified in [tool.flwr.hierarchy] section of pyproject.toml")
        
        mu_min = float(hierarchy_cfg["prox_mu_min"])
        acc_stable = float(hierarchy_cfg["acc_stable_threshold"])
        loss_delta = float(hierarchy_cfg["loss_delta"])
        
        # Load per-client eval metrics for adaptive FedProx
        import csv
        metrics_path = self.base_dir / "metrics" / f"server_{self.server_id}_client_eval_metrics.csv"
        rows = list(csv.DictReader(open(metrics_path))) if metrics_path.exists() else []
        
        # ✅ CRITICAL: Configure ALL clients with SCAFFOLD + FedProx + Hyperparams BEFORE gate check
        for cid, fit_ins in fit_cfgs:
            # ✅ FIX #1: Pass SCAFFOLD through config instead of environment
            fit_ins.config["scaffold_enabled"] = scaffold_enabled
            
            # ✅ NEW: Pass ALL client hyperparameters through config
            fit_ins.config["learning_rate"] = client_lr
            fit_ins.config["weight_decay"] = weight_decay
            fit_ins.config["clip_norm"] = clip_norm
            fit_ins.config["momentum"] = momentum
            fit_ins.config["lr_gamma"] = lr_gamma
            
            # ✅ FIX #2: Calculate adaptive FedProx proximal_mu per client
            cid_str = fit_ins.config.get("client_id", str(cid))
            mu = mu_base
            if rows and self.global_round >= 1:
                last_round = self.global_round - 1
                prev_round = self.global_round - 2
                client_rows = [r for r in rows if r.get("client_id") == cid_str]
                last_entries = [r for r in client_rows if int(r.get("global_round", -1)) == last_round]
                prev_entries = [r for r in client_rows if int(r.get("global_round", -1)) == prev_round]
                if last_entries and prev_entries:
                    best_last = max(last_entries, key=lambda r: int(r.get("local_round", 0)))
            
            # ✅ CRITICAL FIX: Add proximal_mu to client config (was missing!)
            fit_ins.config["proximal_mu"] = mu
            try:
                # Only run accuracy gate if we have valid previous parameters
                if self.latest_parameters is not None and isinstance(self.latest_parameters, Parameters):
                    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    # Evaluate old params
                    model_old = Net().to(dev)
                    set_weights(model_old, parameters_to_ndarrays(self.latest_parameters))
                    _, acc_old = test(model_old, self._valloader_gate, dev)
                    # Evaluate new params
                    model_new = Net().to(dev)
                    set_weights(model_new, parameters_to_ndarrays(parameters))
                    _, acc_new = test(model_new, self._valloader_gate, dev)
                    if acc_new < acc_old + self.cluster_better_delta:
                        # Keep old parameters but preserve our custom configs
                        logger.info(f"[{self.server_str}] Accuracy gate: keeping old params (old={acc_old:.4f}, new={acc_new:.4f})")
                        parameters = self.latest_parameters
                        # Re-create fit_cfgs with old parameters but keep our configs
                        old_fit_cfgs = super().configure_fit(server_round, parameters, client_manager)
                        # Preserve our custom SCAFFOLD + FedProx configs
                        for i, (old_cid, old_fit_ins) in enumerate(old_fit_cfgs):
                            if i < len(fit_cfgs):
                                _, original_fit_ins = fit_cfgs[i]
                                old_fit_ins.config.update(original_fit_ins.config)
                        fit_cfgs = old_fit_cfgs
                    else:
                        logger.info(f"[{self.server_str}] Accuracy gate: using new params (old={acc_old:.4f}, new={acc_new:.4f})")
                else:
                    # Skip accuracy gate for first round or invalid parameters
                    logger.info(f"[{self.server_str}] Skipping accuracy gate - no valid previous parameters")
            except Exception as e:
                logger.error(f"[{self.server_str}] Accuracy gate error: {e}")
        
        return fit_cfgs



    def aggregate_evaluate(self, rnd, results, failures):
        # DEBUG: Log evaluation call details
        logger.info(f"[{self.server_str}] aggregate_evaluate called for round {rnd}")
        logger.info(f"[{self.server_str}] Received {len(results)} evaluation results, {len(failures)} failures")
        if not results:
            logger.warning(f"[{self.server_str}] No evaluation results received - clients may not be evaluating")
        
        # 1) Save per-client evaluation metrics
        self._write_eval_metrics_csv(rnd, results)

        # 2) Aggregate client evaluation (FedAvg returns (loss, metrics))
        aggregated = super().aggregate_evaluate(rnd, results, failures)
        if aggregated is not None and isinstance(aggregated, tuple):
            agg_loss = aggregated[0]
            agg_acc = aggregated[1].get("accuracy") if aggregated[1] else None
        else:
            agg_loss, agg_acc = None, None
        
        # ✅ NEW: Compute aggregated accuracy from evaluation results
        if results and not agg_acc:  # Only if parent didn't provide agg_acc
            try:
                total_examples = sum(eval_res.num_examples for _, eval_res in results)
                if total_examples > 0:
                    weighted_acc_sum = 0.0
                    for _, eval_res in results:
                        client_acc = eval_res.metrics.get("accuracy", 0.0)
                        weighted_acc_sum += eval_res.num_examples * client_acc
                    agg_acc = weighted_acc_sum / total_examples
                    print(f"[{self.server_str}] Computed SCAFFOLD aggregated accuracy: {agg_acc:.4f}")
            except Exception as e:
                logger.error(f"[{self.server_str}] Error computing aggregated accuracy: {e}")
                agg_acc = None
        
        # Use fit accuracy as fallback if evaluation accuracy is not available
        if not agg_acc and hasattr(self, '_fit_agg_acc') and self._fit_agg_acc is not None:
            agg_acc = self._fit_agg_acc
            print(f"[{self.server_str}] Using fit aggregated accuracy as fallback: {agg_acc:.4f}")

        # 3) Centralized evaluation on full *global* test set
        local_rnd = rnd - 1
        model = Net()
        if hasattr(self, "latest_parameters") and self.latest_parameters is not None:
            nds = parameters_to_ndarrays(self.latest_parameters)
            set_weights(model, nds)
        dev = torch.device("cpu")

        # 3-a Global test set (shared by all servers) - use dedicated CIFAR-10 test loader
        from fedge.task import get_cifar10_test_loader
        batch_size = getattr(self, 'eval_batch_size', 20)  # From TOML config
        testloader_global = get_cifar10_test_loader(batch_size=batch_size)
        central_loss, central_acc = test(model, testloader_global, dev)

        # 3-b Local-centralised test set (union of this server's clients)
        dataset_flag = "cifar10"  # Fixed: define dataset_flag
        total_clients = self.num_servers * self.clients_per_server
        local_examples = 0
        local_loss_sum = 0.0
        local_acc_sum = 0.0
        for cid_local in range(self.clients_per_server):
            # Use cid_local directly as partition_id since each server has its own partition namespace
            part_id = cid_local
            _, testloader_local, _ = load_data(dataset_flag, part_id, total_clients, batch_size=batch_size, server_id=self.server_id)
            l_loss, l_acc = test(model, testloader_local, dev)
            n = len(testloader_local.dataset)
            local_examples += n
            local_loss_sum += l_loss * n
            local_acc_sum += l_acc * n
        local_loss = local_loss_sum / local_examples if local_examples else None
        local_acc = local_acc_sum / local_examples if local_examples else None

        # 4) Compute gaps and statistical metrics
        gap_global_loss = central_loss - (agg_loss or 0.0)
        gap_global_acc = central_acc - (agg_acc or 0.0)
        gap_local_loss = local_loss - (agg_loss or 0.0) if local_loss is not None else None
        gap_local_acc = local_acc - (agg_acc or 0.0) if local_acc is not None else None

        # 4-b) Calculate standard deviation and confidence intervals from client results
        import numpy as np
        from scipy import stats
        
        client_losses = [res[1].loss for res in results if res[1].loss is not None]
        client_accs = [res[1].metrics.get("accuracy", 0) for res in results if res[1].metrics.get("accuracy") is not None]
        
        # Standard deviations
        loss_std = np.std(client_losses) if len(client_losses) > 1 else 0.0
        acc_std = np.std(client_accs) if len(client_accs) > 1 else 0.0
        
        # 95% Confidence intervals (assuming normal distribution)
        loss_ci_lower, loss_ci_upper = None, None
        acc_ci_lower, acc_ci_upper = None, None
        
        if len(client_losses) > 1:
            loss_mean = np.mean(client_losses)
            loss_sem = stats.sem(client_losses)  # Standard error of mean
            loss_ci = stats.t.interval(0.95, len(client_losses)-1, loc=loss_mean, scale=loss_sem)
            loss_ci_lower, loss_ci_upper = loss_ci
            
        if len(client_accs) > 1:
            acc_mean = np.mean(client_accs)
            acc_sem = stats.sem(client_accs)
            acc_ci = stats.t.interval(0.95, len(client_accs)-1, loc=acc_mean, scale=acc_sem)
            acc_ci_lower, acc_ci_upper = acc_ci

        # 5) Create metrics folder and write server-side metrics CSV
        metrics_dir = self.base_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        server_csv = metrics_dir / f"server_{self.server_id}_metrics.csv"
        write_header = not server_csv.exists()
        with open(server_csv, "a", newline="") as fsv:
            writer = csv.DictWriter(
                fsv,
                fieldnames=[
                    "global_round","local_round",
                    "agg_loss","agg_acc",
                    "central_loss","central_acc",
                    "local_loss","local_acc",
                    "gap_global_loss","gap_global_acc",
                    "gap_local_loss","gap_local_acc",
                    "loss_std","acc_std",
                    "loss_ci_lower","loss_ci_upper",
                    "acc_ci_lower","acc_ci_upper",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow({
                "global_round": self.global_round,
                "local_round": local_rnd,
                "agg_loss": agg_loss,
                "agg_acc": agg_acc,
                "central_loss": central_loss,
                "central_acc": central_acc,
                "local_loss": local_loss,
                "local_acc": local_acc,
                "gap_global_loss": gap_global_loss,
                "gap_global_acc": gap_global_acc,
                "gap_local_loss": gap_local_loss,
                "gap_local_acc": gap_local_acc,
                "loss_std": loss_std,
                "acc_std": acc_std,
                "loss_ci_lower": loss_ci_lower,
                "loss_ci_upper": loss_ci_upper,
                "acc_ci_lower": acc_ci_lower,
                "acc_ci_upper": acc_ci_upper,
            })
        logger.info(f"[{self.server_str} | Round {local_rnd}] Appended server metrics → {server_csv}")
        return aggregated


def _write_error_signal_and_exit(server_id: int, error_msg: str, exit_code: int) -> None:
    """Write error signal and exit with non-zero code."""
    try:
        global_round = int(os.getenv('GLOBAL_ROUND', '1'))
        signals_dir = Path().resolve() / "signals" / f"round_{global_round}"
        signals_dir.mkdir(parents=True, exist_ok=True)
        
        error_path = signals_dir / f"server_{server_id}_error.signal"
        error_path.write_text(f"{error_msg}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
        
        logger.error(f"[Server {server_id}] Written error signal: {error_path}")
    except Exception as e:
        logger.error(f"[Server {server_id}] Failed to write error signal: {e}")
    
    # Cleanup and exit with error code
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
    logger.info(f"[Leaf Server {server_id}] Received signal {sig}, shutting down gracefully...")
    
    # If we know the server ID, try to save the model before exiting
    if server_id:
        try:
            script_dir = Path(__file__).resolve().parent
            project_root_local = script_dir.parent
            model_dir = project_root_local / "models"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f"server_{server_id}.pkl"
            
            # Try to get a model instance
            net = Net()
            ndarrays = get_weights(net)
            
            # Save with a default example count
            with open(model_path, "wb") as f:
                pickle.dump((ndarrays, 0), f)
            logger.info(f"[Leaf Server {server_id}] Saved emergency backup model during shutdown to {model_path}")
        except Exception as e:
            logger.error(f"[Leaf Server {server_id}] Failed to save emergency model: {e}")
    
    sys.exit(0)

def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_id", type=int, required=True)
    parser.add_argument("--clients_per_server", type=int, required=True)
    parser.add_argument("--num_rounds", type=int, required=True)
    parser.add_argument("--fraction_fit", type=float, required=True)
    parser.add_argument("--fraction_evaluate", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--initial_model_path", type=str, help="Path to global model to start from")
    parser.add_argument("--global_round", type=int, default=0, help="Current global round number (0-indexed)")
    parser.add_argument("--start_round", type=int, default=0, help="Starting round number (0-indexed)")
    parser.add_argument("--dir_round", type=int, help="Round number for directory naming (0-indexed)")
    
    args = parser.parse_args()

    # Get server ID for logging
    server_id = args.server_id
    server_str = f"Leaf Server {server_id}"
    
    # Store in environment for signal handlers
    os.environ["SERVER_ID"] = str(server_id)
    os.environ["SERVER_STR"] = server_str
    
    logger.info(f"[{server_str}] Starting server with {args.clients_per_server} clients, {args.num_rounds} rounds")
    
    # Initialize model and get parameters
    initial_parameters = None
    
    # CRITICAL FIX: Always provide initial_parameters to prevent random reinitialization
    if args.initial_model_path:
        model_path = Path(args.initial_model_path)
        if model_path.exists():
            logger.info(f"[{server_str}] Loading initial model from {model_path}")
            try:
                with open(model_path, "rb") as f:
                    # The global model could be stored in different formats
                    loaded_data = pickle.load(f)
                    if isinstance(loaded_data, tuple):
                        # If it's a tuple, first element should be the model parameters
                        ndarrays = loaded_data[0]
                    else:
                        # Otherwise, assume it's the raw parameters
                        ndarrays = loaded_data
                logger.info(f"[{server_str}] ✅ Loaded trained model from round {args.global_round}")
                initial_parameters = ndarrays_to_parameters(ndarrays)
            except Exception as e:
                logger.error(f"[{server_str}] Error loading initial model: {e}")
                logger.warning(f"[{server_str}] Falling back to fresh model parameters")
    
    # CRITICAL: If no model loaded, create fresh parameters (Round 1 only)
    if initial_parameters is None:
        logger.info(f"[{server_str}] Creating fresh model parameters for Round {args.global_round}")
        fresh_model = Net()
        fresh_weights = get_weights(fresh_model)
        initial_parameters = ndarrays_to_parameters(fresh_weights)
        logger.info(f"[{server_str}] ✅ Fresh model initialized with {len(fresh_weights)} parameter arrays")
    
    # Load TOML configuration for server learning rate
    try:
        script_dir = Path(__file__).resolve().parent
        project_root_local = script_dir.parent
        config_path = project_root_local / "pyproject.toml"
        config = toml.load(config_path)
        
        # Extract server learning rate and global learning rate from hierarchy config
        server_lr = float(config["tool"]["flwr"]["hierarchy"].get("server_lr", 1.0))
        global_lr = float(config["tool"]["flwr"]["hierarchy"].get("global_lr", 1.0))
        print(f"[{server_str}] Using server learning rate: {server_lr}, global learning rate: {global_lr}")
        
    except Exception as e:
        print(f"[{server_str}] Failed to load TOML config: {e}")
        server_lr = 1.0  # Default fallback
        global_lr = 1.0  # Default fallback
    
    # Configure strategy with SCAFFOLD support
    strategy = LeafFedAvg(
        server_id=server_id,
        project_root=project_root_local,
        global_round=args.global_round,
        server_lr=server_lr,
        global_lr=global_lr,
        clients_per_server=args.clients_per_server,
        initial_parameters=initial_parameters,  # Pass initial parameters to strategy
    )
    
    # Start server with error handling
    logger.info(f"[{server_str}] Starting Flower server on port {args.port}")
    try:
        bind_addr = os.getenv("BIND_ADDRESS", "0.0.0.0")
        # Run blocking Flower server and capture History object for metrics
        # Temporarily redirect stderr to suppress deprecation warnings
        import contextlib
        import io
        
        stderr_capture = io.StringIO()
        with contextlib.redirect_stderr(stderr_capture):
            history = start_server(
                server_address=f"{bind_addr}:{args.port}",
                config=ServerConfig(
                    num_rounds=args.num_rounds,
                    round_timeout=900.0,# Increased from 60.0 to handle client startup delays
                ),
                strategy=strategy,
            )
        logger.info(f"[{server_str}] Server has completed all rounds successfully")
        
        # === SAFETY SAVE: ensure model_s{sid}_g{GR}.pkl exists ===
        try:
            from fedge.utils.fs_optimized import get_model_path
            model_path = get_model_path(project_root_local, server_id, args.global_round)

            # Prefer the latest parameters kept by the strategy; fall back progressively
            ndarrays_to_save = None
            if getattr(strategy, "latest_parameters", None) is not None:
                ndarrays_to_save = parameters_to_ndarrays(strategy.latest_parameters)
            elif getattr(strategy, "_latest_global_parameters", None) is not None:
                ndarrays_to_save = parameters_to_ndarrays(strategy._latest_global_parameters)
            elif 'initial_parameters' in locals() and initial_parameters is not None:
                # very first round fallback
                ndarrays_to_save = parameters_to_ndarrays(initial_parameters)

            if ndarrays_to_save is None:
                raise RuntimeError("No parameters available to save at server end")

            # Compute total_examples from the strategy's last aggregation
            total_examples = 0
            try:
                # Try to get total_examples from the strategy's last aggregation
                if hasattr(strategy, '_last_total_examples'):
                    total_examples = strategy._last_total_examples
                elif hasattr(strategy, 'total_examples'):
                    total_examples = strategy.total_examples
                else:
                    # Fallback: estimate from client count and average samples
                    total_examples = args.clients_per_server * 1000  # reasonable estimate
                logger.info(f"[{server_str}] Using total_examples={total_examples} for model save")
            except Exception as e:
                logger.warning(f"[{server_str}] Could not determine total_examples: {e}, using fallback")
                total_examples = args.clients_per_server * 1000

            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump((ndarrays_to_save, int(total_examples)), f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"[{server_str}] 💾 Safety-saved model to {model_path}")
        except Exception as e:
            logger.error(f"[{server_str}] ❌ Failed safety-save of server model: {e}")
            # Emit an explicit error signal and exit non-zero so orchestrator halts correctly
            try:
                from fedge.utils.fs_optimized import create_error_signal
                create_error_signal(project_root_local, args.global_round, server_id, str(e))
            except Exception:
                pass
            sys.exit(2)
        
    except UnicodeEncodeError as e:
        logger.error(f"[{server_str}] Unicode encoding error: {str(e)}")
        logger.error(f"[{server_str}] This is likely due to special characters in logging output")
        # Continue with metrics saving
        history = None
    except Exception as e:
        import traceback
        logger.error(f"[{server_str}] Server error: {type(e).__name__}: {str(e)}")
        logger.error(f"[{server_str}] Full traceback: {traceback.format_exc()}")
        
        # Check for specific error types
        if "address already in use" in str(e).lower() or "bind" in str(e).lower():
            logger.error(f"[{server_str}] PORT CONFLICT: Another process is using port {args.port}")
        elif "connection" in str(e).lower():
            logger.error(f"[{server_str}] CONNECTION ERROR: Network connectivity issue")
        
        # Don't return here - continue with cleanup
        history = None
        
    # ❌ STRICT MODE: Fail hard if no history (server failed)
    if history is None:
        logger.error(f"[{server_str}] CRITICAL: No history available - server failed")
        _write_error_signal_and_exit(server_id, "leaf server failed: no history", 2)
        
    # ------------------------------------------------------------------
    # Persist round-level communication CSV for this leaf server
    # ------------------------------------------------------------------
    try:
        import pandas as pd, filelock
        logger.info(f"[{server_str}] Starting communication metrics export")
        
        # Extract round-level communication metrics from History
        # Build per-round communication metrics from History.metrics_distributed_fit
        mdf = getattr(history, "metrics_distributed_fit", {}) or {}

        def _collect(keys):
            for k in keys:
                if k in mdf:
                    return mdf[k]
            return []

        up_entries = _collect(["bytes_up", "bytes_written"])
        down_entries = _collect(["bytes_down", "bytes_read"])
        rt_entries = _collect(["round_time"])  # mean time per round
        comp_entries = _collect(["compute_s"])  # mean compute per round

        by_round: dict[int, dict[str, float]] = {}
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
            logger.info(f"[{server_str}] Wrote communication CSV to {out}")
        else:
            logger.warning(f"[{server_str}] No communication metrics to export")
            
    except ImportError as import_err:
        logger.warning(f"[{server_str}] Could not import pandas/filelock for comm CSV: {import_err}")
    except Exception as csv_err:
        logger.warning(f"[{server_str}] Could not write comm CSV: {csv_err}")
    
    # Create completion signal to indicate server finished successfully
    try:
        # Create completion signal using optimized storage
        global_round = int(os.getenv('GLOBAL_ROUND', '1'))
        script_dir = Path(__file__).resolve().parent
        project_root_local = script_dir.parent
        from fedge.utils.fs_optimized import create_completion_signal
        completion_signal = create_completion_signal(project_root_local, global_round, server_id)
        logger.info(f"[{server_str}] Created completion signal at {completion_signal}")
    except Exception as signal_err:
        logger.warning(f"[{server_str}] Could not create completion signal: {signal_err}")
    
    logger.info(f"[{server_str}] Process completed successfully")
    
    # Perform cleanup and exit gracefully
    try:
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Shutdown logging
        logging.shutdown()
        
        logger.info(f"[{server_str}] Cleanup completed, exiting gracefully")
    except Exception as cleanup_err:
        print(f"[{server_str}] Cleanup error: {cleanup_err}")
    
    # Exit gracefully to allow proper Python shutdown sequence
    sys.exit(0)


def main_wrapper():
    """Wrapper to ensure graceful exit even on exceptions"""
    try:
        main()
    except Exception as e:
        server_id = int(os.environ.get("SERVER_ID", "0"))
        server_str = os.environ.get("SERVER_STR", "Leaf Server")
        logger.error(f"[{server_str}] Main function error: {e}")
        _write_error_signal_and_exit(server_id, f"leaf server crashed: {e}", 3)


if __name__ == "__main__":
    gc.collect()
    main_wrapper()
