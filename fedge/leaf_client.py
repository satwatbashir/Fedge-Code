#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import signal
import warnings
import pickle
import base64
from fedge.utils.bytes_helper import raw_bytes
import logging
from pathlib import Path

import torch
import grpc
from flwr.client import NumPyClient, start_client

# Import your task utilities:
from fedge.task import Net, load_data, set_weights, train, test, get_weights
try:
    from fedge.scaffold_utils import create_scaffold_manager
except ImportError:
    create_scaffold_manager = None

# ─── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
for name in ("flwr", "ece", "grpc"):
    logging.getLogger(name).setLevel(logging.ERROR)

# =============================================================================
# Flower NumPyClient implementation
# =============================================================================
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        
        # ✅ SCAFFOLD will be initialized when config is received in fit()
        # Don't initialize here since we need config from server
        self.net._scaffold_manager = None
        self._scaffold_initialized = False  # Track initialization state

    def get_properties(self, config):
        from flwr.common import Properties
        cid = os.environ.get("CLIENT_ID", "")
        return Properties(other={"client_id": cid})

    def get_parameters(self, config):
        # Return the model parameters as numpy arrays
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    
    def _initialize_scaffold(self, config):
        """Initialize SCAFFOLD if enabled in config"""
        scaffold_enabled = config.get("scaffold_enabled", False)
        client_id = os.getenv('CLIENT_ID', 'unknown')
        
        # Sync SCAFFOLD flag to environment so task.train can apply control variates
        os.environ["SCAFFOLD_ENABLED"] = str(scaffold_enabled).lower()
        
        if scaffold_enabled and not self._scaffold_initialized:
            if create_scaffold_manager is not None:
                self.net._scaffold_manager = create_scaffold_manager(self.net)
                self._scaffold_initialized = True
                # SCAFFOLD initialization logging removed for brevity
            else:
                logger.warning(f"[Client {client_id}] SCAFFOLD enabled but module not available")
        elif not scaffold_enabled and self._scaffold_initialized:
            # Disable SCAFFOLD if config says so
            self.net._scaffold_manager = None
            self._scaffold_initialized = False
            logger.info(f"[Client {client_id}] SCAFFOLD disabled")
        
        return scaffold_enabled
    
    def _apply_server_control_variates(self, config, scaffold_enabled):
        """Apply server control variates if received from cloud"""
        if not (scaffold_enabled and hasattr(self.net, '_scaffold_manager') and self.net._scaffold_manager):
            return
            
        if "scaffold_server_control" not in config:
            return
            
        client_id = os.getenv('CLIENT_ID', 'unknown')
        try:
            # Deserialize server control variates
            serialized_control = config["scaffold_server_control"]
            server_control = pickle.loads(base64.b64decode(serialized_control.encode('utf-8')))
            
            # Update client's server control variates
            self.net._scaffold_manager.server_control = server_control
            logger.debug(f"[Client {client_id}] ✅ SCAFFOLD server control variates updated from cloud")
        except Exception as e:
            logger.warning(f"[Client {client_id}] Failed to apply server control variates: {e}")
    
    def _extract_training_config(self, config):
        """Extract and validate training hyperparameters from config"""
        client_id = os.getenv('CLIENT_ID', 'unknown')
        
        # ✅ STRICT: Extract ALL hyperparameters from config - FAIL FAST if missing
        required_params = ["learning_rate", "weight_decay", "momentum", "clip_norm", "lr_gamma", "proximal_mu"]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Required parameter '{param}' missing from server config. Server misconfiguration detected.")
        
        training_config = {
            'learning_rate': config["learning_rate"],
            'weight_decay': config["weight_decay"],
            'momentum': config["momentum"],
            'clip_norm': config["clip_norm"],
            'lr_gamma': config["lr_gamma"],
            'proximal_mu': config["proximal_mu"]
        }
        
        return training_config
    
    def _update_scaffold_after_training(self, scaffold_enabled, global_weights, learning_rate):
        """Update SCAFFOLD control variates after training"""
        if not (scaffold_enabled and hasattr(self.net, '_scaffold_manager') and self.net._scaffold_manager):
            return None
            
        client_id = os.getenv('CLIENT_ID', 'unknown')
        try:
            # Create temporary models for SCAFFOLD update
            from fedge.task import Net
            global_model = Net()
            set_weights(global_model, global_weights)
            
            # Update client control variates
            self.net._scaffold_manager.update_client_control(
                local_model=self.net,
                global_model=global_model,
                learning_rate=learning_rate,
                local_epochs=self.local_epochs
            )
            
            # Get updated control variates to send to server
            scaffold_delta = self.net._scaffold_manager.get_client_control()
            # SCAFFOLD update logging removed for brevity
            return scaffold_delta
        except Exception as e:
            logger.warning(f"[Client {client_id}] SCAFFOLD update failed: {e}")
            return None
    
    def _prepare_metrics(self, train_loss, bytes_down, bytes_up, round_time, scaffold_delta, accuracy=None, eval_loss=None):
        """Prepare metrics dictionary for server"""
        client_id = os.environ.get("CLIENT_ID", "")
        metrics = {
            "train_loss": train_loss,
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
            "round_time": round_time,
            "compute_s": round_time,  # same proxy value
            "client_id": client_id,
        }
        
        # ✅ NEW: Include accuracy in fit metrics for server aggregation
        if accuracy is not None:
            metrics["accuracy"] = float(accuracy)
        if eval_loss is not None:
            metrics["eval_loss"] = float(eval_loss)
        
        # ✅ SCAFFOLD: Include control variate delta in metrics for server aggregation
        if scaffold_delta is not None:
            # Serialize control variates for transmission
            try:
                serialized_delta = base64.b64encode(pickle.dumps(scaffold_delta)).decode('utf-8')
                metrics["scaffold_delta"] = serialized_delta
                logger.debug(f"[Client {client_id}] SCAFFOLD delta included in metrics")
            except Exception as e:
                logger.warning(f"[Client {client_id}] Failed to serialize SCAFFOLD delta: {e}")
        
        return metrics

    def fit(self, parameters, config):
        import time
        t0 = time.time()
        
        # Compute download size of incoming parameters
        bytes_down = raw_bytes(parameters)
        ref_weights = [w.copy() for w in parameters]
        set_weights(self.net, parameters)
        
        # Initialize SCAFFOLD and extract configuration
        scaffold_enabled = self._initialize_scaffold(config)
        self._apply_server_control_variates(config, scaffold_enabled)
        training_config = self._extract_training_config(config)
        
        # Store global model weights for SCAFFOLD update
        global_weights = [w.copy() for w in parameters] if scaffold_enabled else None
        
        # SCAFFOLD status logging removed for brevity
        
        # Train the model with extracted configuration
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
            lr=training_config['learning_rate'],
            momentum=training_config['momentum'],
            weight_decay=training_config['weight_decay'],
            gamma=training_config['lr_gamma'],
            clip_norm=training_config['clip_norm'],
            prox_mu=training_config['proximal_mu'],
            ref_weights=ref_weights,
            global_round=config.get("global_round", 0),
            scaffold_enabled=scaffold_enabled,
        )
        
        # Update SCAFFOLD control variates after training
        scaffold_delta = self._update_scaffold_after_training(
            scaffold_enabled, global_weights, training_config['learning_rate']
        )
        
        # ✅ NEW: Evaluate accuracy after training to include in fit metrics
        eval_loss, eval_acc = test(self.net, self.valloader, self.device)
        
        # Prepare and return results
        round_time = time.time() - t0
        bytes_up = raw_bytes(get_weights(self.net))
        metrics = self._prepare_metrics(train_loss, bytes_down, bytes_up, round_time, scaffold_delta, eval_acc, eval_loss)
        
        return get_weights(self.net), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        # Compute download size of evaluation parameters
        bytes_down = raw_bytes(parameters)
        # Update local model, evaluate on validation set
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        cid = os.environ.get("CLIENT_ID", "")
        metrics = {
            "accuracy": accuracy,
            "bytes_down_eval": bytes_down,
            "client_id": cid,
        }
        return loss, len(self.valloader.dataset), metrics

# =============================================================================
# Signal handler for graceful shutdown
# =============================================================================
def handle_signal(sig, frame):
    client_id = os.environ.get("CLIENT_ID", "leaf_client")
    logger.info(f"[{client_id}] Received signal {sig}, shutting down gracefully...")
    sys.exit(0)

# =============================================================================
# Main: parse arguments, load OrgansMNIST shard, start Flower client
# =============================================================================
def main():
    # 1) Catch SIGINT/SIGTERM so we can clean up nicely
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser()
    parser.add_argument("--server_id", type=int, required=True)
    parser.add_argument("--client_id", type=int, required=True)
    # Dataset flag for CIFAR-10
    parser.add_argument(
        "--dataset_flag",
        type=str,
        required=True,
        choices=["cifar10"],
        help="Dataset to use (CIFAR-10 only)",
    )
    parser.add_argument("--local_epochs", type=int, required=True)
    parser.add_argument("--server_addr", type=str, default=os.getenv("LEAF_ADDRESS", "127.0.0.1:6100"))
    parser.add_argument(
        "--max_retries", type=int, default=10, help="Max gRPC connection retries"
    )
    parser.add_argument(
        "--retry_delay", type=int, default=5, help="Seconds between retry attempts"
    )
    args = parser.parse_args()

    # 2) Build a human-readable CLIENT_ID for logging (e.g. "leaf_0_client_3")
    client_id = f"leaf_{args.server_id}_client_{args.client_id}"
    os.environ["CLIENT_ID"] = client_id

    # 3) Load hierarchical partition indices from PARTITIONS_JSON
    indices = None
    parts_path = os.environ.get("PARTITIONS_JSON")
    if parts_path and Path(parts_path).exists():
        with open(parts_path, "r", encoding="utf-8") as fp:
            mapping = json.load(fp)
        # Use server_id and client_id to get the correct partition indices
        indices = mapping[str(args.server_id)][str(args.client_id)]
        logger.debug(f"[Client {args.server_id}_{args.client_id}] Loaded {len(indices)} samples from hierarchical partition")
    else:
        raise RuntimeError(f"PARTITIONS_JSON not found at {parts_path}. Hierarchical partitioning required.")
    
    trainloader, valloader, n_classes = load_data(
        args.dataset_flag,
        0,  # partition_id not used when indices provided
        1,  # num_partitions not used when indices provided  
        indices=indices,
        server_id=args.server_id,
    )

    # 4) Instantiate your Net with appropriate channels & image size
    sample, _ = next(iter(trainloader))
    _, in_ch, H, W = sample.shape
    net = Net()

    # 5) Wrap in the FlowerClient
    client = FlowerClient(net, trainloader, valloader, args.local_epochs)

    # 6) Connect (with retry logic) to the leaf server’s Flower endpoint
    retries = 0
    while retries < args.max_retries:
        try:
            logger.debug(f"[{client_id}] Connecting to server at {args.server_addr}")
            
            # Suppress Flower deprecation warnings during client startup
            import contextlib
            import io
            
            stderr_capture = io.StringIO()
            with contextlib.redirect_stderr(stderr_capture):
                start_client(
                    server_address=args.server_addr,
                    client=client.to_client(),
                )
            logger.debug(f"[{client_id}] Client session completed successfully")
            break
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE and retries < args.max_retries:
                retries += 1
                logger.warning(
                    f"[{client_id}] Connection failed (attempt {retries}/{args.max_retries}): {e.details()}"
                )
                logger.info(f"[{client_id}] Retrying in {args.retry_delay}s …")
                time.sleep(args.retry_delay)
            else:
                logger.error(f"[{client_id}] Unexpected gRPC error: {e.details()}")
                return
        except Exception as e:
            logger.error(f"[{client_id}] Unexpected error: {e}")
            return

if __name__ == "__main__":
    main()
