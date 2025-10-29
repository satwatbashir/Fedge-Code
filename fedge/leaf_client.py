#!/usr/bin/env python3
# FIXED VERSION - Removed duplicate evaluation in fit()
import argparse
import json
import os
import sys
import time
import signal
import warnings
import pickle
import base64
import logging
from pathlib import Path

import torch
import grpc
from flwr.client import NumPyClient, start_client

# Safe import for raw_bytes
try:
    from fedge.utils.bytes_helper import raw_bytes
except Exception:
    import numpy as _np
    def raw_bytes(params):
        try:
            return int(sum(_np.asarray(x).nbytes for x in params))
        except Exception:
            try:
                return int(sum(v.numel() * v.element_size() for v in params.values()))
            except Exception:
                return 0

from fedge.task import Net, load_data, set_weights, train, test, get_weights
try:
    from fedge.scaffold_utils import create_scaffold_manager
except ImportError:
    create_scaffold_manager = None

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")
for name in ("flwr", "ece", "grpc"):
    logging.getLogger(name).setLevel(logging.ERROR)

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.net._scaffold_manager = None
        self._scaffold_initialized = False

    def get_properties(self, config):
        from flwr.common import Properties
        cid = os.environ.get("CLIENT_ID", "")
        return Properties(other={"client_id": cid})

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    
    def _initialize_scaffold(self, config):
        """Initialize SCAFFOLD if enabled in config"""
        scaffold_enabled = config.get("scaffold_enabled", False)
        client_id = os.getenv('CLIENT_ID', 'unknown')
        os.environ["SCAFFOLD_ENABLED"] = str(scaffold_enabled).lower()
        
        if scaffold_enabled and not self._scaffold_initialized:
            if create_scaffold_manager is not None:
                self.net._scaffold_manager = create_scaffold_manager(self.net)
                self._scaffold_initialized = True
            else:
                logger.warning(f"[Client {client_id}] SCAFFOLD enabled but module not available")
        elif not scaffold_enabled and self._scaffold_initialized:
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
            serialized_control = config["scaffold_server_control"]
            server_control = pickle.loads(base64.b64decode(serialized_control.encode('utf-8')))
            self.net._scaffold_manager.server_control = server_control
            logger.debug(f"[Client {client_id}] ✅ SCAFFOLD server control variates updated from cloud")
        except Exception as e:
            logger.warning(f"[Client {client_id}] Failed to apply server control variates: {e}")
    
    def _extract_training_config(self, config):
        """Extract and validate training hyperparameters from config"""
        required_params = ["learning_rate", "weight_decay", "momentum", "clip_norm", "lr_gamma", "proximal_mu"]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Required parameter '{param}' missing from server config.")
        
        return {
            'learning_rate': config["learning_rate"],
            'weight_decay': config["weight_decay"],
            'momentum': config["momentum"],
            'clip_norm': config["clip_norm"],
            'lr_gamma': config["lr_gamma"],
            'proximal_mu': config["proximal_mu"]
        }
    
    def _update_scaffold_after_training(self, scaffold_enabled, global_weights, learning_rate):
        """Update SCAFFOLD control variates after training"""
        if not (scaffold_enabled and hasattr(self.net, '_scaffold_manager') and self.net._scaffold_manager):
            return None
            
        client_id = os.getenv('CLIENT_ID', 'unknown')
        try:
            from fedge.task import Net
            global_model = Net()
            set_weights(global_model, global_weights)
            
            self.net._scaffold_manager.update_client_control(
                local_model=self.net,
                global_model=global_model,
                learning_rate=learning_rate,
                local_epochs=self.local_epochs
            )
            
            scaffold_delta = self.net._scaffold_manager.get_client_control()
            return scaffold_delta
        except Exception as e:
            logger.warning(f"[Client {client_id}] SCAFFOLD update failed: {e}")
            return None
    
    def _prepare_metrics(self, train_loss, bytes_down, bytes_up, round_time, scaffold_delta):
        """✅ FIXED: Removed accuracy/eval_loss - evaluation only in evaluate()"""
        client_id = os.environ.get("CLIENT_ID", "")
        metrics = {
            "train_loss": train_loss,
            "bytes_up": bytes_up,
            "bytes_down": bytes_down,
            "round_time": round_time,
            "compute_s": round_time,
            "client_id": client_id,
        }
        
        if scaffold_delta is not None:
            try:
                serialized_delta = base64.b64encode(pickle.dumps(scaffold_delta)).decode('utf-8')
                metrics["scaffold_delta"] = serialized_delta
                logger.debug(f"[Client {client_id}] SCAFFOLD delta included in metrics")
            except Exception as e:
                logger.warning(f"[Client {client_id}] Failed to serialize SCAFFOLD delta: {e}")
        
        return metrics

    def fit(self, parameters, config):
        """✅ FIXED: Removed duplicate evaluation - only train and return"""
        import time
        t0 = time.time()
        
        bytes_down = raw_bytes(parameters)
        ref_weights = [w.copy() for w in parameters]
        set_weights(self.net, parameters)
        
        scaffold_enabled = self._initialize_scaffold(config)
        self._apply_server_control_variates(config, scaffold_enabled)
        training_config = self._extract_training_config(config)
        global_weights = [w.copy() for w in parameters] if scaffold_enabled else None
        
        # Train the model
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
        
        # Update SCAFFOLD control variates
        scaffold_delta = self._update_scaffold_after_training(
            scaffold_enabled, global_weights, training_config['learning_rate']
        )
        
        # ✅ FIXED: NO evaluation here - evaluation done separately in evaluate()
        # The server will call evaluate() if needed for aggregation
        
        round_time = time.time() - t0
        bytes_up = raw_bytes(get_weights(self.net))
        metrics = self._prepare_metrics(train_loss, bytes_down, bytes_up, round_time, scaffold_delta)
        
        return get_weights(self.net), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        """Evaluate on validation set (called by server if needed)"""
        bytes_down = raw_bytes(parameters)
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        cid = os.environ.get("CLIENT_ID", "")
        metrics = {
            "accuracy": accuracy,
            "bytes_down_eval": bytes_down,
            "client_id": cid,
        }
        return loss, len(self.valloader.dataset), metrics

def handle_signal(sig, frame):
    client_id = os.environ.get("CLIENT_ID", "leaf_client")
    logger.info(f"[{client_id}] Received signal {sig}, shutting down gracefully...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser()
    parser.add_argument("--server_id", type=int, required=True)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--dataset_flag", type=str, required=True, choices=["cifar10"])
    parser.add_argument("--local_epochs", type=int, required=True)
    parser.add_argument("--server_addr", type=str, default=os.getenv("LEAF_ADDRESS", "127.0.0.1:6100"))
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument("--retry_delay", type=int, default=5)
    args = parser.parse_args()

    client_id = f"leaf_{args.server_id}_client_{args.client_id}"
    os.environ["CLIENT_ID"] = client_id

    indices = None
    parts_path = os.environ.get("PARTITIONS_JSON")
    if parts_path and Path(parts_path).exists():
        with open(parts_path, "r", encoding="utf-8") as fp:
            mapping = json.load(fp)
        indices = mapping[str(args.server_id)][str(args.client_id)]
        logger.debug(f"[Client {args.server_id}_{args.client_id}] Loaded {len(indices)} samples")
    else:
        raise RuntimeError(f"PARTITIONS_JSON not found at {parts_path}")
    
    trainloader, valloader, n_classes = load_data(
        args.dataset_flag, 0, 1, indices=indices, server_id=args.server_id
    )

    sample, _ = next(iter(trainloader))
    net = Net()
    client = FlowerClient(net, trainloader, valloader, args.local_epochs)

    retries = 0
    while retries < args.max_retries:
        try:
            logger.debug(f"[{client_id}] Connecting to server at {args.server_addr}")
            import contextlib, io
            stderr_capture = io.StringIO()
            with contextlib.redirect_stderr(stderr_capture):
                start_client(server_address=args.server_addr, client=client.to_client())
            logger.debug(f"[{client_id}] Client session completed successfully")
            break
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE and retries < args.max_retries:
                retries += 1
                logger.warning(f"[{client_id}] Connection failed (attempt {retries}/{args.max_retries})")
                time.sleep(args.retry_delay)
            else:
                logger.error(f"[{client_id}] Unexpected gRPC error: {e.details()}")
                return
        except Exception as e:
            logger.error(f"[{client_id}] Unexpected error: {e}")
            return

if __name__ == "__main__":
    main()
