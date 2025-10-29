# fedge/proxy_client.py - FIXED VERSION
# ✅ Removed incorrect proxy evaluation - proxy only uploads server models
import argparse
import atexit
import csv
import os
import pickle
import signal
import sys
import time
import socket
import hashlib
import traceback
import logging
from pathlib import Path
from typing import Tuple, Dict

import grpc
import torch
from flwr.client import NumPyClient, start_client

from fedge.task import Net, set_weights, test, get_cifar10_test_loader
from fedge.utils.bytes_helper import raw_bytes
from fedge.utils.fs_optimized import get_model_path



logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def diag_addrinfo(host: str, port: int, logger):
    try:
        infos = socket.getaddrinfo(host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
        logger.info(f"[diag] getaddrinfo({host},{port}) -> {len(infos)} results")
        for i, (fam, socktype, proto, canon, sockaddr) in enumerate(infos):
            fam_s = "AF_INET6" if fam == socket.AF_INET6 else ("AF_INET" if fam == socket.AF_INET else str(fam))
            logger.info(f"[diag]  {i+1}. family={fam_s}, addr={sockaddr}")
    except Exception as e:
        logger.error(f"[diag] getaddrinfo failed: {e}")


def diag_tcp(host: str, port: int, timeout_sec: float, logger) -> bool:
    t0 = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout_sec) as s:
            fam_s = "AF_INET6" if s.family == socket.AF_INET6 else ("AF_INET" if s.family == socket.AF_INET else str(s.family))
            peer = s.getpeername()
            logger.info(f"[diag] TCP connect OK -> {peer} family={fam_s} in {time.time()-t0:.3f}s")
            return True
    except Exception as e:
        logger.warning(f"[diag] TCP connect FAIL ({time.time()-t0:.3f}s): {repr(e)}")
        return False




class ProxyClient(NumPyClient):
    """✅ FIXED: Only uploads server model - no evaluation needed"""

    def __init__(self, server_id: int):
        self.server_id = int(server_id)
        self.proxy_id = os.environ.get("PROXY_ID", f"proxy_{self.server_id}")
        self._sent_complete = False

        self.project_root = Path(__file__).resolve().parent.parent
        self.global_round = int(os.environ.get("GLOBAL_ROUND", "1"))
        self.model_path = get_model_path(self.project_root, self.server_id, self.global_round)

        # Wait for model file
        if not self.model_path.exists():
            max_wait = int(os.environ.get("PROXY_WAIT_MODEL_SEC", "300"))
            logger.info(f"[{self.proxy_id}] Waiting for leaf model {self.model_path} (≤ {max_wait}s)")
            t0 = time.time()
            while time.time() - t0 < max_wait and not self.model_path.exists():
                time.sleep(2)
        
        if not self.model_path.exists():
            raise RuntimeError(f"[{self.proxy_id}] Model not found: {self.model_path}")

        # Load model
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
        if not (isinstance(data, tuple) and len(data) == 2):
            raise ValueError(f"[{self.proxy_id}] Unexpected model format")
        
        self.ndarrays, self.total_examples = data
        logger.info(f"[{self.proxy_id}] Loaded model from {self.model_path} "
                    f"(examples={self.total_examples}, bytes={raw_bytes(self.ndarrays)})")
        try:
            _size = os.path.getsize(self.model_path)
            _hash = _md5(self.model_path)
            logger.info(f"[{self.proxy_id}] Model md5={_hash} size={_size}")
        except Exception as _e:
            logger.warning(f"[{self.proxy_id}] md5/size check failed: {_e}")


        # Setup for potential evaluation (not used with fixed evaluate())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.device)
        set_weights(self.net, self.ndarrays)

        # Signals
        self.leaf_dir = self.project_root / "rounds" / f"round_{self.global_round}" / "leaf" / f"server_{self.server_id}"
        self.leaf_dir.mkdir(parents=True, exist_ok=True)
        self.proxy_signals_csv = self.leaf_dir / "proxy_signals.csv"
        self._append_signal_row("started")
        # --- NEW: per-round heartbeat file so cloud can gate dispatch ---
        hb_dir = self.project_root / "signals" / "proxies"
        hb_dir.mkdir(parents=True, exist_ok=True)
        self.hb_path = hb_dir / f"s{self.server_id}_g{self.global_round}.hb"
        try:
            with open(self.hb_path, "w") as _hb:
                _hb.write(str(time.time()))
        except Exception as _e:
            logger.warning(f"[{self.proxy_id}] Failed to write heartbeat {self.hb_path}: {_e}")


        atexit.register(self._write_completion_signal)

    
    def get_properties(self, config):
        """Expose identity info so the cloud can match this proxy to its node_id."""
        # Return a plain dict; the server reads keys directly.
        return {
            "role": "proxy",
            "node_id": f"s{self.server_id}_g{self.global_round}",
            "server_id": int(self.server_id),
            "global_round": int(self.global_round),
        }


    def get_parameters(self, config):
        """Return server's trained model"""
        return self.ndarrays

    def fit(self, parameters, config):
        """Upload server's trained model to cloud and exit cleanly after RPC completes.


        Returns the FitRes so the cloud receives the model update.
        The proxy process will exit naturally after start_client() returns.
        Each proxy is launched per-round with a unique node_id, so no reuse occurs.
        
        IMPORTANT: We do NOT force exit here. Let Flower/gRPC complete the response
        transmission and close the session cleanly. The process will exit naturally
        when start_client() returns.
        """
        logger.info(f"[{self.proxy_id}] fit() received from cloud; starting upload")
        up = raw_bytes(self.ndarrays)
        down = raw_bytes(parameters)
        logger.info(
            f"[{self.proxy_id}] Uploading model: bytes_up={up}, bytes_down={down}, "
            f"examples={self.total_examples}"
        )

        # Return result to cloud for aggregation
        result = (
            self.ndarrays,
            int(self.total_examples),
            {
                "sid": self.server_id,
                "server_id": self.server_id,
                "dataset_flag": "cifar10",
                "bytes_up": up,
                "bytes_down": down,
                "round_time": 0.0,
                "compute_s": 0.0,
            },
        )

        logger.debug(f"[{self.proxy_id}] Returning fit result to cloud (no forced exit)")

        # Write completion signal BEFORE exiting so orchestrator sees progress
        try:
            self._write_completion_signal()
        except Exception:
            pass

        # Schedule a clean hard-exit after the gRPC response is flushed
        # (Timer avoids killing the process before the RPC returns.)
        import threading, os
        def _late_exit():
            try:
                logger.info(f"[{self.proxy_id}] Upload done; exiting proxy process")
                os._exit(0)
            except Exception:
                os._exit(0)
        threading.Timer(0.7, _late_exit).start()

        return result



    def evaluate(self, parameters, config):
        """
        ✅ FIXED: No evaluation needed - proxy only uploads.
        
        Previously, this was evaluating the cloud's global model on server shards,
        which is incorrect. The server already evaluated its own model on its shard
        before saving. The cloud doesn't need per-shard evaluations of the global model.
        
        Returns minimal valid response to satisfy Flower's interface.
        """
        logger.debug(f"[{self.proxy_id}] evaluate() called but skipped (no eval needed)")
        return 0.0, 0, {"accuracy": 0.0, "note": "proxy_no_eval"}

    def _append_signal_row(self, signal_type: str):
        write_header = not self.proxy_signals_csv.exists()
        with open(self.proxy_signals_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["global_round", "proxy_id", "server_id", "signal_type", "timestamp"])
            if write_header:
                w.writeheader()
            w.writerow({
                "global_round": self.global_round,
                "proxy_id": self.proxy_id,
                "server_id": self.server_id,
                "signal_type": signal_type,
                "timestamp": time.time(),
            })

    def _write_completion_signal(self):
        if self._sent_complete:
            return
        self._append_signal_row("complete")
        self._sent_complete = True

def handle_signal(sig, frame):
    proxy_id = os.environ.get("PROXY_ID", "proxy")
    server_id = int(os.environ.get("SERVER_ID", "0"))
    global_round = int(os.environ.get("GLOBAL_ROUND", "0"))
    project_root = Path(__file__).resolve().parent.parent
    leaf_dir = project_root / "rounds" / f"round_{global_round}" / "leaf" / f"server_{server_id}"
    leaf_dir.mkdir(parents=True, exist_ok=True)
    proxy_signals_csv = leaf_dir / "proxy_signals.csv"

    write_header = not proxy_signals_csv.exists()
    with open(proxy_signals_csv, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["global_round", "proxy_id", "server_id", "signal_type", "timestamp"])
        if write_header:
            w.writeheader()
        w.writerow({
            "global_round": global_round,
            "proxy_id": proxy_id,
            "server_id": server_id,
            "signal_type": "complete",
            "timestamp": time.time(),
        })
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser()
    parser.add_argument("--server_id", type=int, required=True)
    parser.add_argument("--cloud_address", default=os.getenv("CLOUD_ADDRESS", "127.0.0.1:6000"))
    parser.add_argument("--max_retries", type=int, default=120)
    parser.add_argument("--retry_delay", type=int, default=2)
    parser.add_argument("--global_round", type=int, default=0)
    parser.add_argument("--dir_round", type=int)
    args = parser.parse_args()
    
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s"
    )
    log = logging.getLogger("proxy")

    server_id = int(args.server_id)
    cloud_address = args.cloud_address
    global_round = int(args.global_round)
    node_id = f"s{server_id}_g{global_round}"

    log.info(f"[proxy] argv={sys.argv}")
    log.info(f"[proxy] start: server_id={server_id} round={global_round} cloud={cloud_address}")
    log.info(f"[proxy] node_id={node_id}")

    # Sanity: parse host/port and raw TCP probe
    try:
        host, port_s = cloud_address.rsplit(":", 1)
        port = int(port_s)
    except Exception as e:
        log.error(f"[proxy] bad cloud address '{cloud_address}': {e}")
        sys.exit(2)

    try:
        with socket.create_connection((host, port), timeout=2.0) as s:
            fam = {socket.AF_INET: "AF_INET", socket.AF_INET6: "AF_INET6"}.get(s.family, str(s.family))
            log.info(f"[proxy] TCP reachability OK -> {s.getpeername()} family={fam}")
    except Exception as e:
        log.warning(f"[proxy] TCP reachability FAIL to {host}:{port}: {e}")


    os.environ["SERVER_ID"] = str(args.server_id)
    if args.global_round is not None:
        os.environ["GLOBAL_ROUND"] = str(args.global_round)
    if "PROXY_ID" not in os.environ:
        os.environ["PROXY_ID"] = f"proxy_{args.server_id}"

    client_node_id = f"s{args.server_id}_g{args.global_round}"  # unique per round

    os.environ["FLWR_CLIENT_NODE_ID"] = client_node_id

    logger.info(f"[{os.environ['PROXY_ID']}] Proxy start: server_id={args.server_id}, "
                f"cloud={args.cloud_address}, node_id={client_node_id}")

    client = ProxyClient(args.server_id)

    # Wait for cloud start signal
    project_root = Path(__file__).resolve().parent.parent

    start_signal = project_root / "signals" / "cloud_started.signal"
    
    # === Diagnostics: cloud target parsing + reachability ===
    cloud_target = args.cloud_address
    if ":" in cloud_target:
        _host, _port_s = cloud_target.rsplit(":", 1)
        try:
            _port = int(_port_s)
        except:
            _port = 6000
    else:
        _host, _port = cloud_target, 6000

    logger.info(f"[{os.environ['PROXY_ID']}] Will dial cloud at {cloud_target} (node_id={client_node_id})")
    diag_addrinfo(_host, _port, logger)
    diag_tcp(_host, _port, timeout_sec=2.0, logger=logger)
    
    # Extra: show established sockets to this port (namespace view)
    try:
        ss = os.popen(f"ss -tn dst :{_port} state established || true").read()
        logger.info(f"[{os.environ['PROXY_ID']}] ss-established(to port {_port})=\n{ss}")
    except Exception as _e:
        logger.debug(f"[proxy] ss check failed: {_e}")


    waited = 0
    max_wait = int(os.environ.get("PROXY_WAIT_CLOUD_START_SEC", "60"))

    while not start_signal.exists() and waited < max_wait:
        logger.info("[proxy] Waiting for cloud start signal...")
        time.sleep(1)
        waited += 1

    # Connect with retry
    success = False
    for attempt in range(1, args.max_retries + 1):
        try:
            logger.info(f"[{os.environ['PROXY_ID']}] Connecting to cloud at {args.cloud_address} (attempt {attempt}/{args.max_retries})")
            # ✅ CRITICAL FIX: Call .to_client() like leaf_client does
            start_client(server_address=args.cloud_address, client=client.to_client())
            logger.info(f"[{os.environ['PROXY_ID']}] Proxy finished successfully")
            success = True
            break         
            
        except grpc.RpcError as e:
            # Log with maximum detail and keep trying unless we've exhausted attempts
            code = getattr(e, "code", lambda: None)()
            details = None
            try:
                details = e.details()
            except Exception:
                pass
            logger.error(f"[proxy] gRPC RpcError: code={code} details={details} repr={repr(e)}")
            if attempt < args.max_retries:
                time.sleep(args.retry_delay)
                continue
            client._write_completion_signal()
            sys.exit(1)

        except Exception as e:
            logger.error(f"[proxy] Unexpected error: {e}\n{traceback.format_exc()}")
            client._write_completion_signal()
            sys.exit(1)

    if success:
        client._write_completion_signal()
        sys.exit(0)

    client._write_completion_signal()
    sys.exit(1)
