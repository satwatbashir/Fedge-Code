################################################################################
# orchestrator.py   ―  Chunk 1 / 3
# ---------------------------------------------------------------------------
# Imports, configuration loading, constants, directory setup, generic helpers
################################################################################
from __future__ import annotations

import os
import sys
import time
import logging
import io
import datetime as _dt
import subprocess as _sp
import socket
import psutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import toml
import pickle
import numpy as np
import json
import shutil
from fedge.utils.fs_optimized import (
    get_server_signal_path,
    get_round_signals_dir,
    check_all_servers_completed,
    get_model_path,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────────────
# Basic logging setup (replaces removed `logging_config` dependency)
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),  # Keep INFO for orchestrator
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# Set specific loggers to WARNING to reduce noise
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("fedge.task").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger("orchestrator")

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers replacing removed logging_config utilities
# ──────────────────────────────────────────────────────────────────────────────

def create_run_summary(run_dir: Path | None = None) -> Path:
    """Create a minimal run summary file and return its path."""
    run_dir = run_dir or Path(os.environ.get("RUN_DIR", "."))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_file = run_dir / "run_summary.txt"
    try:
        with open(summary_file, "w", encoding="utf-8") as fsum:
            fsum.write(f"Run started: {time.asctime()}\n")
            fsum.write(f"Command: {' '.join(sys.argv)}\n")
    except Exception as exc:
        logger.warning("Could not write run summary: %s", exc)
    return summary_file


def log_system_info(log: logging.Logger | None = None) -> None:
    """Log basic system information (lightweight replacement)."""
    log = log or logger
    import platform, psutil

    log.info("=== SYSTEM INFORMATION ===")
    log.info("Platform: %s", platform.platform())
    log.info("Python: %s", sys.version.replace('\n', ' '))
    log.info("CPU Count: %s", os.cpu_count())
    try:
        vm = psutil.virtual_memory()
        log.info("Memory: %.1f GB", vm.total / 1024**3)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
#  Project & run directories
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RUN_ID = os.getenv("FL_RUN_ID") or _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
RUN_DIR = PROJECT_ROOT / "runs" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
(RUN_DIR / "logs").mkdir(exist_ok=True)

# Expose for child processes (Flower picks these up)
os.environ.setdefault("RUN_DIR", str(RUN_DIR))
os.environ.setdefault("FLWR_LOGGING_FORMAT", "json")
os.environ.setdefault("FLWR_LOGGING", "json")

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration helpers
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def _load_config() -> Dict:
    """Load and cache pyproject-based FL hierarchy config."""
    cfg_path = PROJECT_ROOT / "pyproject.toml"
    if not cfg_path.exists():
        logger.error(f"Cannot find {cfg_path}")
        sys.exit(1)
    return toml.load(cfg_path)


def _create_partitions_if_needed() -> None:
    """Create hierarchical partitions using the working partitioning.py system."""
    rounds_dir = PROJECT_ROOT / "rounds"
    parts_json = rounds_dir / "partitions.json"
    
    if parts_json.exists():
        os.environ["PARTITIONS_JSON"] = str(parts_json)   # ← add this
        logger.info(f"✅ Using existing partitions: {parts_json}")
        # Log partition summary
        try:
            import json
            with open(parts_json, "r") as f:
                mapping = json.load(f)
            total_clients = sum(len(server_clients) for server_clients in mapping.values())
            logger.info(f"📊 Partition summary: {len(mapping)} servers, {total_clients} total clients")
            for server_id, clients in mapping.items():
                logger.info(f"   Server {server_id}: {len(clients)} clients")
        except Exception as e:
            logger.warning(f"Could not read partition summary: {e}")
        return
        
    logger.info(f"🔄 Creating NEW hierarchical Dirichlet partitions for dataset: {DATASET_FLAG}")
    logger.info(f"📁 Will save to: {parts_json}")
    logger.info(f"🏗️  Configuration: {NUM_SERVERS} servers, clients per server: {CLIENTS_PER_SERVER_LIST}")
    rounds_dir.mkdir(exist_ok=True, parents=True)
    
    # Import the working partitioning system
    from fedge.partitioning import hier_dirichlet_indices, write_partitions
    from fedge.task import load_cifar10_hf
    import numpy as np
    
    logger.info("🎯 Creating CIFAR-10 partitions with hierarchical Dirichlet distribution")
    
    # Load CIFAR-10 training data (torchvision-backed)
    data = load_cifar10_hf(seed=42)
    labels_np = data.train_labels
    
    # Log dataset statistics
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    total_samples = len(labels_np)
    logger.info(f"📊 CIFAR-10 dataset: {total_samples} samples, {len(unique_labels)} classes")
    
    logger.info(f"🔀 Creating partitions: {NUM_SERVERS} servers, {CLIENTS_PER_SERVER_LIST} clients per server")
    
    mapping = hier_dirichlet_indices(
        labels_np,
        NUM_SERVERS,
        CLIENTS_PER_SERVER_LIST,
    )
    
    # Log partition statistics for each server (simplified)
    for server_id, client_partitions in mapping.items():
        partition_sizes = [len(indices) for indices in client_partitions.values()]
        logger.info(f"   ✅ Server {server_id}: {len(client_partitions)} clients (avg: {sum(partition_sizes)/len(partition_sizes):.0f} samples)")
    
    write_partitions(parts_json, mapping)
    
    # Final summary
    total_clients = sum(len(server_clients) for server_clients in mapping.values())
    total_samples = sum(
        sum(len(client_indices) for client_indices in server_clients.values())
        for server_clients in mapping.values()
    )
    logger.info(f"✅ Partitions created: {len(mapping)} servers, {total_clients} clients")
    
    # Expose to subprocesses
    os.environ["PARTITIONS_JSON"] = str(parts_json)


def _hier() -> Dict:
    """Return hierarchy section i.e. `[tool.flwr.hierarchy]`."""
    return _load_config()["tool"]["flwr"]["hierarchy"]


def _app_cfg() -> Dict:
    """Return app-level config from hierarchy section."""
    return _load_config()["tool"]["flwr"]["hierarchy"]


# ──────────────────────────────────────────────────────────────────────────────
#  Constants derived from config (env > CLI > TOML is enforced in ONE place)
# ──────────────────────────────────────────────────────────────────────────────
HIER = _hier()
APP_CFG = _app_cfg()

# Number of leaf servers / clients per server
_raw_cps = HIER["clients_per_server"]
if isinstance(_raw_cps, Sequence):
    CLIENTS_PER_SERVER_LIST: List[int] = list(map(int, _raw_cps))
    NUM_SERVERS: int = len(CLIENTS_PER_SERVER_LIST)
else:
    if "num_servers" not in HIER:
        raise ValueError("num_servers must be specified in [tool.flwr.hierarchy] section of pyproject.toml")
    NUM_SERVERS = int(HIER["num_servers"])
    CLIENTS_PER_SERVER_LIST = [int(_raw_cps)] * NUM_SERVERS

GLOBAL_ROUNDS: int = int(HIER["global_rounds"])  # No environment override
SERVER_ROUNDS_PER_GLOBAL: int = HIER["server_rounds_per_global"]
CLOUD_PORT: int = HIER["cloud_port"]

# Dataset strategy - require explicit TOML value
if "dataset_flag" not in HIER:
    raise ValueError("dataset_flag must be specified in [tool.flwr.hierarchy] section of pyproject.toml")
DATASET_FLAG = HIER["dataset_flag"]
logger.info(f"Dataset distribution strategy → {DATASET_FLAG}")

# Enable dynamic clustering for CIFAR-10
if DATASET_FLAG == "cifar10":
    dynamic_clustering = HIER.get("dynamic_clustering", True)
    os.environ["DYNAMIC_CLUSTERING"] = str(dynamic_clustering).lower()
    logger.info(f"🔄 Set DYNAMIC_CLUSTERING={dynamic_clustering} for CIFAR-10 (weight-based clustering)")
else:
    logger.error(f"❌ Unsupported dataset_flag '{DATASET_FLAG}', only 'cifar10' is supported")
    sys.exit(1)

# Learning-rate scheduling (centralised source-of-truth) - require explicit TOML values
if "lr_init" not in HIER:
    raise ValueError("lr_init must be specified in [tool.flwr.hierarchy] section of pyproject.toml")
LR_INIT: float = float(HIER["lr_init"])
LR_DECAY: float = float(HIER.get("lr_decay", 1.0))  # Optional, defaults to no decay
LR_GAMMA: float = float(HIER.get("lr_gamma", 1.0))  # Optional, defaults to no gamma

# Local training hyper-params
ORIG_LOCAL_EPOCHS: int = APP_CFG["local_epochs"]
EPOCH_BOOST_FACTOR: float = float(os.getenv("EPOCH_BOOST_FACTOR", "1.0"))
FRACTION_FIT: float = 1.0  # Use all clients
FRACTION_EVAL: float = 1.0  # Evaluate all clients
TOTAL_LEAF_ROUNDS: int = APP_CFG["server_rounds_per_global"]

# ──────────────────────────────────────────────────────────────────────────────
#  Paths for signal files & rounds
# ──────────────────────────────────────────────────────────────────────────────
ROUND_DIR = PROJECT_ROOT / "rounds"
SIGNALS_DIR = PROJECT_ROOT / "signals"
for p in (ROUND_DIR, SIGNALS_DIR):
    p.mkdir(parents=True, exist_ok=True)

def round_dir(round_no: int) -> Path:
    return ROUND_DIR / f"round_{round_no}"


def global_complete_signal(round_no: int) -> Path:
    return round_dir(round_no) / "global" / "complete.signal"


def global_model_path(round_no: int) -> Path:
    return round_dir(round_no) / "global" / "model.pkl"


# ──────────────────────────────────────────────────────────────────────────────
#  Sub-process book-keeping
# ──────────────────────────────────────────────────────────────────────────────
active_processes: List[Tuple[str, _sp.Popen]] = []
leaf_server_procs: List[Tuple[int, _sp.Popen]] = []
leaf_client_procs: List[Tuple[Tuple[int, int], _sp.Popen]] = []  # ((sid,cid),proc)
proxy_client_procs: List[Tuple[int, _sp.Popen]] = []
cloud_proc: _sp.Popen | None = None
# Track log file handles to avoid descriptor leaks
LOG_HANDLES: dict[int, Tuple[io.TextIOWrapper, io.TextIOWrapper]] = {}


# ──────────────────────────────────────────────────────────────────────────────
#  Generic helpers
# ──────────────────────────────────────────────────────────────────────────────
def build_env(role: str, extra: Dict[str, str] | None = None) -> Dict[str, str]:
    """Return a fresh environment dict for a child process."""
    env = os.environ.copy()
    env["ROLE"] = role
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def spawn(cmd: Sequence[str], name: str, env: Dict[str, str] | None = None,
          cwd: Path | None = None) -> _sp.Popen:
    """Start a subprocess and register it for later cleanup.
    Redirect stdout/stderr to per-process log files under RUN_DIR/logs.
    """
    logs_dir = RUN_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    stdout_path = logs_dir / f"{name}_{ts}.out"
    stderr_path = logs_dir / f"{name}_{ts}.err"
    out_fh = open(stdout_path, "a", encoding="utf-8", buffering=1)
    err_fh = open(stderr_path, "a", encoding="utf-8", buffering=1)
    try:
        proc = _sp.Popen(
            list(cmd),
            cwd=str(cwd or PROJECT_ROOT),
            env=env or os.environ.copy(),
            stdout=out_fh,
            stderr=err_fh,
        )
        active_processes.append((name, proc))
        LOG_HANDLES[proc.pid] = (out_fh, err_fh)
        logger.debug(f"Spawned {name}: {' '.join(cmd)} (pid={proc.pid}) → logs: {stdout_path.name}, {stderr_path.name}")
        return proc
    except Exception:
        # Close file handles on spawn failure
        try:
            out_fh.close()
        finally:
            err_fh.close()
        raise


def wait_for_file(path: Path, timeout: int, poll: float = 1.0) -> bool:
    """Return True if file appears within `timeout` seconds."""
    logger.debug(f"Waiting for {path} (timeout={timeout}s)…")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists():
            return True
        time.sleep(poll)
    return False


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False


def wait_for_port_release(port: int, timeout: int = 30, host: str = "127.0.0.1") -> bool:
    """Wait for a port to become available, with timeout."""
    logger.debug(f"Waiting for port {port} to be released (timeout={timeout}s)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_available(port, host):
            logger.debug(f"Port {port} is now available")
            return True
        time.sleep(0.5)
    logger.warning(f"Port {port} still not available after {timeout}s")
    return False


def kill_processes_on_ports(ports: List[int]) -> None:
    """Kill any processes using the specified ports."""
    for port in ports:
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                    try:
                        proc = psutil.Process(conn.pid)
                        logger.info(f"Killing process {proc.pid} ({proc.name()}) using port {port}")
                        proc.terminate()
                        # Give it a moment to terminate gracefully
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except Exception as e:
            logger.debug(f"Error checking port {port}: {e}")


def cleanup_processes() -> None:
    """Terminate any still-running child processes and ensure ports are released.
    NOTE: Does NOT terminate the long-running cloud server.
    """
    # First, collect all ports that should be freed
    ports_to_free = []
    for sid in range(NUM_SERVERS):
        ports_to_free.append(SERVER_BASE_PORT + sid)
    
    # Terminate our tracked processes (EXCEPT cloud server)
    for name, proc in active_processes:
        if proc.poll() is None:
            # Skip terminating the long-running cloud server
            if name == "cloud_server":
                logger.debug(f"Keeping long-running {name} (pid={proc.pid}) alive")
                continue
            logger.info(f"Terminating {name} (pid={proc.pid})")
            try:
                proc.terminate()
            except Exception:  # pragma: no cover
                pass
    
    # Wait a bit for graceful termination
    time.sleep(2.0)
    
    # Force kill any remaining processes (EXCEPT cloud server)
    for name, proc in active_processes:
        if proc.poll() is None:
            # Skip force-killing the long-running cloud server
            if name == "cloud_server":
                continue
            logger.info(f"Force killing {name} (pid={proc.pid})")
            try:
                proc.kill()
            except Exception:
                pass
    
    # Kill any processes still using our ports
    kill_processes_on_ports(ports_to_free)
    
    # Wait for ports to be released
    for port in ports_to_free:
        wait_for_port_release(port, timeout=10)
    
    # Clear all process lists
    active_processes.clear()
    leaf_server_procs.clear()
    leaf_client_procs.clear()
    proxy_client_procs.clear()
    # Close any remaining log handles
    for pid, (out_fh, err_fh) in list(LOG_HANDLES.items()):
        try:
            if not out_fh.closed:
                out_fh.close()
        finally:
            if not err_fh.closed:
                err_fh.close()
        LOG_HANDLES.pop(pid, None)
    
    logger.info("Process cleanup completed")
################################################################################
#  end of chunk 1  ─────────────────────────────────────────────────────────────
################################################################################




################################################################################
# orchestrator.py   ―  Chunk 2 / 3
# ---------------------------------------------------------------------------
# Cloud server, leaf server / client, and proxy client launch logic,
# together with round-level wait utilities.
################################################################################

# ──────────────────────────────────────────────────────────────────────────────
#  Extra constants & derived paths
# ──────────────────────────────────────────────────────────────────────────────
PACKAGE_DIR          = PROJECT_ROOT / "fedge"
SERVER_BASE_PORT     = HIER.get("server_base_port", 5000)  # Read from TOML, fallback to 5000
SERVER_START_STAGGER = {0: 2, 1: 4, 2: 6}          # seconds - reduced from 15-35s to prevent timeout issues
MAX_WAIT_SEC_LEAF    = SERVER_ROUNDS_PER_GLOBAL * 900 + 180   # generous buffer
MAX_WAIT_SEC_PROXY   = 600
MAX_WAIT_SEC_CLOUD   = GLOBAL_ROUNDS * SERVER_ROUNDS_PER_GLOBAL * 900 + 180

# Early-stopping / LR-on-plateau - require explicit TOML values
if "es_patience" not in HIER:
    raise ValueError("es_patience must be specified in [tool.flwr.hierarchy] section of pyproject.toml")
if "es_delta" not in HIER:
    raise ValueError("es_delta must be specified in [tool.flwr.hierarchy] section of pyproject.toml")
ES_PATIENCE = int(HIER["es_patience"])  # No environment override
ES_DELTA    = float(HIER["es_delta"])   # No environment override
BEST_VAL_LOSS: float = float("inf")
NO_IMPROVE: int      = 0
LR_SCALE: float      = 1.0                 # multiplicative factor
os.environ.setdefault("LR_SCALE", str(LR_SCALE))

# Process tracking variables


# ──────────────────────────────────────────────────────────────────────────────
#  Learning-rate schedule helper (centralised!)
# ──────────────────────────────────────────────────────────────────────────────
def lr_for_round(global_round: int) -> float:
    """Return LR for *clients* this global round, applying LR_DECAY / LR_GAMMA."""
    decay_steps = max(global_round - 1, 0)
    lr = LR_INIT * (LR_DECAY ** decay_steps) * LR_SCALE
    return round(lr, 8)


# ──────────────────────────────────────────────────────────────────────────────
#  1) Cloud Aggregator
# ──────────────────────────────────────────────────────────────────────────────
def start_cloud_server_once() -> None:
    """
    Launch single long-running `cloud_flower.py` that handles all global rounds.
    More efficient than spawning/killing cloud server per round.
    """
    global cloud_proc

    # Only start if not already running
    if cloud_proc and cloud_proc.poll() is None:
        logger.info("⛅ Cloud server already running")
        return

    # Clean up old signal files from previous runs
    signals_dir = PROJECT_ROOT / "signals"
    old_signals = list(signals_dir.glob("cloud_*.signal"))
    for signal_file in old_signals:
        signal_file.unlink()
        logger.debug(f"Removed old signal: {signal_file}")

    logger.info(f"⛅ Starting long-running cloud server on port {CLOUD_PORT}")

    env = build_env(
        role="cloud",
        extra={
            "SERVER_ID": "cloud",
            "TOTAL_GLOBAL_ROUNDS": str(GLOBAL_ROUNDS),  # Tell cloud how many rounds total
            "USE_NEW_DIR_STRUCTURE": "1",
            "LR_INIT": str(LR_INIT),
            "LR_DECAY": str(LR_DECAY),
            "LR_GAMMA": str(LR_GAMMA),
            "DATASET_FLAG": DATASET_FLAG,
            "CLOUD_PORT": str(CLOUD_PORT),
            "NUM_SERVERS": str(NUM_SERVERS),
            "SERVER_ROUNDS": str(SERVER_ROUNDS_PER_GLOBAL),
        },
    )

    cloud_script = (PROJECT_ROOT / "fedge" / "cloud_flower.py").resolve()
    if not cloud_script.exists():
        logger.error(f"Cloud server script not found at {cloud_script}")
        raise FileNotFoundError(f"Cloud server script not found at {cloud_script}")
    cmd = [sys.executable, str(cloud_script)]
    cloud_proc = spawn(cmd, name="cloud_server", env=env, cwd=PROJECT_ROOT)

    # Wait (max 60 s) for cloud_started.signal
    # Cloud server creates signal in signals/ directory
    signal_path = PROJECT_ROOT / "signals" / "cloud_started.signal"
    if not wait_for_file(signal_path, timeout=60):
        logger.warning("Cloud did not emit cloud_started.signal – continuing anyway.")
    else:
        logger.info("✅ Cloud server started successfully")
        # Keep signal file for debugging/analysis (no cleanup)


# ──────────────────────────────────────────────────────────────────────────────
#  2) Leaf servers + their local clients
# ──────────────────────────────────────────────────────────────────────────────
def launch_leaf_servers(global_round: int, prev_local_rounds: int) -> int:
    """
    Spawn NUM_SERVERS × `leaf_server.py`, each with its own leaf clients.
    Returns the number of local rounds each server will execute this pass.
    """
    # Clean up any previous processes and ensure ports are available
    logger.info("🧹 Cleaning up previous processes before starting new servers")
    cleanup_processes()
    
    # Create shared initial model for GR-1 to align model orientations
    if global_round == 1:
        init_path = PROJECT_ROOT / "models" / "init.pkl"
        if not init_path.exists():
            from fedge.task import Net, get_weights
            init_path.parent.mkdir(exist_ok=True)
            import pickle
            w = get_weights(Net())
            with open(init_path, "wb") as f:
                pickle.dump(w, f)
            logger.info(f"🎯 Created shared initial model for GR-1: {init_path}")
        else:
            logger.info(f"🎯 Using existing shared initial model: {init_path}")
    
    rounds_this_pass = SERVER_ROUNDS_PER_GLOBAL
    logger.info(
        f"🚀 Launching {NUM_SERVERS} leaf servers "
        f"(each runs {rounds_this_pass} local rounds) for GR {global_round}"
    )
    
    # Verify all required ports are available before starting any servers
    required_ports = [SERVER_BASE_PORT + sid for sid in range(NUM_SERVERS)]
    for port in required_ports:
        if not is_port_available(port):
            logger.error(f"Port {port} is not available. Attempting to free it...")
            kill_processes_on_ports([port])
            if not wait_for_port_release(port, timeout=15):
                raise RuntimeError(f"Unable to free port {port} for server startup")
            logger.info(f"Port {port} is now available")

    for sid in range(NUM_SERVERS):
        port = SERVER_BASE_PORT + sid
        n_clients = CLIENTS_PER_SERVER_LIST[sid]

        # ── Server env & command ────────────────────────────────────────────
        senv = build_env(
            role="leaf_server",
            extra={
                "SERVER_ID": str(sid),
                "GLOBAL_ROUND": str(global_round),
                "USE_NEW_DIR_STRUCTURE": "1",
                "FRACTION_FIT": str(FRACTION_FIT),
                "FRACTION_EVAL": str(FRACTION_EVAL),
                "DATASET_FLAG": DATASET_FLAG,
                "LR_INIT": str(LR_INIT),
            },
        )

        # Check for existing global model from previous round
        if global_round > 1:
            # ── Model continuity: determine initial model path ────
            prev_round = global_round - 1
            candidate_paths = [
                PROJECT_ROOT / "models" / f"head_{sid}.pkl",  # Prefer cluster head
                PROJECT_ROOT / "rounds" / f"round_{prev_round}" / "global" / "model.pkl",
                PROJECT_ROOT / "models" / f"model_global_g{prev_round}.pkl",
                PROJECT_ROOT / "models" / f"global_model_round_{prev_round}.pkl"
            ]
            model_path = next((p for p in candidate_paths if p.exists()), None)
        else:
            # GR-1: Use shared initial model to align orientations
            model_path = PROJECT_ROOT / "models" / "init.pkl"
        
        scmd = [
            sys.executable,
            str(PACKAGE_DIR / "leaf_server.py"),
            "--server_id", str(sid),
            "--clients_per_server", str(n_clients),
            "--num_rounds", str(rounds_this_pass),
            "--fraction_fit", str(FRACTION_FIT),
            "--fraction_evaluate", str(FRACTION_EVAL),
            "--port", str(port),
            "--global_round", str(global_round),
        ]
        
        # Add initial model path if it exists (for model continuity across global rounds)
        if model_path and model_path.exists():
            scmd.extend(["--initial_model_path", str(model_path)])
            if global_round == 1:
                logger.info(f"🎯 Server {sid} will start from shared init: {model_path}")
            else:
                logger.info(f"🔄 Server {sid} will continue from saved model: {model_path}")
        else:
            logger.info(f"🆕 Server {sid} will start with fresh model (no previous model found)")
        
        # Retry server startup with exponential backoff in case of transient issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Double-check port availability right before starting
                if not is_port_available(port):
                    logger.warning(f"Port {port} became unavailable, attempting to free it...")
                    kill_processes_on_ports([port])
                    wait_for_port_release(port, timeout=5)
                
                srv_proc = spawn(scmd, name=f"leaf_server_{sid}", env=senv)
                leaf_server_procs.append((sid, srv_proc))
                logger.info(f"✅ Server {sid} started successfully on port {port}")
                break
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to start server {sid}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to start server {sid} after {max_retries} attempts")
                    raise RuntimeError(f"Unable to start leaf server {sid} on port {port}")

        # Optional stagger to avoid port-storm
        time.sleep(SERVER_START_STAGGER.get(sid, 5))

        # ── Client env & command ────────────────────────────────────────────
        for cid in range(n_clients):
            cenv = build_env(
                role="leaf_client",
                extra={
                    "SERVER_ID": str(sid),
                    "CLIENT_ID": f"{sid}_{cid}",
                    "GLOBAL_ROUND": str(global_round),
                    "LR": str(lr_for_round(global_round)),
                    "LOCAL_EPOCHS": str(ORIG_LOCAL_EPOCHS),
                },
            )
            ccmd = [
                sys.executable,
                str(PACKAGE_DIR / "leaf_client.py"),
                "--server_id", str(sid),
                "--client_id", str(cid),
                "--dataset_flag", DATASET_FLAG,
                "--local_epochs", str(ORIG_LOCAL_EPOCHS),
                "--server_addr", f"127.0.0.1:{port}",
            ]
            cproc = spawn(ccmd, name=f"leaf_client_{sid}_{cid}", env=cenv)
            leaf_client_procs.append(((sid, cid), cproc))

            time.sleep(0.2)  # small spacing

    return rounds_this_pass




# ──────────────────────────────────────────────────────────────────────────────
#  4) Wait helpers
# ──────────────────────────────────────────────────────────────────────────────
def _wait(procs: List[Tuple[int, _sp.Popen]], label: str, timeout: int) -> Tuple[set[int], set[int]]:
    """Wait for processes and return (successful_pids, failed_pids)."""
    logger.info(f"⏳ Waiting for all {label} to finish …")
    done: set[int] = set()
    failed: set[int] = set()
    start = time.time()
    while len(done) + len(failed) < len(procs) and (time.time() - start) < timeout:
        for pid, proc in procs:
            if pid in done or pid in failed:
                continue
            if proc.poll() is not None:
                if proc.returncode == 0:
                    logger.info(f"✅ {label.capitalize()} {pid} exited successfully (code {proc.returncode})")
                    done.add(pid)
                else:
                    logger.error(f"❌ {label.capitalize()} {pid} failed (code {proc.returncode})")
                    failed.add(pid)
        time.sleep(1)
    if len(done) + len(failed) < len(procs):
        logger.warning(f"Timeout waiting for {label}.   finished={len(done) + len(failed)}/{len(procs)}")
    return done, failed


 



 
################################################################################
#  end of chunk 2  ─────────────────────────────────────────────────────────────
################################################################################


################################################################################
# orchestrator.py   ―  Chunk 3 / 3
# ---------------------------------------------------------------------------
# Main orchestration loop, early-stopping logic, graceful shutdown
################################################################################

# ──────────────────────────────────────────────────────────────────────────────
#  Validation-loss helper (optional but useful)
# ──────────────────────────────────────────────────────────────────────────────
def _read_val_loss(global_round: int) -> float | None:
    """
    Return validation loss written by cloud for this global round.
    Expects a file `<round_dir>/global/val_loss.txt` containing a single float.
    """
    path = round_dir(global_round) / "global" / "val_loss.txt"
    try:
        return float(path.read_text().strip())
    except FileNotFoundError:
        logger.warning(f"No val_loss.txt for GR {global_round}")
    except ValueError:
        logger.error(f"Malformed val_loss.txt for GR {global_round}")
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  Early-stop & LR-plateau bookkeeping
# ──────────────────────────────────────────────────────────────────────────────
def _update_early_stopping(val_loss: float | None) -> bool:
    """
    Update BEST_VAL_LOSS / NO_IMPROVE counters.
    Reduce LR_SCALE if plateau persists.
    Return True if training should halt.
    """
    global BEST_VAL_LOSS, NO_IMPROVE, LR_SCALE

    if val_loss is None:
        return False  # skip

    if val_loss < BEST_VAL_LOSS - ES_DELTA:
        BEST_VAL_LOSS = val_loss
        NO_IMPROVE = 0
        logger.info(f"✨  New best val-loss = {val_loss:.6f}")
    else:
        NO_IMPROVE += 1
        logger.info(
            f"Val-loss did not improve ({val_loss:.6f})  "
            f"→ plateau {NO_IMPROVE}/{ES_PATIENCE}"
        )
        if NO_IMPROVE % ES_PATIENCE == 0:
            LR_SCALE *= LR_GAMMA
            os.environ["LR_SCALE"] = str(LR_SCALE)
            logger.info(
                f"Reducing LR scale to {LR_SCALE:.5f} "
                f"after {NO_IMPROVE} stagnant rounds."
            )
    return NO_IMPROVE >= ES_PATIENCE


# ──────────────────────────────────────────────────────────────────────────────
#  3) Proxy Clients (upload leaf server models to cloud)
# ──────────────────────────────────────────────────────────────────────────────

def start_proxy_clients(global_round: int) -> None:
    """
    Launch proxy clients to upload leaf server models to cloud server.
    Each proxy client gets a deterministic numeric client_id matching its server_id.
    """
    logger.info(f"🚀 Starting {NUM_SERVERS} proxy clients for global round {global_round}")
    
    for server_id in range(NUM_SERVERS):
        proxy_env = build_env("proxy", {
            "SERVER_ID": str(server_id),
            "PROXY_ID": f"proxy_{server_id}",
            "GLOBAL_ROUND": str(global_round),
            "DIR_ROUND": str(global_round),
            # Let proxies know how many server rounds the cloud will run this pass
            "TOTAL_SERVER_ROUNDS_THIS_CLOUD": str(SERVER_ROUNDS_PER_GLOBAL),
            "CLOUD_ADDRESS": f"127.0.0.1:{HIER.get('cloud_port', 6000)}",
            "DATASET_FLAG": DATASET_FLAG,
        })
        
        proxy_cmd = [
            sys.executable, "-m", "fedge.proxy_client",
            "--server_id", str(server_id),
            "--cloud_address", f"127.0.0.1:{HIER.get('cloud_port', 6000)}",
            "--global_round", str(global_round),
            "--dir_round", str(global_round),
        ]
        
        # Use spawn() which registers in active_processes, then also track in proxy_client_procs
        proc = spawn(proxy_cmd, f"proxy_{server_id}", env=proxy_env, cwd=PROJECT_ROOT)
        proxy_client_procs.append((server_id, proc))
        logger.info(f"   ✅ Started proxy client {server_id} (node_id enforced if ClientConfig available)")


def wait_for_proxy_clients() -> bool:
    """
    Wait for all proxy clients to complete uploading models to cloud.
    Returns True if all succeeded, False if any failed.
    """
    logger.info("⏳ Waiting for proxy clients to complete...")
    
    # Proxy clients should complete quickly (just upload models)
    timeout = 120  # 2 minutes should be plenty
    start_time = time.time()
    completed = set()
    failed = set()
    
    while time.time() - start_time < timeout:
        # Check proxy processes and their exit codes
        for server_id, proc in proxy_client_procs:
            if server_id in completed or server_id in failed:
                continue
            if proc.poll() is not None:
                if proc.returncode == 0:
                    logger.info(f"✅ Proxy client {server_id} completed successfully (code {proc.returncode})")
                    completed.add(server_id)
                else:
                    logger.error(f"❌ Proxy client {server_id} failed (code {proc.returncode})")
                    failed.add(server_id)
        
        if len(completed) + len(failed) == len(proxy_client_procs):
            if failed:
                logger.error(f"❌ {len(failed)} proxy clients failed: {failed}")
                return False
            logger.info(f"✅ All {len(completed)} proxy clients completed successfully")
            return True
        
        time.sleep(2)
    
    logger.warning(f"⚠️ Proxy clients did not complete within {timeout}s")
    return False


def wait_for_leaf_servers(global_round: int) -> Tuple[set[int], set[int]]:
     """
     Wait for leaf server subprocesses to exit and for their completion signals to appear.
     Signals are created by leaf servers via fs_optimized at:
       signals/round_{global_round}/server_{sid}_completion.signal
     Returns (done_server_ids, failed_server_ids) based on subprocess exit codes.
     """
     logger.info(f"⏳ Waiting for {NUM_SERVERS} leaf servers to complete (processes + signals)…")

     # 1) Wait for subprocess exit across all servers
     done, failed = _wait(leaf_server_procs, "leaf server", MAX_WAIT_SEC_LEAF)

     # 2) After processes finish, wait briefly for expected signal files
     round_sig_dir = get_round_signals_dir(PROJECT_ROOT, global_round)
     logger.info(f"🔎 Expecting completion signals under: {round_sig_dir}")

     missing: List[Tuple[int, Path]] = []
     for sid in range(NUM_SERVERS):
         sig_path = get_server_signal_path(PROJECT_ROOT, global_round, sid)
         if wait_for_file(sig_path, timeout=60):
             logger.info(f"   ✅ Signal present for server {sid}: {sig_path}")
         else:
             logger.warning(f"   ⏰ Signal NOT found for server {sid} within 60s: {sig_path}")
             missing.append((sid, sig_path))

     if missing:
         try:
             contents = sorted(p.name for p in round_sig_dir.iterdir())
             logger.info(f"📂 Contents of {round_sig_dir}: {contents}")
         except Exception as e:
             logger.warning(f"Could not list {round_sig_dir}: {e}")

     return done, failed


def _enforce_strict_server_barrier(global_round: int, done: set[int], failed: set[int]) -> None:
    """Strict barrier: abort immediately if any server fails or is missing artifacts."""
    round_sig_dir = get_round_signals_dir(PROJECT_ROOT, global_round)
    error_signals = sorted(round_sig_dir.glob("server_*_error.signal")) if round_sig_dir.exists() else []
    missing_completion = []
    missing_models = []

    for sid in range(NUM_SERVERS):
        comp = get_server_signal_path(PROJECT_ROOT, global_round, sid)
        if not comp.exists():
            missing_completion.append((sid, comp))
        
        # ⬇️ use the canonical writer path
        model_path = get_model_path(PROJECT_ROOT, sid, global_round)
        if not model_path.exists():
            missing_models.append((sid, model_path))

    if failed or error_signals or missing_completion or missing_models or len(done) != NUM_SERVERS:
        logger.error("❌ STRICT MODE: at least one leaf server failed this round; aborting.")
        if failed:
            logger.error(f"   • Failed processes: {sorted(failed)}")
        if error_signals:
            logger.error(f"   • Error signals: {[p.name for p in error_signals]}")
        if missing_completion:
            logger.error(f"   • Missing completion signals: {missing_completion}")
        if missing_models:
            logger.error(f"   • Missing model files: {missing_models}")

        cleanup_processes()  # terminate all children
        raise SystemExit(1)

def _assert_cloud_outputs(global_round: int) -> None:
    """
    Hard check that cloud aggregation produced the expected global model files.
    Raises RuntimeError if cloud outputs are missing.
    """
    models_dir = PROJECT_ROOT / "models"
    legacy = models_dir / f"model_global_g{global_round}.pkl"
    rounds_global = PROJECT_ROOT / "rounds" / f"round_{global_round}" / "global" / "model.pkl"
    if not (legacy.exists() or rounds_global.exists()):
        raise RuntimeError(
            f"Cloud aggregation produced no global model for round {global_round}. "
            f"Expected {legacy} or {rounds_global}."
        )


def _distribute_cluster_heads(global_round: int) -> None:
    """
    Distribute cluster-specific heads to each server for warm-start.
    If clustering produced K clusters, copy the appropriate cluster model
    to head_{sid}.pkl for each server based on cluster assignments.
    """
    round_dir = PROJECT_ROOT / "rounds" / f"round_{global_round}" / "cloud"
    models_dir = PROJECT_ROOT / "models"
    server_ids = list(range(NUM_SERVERS))
    
    # 1) Try to use cluster-specific heads if clustering happened
    # Check both canonical location and legacy location
    clusters_json = None
    for clusters_path in [
        round_dir / f"clusters_g{global_round}.json",
        models_dir / f"clusters_g{global_round}.json",
    ]:
        if clusters_path.exists():
            clusters_json = clusters_path
            break
    
    if clusters_json:
        try:
            with open(clusters_json) as f:
                data = json.load(f)
            assignments = data.get("assignments", {})
            logger.info(f"📋 Found cluster assignments for round {global_round}: {assignments}")
            
            for sid in server_ids:
                lab = assignments.get(str(sid))
                if lab is None:
                    logger.warning(f"No cluster label for server {sid} in {clusters_json}")
                    continue
                
                # Try both locations for cluster models
                cluster_model = None
                for src in [
                    round_dir / f"model_cluster{lab}_g{global_round}.pkl",
                    models_dir / f"model_cluster{lab}_g{global_round}.pkl",
                ]:
                    if src.exists():
                        cluster_model = src
                        break
                
                if cluster_model:
                    dst = models_dir / f"head_{sid}.pkl"
                    shutil.copyfile(cluster_model, dst)
                    logger.info(f"📋 Created cluster head for server {sid}: {cluster_model.name} → {dst.name}")
                else:
                    logger.error(f"No model for cluster {lab} of round {global_round} found")
                    
        except Exception as e:
            logger.warning(f"Failed to process cluster assignments: {e}")
            clusters_json = None  # Fall back to global heads
    
    # 2) Fallback to global heads if no clustering or clustering failed
    if not clusters_json:
        logger.warning("No clusters mapping found; falling back to global head")
        # Try both locations for global model
        global_model = None
        for global_path in [
            round_dir / "model.pkl",
            models_dir / f"model_global_g{global_round}.pkl",
        ]:
            if global_path.exists():
                global_model = global_path
                break
        
        if global_model:
            for sid in server_ids:
                dst = models_dir / f"head_{sid}.pkl"
                shutil.copyfile(global_model, dst)
                logger.info(f"📋 Created global fallback head for server {sid}: {dst}")
        else:
            logger.warning(f"No global model found for round {global_round} fallback heads")


def wait_for_global_completion(global_round: int) -> bool:
    """
    Wait for cloud server to complete aggregation for specific global round.
    """
    completion_signal = PROJECT_ROOT / "signals" / f"cloud_round_{global_round}_completed.signal"
    if wait_for_file(completion_signal, timeout=MAX_WAIT_SEC_CLOUD):
        logger.info(f"✅ Cloud server completed round {global_round} successfully")
        return True
    logger.error(f"❌ Cloud server did not complete round {global_round} within {MAX_WAIT_SEC_CLOUD}s")
    return False


# ──────────────────────────────────────────────────────────────────────────────
#  Server-level FedAvg aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_server_models(global_round: int) -> None:
    """Weighted FedAvg aggregation of leaf server models.

    Leaf servers save their final model and sample count to:
        models/model_s{sid}_g{global_round}.pkl

    Each pickle contains a tuple (parameters_ndarrays, num_examples). We load all
    servers, perform a weighted average using the number of examples as weights,
    and write the aggregated parameters to models/head_{sid}.pkl so every server
    starts the next global round from the same model.
    """
    from fedge.utils.fs_optimized import get_model_path
    
    model_entries: List[Tuple[List[np.ndarray], int]] = []

    for sid in range(NUM_SERVERS):
        model_file = get_model_path(PROJECT_ROOT, sid, global_round)

        if not model_file.exists():
            logger.error(f"❌ Cannot aggregate – model file not found: {model_file}")
            return

        try:
            with open(model_file, "rb") as f:
                loaded = pickle.load(f)

            if isinstance(loaded, tuple) and len(loaded) == 2:
                params, num_examples = loaded
            else:
                params, num_examples = loaded, 0  # Fallback if no sample count stored
                logger.warning(
                    f"Model file {model_file} did not include sample count; weight set to 0."
                )

            model_entries.append((params, num_examples))
            logger.info(f"📥 Loaded model from server {sid} ({num_examples} samples)")
        except Exception as exc:
            logger.error(f"❌ Failed to read {model_file}: {exc}")
            return

    total_examples = sum(n for _, n in model_entries)
    if total_examples <= 0:
        logger.error("❌ No valid sample counts found; skipping aggregation.")
        return

    num_layers = len(model_entries[0][0])
    aggregated: List[np.ndarray] = []

    for layer_idx in range(num_layers):
        layer_sum = sum(params[layer_idx] * weight for params, weight in model_entries)
        aggregated.append(layer_sum / total_examples)

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    for sid in range(NUM_SERVERS):
        out_path = models_dir / f"head_{sid}.pkl"
        try:
            with open(out_path, "wb") as f:
                pickle.dump(aggregated, f)
            logger.info(f"💾 Saved aggregated model to {out_path}")
        except Exception as exc:
            logger.error(f"❌ Could not write {out_path}: {exc}")

    logger.info(f"⭐ FedAvg aggregation for global round {global_round} completed.")


def evaluate_and_save_global_metrics(global_round: int) -> None:
    """Evaluate aggregated global model on CIFAR-10 test set and save metrics."""
    try:
        # CIFAR-10 only: use dedicated test loader and Net
        from fedge.task import Net, test, get_cifar10_test_loader, set_weights
        import torch
        import pickle
        
        # Optimized metrics storage helpers
        from fedge.utils.fs_optimized import save_global_metrics_optimized, get_global_model_path
        
        # Determine which aggregated model to load
        candidate_paths = []
        # Prefer the new rounds directory structure
        rounds_model_path = PROJECT_ROOT / "rounds" / f"round_{global_round}" / "global" / "model.pkl"
        candidate_paths.append(rounds_model_path)
        # Legacy paths for compatibility
        models_dir = PROJECT_ROOT / "models"
        candidate_paths.append(get_global_model_path(PROJECT_ROOT, global_round))
        candidate_paths.append(models_dir / f"model_global_g{global_round}.pkl")
        candidate_paths.append(models_dir / f"global_model_round_{global_round}.pkl")
        # Fallback to aggregated head saved by FedAvg (identical for all heads)
        candidate_paths.append(models_dir / "head_0.pkl")
        # As a last resort, try any server head
        for sid in range(NUM_SERVERS):
            candidate_paths.append(models_dir / f"head_{sid}.pkl")
        
        model_path = next((p for p in candidate_paths if p.exists()), None)
        if model_path is None:
            logger.warning(f"[Global Eval] No aggregated model found for round {global_round} under {models_dir}")
            return
        
        with open(model_path, "rb") as f:
            global_weights = pickle.load(f)
        
        # Initialize model and device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net().to(device)
        set_weights(model, global_weights)
        
        # Evaluate on the CIFAR-10 test set
        test_loader = get_cifar10_test_loader(batch_size=64)
        loss, accuracy = test(model, test_loader, device)
        total_samples = len(test_loader.dataset)
        
        # Save global metrics using consolidated storage
        global_metrics = {
            'global_accuracy': accuracy,
            'global_loss': loss,
            'total_samples': total_samples,
            'convergence_rate': 0.0,
            'generalization_gap': 0.0,
        }
        save_global_metrics_optimized(PROJECT_ROOT, global_round, global_metrics)
        logger.info(f"[Global Eval] Global Model: {accuracy:.4f} accuracy, {loss:.4f} loss, {total_samples} samples")
        logger.info("💾 Global metrics saved using consolidated storage")
        
    except Exception as e:
        logger.error(f"[Global Eval] Failed to evaluate global model for round {global_round}: {e}")
        import traceback
        traceback.print_exc()

# ──────────────────────────────────────────────────────────────────────────────
#  Main driver
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    """Entry-point: orchestrate hierarchical federated learning with global model evaluation."""
    # Initialize comprehensive logging
    summary_file = create_run_summary()
    # Removed verbose system info logging - kept in summary file only
    
    # Create partitions once at startup
    _create_partitions_if_needed()
    
    try:
        for gr in range(1, GLOBAL_ROUNDS + 1):
            logger.info(f"\n══════════════════════  GLOBAL ROUND {gr}/{GLOBAL_ROUNDS}  ══════════════════════")
            
            # Set global round environment variable for all components
            os.environ["GLOBAL_ROUND"] = str(gr)

            # HIERARCHICAL FL FLOW:
            # 1) Start leaf servers (federated servers with local clients)
            logger.info("🚀 Step 1: Starting leaf servers for local federated learning")
            local_rounds = launch_leaf_servers(global_round=gr, prev_local_rounds=0)
            
            # 2) Wait for leaf servers to complete their local FL rounds
            logger.info("⏳ Step 2: Waiting for leaf servers to complete local training")
            done, failed = wait_for_leaf_servers(gr)
            
            # ❌ HARD STOP GATE: Abort immediately if any server fails
            if failed:
                ids = ", ".join(map(str, sorted(failed)))
                logger.error(f"❌ Aborting: leaf server(s) failed: {ids}. No fallback, no skipping.")
                cleanup_processes()
                sys.exit(2)
            
            # ❌ STRICT BARRIER: Abort immediately if any server fails
            _enforce_strict_server_barrier(gr, done, failed)
            
            # Print error signal contents for debugging
            if failed:
                round_sig_dir = get_round_signals_dir(PROJECT_ROOT, gr)
                for sid in sorted(failed):
                    err = round_sig_dir / f"server_{sid}_error.signal"
                    if err.exists():
                        try:
                            logger.error(f"🧾 Error from leaf server {sid}:\n{err.read_text(encoding='utf-8')}")
                        except Exception as e:
                            logger.error(f"Could not read error signal for server {sid}: {e}")
            
            # Double-check all completion signals are present
            if not check_all_servers_completed(PROJECT_ROOT, gr, NUM_SERVERS):
                logger.error(f"❌ Missing completion signals for GR {gr}; see logs above for missing files.")
                continue

            # ── Step 3: Ensure cloud server is running ──
            if gr == 1:
                logger.info("⛅ Step 3: Starting long-running cloud server for all rounds")
                start_cloud_server_once()
            else:
                logger.info(f"⛅ Step 3: Cloud server ready for round {gr}")
            
            # Give cloud server time to fully start and listen for connections
            import time
            logger.info("⏳ Waiting 5 seconds for cloud server to be ready...")
            time.sleep(5)
            
            # ── Step 4: Upload server models via proxy clients ──
            logger.info("📤 Step 4: Uploading server models to cloud via proxy clients")
            start_proxy_clients(gr)
            
            # ── Step 5: Wait for cloud clustering and aggregation ──
            logger.info("⏳ Step 5: Waiting for cloud clustering and aggregation")
            if not wait_for_global_completion(gr):
                logger.error(f"❌ Cloud server failed for round {gr}, aborting")
                cleanup_processes()
                sys.exit(3)
            
            # ── Step 5.1: Validate cloud outputs ──
            logger.info("🔍 Step 5.1: Validating cloud outputs")
            _assert_cloud_outputs(gr)
            
            # ── Step 5.2: Distribute cluster heads for next round warm-start ──
            logger.info("📋 Step 5.2: Distributing cluster heads for warm-start")
            _distribute_cluster_heads(gr)
                
            # ── Step 6: Evaluate global model and save metrics ──
            logger.info("📊 Step 6: Evaluating global model performance")
            evaluate_and_save_global_metrics(gr)

            logger.info(f"✅ Global round {gr} completed successfully!")

    except KeyboardInterrupt:
        logger.warning("⛔  Interrupted by user (Ctrl-C).")

    finally:
        logger.info("🧹  Cleaning up child processes …")
        # Terminate cloud server only at the very end of all rounds
        if cloud_proc and cloud_proc.poll() is None:
            logger.info("Terminating long-running cloud server")
            cloud_proc.terminate()
            cloud_proc.wait()
        cleanup_processes()
        logger.info("🏁  Orchestration finished.")


# ──────────────────────────────────────────────────────────────────────────────
#  Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
################################################################################
#  end of orchestrator.py  ─────────────────────────────────────────────────────
################################################################################
