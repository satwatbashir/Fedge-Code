# cloud_flower.py
# OPTIMIZED VERSION - Dynamic clustering every round, no redundant evaluations
"""
Cloud server for hierarchical FL with dynamic clustering.

Key features:
- Clustering EVERY round based on similarity (dynamic clusters)
- No global model evaluation after round 1 (not used)
- No cluster evaluations (servers already evaluated)
- Each server gets ONLY its cluster model (no fallback)
- Fixed race conditions and None parameter issues
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="flwr")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")

# Ensure project root on sys.path


import csv
import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import toml
import torch
from flwr.common import NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays, FitIns
from flwr.common.typing import Properties
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg

from fedge.task import Net, get_weights, set_weights
from fedge.utils.cluster_utils import cifar10_weight_clustering
# Flower compatibility: PropertiesIns exists in newer versions; older versions don’t have it
try:
    from flwr.common import PropertiesIns  # newer API
except Exception:
    PropertiesIns = None                   # older API path


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cloud")

    
# Ensure project root on sys.path AND pin CWD to repo
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)  # <<< critical: make all relative paths land in the repo

# ─────────────────────── Helper Functions ───────────────────────
def _signals_dir() -> Path:
    p = PROJECT_ROOT / "signals"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _metrics_dir() -> Path:
    p = PROJECT_ROOT / "metrics" / "cloud"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _round_dir(r: int) -> Path:
    return PROJECT_ROOT / "rounds" / f"round_{r}"

def _write_signal(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            f.write(f"{message}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        logger.error(f"Failed to write signal {path}: {e}")


def _append_csv(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fieldnames})


# ─────────────────────── Cloud Strategy ───────────────────────
class CloudFedAvg(FedAvg):
    """FedAvg with dynamic clustering every round + per-round client gating"""

    def __init__(self, *args, cloud_cluster_cfg: Optional[Dict] = None, num_servers: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cloud_cluster_cfg = cloud_cluster_cfg or {}
        self.num_servers = int(num_servers) if num_servers is not None else max(
            getattr(self, "min_fit_clients", 1),
            getattr(self, "min_available_clients", 1),
        )

        # Extract clustering configuration
        self.clustering_enabled = self.cloud_cluster_cfg.get("enable", False)
        self.cluster_start = self.cloud_cluster_cfg.get("start_round", 1)
        self.cluster_frequency = self.cloud_cluster_cfg.get("frequency", 1)
        self.cluster_method = self.cloud_cluster_cfg.get("method", "cosine_similarity")
        self.cluster_tau = self.cloud_cluster_cfg.get("tau", 0.01)

        logger.info(f"[Cloud Strategy] Clustering enabled: {self.clustering_enabled}")
        if self.clustering_enabled:
            logger.info(
                f"[Cloud Strategy] Clustering config: method={self.cluster_method}, "
                f"tau={self.cluster_tau}, start={self.cluster_start}, frequency={self.cluster_frequency}"
            )

    def _expected_node_ids(self, server_round: int) -> List[str]:
        # Proxies set: FLWR_CLIENT_NODE_ID = f"s{sid}_g{global_round}"
        return [f"s{sid}_g{server_round}" for sid in range(self.num_servers)]
        

    def _refresh_properties(self, client_manager, server_round: int):
        """Populate cid<->node_id maps *strictly* from per-round node_id (s{sid}_g{server_round}).
        Any client without a valid node_id for THIS round is ignored.
        """
        # Collect connected clients (Flower API is version-dependent)
        try:
            all_clients = client_manager.all() if hasattr(client_manager, "all") else {}
        except Exception:
            all_clients = {}

        # Init maps once
        if not hasattr(self, "_cid_to_node"):
            self._cid_to_node, self._node_to_cid = {}, {}

        # Only accept node_ids that belong to THIS round
        expected = set(self._expected_node_ids(server_round))

        # Prepare PropertiesIns for newer Flower; otherwise accept plain dicts
        try:
            from flwr.common import PropertiesIns as _PI
        except Exception:
            _PI = None
        props_ins = _PI(config={}) if _PI is not None else None

        # Helper: derive node_id from a properties dict
        def _derive_node_id(props: dict) -> str | None:
            if not isinstance(props, dict):
                return None
            nid = props.get("node_id")
            if isinstance(nid, str):
                return nid
            sid = props.get("server_id")
            grd = props.get("global_round")
            try:
                if sid is not None and grd is not None:
                    return f"s{int(sid)}_g{int(grd)}"
            except Exception:
                pass
            alt = props.get("client_name") or props.get("name")
            if isinstance(alt, str):
                return alt
            return None

        for cid, cp in list(all_clients.items()):
            # Fetch properties with broad signature compatibility
            props = None
            try:
                if props_ins is None:
                    for call in (
                        lambda: cp.get_properties(timeout=10.0),
                        lambda: cp.get_properties({}, timeout=10.0),
                        lambda: cp.get_properties(config={}, timeout=10.0),
                        lambda: cp.get_properties({}),
                        lambda: cp.get_properties(),
                    ):
                        try:
                            props = call()
                            break
                        except TypeError:
                            continue
                else:
                    try:
                        res = cp.get_properties(props_ins, timeout=10.0)      # positional
                    except TypeError:
                        res = cp.get_properties(ins=props_ins, timeout=10.0)  # keyword
                    props = getattr(res, "properties", res)
            except Exception as e:
                logger.debug(f"[_refresh_properties] get_properties failed for cid={getattr(cp,'cid',cid)}: {e}")
                props = None

            if hasattr(props, "properties"):
                props = props.properties

            node_id = _derive_node_id(props) if isinstance(props, dict) else None
            if not isinstance(node_id, str):
                # No usable node_id → ignore for gating
                continue

            if node_id not in expected:
                # Not a proxy for THIS round → ignore
                continue

            # Update maps (idempotent)
            self._cid_to_node[cid] = node_id
            self._node_to_cid[node_id] = cid

        return all_clients


    def configure_fit(self, server_round: int, parameters, client_manager):
        import os, time
        from pathlib import Path

        expected_ids = [f"s{i}_g{server_round}" for i in range(self.num_servers)]
        logger.info(f"[Cloud Round {server_round}] ENTER configure_fit (expected={expected_ids})")

        # --- Settings (env/attrs) ---
        timeout_sec = getattr(self, "proxy_wait_timeout_sec", 120)
        adaptive_extend_sec = int(os.getenv("CLOUD_ADAPTIVE_EXTEND_SEC", "600"))
        log_every = float(os.getenv("CLOUD_HEARTBEAT_LOG_EVERY_SEC", "5.0"))
        sleep_step = float(os.getenv("CLOUD_WAIT_POLL_SEC", "0.5"))

        # Optional guard rails via files
        shutdown_file = Path("signals/shutdown_requested.signal")
        crash_file = Path("signals/cloud_crashed.signal")
        signals_dir = Path("signals")
        round_dir = signals_dir / f"round_{server_round}"

        deadline = None if timeout_sec is None or timeout_sec <= 0 else (time.monotonic() + timeout_sec)
        last_log = 0.0
        connected_once = set()
        last_present_count = -1
        last_missing_count = -1
        last_done_servers = tuple()

        def _server_completions():
            done = []
            for i in range(self.num_servers):
                f = round_dir / f"server_{i}_completion.signal"
                if f.exists():
                    done.append(i)
            return tuple(done)

        def _snapshot_presence():
            # (Re)build node_id maps if possible. If props aren’t supported, maps may stay empty.
            try:
                _ = self._refresh_properties(client_manager, server_round=server_round)
            except Exception as e:
                logger.debug(f"[Cloud Round {server_round}] refresh_properties failed: {e}")
            node_to_cid = getattr(self, "_node_to_cid", {})
            present = [nid for nid in expected_ids if nid in node_to_cid]
            missing = [nid for nid in expected_ids if nid not in node_to_cid]
            all_clients = client_manager.all() or {}
            return all_clients, present, missing

        # Helper: count per-round proxy heartbeats (created by proxies before dialing)
        def _hb_ready():
            hb_dir = signals_dir / "proxies"
            try:
                need = {f"s{i}_g{server_round}.hb" for i in range(self.num_servers)}
                have = set(p.name for p in hb_dir.glob(f"s*_g{server_round}.hb"))
                return need.issubset(have)
            except Exception:
                return False

        # ── Wait loop (with adaptive extension on first-time connects) ─────────
        while True:
            now = time.monotonic()
            available, present, missing = _snapshot_presence()
            hb_all = _hb_ready()
            newly_seen = [nid for nid in present if nid not in connected_once]
            if newly_seen:
                connected_once.update(newly_seen)
                if deadline is not None:
                    deadline += adaptive_extend_sec
                    logger.info(
                        f"[Cloud Round {server_round}] progress: {len(connected_once)}/{self.num_servers} proxies connected; "
                        f"extended deadline by {adaptive_extend_sec}s"
                    )

            present_count = len(present)
            missing_count = len(missing)
            done_servers = _server_completions()

            # Rate-limited heartbeat
            if (
                present_count != last_present_count
                or missing_count != last_missing_count
                or done_servers != last_done_servers
                or (log_every > 0 and (now - last_log) >= log_every)
            ):
                logger.info(
                    f"[Cloud Round {server_round}] heartbeat: present={present_count} "
                    f"all_clients={len(available)} missing={missing_count} "
                    f"(server_done={list(done_servers)})"
                )
                last_log = now
                last_present_count = present_count
                last_missing_count = missing_count
                last_done_servers = done_servers

            # Exit condition A: all expected proxies identified via node_id
            if not missing and present_count == self.num_servers:
                logger.info(f"[Cloud Round {server_round}] ALL expected proxies present (node_id)")
                break

            # Exit condition B: all leaf servers for this round are done AND heartbeat files exist
            if set(done_servers) == set(range(self.num_servers)) and hb_all:
                logger.info(f"[Cloud Round {server_round}] All leaf servers done and heartbeat files present")
                break

            # Abort flags / timeout
            if shutdown_file.exists():
                raise RuntimeError(f"Round {server_round}: shutdown requested; still missing → {missing}")
            if crash_file.exists():
                raise RuntimeError(f"Round {server_round}: crash flag present; still missing → {missing}")
            if deadline is not None and now >= deadline:
                raise RuntimeError(
                    f"Round {server_round}: proxies missing after {timeout_sec}s (+{adaptive_extend_sec}s per progress) → {missing}"
                )

            time.sleep(sleep_step)

        # ── Selection (prefer node_id mapping; fallback allowed only if HB files exist) ───
        try:
            _ = self._refresh_properties(client_manager, server_round=server_round)
        except Exception:
            pass

        node_to_cid = getattr(self, "_node_to_cid", {})
        selected_cids = [node_to_cid[nid] for nid in sorted(expected_ids) if nid in node_to_cid]

        clients = []
        if len(selected_cids) == self.num_servers:
            # Perfect match via properties
            for cid in selected_cids:
                cp = client_manager.get(cid)
                if cp is not None:
                    clients.append(cp)
        else:
            # Only allow anonymous fallback if HB files say proxies for THIS round exist
            if _hb_ready():              
                pool_map = (client_manager.all() or {})
                if len(pool_map) < self.num_servers:
                    logger.error(f"[Cloud Round {server_round}] Not enough connected clients for fallback: {len(pool_map)} available")
                    return []
                # Prefer newest connections if cids are monotonic / comparable
                try:
                    ordered_cids = sorted(pool_map.keys(), reverse=True)
                    pool = [pool_map[cid] for cid in ordered_cids]
                except Exception:
                    pool = list(pool_map.values())
                clients = pool[: self.num_servers]

                
                logger.warning(
                    f"[Cloud Round {server_round}] Using anonymous selection of {len(clients)} clients "
                    f"(node_id not visible, but heartbeat files confirm round presence)"
                )
            else:
                logger.info(
                    f"[Cloud Round {server_round}] Waiting for per-round proxies to appear "
                    f"(no node_id, no heartbeats yet)"
                )
                # No clients selected this tick → ask Flower to try again shortly
                return []

        # Build per-round config WITHOUT calling parent.configure_fit (avoids double dispatch)
        if hasattr(self, "on_fit_config_fn") and self.on_fit_config_fn is not None:
            try:
                base_config = self.on_fit_config_fn(server_round)
            except TypeError:
                # Older Flower may not pass the round
                base_config = self.on_fit_config_fn()
        else:
            base_config = {}

        fit_ins = FitIns(parameters=parameters, config=dict(base_config))
        selected = [(cp, fit_ins) for cp in clients]

        logger.info(
            f"[Cloud Round {server_round}] ⚡ DISPATCH: sending FitIns to {len(selected)}/{self.num_servers} proxies"
        )
        return selected





    def aggregate_fit(self, server_round: int, results, failures):
        """✅ OPTIMIZED: Dynamic clustering every round, fixed call signature + dual signal paths."""
        logger.info(f"[Cloud Round {server_round}] aggregate_fit called: results={len(results)} failures={len(failures)}")
        if not results:
            logger.error(
                f"[Cloud Round {server_round}] No results to aggregate. "
                f"This usually means proxies disconnected before transmitting results. "
                f"Check: (1) proxy_client.py timing issues, (2) network connectivity, "
                f"(3) CLOUD_WAIT_CLIENTS_SEC environment variable."
            )
            return None

        logger.info(f"[Cloud Round {server_round}] Aggregating {len(results)} server results")

        # 1) Standard FedAvg aggregation via parent
        aggregation_start = time.time()
        parent_result = super().aggregate_fit(server_round, results, failures)
        aggregation_time = time.time() - aggregation_start

        if parent_result is None:
            logger.error(f"[Cloud Round {server_round}] Parent aggregation failed")
            return None

        aggregated, metrics = parent_result
        aggregated_ndarrays = parameters_to_ndarrays(aggregated)
        logger.info(f"[Cloud Round {server_round}] FedAvg aggregation completed in {aggregation_time:.2f}s")

        # 2) Prepare directories
        models_dir = PROJECT_ROOT / "models"

        models_dir.mkdir(parents=True, exist_ok=True)

        rdir = _round_dir(server_round)
        gdir = rdir / "global"
        cdir = rdir / "cloud"
        gdir.mkdir(parents=True, exist_ok=True)
        cdir.mkdir(parents=True, exist_ok=True)

        # 3) Save global model
        global_path = gdir / "model.pkl"
        with open(global_path, "wb") as f:
            pickle.dump(aggregated_ndarrays, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        shutil.copy2(global_path, models_dir / f"model_global_g{server_round}.pkl")
        logger.debug(f"[Cloud Round {server_round}] Global model saved to {global_path}")

        # 4) Dynamic clustering
        if self.clustering_enabled:
            should_cluster = (
                server_round >= self.cluster_start and
                (server_round - self.cluster_start) % self.cluster_frequency == 0
            )

            if should_cluster and len(results) >= 2:
                try:
                    clustering_start = time.time()
                    logger.info(f"[Cloud Round {server_round}] ✅ DYNAMIC CLUSTERING: Performing similarity-based clustering")

                    # --- Build clustering inputs ---
                    # ───────────────────────────────────────────────────────────────
                    # --- Build clustering inputs (STRICT: require metrics['server_id']) ---
                    server_ids: List[int] = []
                    server_models: Dict[int, List[np.ndarray]] = {}
                    server_examples: Dict[int, int] = {}

                    for proxy, pr in results:
                        sid = pr.metrics.get("server_id")
                        if sid is None:
                            raise RuntimeError("Missing metrics['server_id'] in FitRes (strict mode).")
                        if isinstance(sid, str):
                            if not sid.isdigit():
                                raise RuntimeError(f"Non-numeric metrics['server_id']={sid!r} (strict mode).")
                            sid = int(sid)
                        elif not isinstance(sid, (int, np.integer)):
                            raise RuntimeError(f"Invalid metrics['server_id'] type={type(sid).__name__} (strict mode).")

                        sid = int(sid)
                        server_ids.append(sid)
                        server_models[sid] = parameters_to_ndarrays(pr.parameters)
                        server_examples[sid] = pr.num_examples

                    server_weights_list = [server_models[sid] for sid in server_ids]
                    global_weights = aggregated_ndarrays



                    # --- Correct call signature ---
                    labels, S, _ = cifar10_weight_clustering(
                        server_weights_list=server_weights_list,
                        global_weights=global_weights,
                        reference_imgs=None,
                        round_num=server_round,
                        tau=self.cluster_tau,
                        stability_history=None,
                    )

                    assignments = {sid: int(labels[i]) for i, sid in enumerate(server_ids)}
                    clustering_time = time.time() - clustering_start
                    labs = sorted(set(assignments.values()))
                    logger.info(
                        f"[Cloud Round {server_round}] Clustering completed: {len(labs)} clusters, "
                        f"{len(server_ids)} servers, time={clustering_time:.2f}s"
                    )

                    # --- Save cluster assignments ---
                    assign_csv = cdir / f"clusters_g{server_round}.csv"
                    with open(assign_csv, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["server_id", "cluster"])
                        for sid in server_ids:
                            w.writerow([sid, assignments[sid]])
                    
                    # NEW: also write JSON so net_topo can resolve previous cluster per server
                    assign_json = cdir / f"clusters_g{server_round}.json"
                    with open(assign_json, "w") as jf:
                        json.dump({"assignments": {str(sid): int(assignments[sid]) for sid in server_ids}}, jf)
                    shutil.copy2(assign_json, models_dir / assign_json.name)

                    # --- Save similarity matrix (optional) ---
                    try:
                        sim_csv = cdir / f"similarity_g{server_round}.csv"
                        with open(sim_csv, "w", newline="") as f:
                            w = csv.writer(f)
                            w.writerow([""] + [f"server_{sid}" for sid in server_ids])
                            for i, sid_i in enumerate(server_ids):
                                row = [f"server_{sid_i}"] + [f"{S[i, j]:.6f}" for j in range(len(server_ids))]
                                w.writerow(row)
                        self._last_similarity_matrix = S
                    except Exception as e:
                        logger.warning(f"[Cloud Round {server_round}] Could not write similarity CSV: {e}")

                    # --- Create per-cluster aggregated models ---
                    cluster_creation_start = time.time()
                    for lab in labs:
                        members = [sid for sid, labv in assignments.items() if labv == lab]
                        if not members:
                            continue

                        total_w = sum(max(1, int(server_examples.get(sid, 0))) for sid in members)

                        cluster_model = None
                        for sid in members:
                            w = max(1, int(server_examples.get(sid, 0))) / float(total_w)
                            weights = server_models[sid]
                            if cluster_model is None:
                                cluster_model = [w * np.copy(arr) for arr in weights]
                            else:
                                for i in range(len(cluster_model)):
                                    cluster_model[i] += w * weights[i]

                        cpath = cdir / f"model_cluster{lab}_g{server_round}.pkl"
                        with open(cpath, "wb") as f:
                            pickle.dump(cluster_model, f, protocol=pickle.HIGHEST_PROTOCOL)
                            f.flush()
                            os.fsync(f.fileno())
                        shutil.copy2(cpath, models_dir / cpath.name)
                        logger.info(f"[Cloud Round {server_round}] Cluster {lab} model saved: {cpath.name}")

                    cluster_creation_time = time.time() - cluster_creation_start
                    logger.debug(f"[Cloud Round {server_round}] Cluster models created in {cluster_creation_time:.2f}s")

                except Exception as e:
                    logger.error(
                        f"[Cloud Round {server_round}] Clustering failed: {e.__class__.__name__}: {e}",
                        exc_info=True,
                    )

        # 5) Completion signals
        round_signal = cdir / f"cloud_completed_g{server_round}.signal"
        _write_signal(round_signal, f"Round {server_round} completed")

        # also write one in signals/ for net_topo
        signals_dir = _signals_dir()
        net_topo_signal = signals_dir / f"cloud_round_{server_round}_completed.signal"
        _write_signal(net_topo_signal, f"Round {server_round} completed")

        logger.info(f"[Cloud Round {server_round}] ✅ Completion signals created")

        return parent_result

        
    def aggregate_evaluate(self, server_round: int, results, failures):
        """
        ✅ FIXED: Simplified - proxy evaluations not needed.
        
        Servers already evaluated on their own shards before uploading.
        Proxy evaluating cloud's global model on shards is incorrect.
        """
        if failures:
            logger.warning(f"[Cloud Round {server_round}] {len(failures)} proxy eval failures (ignored)")
        
        if not results:
            logger.debug(f"[Cloud Round {server_round}] No evaluation results (expected - proxies don't eval)")
            return None, {}
        
        # If we get results, log them but don't use for anything critical
        logger.debug(f"[Cloud Round {server_round}] Received {len(results)} eval results (informational only)")
        
        return None, {}

# ─────────────────────── Server Runner ───────────────────────
def run_server():
    logger.info("[Cloud Server] Starting long-running cloud aggregation server")
    
    # Signals
    sigdir = _signals_dir()
    _write_signal(sigdir / "cloud_started.signal", "Cloud server started")
    logger.info("[Cloud Server] Start signal created")
    
    # Configuration
    total_rounds = int(os.getenv("TOTAL_GLOBAL_ROUNDS", os.getenv("ROUNDS", "3")))
    num_servers = int(os.getenv("NUM_SERVERS", "2"))
    logger.info(f"[Cloud Server] Configuration: {total_rounds} rounds, {num_servers} servers")
    
    cfg_path = PROJECT_ROOT / "pyproject.toml"

    if not cfg_path.exists():
        raise FileNotFoundError("pyproject.toml not found")
    
    cfg = toml.load(cfg_path)
    hier = cfg.get("tool", {}).get("flwr", {}).get("hierarchy", {})
    cloud_cluster_cfg = cfg.get("tool", {}).get("flwr", {}).get("cloud_cluster", {})
    
    cloud_port = int(hier.get("cloud_port", os.getenv("CLOUD_PORT", 6000)))
    
    # Initial parameters (fresh model)
    init_model = Net()
    init_params = ndarrays_to_parameters(get_weights(init_model))
    logger.info("[Cloud Server] Fresh model parameters initialized")  
    
    strategy = CloudFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,          # ✅ No proxy eval
        min_fit_clients=num_servers,
        min_evaluate_clients=0,         # ✅ No eval clients
        min_available_clients=num_servers,
        initial_parameters=init_params,
        cloud_cluster_cfg=cloud_cluster_cfg,
        accept_failures=False,  # ← force waiting for all 3 proxies
        num_servers=num_servers,         # <-- add this
    )
    # ✅ No evaluate_fn (their suggestion)

    # Wait policy: <=0 means infinite; otherwise finite with adaptive extension in configure_fit()
    try:
        strategy.proxy_wait_timeout_sec = int(os.getenv("CLOUD_WAIT_CLIENTS_SEC", "600"))
    except Exception:
        strategy.proxy_wait_timeout_sec = 600
    logger.info(
        f"[Cloud Server] Proxy wait timeout set to {strategy.proxy_wait_timeout_sec}s "
        f"({'infinite' if strategy.proxy_wait_timeout_sec <= 0 else 'finite'})"
    )

    server_address = f"0.0.0.0:{cloud_port}"
    logger.info(f"[Cloud Server] Binding Flower to {server_address} (num_rounds={total_rounds})")

    # Write a PID file so we can confirm liveness later
    try:
        (_signals_dir() / "cloud_server.pid").write_text(str(os.getpid()))
    except Exception:
        pass

    try:
        history = start_server(
            server_address=server_address,
            config=ServerConfig(num_rounds=total_rounds, round_timeout=None),
            strategy=strategy,
        )
        logger.info("[Cloud Server] start_server() returned normally")
        return history
    except Exception as e:
        import traceback
        crash_path = _signals_dir() / "cloud_crashed.signal"
        msg = f"Cloud crashed: {e}\n{traceback.format_exc()}"
        logger.error(msg)
        _write_signal(crash_path, msg)
        raise
    finally:
        logger.info("[Cloud Server] EXIT run_server()")

    
    return history


if __name__ == "__main__":
    run_server()

