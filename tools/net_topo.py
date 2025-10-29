#!/usr/bin/env python3
# tools/net_topo.py — Mininet driver for hierarchical (Fedge) FL
#
# Finalized (2025-10-19):
# - Feed previous round model into leaves (clustered first, else global)
# - Clear per-round signal dir before each round (avoid stale files)
# - Hard readiness gate: wait until leaf servers LISTEN on port
# - Generate partitions once
# - No placeholders or duplicate helpers

import os
import sys
import time
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mininet.net import Mininet
from mininet.node import OVSController, Host
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.cli import CLI

# ───────────────────────── Project paths / config ─────────────────────────
PROJECT   = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
ROUND_DIR = PROJECT / "rounds"
SIGNALS   = PROJECT / "signals"
LOG       = PROJECT  # logs in repo root

CLOUD_PY        = PROJECT / "cloud_flower.py"
LEAF_SERVER_PY  = PROJECT / "fedge" / "leaf_server.py"
LEAF_CLIENT_PY  = PROJECT / "fedge" / "leaf_client.py"
PROXY_CLIENT_PY = PROJECT / "fedge" / "proxy_client.py"

# Python interpreter resolution
VENV_PATH = os.getenv("VENV_PATH", "")
venv_py   = str(Path(VENV_PATH) / "bin" / "python") if VENV_PATH else ""
PY        = os.getenv("PYTHON_EXE", venv_py or sys.executable)

SCENARIO     = os.getenv("SCENARIO", "uniform_good")
DATASET_FLAG = os.getenv("DATASET_FLAG", "cifar10")
_active_processes = []  # Track all spawned processes
_log_handles = {}       # Track file handles to prevent leaks
CLOUD_PID: Optional[int] = None

# ───────────────────────── Load hierarchy from env or toml ────────────────
def _load_hier():
    num_servers = int(os.getenv("NUM_SERVERS", "2"))
    cps_env     = os.getenv("CLIENTS_PER_SERVER", "5")
    cps = [int(x) for x in cps_env.split(",")] if "," in cps_env else int(cps_env)
    cfg = {
        "num_servers": num_servers,
        "clients_per_server": cps,
        "cloud_port": int(os.getenv("CLOUD_PORT", "6000")),
        "server_base_port": int(os.getenv("SERVER_BASE_PORT", "5000")),
        "global_rounds": int(os.getenv("GLOBAL_ROUNDS", os.getenv("ROUNDS", "3"))),
        "server_rounds_per_global": int(os.getenv("SERVER_ROUNDS", "1")),
        "local_epochs": int(os.getenv("LOCAL_EPOCHS", "1")),
    }
    try:
        try:
            import tomllib
            data = tomllib.loads((PROJECT / "pyproject.toml").read_text())
        except Exception:
            import toml
            data = toml.load(PROJECT / "pyproject.toml")
        H = data.get("tool", {}).get("flwr", {}).get("hierarchy", {})
        def _pick(k, default):
            return H.get(k, cfg[k])
        cfg = {
            "num_servers": int(_pick("num_servers", cfg["num_servers"])),
            "clients_per_server": _pick("clients_per_server", cfg["clients_per_server"]),
            "cloud_port": int(_pick("cloud_port", cfg["cloud_port"])),
            "server_base_port": int(_pick("server_base_port", cfg["server_base_port"])),
            "global_rounds": int(_pick("global_rounds", cfg["global_rounds"])),
            "server_rounds_per_global": int(_pick("server_rounds_per_global", cfg["server_rounds_per_global"])),
            "local_epochs": int(_pick("local_epochs", cfg["local_epochs"])),
        }
    except Exception:
        pass
    return cfg

HIER = _load_hier()
NUM_SERVERS   = int(HIER["num_servers"])
CPS           = HIER["clients_per_server"]
CLOUD_PORT    = int(HIER["cloud_port"])
SERVER_BASE   = int(HIER["server_base_port"])
GLOBAL_ROUNDS = int(HIER["global_rounds"])
SERVER_ROUNDS = int(HIER["server_rounds_per_global"])
LOCAL_EPOCHS  = int(HIER["local_epochs"])

# ───────────────────────── Scenarios ──────────────────────────────────────
SCENARIOS = {
    "uniform_good": {
        "srv":    {"bw": 100, "delay": "5ms",  "loss": 0},
        "client": {"bw":  50, "delay": "10ms", "loss": 0},
        "note": "Fast/clean everywhere",
    },
    "jitter": {
        "srv":    {"bw": 50, "delay": "20ms", "loss": 0},
        "client": {"bw": 20, "delay": "50ms", "loss": 1},
        "client_jitter": {
            "bw_mbit":  [1.0, 50.0],
            "delay_ms": [10, 300],
            "loss_pct": [0.0, 10.0],
        },
        "note": "Each client link fluctuates (once per round)",
    },
}

# ───────────────────────── Helpers ────────────────────────────────────────
def env_for(role: str, extra: Dict[str, str] = None) -> Dict[str, str]:
    e = os.environ.copy()
    e["ROLE"] = role
    e["PYTHONPATH"] = str(PROJECT)
    e["DATASET_FLAG"] = DATASET_FLAG
    e["PYTHONUNBUFFERED"] = "1"
    if extra:
        e.update({k: str(v) for k, v in extra.items()})
    return e

    
def popen(h: Host, cmd: List[str], logf: Path, env: Dict[str, str]) -> subprocess.Popen:
    """Tracked popen that prevents zombies and file handle leaks."""
    logf.parent.mkdir(parents=True, exist_ok=True)
    out = open(logf, "w", buffering=1)
    err = open(str(logf) + ".err", "w", buffering=1)
    
    proc = h.popen(cmd, stdout=out, stderr=err, env=env)
    
    # ✅ Track process and file handles
    _active_processes.append(proc)
    _log_handles[proc.pid] = (out, err)
    
    return proc    
    

def wait_for(path: Path, timeout: int) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists():
            return True
        time.sleep(1)
    return path.exists()
    
def reap_finished_processes():
    """Reap any finished processes to prevent zombies (like orchestrator.py fix)."""
    for proc in list(_active_processes):
        if proc.poll() is not None:
            # ✅ Process finished - reap it immediately!
            try:
                proc.wait(timeout=0)
            except Exception:
                pass
            
            # Close file handles
            if proc.pid in _log_handles:
                out, err = _log_handles[proc.pid]
                try:
                    out.close()
                    err.close()
                except Exception:
                    pass
                del _log_handles[proc.pid]
            
            # Remove from tracking
            try:
                _active_processes.remove(proc)
            except ValueError:
                pass

def cleanup_round_processes(keep_cloud: bool = True):
    """Clean up processes from current round. Optionally keep the cloud process alive."""
    global CLOUD_PID

    cloud_proc = None
    # Make a copy since we'll remove during iteration
    for proc in list(_active_processes):
        pid = getattr(proc, "pid", None)

        # Keep cloud if requested and pid matches the recorded CLOUD_PID
        if keep_cloud and CLOUD_PID is not None and pid == CLOUD_PID:
            cloud_proc = proc
            continue

        # Terminate non-cloud (or all if keep_cloud=False)
        try:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=1)
                    except Exception:
                        pass
            else:
                # Already finished - reap return code immediately
                try:
                    proc.wait(timeout=0)
                except Exception:
                    pass
        finally:
            # Close file handles
            if pid in _log_handles:
                out, err = _log_handles.pop(pid, (None, None))
                try:
                    if out:
                        out.close()
                    if err:
                        err.close()
                except Exception:
                    pass
            # Remove from tracking
            try:
                _active_processes.remove(proc)
            except ValueError:
                pass

    # Restore tracking to include only the cloud if we kept it
    if keep_cloud and cloud_proc:
        _active_processes.clear()
        _active_processes.append(cloud_proc)
    elif not keep_cloud:
        # If we’re killing everything, also clear CLOUD_PID
        CLOUD_PID = None


def cloud_round_complete(g: int) -> Path:
    return SIGNALS / f"cloud_round_{g}_completed.signal"

def server_complete(g: int, sid: int) -> Path:
    # fedge.utils.fs_optimized.create_completion_signal writes exactly this path
    return SIGNALS / f"round_{g}" / f"server_{sid}_completion.signal"

def server_error(g: int, sid: int) -> Path:
    return SIGNALS / f"round_{g}" / f"server_{sid}_error.signal"

def _cps_for_sid(sid: int) -> int:
    return CPS[sid] if isinstance(CPS, list) else int(CPS)

# ───────────────────────── Partitions (generate once) ─────────────────────
def ensure_partitions():
    parts = ROUND_DIR / "partitions.json"
    if parts.exists():
        return
    code = r"""
import os, json
from pathlib import Path
from fedge.task import load_cifar10_hf
from fedge.partitioning import hier_dirichlet_indices, write_partitions
d = load_cifar10_hf(seed=42)
num_servers = int(os.getenv("NUM_SERVERS", "2"))
_cps = os.getenv("CLIENTS_PER_SERVER", "5")
clients_per_server = [int(x) for x in _cps.split(",")] if "," in _cps else int(_cps)
(out := Path('rounds')).mkdir(exist_ok=True)
mapping = hier_dirichlet_indices(labels=d.train_labels, num_servers=num_servers, clients_per_server=clients_per_server)
write_partitions(out / 'partitions.json', mapping)
print('Wrote', out / 'partitions.json')
"""
    env = env_for("partgen")
    env["NUM_SERVERS"] = str(NUM_SERVERS)
    env["CLIENTS_PER_SERVER"] = ",".join(map(str, CPS)) if isinstance(CPS, list) else str(CPS)
    subprocess.run([PY, "-u", "-c", code], env=env, check=True, cwd=str(PROJECT))

# ───────────────────────── Mininet build ──────────────────────────────────
def build_net():
    net = Mininet(controller=OVSController, link=TCLink, autoSetMacs=True, autoStaticArp=True)
    net.addController("c0")
    s = net.addSwitch("s1")
    cloud = net.addHost("cloud")
    srvs = [net.addHost(f"srv{i+1}") for i in range(NUM_SERVERS)]
    clients = []
    net.addLink(cloud, s)
    for h in srvs:
        net.addLink(h, s)
    for sid in range(NUM_SERVERS):
        for cid in range(_cps_for_sid(sid)):
            h = net.addHost(f"c{sid}_{cid}")
            net.addLink(h, s)
            clients.append((sid, cid, h))
    net.start()
    sc = SCENARIOS[SCENARIO]
    for h in srvs:
        h.defaultIntf().config(**sc["srv"])
    for (_, _, h) in clients:
        h.defaultIntf().config(**sc["client"])
    cloud_ip = cloud.IP()
    srv_ips = [h.IP() for h in srvs]
    return net, cloud, srvs, clients, cloud_ip, srv_ips, sc

# ───────────────────────── Jitter once per round ─────────────────────────
def apply_jitter_once(clients, jitter_cfg, round_idx):
    if not jitter_cfg:
        return
    lo_bw, hi_bw = jitter_cfg["bw_mbit"]
    lo_d, hi_d   = jitter_cfg["delay_ms"]
    lo_l, hi_l   = jitter_cfg["loss_pct"]
    print(f"[Jitter] Applying random link conditions for round {round_idx}")
    for (_, _, h) in clients:
        bw  = round(random.uniform(lo_bw, hi_bw), 2)
        dms = random.randint(lo_d, hi_d)
        lpc = round(random.uniform(lo_l, hi_l), 2)
        try:
            h.defaultIntf().config(bw=bw, delay=f"{dms}ms", loss=lpc)
        except Exception:
            pass

# ───────────────────────── LR_INIT: env first, then TOML ─────────────────
def _lr_init() -> str:
    if os.getenv("LR_INIT"):
        return os.getenv("LR_INIT")
    try:
        try:
            import tomllib as _toml
            data = _toml.loads((PROJECT / "pyproject.toml").read_text())
        except Exception:
            import toml as _toml
            data = _toml.load(PROJECT / "pyproject.toml")
        return str(data["tool"]["flwr"]["hierarchy"].get("lr_init", "0.01"))
    except Exception:
        return "0.01"

# ───────────────────────── Previous model resolver (CRITICAL) ────────────
def prev_model_path_for_sid(g: int, sid: int) -> Optional[str]:
    """
    Return the correct model path for server `sid` at round `g`.

    Priority order:
      1. Its own cluster model from previous round (using clusters_g{g-1}.json)
      2. Any available cluster model from the previous round (fallback)
      3. The global model from previous round
      4. The per-round global/model.pkl if global model not found
    """
    if g <= 1:
        return None

    rounds_cloud_dir = ROUND_DIR / f"round_{g-1}" / "cloud"
    models_dir = PROJECT / "models"

    import json
    cluster_json_candidates = [
        models_dir / f"clusters_g{g-1}.json",
        rounds_cloud_dir / f"clusters_g{g-1}.json",
    ]
    for meta_path in cluster_json_candidates:
        if meta_path.is_file():
            try:
                data = json.loads(meta_path.read_text())
                assign = data.get("assignments", {})
                cid = assign.get(str(sid)) if isinstance(assign, dict) else None
                if cid is None:
                    cid = assign.get(sid)
                if cid is not None:
                    cpath = rounds_cloud_dir / f"model_cluster{cid}_g{g-1}.pkl"
                    if cpath.is_file():
                        return str(cpath)
            except Exception as e:
                print(f"[prev_model_path_for_sid] Warning: failed to read {meta_path}: {e}")
            break

    for p in sorted(rounds_cloud_dir.glob(f"model_cluster*_g{g-1}.pkl")):
        if p.is_file():
            return str(p)

    g1 = models_dir / f"model_global_g{g-1}.pkl"
    if g1.is_file():
        return str(g1)

    g2 = rounds_cloud_dir.parent / "global" / "model.pkl"
    if g2.is_file():
        return str(g2)

    return None


# ───────────────────────── Process launchers ──────────────────────────────
    
def launch_cloud(hcloud: Host, cloud_ip: str):
    """Launch cloud server and WAIT for it to be ready (not just sleep)."""
    SIGNALS.mkdir(exist_ok=True, parents=True)
    env = env_for("cloud", {
        "TOTAL_GLOBAL_ROUNDS": GLOBAL_ROUNDS,
        "NUM_SERVERS": NUM_SERVERS,
        "CLOUD_WAIT_CLIENTS_SEC": "0",           # <=0 means "infinite wait" inside cloud_flower.py
        "CLOUD_HEARTBEAT_LOG_EVERY_SEC": "5.0",  # nicer heartbeat cadence
        "CLOUD_PORT": CLOUD_PORT,
    })
    cmd = [PY, str(CLOUD_PY)]
    
    proc = popen(hcloud, cmd, LOG / "cloud.log", env)
    global CLOUD_PID
    CLOUD_PID = proc.pid
    
    # ✅ Wait for cloud to actually be listening on port (not just sleep!)
    print(f"[cloud] Waiting for cloud server to listen on {cloud_ip}:{CLOUD_PORT}")
    if not wait_port_ns(hcloud, cloud_ip, CLOUD_PORT, timeout=60):
        print(f"ERROR: Cloud server did not start listening on port {CLOUD_PORT}")
        # Try to get error from log
        try:
            with open(LOG / "cloud.log.err", "r") as f:
                err_content = f.read()
                if err_content:
                    print(f"Cloud error log:\n{err_content}")
        except Exception:
            pass
        raise RuntimeError("Cloud server failed to start")
    
    print(f"[cloud] ✓ Cloud server ready on {cloud_ip}:{CLOUD_PORT}")
    return proc
    


def launch_leaf_server(hsrv: Host, sid: int, g: int, srv_ip: str):
    port      = SERVER_BASE + sid
    parts     = str(ROUND_DIR / "partitions.json")
    frac_fit  = os.getenv("FRACTION_FIT", "1.0")
    frac_eval = os.getenv("FRACTION_EVALUATE", "0.2")
    lr_init   = _lr_init()
    init_path = prev_model_path_for_sid(g, sid)

    env = env_for("leaf_server", {
        "SERVER_ID": sid,
        "GLOBAL_ROUND": g,
        "LOCAL_EPOCHS": LOCAL_EPOCHS,
        "PYTHONPATH": str(PROJECT),
        "PARTITIONS_JSON": parts,
        "CLOUD_PORT": CLOUD_PORT,
        "SERVER_BASE_PORT": SERVER_BASE,
        "BIND_ADDRESS": srv_ip,
        "LR_INIT": lr_init,
        # Ensure fractions visible to leaf_server (it reads env again)
        "FRACTION_FIT": frac_fit,
        "FRACTION_EVALUATE": frac_eval,
    })

    cmd = [
        PY, str(LEAF_SERVER_PY),
        "--server_id", str(sid),
        "--clients_per_server", str(_cps_for_sid(sid)),
        "--server_rounds", str(SERVER_ROUNDS),
        "--fraction_fit", frac_fit,
        "--fraction_evaluate", frac_eval,
        "--port", str(port),
        "--global_round", str(g),
    ]
    if init_path:
        cmd += ["--initial_model_path", init_path]

    # Append a one-liner to server stdout for easy grep later
    with open(LOG / f"server{sid}.log", "a") as lf:
        lf.write(f"[Orch] g={g} sid={sid} init_model={init_path}\n")

    return popen(hsrv, cmd, LOG / f"server{sid}.log", env)

def launch_leaf_client(hcli: Host, sid: int, cid: int, g: int, server_ip: str):
    server_addr = f"{server_ip}:{SERVER_BASE + sid}"
    parts = str(ROUND_DIR / "partitions.json")
    env = env_for("leaf_client", {
        "SERVER_ID": sid,
        "CLIENT_ID": f"{sid}_{cid}",
        "GLOBAL_ROUND": g,
        "LOCAL_EPOCHS": LOCAL_EPOCHS,
        "SERVER_ADDR": server_addr,
        "PARTITIONS_JSON": parts,
    })
    cmd = [
        PY, str(LEAF_CLIENT_PY),
        "--server_id", str(sid),
        "--client_id", str(cid),
        "--dataset_flag", DATASET_FLAG,
        "--local_epochs", str(LOCAL_EPOCHS),
        "--server_addr", server_addr,
    ]
    return popen(hcli, cmd, LOG / f"client_{sid}_{cid}.log", env)


def launch_proxy(hsrv: Host, sid: int, g: int, cloud_ip: str):
    proxy_path = PROXY_CLIENT_PY
    if not proxy_path.exists():
        print(f"[proxy-launch] ERROR: Proxy script not found at {proxy_path}")
    else:
        print(f"[proxy-launch] Using proxy script: {proxy_path}")

    env = env_for("proxy", {
        "PROXY_ID": f"proxy_{sid}",
        "SERVER_ID": sid,
        "GLOBAL_ROUND": g,
        "CLOUD_ADDRESS": f"{cloud_ip}:{CLOUD_PORT}",
        "TOTAL_SERVER_ROUNDS_THIS_CLOUD": str(SERVER_ROUNDS),
        "PROXY_WAIT_CLOUD_START_SEC": "300",  # proxy waits up to 5min for cloud start signal
    })
    cmd = [
        PY, "-u", str(proxy_path),
        "--server_id", str(sid),
        "--max_retries", "120",
        "--retry_delay", "2",
        "--cloud_address", f"{cloud_ip}:{CLOUD_PORT}",
        "--global_round", str(g),
    ]
    logf = LOG / f"proxy_{sid}.log"
    print(f"[proxy-launch] cmd={cmd}")
    print(f"[proxy-launch] log={logf}  err={logf}.err")
    # Quick TCP probe from the server namespace (not the host!)
    try:
        res = hsrv.cmd(f"timeout 2 bash -lc 'echo > /dev/tcp/{cloud_ip}/{CLOUD_PORT}' && echo OK || echo FAIL")
        print(f"[proxy-launch] ns-probe {cloud_ip}:{CLOUD_PORT} -> {res.strip()}")
    except Exception as _e:
        print(f"[proxy-launch] ns-probe error: {_e}")

    proc = popen(hsrv, cmd, logf, env)
    try:
        pid = proc.pid
    except Exception:
        pid = None
    print(f"[proxy-launch] spawned sid={sid} pid={pid}")

    # Sanity: ensure logs exist immediately (we opened them before popen)
    try:
        print(f"[proxy-launch] logfile exists? {logf.exists()}  err? {(LOG / f'proxy_{sid}.log.err').exists()}")
    except Exception:
        pass

    return proc


# ───────────────────────── Readiness gate (CRITICAL) ──────────────────────
def wait_port_ns(hsrv: Host, ip: int, port: int, timeout: float = 180.0) -> bool:
    t0 = time.time()
    quick_phase_until = t0 + 5.0
    while time.time() - t0 < timeout:
        out = hsrv.cmd(f"ss -ltn 'sport = :{port}' || true")
        if out.strip():
            lines = [ln.strip() for ln in out.splitlines() if "LISTEN" in ln and f":{port}" in ln]
            for ln in lines:
                if "127.0.0.1" in ln or "[::1]" in ln:
                    continue
                return True
        time.sleep(0.2 if time.time() < quick_phase_until else 0.5)
    return False
   
    
def launch_proxies_synchronized(srvs, srv_ips, g, cloud_ip):
    """Launch proxies and wait for them all to complete (prevent race conditions)."""
    proxy_procs = []
    
    # Launch all proxies
    for sid, hsrv in enumerate(srvs):
        print(f"[proxy] Launching proxy {sid} for round {g}")
        proc = launch_proxy(hsrv, sid, g, cloud_ip)
        proxy_procs.append((sid, proc))
    
    # Wait for all proxies to complete
    timeout = 300  # 5 minutes
    start = time.time()
    completed = set()
    failed = set()
    
    while time.time() - start < timeout and (len(completed) + len(failed) < len(proxy_procs)):
        for sid, proc in list(proxy_procs):
            # Still running?
            rc = proc.poll()
            if rc is None:
                # Extra: confirm process is visible in srv namespace
                # (cheap check: /proc/<pid> exists in host; visibility means it's alive)
                print(f"[proxy-wait] sid={sid} pid={proc.pid} state=RUNNING")
                continue

            # Finished: reap now
            try:
                proc.wait(timeout=0)
            except Exception:
                pass

            if rc == 0:
                completed.add(sid)
                print(f"[proxy] ✓ Proxy {sid} completed successfully")
            else:
                failed.add(sid)
                print(f"[proxy] ✗ Proxy {sid} failed (rc={rc})")

        if len(completed) + len(failed) < len(proxy_procs):
            time.sleep(1)
    
    # Check results
    if failed:
        print(f"ERROR: {len(failed)} proxy(ies) failed: {sorted(failed)}")
        return False
    
    if len(completed) < len(proxy_procs):
        missing = len(proxy_procs) - len(completed)
        print(f"ERROR: Only {len(completed)}/{len(proxy_procs)} proxies completed within timeout ({missing} missing)")
        return False
    
    print(f"[proxy] ✓ All {len(completed)} proxies completed successfully")
    return True


def stop_cloud(proc):
    """Gracefully stop the cloud process if it's still alive."""
    if not proc:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    except Exception:
        pass


# ───────────────────────── Main loop ──────────────────────────────────────
def main():
    setLogLevel("info")
    (PROJECT / "metrics").mkdir(exist_ok=True, parents=True)
    ensure_partitions()
    net, cloud, srvs, clients, cloud_ip, srv_ips, sc = build_net()
    try:
        print(
            "[net] Scenario={}  NUM_SERVERS={}  CPS={}  ROUNDS={}  SERVER_ROUNDS={}  LOCAL_EPOCHS={}".format(
                SCENARIO, NUM_SERVERS, CPS, GLOBAL_ROUNDS, SERVER_ROUNDS, LOCAL_EPOCHS
            )
        )
        cloud_proc = launch_cloud(cloud, cloud_ip)  # ✅ Now waits for ready
        print(f"[cloud] PID={cloud_proc.pid} (watch with: tail -F signals/cloud_crashed.signal)")


        for g in range(1, GLOBAL_ROUNDS + 1):
            if g > 1:
                print("[cleanup] Cleaning up previous round processes")
                cleanup_round_processes(keep_cloud=True)
            
            # ✅ Reap any zombies before starting new round
            reap_finished_processes()
            # Fresh signal dir for this round to avoid stale flags
            rdir = SIGNALS / f"round_{g}"
            if rdir.exists():
                for p in rdir.glob("*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
            rdir.mkdir(parents=True, exist_ok=True)


            # ⬇️ ADD THESE 4 LINES (clear top-level cloud signal for this round)
            cloud_sig = SIGNALS / f"cloud_round_{g}_completed.signal"
            if cloud_sig.exists():
                try: cloud_sig.unlink()
                except Exception: pass
            
            print("\n==== GLOBAL ROUND {}/{} ====".format(g, GLOBAL_ROUNDS))
            if SCENARIO == "jitter":
                apply_jitter_once(clients, SCENARIOS[SCENARIO].get("client_jitter"), g)

            # 1) Start all leaf servers
            for sid, hsrv in enumerate(srvs):
                print("[spawn] srv{} -> {}:{}".format(sid + 1, srv_ips[sid], SERVER_BASE + sid))
                launch_leaf_server(hsrv, sid, g, srv_ips[sid])

            # 2) Wait for servers to listen
            all_ok = True
            for sid, hsrv in enumerate(srvs):
                ip = srv_ips[sid]; port = SERVER_BASE + sid
                is_up = wait_port_ns(hsrv, ip, port, timeout=180.0)
                print("[health] srv{} wait_port {}:{} -> {}".format(sid + 1, ip, port, "UP" if is_up else "DOWN"))
                if not is_up:
                    print("[diag] srv{} ss -ltn:".format(sid + 1))
                    print(hsrv.cmd("ss -ltn || true"))
                    logfile = LOG / ("server{}.log".format(sid))
                    tail_cmd = "tail -n 120 {} 2>/dev/null || true".format(logfile)
                    print("[diag] last 120 lines of {}:".format(logfile))
                    print(hsrv.cmd(tail_cmd))
                    all_ok = False
            if not all_ok:
                print("ERROR: one or more leaf servers never opened their port; dropping to Mininet CLI.")
                stop_cloud(cloud_proc)
                cleanup_round_processes(keep_cloud=False)
                CLI(net); return
                

            # 3) Start all clients
            for sid, cid, h in clients:
                launch_leaf_client(h, sid, cid, g, srv_ips[sid])

            # 4) Wait for leaf servers to complete (or error) this round
            deadline = time.time() + int(os.getenv("LEAF_DEADLINE_SEC", "5400"))
            missing, errored = set(range(NUM_SERVERS)), set()
            while time.time() < deadline and missing:
                for sid in list(missing):
                    if server_complete(g, sid).exists():
                        missing.discard(sid)
                    elif server_error(g, sid).exists():
                        errored.add(sid); missing.discard(sid)
                        print("ERROR: server_{} reported error (see {})".format(sid, server_error(g, sid)))
                if missing:
                    time.sleep(2)

            if missing:
                print("ERROR: leaf servers incomplete for round {}: {}".format(g, sorted(missing)))
                stop_cloud(cloud_proc)
                cleanup_round_processes(keep_cloud=False)
                CLI(net); return
            if errored:
                print("ERROR: {} leaf server(s) errored; dropping to CLI.".format(sorted(errored)))
                stop_cloud(cloud_proc)
                cleanup_round_processes(keep_cloud=False)
                CLI(net); return
                
            # ✅ Small delay to ensure all file system sync operations complete
            time.sleep(5)
            print(f"[round {g}] Leaf servers complete, ensuring file sync...")

            # 5) ✅ Proxies upload to cloud with proper synchronization
            print(f"[round {g}] Launching proxies to upload models to cloud")
            for sid, hsrv in enumerate(srvs):
                print(f"[peek] srv{sid+1} which python: {hsrv.cmd('which python || which python3 || echo NO_PY').strip()}")
                print(f"[peek] srv{sid+1} cwd: {hsrv.cmd('pwd').strip()}")
                print(f"[peek] srv{sid+1} ls fedge/proxy_client.py: {hsrv.cmd('ls -l fedge/proxy_client.py 2>/dev/null || echo missing').strip()}")

            if not launch_proxies_synchronized(srvs, srv_ips, g, cloud_ip):
                print(f"ERROR: Proxy upload failed for round {g}")
                stop_cloud(cloud_proc)
                cleanup_round_processes(keep_cloud=False)
                CLI(net)
                return
            
            # ✅ Reap any finished processes after proxy completion
            reap_finished_processes()

            # 6) Wait for cloud aggregation to finish this round
            if not wait_for(cloud_round_complete(g), 1200):
                print("ERROR: cloud did not complete round {}".format(g))
                stop_cloud(cloud_proc)
                cleanup_round_processes(keep_cloud=False)
                CLI(net); return

            print("✓ Global round {} completed.".format(g))
            reap_finished_processes()

        print("Done. Opening Mininet CLI (type 'exit' to quit).")
        CLI(net)

    finally:
        # ✅ Final cleanup of all processes
        print("[cleanup] Final cleanup of all processes")
        try:
            cleanup_round_processes(keep_cloud=False)  # Kill everything including cloud
        except Exception as e:
            print(f"[cleanup] Warning during cleanup: {e}")
        
        # Close any remaining file handles
        for pid, (out, err) in list(_log_handles.items()):
            try:
                out.close()
                err.close()
            except Exception:
                pass
        _log_handles.clear()
        _active_processes.clear()
        
        net.stop()

if __name__ == "__main__":
    main()

