from pathlib import Path

def get_signals_dir(project_root: Path) -> Path:
    """Get or create the global signals directory (for cloud start/complete signals)."""
    signals_dir = project_root / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    return signals_dir

def get_global_rounds_dir(project_root: Path) -> Path:
    """Get or create the base directory for global rounds."""
    rounds_dir = project_root / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    return rounds_dir

def get_leaf_rounds_dir(project_root: Path) -> Path:
    """Get or create the base directory for leaf rounds."""
    rounds_dir = project_root / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    return rounds_dir

def get_global_round_dir(project_root: Path, round_num: int) -> Path:
    """
    Get or create the directory for a specific global round.
    Uses consistent rounds/round_X/global/ structure everywhere.
    """
    # Use consistent rounds/round_X/global/ structure
    round_dir = get_global_rounds_dir(project_root) / f"round_{round_num}" / "global"
    round_dir.mkdir(parents=True, exist_ok=True)
    return round_dir

def get_leaf_round_dir(project_root: Path, round_num: int) -> Path:
    """
    Get or create the directory for a specific leaf round.
    Uses consistent rounds/round_X/leaf/ structure everywhere.
    """
    # Use consistent rounds/round_X/leaf/ structure
    round_dir = get_leaf_rounds_dir(project_root) / f"round_{round_num}" / "leaf"
    round_dir.mkdir(parents=True, exist_ok=True)
    return round_dir

# ──────────────────────────────────────────────────────────────────────────────
#  NEW flat-directory helpers (2025-06 refactor)
# ──────────────────────────────────────────────────────────────────────────────

def leaf_server_dir(project_root: Path, server_id: int, global_round: int = 0) -> Path:
    """Return <root>/rounds/round_X/leaf/server_<sid> and ensure it exists."""
    round_dir = get_leaf_rounds_dir(project_root) / f"round_{global_round}" / "leaf"
    d = round_dir / f"server_{server_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def global_dir(project_root: Path, global_round: int = 0) -> Path:
    """Return <root>/rounds/round_X/global and ensure it exists."""
    round_dir = get_global_rounds_dir(project_root) / f"round_{global_round}" / "global"
    round_dir.mkdir(parents=True, exist_ok=True)
    return round_dir

# ──────────────────────────────────────────────────────────────────────────────
#  Legacy helpers (kept for backward compatibility; do NOT use in new code)
# ──────────────────────────────────────────────────────────────────────────────

def get_server_dir(project_root: Path, round_num: int, server_id: int) -> Path:
    """DEPRECATED: Use leaf_server_dir instead."""
    # Forward round_num to new helper (no longer ignored)
    return leaf_server_dir(project_root, server_id, round_num)

def get_proxy_dir(project_root: Path, round_num: int, proxy_id: int) -> Path:
    """DEPRECATED: Use flat helpers instead."""
    """Get or create the proxy directory for a given round and proxy ID."""
    round_dir = get_leaf_round_dir(project_root, round_num)
    proxy_dir = round_dir / "proxies" / f"proxy_{proxy_id}"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    return proxy_dir

def get_logs_dir(project_root: Path, round_num: int) -> Path:
    """Get or create the logs directory for a given round."""
    logs_dir = project_root / "logs" / f"round_{round_num}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

# Duplicate function definitions removed - use the ones defined above (lines 21-39)
