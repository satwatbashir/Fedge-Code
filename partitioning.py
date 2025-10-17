"""Hierarchical Dirichlet partition utilities.

This module produces a two-level Dirichlet split compatible with the
HierFL paper/codebase:
    1.  Outer Dirichlet across *servers* (alpha_server)
    2.  Inner Dirichlet across *clients within each server* (alpha_client)

The result is stored as a JSON mapping
    {
        "0": {"0": [..indices..], "1": [...], ...},
        "1": {...},
        ...
    }
where top-level keys are server_ids (str), second-level keys are
client_ids (str) and values are *global* indices into the original train
set.  Indices are plain Python ints so the file is portable and does not
require NumPy when loading.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import toml

__all__ = [
    "hier_dirichlet_indices",
    "write_partitions",
]

def hier_dirichlet_indices(
    labels: Sequence[int] | np.ndarray,
    num_servers: int,
    clients_per_server: Union[int, Sequence[int]],
    *,
    alpha_server: float = None,
    alpha_client: float = None,
    seed: int = None,
) -> Dict[str, Dict[str, List[int]]]:
    """Return a hierarchical mapping of indices for all clients using NumPy only.

    Parameters
    ----------
    labels : Sequence[int] | np.ndarray
        Labels aligned with the training set order (length = num_samples).
    num_servers : int
        Number of leaf servers.
    clients_per_server : int | Sequence[int]
        Number of clients *inside each* leaf server.
    alpha_server/alpha_client : float
        Dirichlet concentration parameters for outer / inner split.
    seed : int
        Random seed for reproducibility.
    """
    # Read configuration from TOML if not provided
    if alpha_server is None or alpha_client is None or seed is None:
        project_root = Path(__file__).resolve().parent.parent
        cfg = toml.load(project_root / "pyproject.toml")
        dirichlet_config = cfg["tool"]["flwr"]["hierarchy"]["dirichlet"]
        
        if alpha_server is None:
            alpha_server = dirichlet_config["alpha_server"]
        if alpha_client is None:
            alpha_client = dirichlet_config["alpha_client"]
        if seed is None:
            seed = dirichlet_config["seed"]
    
    # Standardise clients_per_server into a list per server
    if isinstance(clients_per_server, int):
        cps_list: list[int] = [clients_per_server] * num_servers
    else:
        cps_list = list(clients_per_server)
        if len(cps_list) != num_servers:
            raise ValueError(f"len(clients_per_server)={len(cps_list)} does not match num_servers={num_servers}")
    
    labels_np = np.asarray(labels, dtype=np.int64)
    n = labels_np.shape[0]
    unique_labels = np.unique(labels_np)
    rng_outer = np.random.default_rng(int(seed))

    # ── Outer split: across servers ────────────────────────────────────────
    server_to_indices: list[list[int]] = [[] for _ in range(num_servers)]
    for cls in unique_labels:
        cls_idx = np.where(labels_np == cls)[0]
        rng_outer.shuffle(cls_idx)
        probs = rng_outer.dirichlet([alpha_server] * num_servers)
        counts = rng_outer.multinomial(cls_idx.size, probs)
        start = 0
        for sid, cnt in enumerate(counts):
            if cnt > 0:
                server_to_indices[sid].extend(cls_idx[start:start+cnt].tolist())
            start += cnt
    for sid in range(num_servers):
        rng_outer.shuffle(server_to_indices[sid])

    # ── Inner split: per-server across that server's clients ───────────────
    mapping: Dict[str, Dict[str, List[int]]] = {}
    for sid in range(num_servers):
        s_indices = np.array(server_to_indices[sid], dtype=np.int64)
        s_labels = labels_np[s_indices]
        n_clients = int(cps_list[sid])
        rng_inner = np.random.default_rng(int(seed) + sid)

        client_lists: list[list[int]] = [[] for _ in range(n_clients)]

        if s_indices.size == 0:
            # Edge case: no data for this server – keep all clients empty
            mapping[str(sid)] = {str(cid): [] for cid in range(n_clients)}
            continue

        for cls in np.unique(s_labels):
            cls_abs_idx = s_indices[s_labels == cls]
            rng_inner.shuffle(cls_abs_idx)
            probs = rng_inner.dirichlet([alpha_client] * n_clients)
            counts = rng_inner.multinomial(cls_abs_idx.size, probs)
            start = 0
            for cid, cnt in enumerate(counts):
                if cnt > 0:
                    client_lists[cid].extend(cls_abs_idx[start:start+cnt].tolist())
                start += cnt

        # Shuffle client lists and ensure no empty client if possible
        sizes = [len(lst) for lst in client_lists]
        if any(sz == 0 for sz in sizes):
            # Move a single sample from the largest to any empty client(s)
            for cid, sz in enumerate(sizes):
                if sz == 0:
                    src = int(np.argmax([len(lst) for lst in client_lists]))
                    if len(client_lists[src]) > 0:
                        client_lists[cid].append(client_lists[src].pop())
        for lst in client_lists:
            rng_inner.shuffle(lst)

        mapping[str(sid)] = {str(cid): client_lists[cid] for cid in range(n_clients)}

    return mapping

def write_partitions(path: Path | str, mapping: Dict[str, Dict[str, List[int]]]) -> None:
    """Write JSON mapping to *path* (overwrites if exists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(mapping, fp)
