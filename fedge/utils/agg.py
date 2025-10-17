"""Aggregation helpers for Flower communication‐cost metrics."""
from typing import Dict, List, Tuple, Any

__all__ = ["sum_bytes", "sum_metrics"]


def sum_bytes(results: List[Tuple[int, Dict[str, Any]]]):
    """Aggregate any metric key that starts with ``bytes_``.

    Parameters
    ----------
    results : list
        List of tuples ``(num_examples, metrics_dict)`` exactly as Flower hands
        to ``fit_metrics_aggregation_fn``.

    Returns
    -------
    Dict[str, int]
        Dictionary with summed byte counts.
    """
    agg: Dict[str, int] = {}
    for _, metrics in results:
        if not metrics:
            continue
        for k, v in metrics.items():
            if k.startswith("bytes_"):
                agg[k] = agg.get(k, 0) + int(v)
    return agg


def sum_metrics(results: List[Tuple[int, Dict[str, Any]]]):
    """Aggregate bytes_* plus mean wall-clock and compute seconds.

    Parameters
    ----------
    results : list
        List of tuples ``(num_examples, metrics_dict)`` as Flower passes to
        *fit_metrics_aggregation_fn*.

    Returns
    -------
    Dict[str, Any]
        Aggregated metrics dict.
    """
    total_up: int = 0
    total_down: int = 0
    total_time: float = 0.0
    total_compute: float = 0.0
    n_clients = 0

    for _, metrics in results:
        if not metrics:
            continue
        total_up += int(metrics.get("bytes_up", 0))
        # include bytes_down_eval for completeness
        total_down += int(metrics.get("bytes_down", 0)) + int(metrics.get("bytes_down_eval", 0))

        # average timing over participating clients
        total_time += float(metrics.get("round_time", 0.0))
        total_compute += float(metrics.get("compute_s", 0.0))
        n_clients += 1

    agg: Dict[str, Any] = {
        "bytes_up": total_up,
        "bytes_down": total_down,
    }
    if n_clients:
        agg["round_time"] = total_time / n_clients
        agg["compute_s"] = total_compute / n_clients
    return agg
