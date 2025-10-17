"""Utility to compute raw byte size of a list of NumPy ndarrays.

We deliberately ignore pickle/gRPC framing; the Flower *message_size_mod*
records the wire-exact size.  This helper is only for round-level metrics.
"""
from typing import List
import numpy as np

__all__ = ["raw_bytes"]

def raw_bytes(arrs: List[np.ndarray]) -> int:  # type: ignore[name-defined]
    """Return the sum of ``.nbytes`` for every ndarray in *arrs*.

    Parameters
    ----------
    arrs : list of np.ndarray
        The parameter tensors.

    Returns
    -------
    int
        Total bytes (float).
    """
    return int(sum(a.nbytes for a in arrs))
