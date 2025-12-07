"""
johansen.py

Johansen cointegration helper functions.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def johansen_weights(
    log_prices: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
    r: int = 1,
    normalize_index: int = -1,
) -> np.ndarray:
    """
    Compute cointegrating vector via Johansen test.

    Parameters
    ----------
    log_prices : DataFrame
        Log price series for assets.
    det_order : int
        Deterministic trend order (0: constant, etc.).
    k_ar_diff : int
        Number of lagged differences in the VECM.
    r : int
        Number of cointegration relations. We use the first one.
    normalize_index : int
        Which asset to normalize on (weight = 1).

    Returns
    -------
    weights : ndarray
        Cointegration weights (length = num assets).
    """
    values = log_prices.values
    result = coint_johansen(values, det_order, k_ar_diff)

    # First eigenvector corresponds to strongest relation
    w = result.evec[:, 0]

    # Normalize
    w = w / w[normalize_index]
    return w


def compute_spread(log_prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Linear combination of log prices with given weights.

    spread_t = sum_i w_i * log_price_{i, t}
    """
    spread_values = log_prices.values @ weights
    return pd.Series(spread_values, index=log_prices.index, name="spread")
