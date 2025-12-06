"""
data_loader.py

Utilities to download and preprocess price data.
"""

from __future__ import annotations
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(
    tickers: List[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance and return Adj Close prices.

    Parameters
    ----------
    tickers : list of str
        Asset tickers (e.g., ["HDFCBANK.NS", "ICICIBANK.NS"]).
    start : str
        Start date (YYYY-MM-DD).
    end : str or None
        End date (YYYY-MM-DD). None = today.
    interval : str
        Sampling interval ("1d", "1h", etc.)

    Returns
    -------
    prices : DataFrame
        Adjusted close prices with Date index and columns=tickers.
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )["Adj Close"]

    if isinstance(df, pd.Series):
        # single ticker case
        df = df.to_frame(name=tickers[0])

    df = df.dropna(how="any")
    df.columns = tickers
    return df


def to_log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert price levels to log prices."""
    return np.log(prices)


def save_prices(prices: pd.DataFrame, path: str) -> None:
    """Save prices to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prices.to_csv(path, index=True)


def load_prices(path: str) -> pd.DataFrame:
    """Load prices from CSV."""
    return pd.read_csv(path, index_col=0, parse_dates=True)
