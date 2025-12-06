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


if __name__ == "__main__":
    # Small test so you see it working
    tickers = ["HDFCBANK.NS", "ICICIBANK.NS"]
    prices = download_prices(tickers, start="2020-01-01")
    print(prices.head())

    log_prices = to_log_prices(prices)
    print("\nLog prices head:")
    print(log_prices.head())

    save_prices(prices, "data/raw/banks.csv")
    print("\nSaved to data/raw/banks.csv")
