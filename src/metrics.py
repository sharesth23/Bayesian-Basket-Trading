"""
metrics.py

Performance and risk metrics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (excess return / volatility)."""
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return np.sqrt(periods_per_year) * mu / sigma


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum drawdown in % (negative number).

    equity_curve : cumulative returns or portfolio value.
    """
    cum_max = equity_curve.cummax()
    drawdown = equity_curve / cum_max - 1.0
    return drawdown.min()


def calmar_ratio(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Calmar ratio = annual return / |max_drawdown|."""
    equity = (1 + returns).cumprod()
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    ann_ret = (1 + returns.mean()) ** periods_per_year - 1
    return ann_ret / abs(mdd)


def half_life_of_mean_reversion(spread: pd.Series) -> float:
    """
    Estimate half-life of mean reversion using an AR(1) model:
        Δspread_t = α + β * spread_{t-1} + ε_t
    half_life = -ln(2) / ln(1 + β)

    Returns days of half-life.
    """
    spread = spread.dropna()
    lagged = spread.shift(1).iloc[1:]
    delta = (spread - lagged).iloc[1:]

    X = np.vstack([np.ones(len(lagged) - 1), lagged.iloc[1:]]).T
    y = delta.values

    # OLS estimate
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0][1]

    if beta_hat >= 0:
        return np.inf

    half_life = -np.log(2) / np.log(1 + beta_hat)
    return float(half_life)
