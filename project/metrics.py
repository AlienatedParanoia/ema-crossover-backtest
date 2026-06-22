"""metrics.py — risk-adjusted performance metrics."""

import numpy as np
import pandas as pd


# ── Sharpe Ratio ──────────────────────────────────────────────────────────────
#
#   Formula:
#       daily_returns  = portfolio_t / portfolio_{t-1} - 1
#       excess_return  = mean(daily_returns) - (risk_free_rate / 252)
#       Sharpe         = (excess_return / std(daily_returns)) × √252
#
#   Annualisation by √252 converts the daily Sharpe to an annualised figure
#   assuming 252 trading days per year (LSE convention).
#
def sharpe_ratio(portfolio_series: pd.Series, risk_free_rate: float) -> float:
    """Compute the annualised Sharpe Ratio.

    Parameters
    ----------
    portfolio_series : pd.Series
        Daily portfolio value series.
    risk_free_rate : float
        Annualised risk-free rate as a decimal (e.g. 0.025 for 2.5 %).

    Returns
    -------
    float
        Annualised Sharpe Ratio.  Returns 0.0 if standard deviation is zero
        (flat portfolio — avoids division by zero).
    """
    daily_returns = portfolio_series.pct_change().dropna()

    daily_rf      = risk_free_rate / 252          # convert annual RF to daily
    excess        = daily_returns - daily_rf
    std           = daily_returns.std()

    if std == 0:
        return 0.0

    return (excess.mean() / std) * np.sqrt(252)   # annualise by √252


# ── Maximum Drawdown ──────────────────────────────────────────────────────────
#
#   Formula:
#       rolling_peak_t = max(portfolio_0 … portfolio_t)
#       drawdown_t     = (portfolio_t - rolling_peak_t) / rolling_peak_t
#       MaxDrawdown    = min(drawdown_t)     ← most negative value
#
#   Maximum drawdown measures the largest peak-to-trough decline, expressed
#   as a fraction of the peak value.  It is the primary downside-risk metric
#   for practitioners because it captures the worst observed loss.
#
def max_drawdown(portfolio_series: pd.Series) -> float:
    """Compute the maximum drawdown (most negative peak-to-trough decline).

    Parameters
    ----------
    portfolio_series : pd.Series
        Daily portfolio value series.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal (e.g. -0.25 means -25 %).
    """
    rolling_peak = portfolio_series.cummax()
    drawdown     = (portfolio_series - rolling_peak) / rolling_peak
    return float(drawdown.min())


# ── Total Return ──────────────────────────────────────────────────────────────
#
#   Formula:
#       TotalReturn = (portfolio_final - portfolio_initial) / portfolio_initial
#
def total_return(portfolio_series: pd.Series) -> float:
    """Compute the total percentage return over the full period.

    Parameters
    ----------
    portfolio_series : pd.Series
        Daily portfolio value series.

    Returns
    -------
    float
        Total return as a decimal (e.g. 0.35 means +35 %).
    """
    return (portfolio_series.iloc[-1] - portfolio_series.iloc[0]) / portfolio_series.iloc[0]


def compute_all_metrics(
    ema_portfolio: pd.Series,
    bh_portfolio: pd.Series,
    risk_free_rate: float,
) -> dict:
    """Compute all performance metrics for both strategies.

    Parameters
    ----------
    ema_portfolio : pd.Series
        Daily portfolio value series for the EMA crossover strategy.
    bh_portfolio : pd.Series
        Daily portfolio value series for the Buy-and-Hold benchmark.
    risk_free_rate : float
        Annualised risk-free rate as a decimal.

    Returns
    -------
    dict
        Nested dictionary with keys "ema" and "bh", each containing:
            - "sharpe"       : annualised Sharpe Ratio (float)
            - "max_drawdown" : maximum drawdown (float, ≤ 0)
            - "total_return" : total return (float)
    """
    return {
        "ema": {
            "sharpe":       sharpe_ratio(ema_portfolio, risk_free_rate),
            "max_drawdown": max_drawdown(ema_portfolio),
            "total_return": total_return(ema_portfolio),
        },
        "bh": {
            "sharpe":       sharpe_ratio(bh_portfolio, risk_free_rate),
            "max_drawdown": max_drawdown(bh_portfolio),
            "total_return": total_return(bh_portfolio),
        },
    }
