"""visualisations.py — chart generation for equity curves, signals, and sensitivity."""

import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

import config


def _ensure_charts_dir() -> None:
    """Create the charts output directory if it does not already exist."""
    os.makedirs(config.CHARTS_DIR, exist_ok=True)


def plot_equity_curves(
    ema_portfolio: pd.Series,
    bh_portfolio: pd.Series,
    ticker: str,
) -> None:
    """Plot and save an equity-curve comparison chart.

    Overlays the EMA crossover strategy and Buy-and-Hold benchmark on the
    same axes so the divergence in cumulative performance is immediately
    visible.  The chart is saved as a PNG file for use as a figure in the
    academic essay.

    Parameters
    ----------
    ema_portfolio : pd.Series
        Daily portfolio value series for the EMA strategy (test period).
    bh_portfolio : pd.Series
        Daily portfolio value series for the Buy-and-Hold benchmark (test period).
    ticker : str
        Stock ticker (used in title and filename).
    """
    _ensure_charts_dir()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(ema_portfolio.index, ema_portfolio.values, label="EMA Crossover",
            color="#1f77b4", linewidth=1.5)
    ax.plot(bh_portfolio.index, bh_portfolio.values, label="Buy & Hold",
            color="#ff7f0e", linewidth=1.5, linestyle="--")

    ax.set_title(f"{ticker} — EMA Crossover vs Buy-and-Hold (Test Period)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (£)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(config.CHARTS_DIR, f"{ticker}_equity_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[vis] Saved {path}")


def plot_signals(
    price_series: pd.Series,
    signals: pd.Series,
    ticker: str,
) -> None:
    """Plot and save a price chart annotated with buy/sell signals.

    Buy signals (1 → -1 transition becomes 1 entry) are shown as green
    upward triangles; sell signals are shown as red downward triangles.
    This chart allows visual inspection of how frequently the strategy trades
    and whether signals align with price movements.

    Parameters
    ----------
    price_series : pd.Series
        Daily adjusted close prices for a single stock (test period).
    signals : pd.Series
        Signal series aligned to price_series.index.
    ticker : str
        Stock ticker (used in title and filename).
    """
    _ensure_charts_dir()

    # Detect signal changes to identify actual entry/exit days
    signal_changes = signals.diff().fillna(0)
    buy_dates  = price_series[signal_changes == 2].index   # -1 → 1 (diff = +2)
    sell_dates = price_series[signal_changes == -2].index  # 1 → -1 (diff = -2)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(price_series.index, price_series.values,
            color="#333333", linewidth=1.2, label="Price")

    ax.scatter(buy_dates, price_series.loc[buy_dates],
               marker="^", color="green", s=80, zorder=5, label="Buy signal")
    ax.scatter(sell_dates, price_series.loc[sell_dates],
               marker="v", color="red", s=80, zorder=5, label="Sell signal")

    ax.set_title(f"{ticker} — EMA Crossover Signals (Test Period)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price (£)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(config.CHARTS_DIR, f"{ticker}_signals.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[vis] Saved {path}")


def plot_sensitivity_heatmap(results_df: pd.DataFrame, ticker: str) -> None:
    """Plot and save a Sharpe Ratio heatmap across the EMA parameter grid.

    Rows represent short EMA windows; columns represent long EMA windows.
    Cells are NaN for invalid combinations (short >= long) and appear grey.
    This chart supports the sensitivity analysis section of the essay.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of ``sensitivity.run_sensitivity`` containing columns
        short_window, long_window, sharpe_ema.
    ticker : str
        Stock ticker (used in title and filename).
    """
    _ensure_charts_dir()

    pivot = results_df.pivot(
        index="short_window",
        columns="long_window",
        values="sharpe_ema",
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",    # red = poor Sharpe, green = strong Sharpe
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Annualised Sharpe Ratio"},
    )

    ax.set_title(f"{ticker} — EMA Sensitivity: Sharpe Ratio by Parameter Pair",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Long EMA Window (days)")
    ax.set_ylabel("Short EMA Window (days)")

    fig.tight_layout()
    path = os.path.join(config.CHARTS_DIR, f"{ticker}_sensitivity_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[vis] Saved {path}")
