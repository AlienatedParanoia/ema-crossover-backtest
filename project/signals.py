"""signals.py — EMA computation and crossover signal generation."""

import pandas as pd


def compute_emas(
    price_series: pd.Series,
    short_window: int,
    long_window: int,
) -> tuple[pd.Series, pd.Series]:
    """Calculate short and long exponential moving averages.

    Parameters
    ----------
    price_series : pd.Series
        Daily adjusted close prices for a single stock.
    short_window : int
        Span (in trading days) for the fast EMA.
    long_window : int
        Span (in trading days) for the slow EMA.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (short_ema, long_ema) aligned to the same index as price_series.

    Notes
    -----
    ``adjust=False`` is used deliberately.  With adjust=True, pandas computes
    a weighted average over the entire history (batch mode), which is not
    reproducible in a live trading system where only the previous EMA value is
    known.  adjust=False uses the standard recursive formula:
        EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1},  alpha = 2 / (span + 1)
    This exactly matches how EMAs are calculated on Bloomberg, Reuters, and
    most professional trading platforms.
    """
    short_ema = price_series.ewm(
        span=short_window,
        adjust=False,   # recursive formula — see docstring above
    ).mean()

    long_ema = price_series.ewm(
        span=long_window,
        adjust=False,   # consistent with short_ema for a fair comparison
    ).mean()

    return short_ema, long_ema


def generate_signals(
    price_series: pd.Series,
    short_window: int,
    long_window: int,
) -> pd.Series:
    """Generate EMA crossover trading signals without look-ahead bias.

    A buy signal (1) is issued when the short EMA crosses above the long EMA,
    indicating upward momentum.  A sell signal (-1) is issued on the reverse
    crossover.  Between crossovers the signal is held (0 = stay in current
    position or cash).

    Look-ahead bias prevention
    --------------------------
    The raw crossover signal is computed on day t using prices known at the
    close of day t.  To prevent the strategy from trading *on* that same
    close, the signal series is shifted forward by one day with ``.shift(1)``.
    This means a crossover observed at the close of day t triggers a trade at
    the open of day t+1 — reflecting realistic execution conditions.  Without
    this shift the backtest would implicitly assume the trader can act on
    information from the future, inflating returns.

    Parameters
    ----------
    price_series : pd.Series
        Daily adjusted close prices for a single stock.
    short_window : int
        Span for the fast EMA (days).
    long_window : int
        Span for the slow EMA (days).

    Returns
    -------
    pd.Series
        Signal series aligned to price_series.index:
            1  = long (hold position)
           -1  = flat / exit position
            0  = initialisation period (not enough data for both EMAs)
    """
    short_ema, long_ema = compute_emas(price_series, short_window, long_window)

    # Raw position: 1 when short EMA is above long EMA, else -1
    raw_position = (short_ema > long_ema).astype(int) * 2 - 1

    # Shift by 1 day so the signal observed at close of day t
    # only influences trading from day t+1 onward.
    signal = raw_position.shift(1)

    # Fill the first row (NaN from shift) with 0 — no position before first signal
    signal.iloc[0] = 0

    return signal
