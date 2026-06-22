"""backtest.py — portfolio simulation for EMA strategy and Buy-and-Hold."""

import pandas as pd


def run_ema_strategy(
    price_series: pd.Series,
    signals: pd.Series,
    starting_capital: float,
    transaction_cost: float,
) -> pd.Series:
    """Simulate the EMA crossover strategy with transaction costs.

    The portfolio holds either a 100 % long equity position (signal = 1) or
    100 % cash (signal = -1).  A transaction cost is deducted on every day
    where the position changes (entry or exit), reflecting broker commissions
    and bid-ask spread.

    Parameters
    ----------
    price_series : pd.Series
        Daily adjusted close prices for a single stock (test period only).
    signals : pd.Series
        Signal series from ``signals.generate_signals``, aligned to the same
        index as price_series.
    starting_capital : float
        Initial portfolio value in £.
    transaction_cost : float
        Fractional cost per trade (e.g. 0.001 for 0.1 %).

    Returns
    -------
    pd.Series
        Daily portfolio value in £, indexed identically to price_series.
    """
    # Align signals to the price index (test slice may differ from full series)
    signals = signals.reindex(price_series.index).fillna(0)

    portfolio_values = pd.Series(index=price_series.index, dtype=float)

    capital    = starting_capital
    position   = 0          # shares held
    prev_signal = 0         # track previous signal to detect position changes

    for date, price in price_series.items():
        signal = signals.loc[date]

        # ── Position change: close old, open new ─────────────────────────────
        if signal != prev_signal:
            if prev_signal == 1 and position > 0:
                # Sell: liquidate position
                proceeds = position * price
                cost     = proceeds * transaction_cost   # cost on sell
                capital  = proceeds - cost
                position = 0

            if signal == 1:
                # Buy: invest all capital into the stock
                cost     = capital * transaction_cost    # cost on buy
                capital -= cost
                position = capital / price               # shares purchased
                capital  = 0                             # fully invested

            prev_signal = signal

        # ── Daily portfolio value ─────────────────────────────────────────────
        if position > 0:
            portfolio_values.loc[date] = position * price
        else:
            portfolio_values.loc[date] = capital

    return portfolio_values


def run_buy_and_hold(
    price_series: pd.Series,
    starting_capital: float,
) -> pd.Series:
    """Simulate a passive Buy-and-Hold strategy as a benchmark.

    The full starting capital is invested at the closing price on the first
    day of the period.  No further trades are made.  This provides a baseline
    that represents what a passive index-style investor would have achieved
    without any active management.

    Note: no transaction cost is applied to the initial buy.  In a strict
    comparison this is a slight advantage for B&H; it is the standard academic
    convention and reflects that a single-trade strategy's cost is negligible
    over multi-year holding periods.

    Parameters
    ----------
    price_series : pd.Series
        Daily adjusted close prices for a single stock (test period only).
    starting_capital : float
        Initial portfolio value in £.

    Returns
    -------
    pd.Series
        Daily portfolio value in £, indexed identically to price_series.
    """
    # Shares bought at close on day 1
    entry_price = price_series.iloc[0]
    shares      = starting_capital / entry_price

    # Portfolio value on each day = shares × daily price
    portfolio_values = shares * price_series

    return portfolio_values
