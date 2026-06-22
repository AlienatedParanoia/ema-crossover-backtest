"""sensitivity.py — parameter grid search for EMA crossover robustness."""

import pandas as pd

import config
import signals as sig
import backtest
import metrics
import visualisations


def run_sensitivity(
    full_price_series: pd.Series,
    test_price_series: pd.Series,
    starting_capital: float,
    transaction_cost: float,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Run EMA crossover backtest across the full parameter grid for one stock.

    Iterates every combination of short EMA window × long EMA window defined
    in config.SHORT_WINDOWS and config.LONG_WINDOWS.  Combinations where the
    short window is greater than or equal to the long window are skipped
    because they are economically meaningless (the "fast" line would always
    lag the "slow" line).

    EMA warm-up correctness
    -----------------------
    Signals are generated on the *full* price history (train + test) so the
    exponential moving averages are fully converged by the time the test period
    begins.  Only the test-period slice of the signal series is passed to the
    backtest engine.  Generating signals on the test period alone would cause a
    cold-start artefact where the EMA begins from scratch at the test boundary,
    producing unreliable early signals.  This mirrors the approach used in the
    main backtest in ``main.py``.

    Parameters
    ----------
    full_price_series : pd.Series
        Daily adjusted close prices for a single stock across the *full* date
        range (train + test), used solely for EMA warm-up.
    test_price_series : pd.Series
        Daily adjusted close prices for the test period only.  Backtest
        performance is evaluated on this slice.
    starting_capital : float
        Initial portfolio value in £.
    transaction_cost : float
        Fractional cost per trade (e.g. 0.001 for 0.1 %).
    risk_free_rate : float
        Annualised risk-free rate as a decimal.

    Returns
    -------
    pd.DataFrame
        One row per valid parameter combination with columns:
            short_window, long_window, sharpe_ema, sharpe_bh,
            max_drawdown_ema, total_return_ema
    """
    bh_portfolio = backtest.run_buy_and_hold(test_price_series, starting_capital)
    bh_sharpe    = metrics.sharpe_ratio(bh_portfolio, risk_free_rate)

    rows = []

    for short_w in config.SHORT_WINDOWS:
        for long_w in config.LONG_WINDOWS:
            if short_w >= long_w:
                # Skip invalid combinations: fast EMA must be shorter than slow EMA
                continue

            # Generate signals on full history to avoid cold-start bias,
            # then slice to the test period for the backtest.
            full_signals  = sig.generate_signals(full_price_series, short_w, long_w)
            test_signals  = full_signals.reindex(test_price_series.index)

            ema_portfolio = backtest.run_ema_strategy(
                test_price_series, test_signals, starting_capital, transaction_cost
            )

            rows.append({
                "short_window":     short_w,
                "long_window":      long_w,
                "sharpe_ema":       metrics.sharpe_ratio(ema_portfolio, risk_free_rate),
                "sharpe_bh":        bh_sharpe,
                "max_drawdown_ema": metrics.max_drawdown(ema_portfolio),
                "total_return_ema": metrics.total_return(ema_portfolio),
            })

    return pd.DataFrame(rows)


def run_all_sensitivity(
    full_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run sensitivity analysis for all tickers and save heatmaps.

    Calls ``run_sensitivity`` for each ticker in the test dataset, saves a
    Sharpe Ratio heatmap for each, and returns all result DataFrames keyed
    by ticker symbol.

    Parameters
    ----------
    full_data : pd.DataFrame
        Complete price DataFrame (train + test) used for EMA warm-up.
    test_data : pd.DataFrame
        Test-period price DataFrame used for backtest evaluation.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker → sensitivity results DataFrame.
    """
    all_results = {}

    for ticker in test_data.columns:
        print(f"\n[sensitivity] Running grid search for {ticker} …")
        full_prices = full_data[ticker].dropna()
        test_prices = test_data[ticker].dropna()

        results_df = run_sensitivity(
            full_price_series   = full_prices,
            test_price_series   = test_prices,
            starting_capital    = config.STARTING_CAPITAL,
            transaction_cost    = config.TRANSACTION_COST,
            risk_free_rate      = config.RISK_FREE_RATE,
        )

        all_results[ticker] = results_df

        # Save heatmap immediately after each ticker completes
        visualisations.plot_sensitivity_heatmap(results_df, ticker)

    return all_results
