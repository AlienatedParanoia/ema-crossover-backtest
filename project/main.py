"""main.py — end-to-end orchestration of the EMA crossover backtesting study."""

import os
import warnings

import pandas as pd

import config
import data
import signals as sig
import backtest
import metrics
import visualisations
import sensitivity

warnings.filterwarnings("ignore")   # suppress yfinance deprecation noise


def print_results_table(results: dict) -> None:
    """Print a formatted comparison table of all metrics across all stocks.

    Parameters
    ----------
    results : dict
        Nested dict: results[ticker]["ema" | "bh"][metric_name] → float
    """
    header = (
        f"{'Ticker':<10}"
        f"{'EMA Sharpe':>12}"
        f"{'B&H Sharpe':>12}"
        f"{'EMA MaxDD':>12}"
        f"{'B&H MaxDD':>12}"
        f"{'EMA Return':>12}"
        f"{'B&H Return':>12}"
    )
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  EMA CROSSOVER vs BUY-AND-HOLD — TEST PERIOD RESULTS")
    print("=" * len(header))
    print(header)
    print(separator)

    for ticker, m in results.items():
        print(
            f"{ticker:<10}"
            f"{m['ema']['sharpe']:>+12.3f}"
            f"{m['bh']['sharpe']:>+12.3f}"
            f"{m['ema']['max_drawdown']:>+12.2%}"
            f"{m['bh']['max_drawdown']:>+12.2%}"
            f"{m['ema']['total_return']:>+12.2%}"
            f"{m['bh']['total_return']:>+12.2%}"
        )

    print(separator)
    print("Sharpe: annualised (√252). Max drawdown: peak-to-trough fraction.")
    print("Risk-free rate used: {:.1%}. Transaction cost: {:.2%} per trade.\n".format(
        config.RISK_FREE_RATE, config.TRANSACTION_COST
    ))


def main() -> None:
    """Run the full backtesting pipeline from data download to chart export."""

    # ── 0. Setup ──────────────────────────────────────────────────────────────
    os.makedirs(config.CHARTS_DIR, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Fetching and preparing data")
    print("=" * 60)

    prices = data.fetch_data(config.TICKERS, config.START_DATE, config.END_DATE)
    prices = data.clean_data(prices)
    train_data, test_data = data.split_data(prices, config.SPLIT_DATE)

    # ── 2. Strategy backtest (test period) ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Running backtests on test period")
    print("=" * 60)

    results = {}

    for ticker in config.TICKERS:
        print(f"\n[main] Processing {ticker} …")

        # Full price series needed for EMA warm-up on train data;
        # signals are generated on the full history but only applied to test prices.
        full_price  = prices[ticker].dropna()
        test_prices = test_data[ticker].dropna()

        # Generate signals on the full history so the EMA is properly warmed up
        # by the time the test period begins — avoids cold-start artefacts.
        signal_series = sig.generate_signals(
            full_price,
            config.SHORT_EMA,
            config.LONG_EMA,
        )

        # Slice signals to the test period only
        test_signals = signal_series.reindex(test_prices.index)

        ema_portfolio = backtest.run_ema_strategy(
            test_prices,
            test_signals,
            config.STARTING_CAPITAL,
            config.TRANSACTION_COST,
        )

        bh_portfolio = backtest.run_buy_and_hold(
            test_prices,
            config.STARTING_CAPITAL,
        )

        results[ticker] = metrics.compute_all_metrics(
            ema_portfolio,
            bh_portfolio,
            config.RISK_FREE_RATE,
        )

        # Store portfolios for charting (attach to results dict for convenience)
        results[ticker]["_ema_portfolio"] = ema_portfolio
        results[ticker]["_bh_portfolio"]  = bh_portfolio
        results[ticker]["_test_signals"]  = test_signals
        results[ticker]["_test_prices"]   = test_prices

    # ── 3. Results table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Summary results")
    print("=" * 60)

    # Build a clean copy of results (without private portfolio series) for printing
    clean_results = {
        t: {k: v for k, v in m.items() if not k.startswith("_")}
        for t, m in results.items()
    }
    print_results_table(clean_results)

    # ── 4. Equity curve and signal charts ─────────────────────────────────────
    print("=" * 60)
    print("STEP 4: Generating equity curve and signal charts")
    print("=" * 60)

    for ticker, m in results.items():
        visualisations.plot_equity_curves(
            m["_ema_portfolio"],
            m["_bh_portfolio"],
            ticker,
        )
        visualisations.plot_signals(
            m["_test_prices"],
            m["_test_signals"],
            ticker,
        )

    # ── 5. Sensitivity analysis ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Sensitivity analysis (EMA parameter grid)")
    print("=" * 60)

    # Pass full price history so EMAs warm up on train data before test evaluation
    sensitivity_results = sensitivity.run_all_sensitivity(prices, test_data)

    # Print a compact sensitivity summary
    print("\n── Sensitivity summary (Sharpe Ratio, EMA strategy) ──")
    for ticker, df in sensitivity_results.items():
        best_row = df.loc[df["sharpe_ema"].idxmax()]
        print(
            f"  {ticker:<10} best params: short={int(best_row['short_window'])}, "
            f"long={int(best_row['long_window'])}, "
            f"Sharpe={best_row['sharpe_ema']:+.3f}"
        )

    # ── Done ──────────────────────────────────────────────────────────────────
    chart_files = [f for f in os.listdir(config.CHARTS_DIR) if f.endswith(".png")]
    print(f"\nAll done. {len(chart_files)} chart(s) saved to '{config.CHARTS_DIR}/'.")


if __name__ == "__main__":
    main()
