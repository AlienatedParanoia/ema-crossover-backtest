# EMA Crossover vs Buy-and-Hold — A Backtesting Study

An out-of-sample backtest of the canonical **20/50 EMA crossover** trend-following strategy
against a passive **buy-and-hold** benchmark, across five FTSE 100 equities.

> **Headline finding:** over the out-of-sample test window (Jul 2022 – Dec 2024), the 20/50 EMA
> crossover **underperformed** buy-and-hold on a risk-adjusted basis (Sharpe ratio) for **all
> five** stocks. Its only win anywhere was a shallower drawdown on Tesco.

## Universe & period

| | |
|---|---|
| **Equities** | Shell (`SHEL.L`), AstraZeneca (`AZN.L`), HSBC (`HSBA.L`), Tesco (`TSCO.L`), BAE Systems (`BA.L`) |
| **Data span** | 2019-01-01 → 2024-12-31 (daily, dividend/split-adjusted close) |
| **Warm-up** | 2019 → Jun 2022 (lets the EMAs converge; not scored) |
| **Test (reported)** | Jul 2022 → Dec 2024, out-of-sample |
| **Costs** | 0.1% per trade · 2.5% annualised risk-free rate · £10,000 starting capital |

## Results (out-of-sample)

| Stock | EMA Sharpe | B&H Sharpe | EMA MaxDD | B&H MaxDD | EMA Return | B&H Return |
|---|---|---|---|---|---|---|
| SHEL.L | +0.007 | +0.255 | −20.19% | −18.70% | +3.10% | +15.25% |
| AZN.L | −0.588 | −0.062 | −31.63% | −26.75% | −17.48% | −3.17% |
| HSBA.L | +0.064 | +0.679 | −32.58% | −20.10% | +5.30% | +46.54% |
| TSCO.L | +0.364 | +0.736 | −14.10% | −26.49% | +17.76% | +43.62% |
| BA.L | −0.008 | +0.567 | −25.38% | −18.21% | +1.41% | +36.66% |

## Repository layout

```
project/            Modular Python pipeline
  config.py         All parameters (tickers, dates, EMA windows, costs)
  data.py           Download, clean, train/test split (with local cache)
  signals.py        EMA computation + crossover signals (look-ahead controlled)
  backtest.py       Portfolio simulation: EMA strategy & buy-and-hold
  metrics.py        Sharpe ratio, max drawdown, total return
  sensitivity.py    Parameter-grid robustness sweep
  visualisations.py Equity-curve, signal, and heatmap charts
  main.py           End-to-end orchestration
  charts/           Generated PNGs
website/            Multi-page research-paper site (static HTML/CSS, offline)
ema_backtest.py     Single-file notebook-style version of the same study
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r project/requirements.txt
cd project && python main.py
```

This downloads the data (cached to `data_cache/`), runs the backtests, prints the summary
table, and writes all charts to `project/charts/`.

> **Note:** Yahoo Finance may rate-limit automated requests. If you hit `Too Many Requests`,
> fetch via a `curl_cffi` Chrome-impersonating session (`pip install curl_cffi`, pass a
> `Session(impersonate="chrome")` to `yfinance`).

## Viewing the website

Open `website/index.html` in a browser (it's fully self-contained — no internet required), or
serve it locally:

```bash
cd website && python -m http.server 8000   # then visit http://localhost:8000
```

## Limitations

The reported figures reflect the pipeline as built. Notably, the sensitivity "best parameters"
are selected on the test set (a robustness illustration, **not** a tradeable result), buy-and-hold
pays no entry cost, the universe is small and survivorship-biased, and the conclusion is specific
to the 2022–2024 regime. See the website's Conclusion page for full detail.
