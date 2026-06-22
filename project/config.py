# config.py — all global parameters in one place
# No logic here; every value is imported by the other modules.

# ── Stocks ────────────────────────────────────────────────────────────────────
TICKERS = ["SHEL.L", "AZN.L", "HSBA.L", "TSCO.L", "BA.L"]

# ── Date range ────────────────────────────────────────────────────────────────
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"

# Train/test split: ~70 % train (Jan 2019 – Jun 2022), ~30 % test (Jul 2022 – Dec 2024)
SPLIT_DATE = "2022-07-01"

# ── EMA parameters ────────────────────────────────────────────────────────────
SHORT_EMA = 20   # fast EMA window (days)
LONG_EMA  = 50   # slow EMA window (days)

# ── Sensitivity-analysis grid ─────────────────────────────────────────────────
SHORT_WINDOWS = [10, 20, 50]       # candidate short-EMA spans
LONG_WINDOWS  = [50, 100, 200]     # candidate long-EMA spans

# ── Transaction costs ─────────────────────────────────────────────────────────
TRANSACTION_COST = 0.001   # 0.1 % charged on every buy or sell execution

# ── Risk-free rate ────────────────────────────────────────────────────────────
# Blended average of 10-year UK Gilt yield across 2019–2024
RISK_FREE_RATE = 0.025     # 2.5 % annualised

# ── Capital ───────────────────────────────────────────────────────────────────
STARTING_CAPITAL = 10_000  # £10,000 initial portfolio value

# ── Output ────────────────────────────────────────────────────────────────────
CHARTS_DIR = "charts"      # relative path; created by main.py if absent
