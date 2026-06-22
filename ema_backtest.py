# ============================================================
# CELL 1 — Imports & Configuration
# ============================================================
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('seaborn-whitegrid')

TICKERS = ['SHEL.L', 'AZN.L', 'HSBA.L', 'TSCO.L', 'BA.L']
NAMES   = {
    'SHEL.L': 'Shell',
    'AZN.L' : 'AstraZeneca',
    'HSBA.L': 'HSBC',
    'TSCO.L': 'Tesco',
    'BA.L'  : 'BAE Systems',
}

START      = '2019-01-01'
END        = '2024-12-31'
SPLIT      = '2022-07-01'   # ~70 / 30 train–test boundary

SHORT_SPAN = 20
LONG_SPAN  = 50
COST       = 0.001          # 0.1 % per side
RF_ANNUAL  = 0.025          # 10-yr UK Gilt period-average (approx.)
RF_DAILY   = RF_ANNUAL / 252


# ============================================================
# CELL 2 — Data Download & Verification
# ============================================================
raw    = yf.download(TICKERS, start=START, end=END,
                     auto_adjust=True, progress=False)
prices = raw['Close']
prices = prices.ffill()     # forward-fill any isolated missing trading days

print(f"Shape : {prices.shape}  (rows = trading days, cols = tickers)")
print(f"From  : {prices.index[0].date()}  →  {prices.index[-1].date()}")
print(f"\nFirst 3 rows:\n{prices.head(3).round(2)}")
print(f"\nLast  3 rows:\n{prices.tail(3).round(2)}")
print(f"\nMissing values per ticker:\n{prices.isnull().sum()}")


# ============================================================
# CELL 3 — EMA Calculation (Shell walkthrough)
# ============================================================
# adjust=False uses the standard recursive formula:
#   EMA_t = α × price_t + (1 − α) × EMA_{t-1},  α = 2 / (span + 1)
# adjust=True uses an expanding weighted sum that is NOT the standard
# financial EMA and gives different (non-standard) values early in
# the series.

shell    = prices['SHEL.L']
ema20_sh = shell.ewm(span=SHORT_SPAN, adjust=False).mean()
ema50_sh = shell.ewm(span=LONG_SPAN,  adjust=False).mean()

print(pd.DataFrame({
    'Price' : shell,
    'EMA-20': ema20_sh,
    'EMA-50': ema50_sh,
}).tail(10).round(2))


# ============================================================
# CELL 4 — Signal Generation (Shell walkthrough)
# ============================================================
# signal[t]   = 1 if EMA-20[t] > EMA-50[t], else 0
# position[t] = signal[t-1]   ← .shift(1) prevents look-ahead bias
#
# Without .shift(1) you would trade *on* the closing crossover, which
# is impossible in practice — you can only act the next day.

signal_sh   = (ema20_sh > ema50_sh).astype(int)
position_sh = signal_sh.shift(1)

daily_ret_sh = shell.pct_change()
cost_sh      = COST * position_sh.diff().abs()
strat_sh     = (position_sh * daily_ret_sh - cost_sh).dropna()

n_trades = int(position_sh.diff().abs().dropna().sum())
print(f"Total position changes (Shell, full period): {n_trades}")

test_strat_sh = strat_sh[SPLIT:]
equity_sh     = (1 + test_strat_sh).cumprod()
print(f"Test-period rows       : {len(test_strat_sh)}")
print(f"Final equity (£1 start): £{equity_sh.iloc[-1]:.4f}")


# ============================================================
# CELL 5 — Helper Functions
# ============================================================

def compute_metrics(returns: pd.Series) -> dict:
    """
    Total return, annualised Sharpe Ratio, and maximum drawdown.

    Sharpe  : SR = mean(r_excess) / std(r_excess) × √252
    Max DD  : MDD = min( (V_t − peak_t) / peak_t )
              where V is the compounding equity curve and peak_t = max(V_0 … V_t)
    """
    returns = returns.dropna()
    equity  = (1 + returns).cumprod()

    total_return = equity.iloc[-1] - 1

    excess = returns - RF_DAILY
    std    = excess.std()
    sharpe = (excess.mean() / std) * np.sqrt(252) if std > 0 else np.nan

    rolling_peak = equity.cummax()
    drawdown     = (equity - rolling_peak) / rolling_peak
    max_dd       = drawdown.min()

    return {
        'total_return': total_return,
        'sharpe'      : sharpe,
        'max_drawdown': max_dd,
    }


def backtest(price: pd.Series,
             short: int   = SHORT_SPAN,
             long_: int   = LONG_SPAN,
             cost:  float = COST) -> dict:
    """
    Run EMA crossover + Buy-and-Hold on a price series.
    All metrics are computed on the test period only.
    EMAs are calculated on the full series so the test period
    inherits a fully warmed-up indicator.
    """
    # ── EMA strategy ─────────────────────────────────────
    ema_s    = price.ewm(span=short,  adjust=False).mean()
    ema_l    = price.ewm(span=long_,  adjust=False).mean()
    signal   = (ema_s > ema_l).astype(int)
    position = signal.shift(1)

    daily_ret  = price.pct_change()
    trade_cost = cost * position.diff().abs()
    strat_ret  = (position * daily_ret - trade_cost).dropna()

    # ── Test-period slice ─────────────────────────────────
    test_ret   = strat_ret[SPLIT:]
    test_pos   = position[SPLIT:]
    test_price = price[SPLIT:]

    # ── Buy-and-Hold (test period) ────────────────────────
    # daily_ret computed on full series so the first test-day return
    # (from last training day → first test day) is preserved.
    bnh = daily_ret[SPLIT:].dropna().copy()
    bnh.iloc[0]  -= cost   # entry cost on first test day
    bnh.iloc[-1] -= cost   # exit cost on last test day

    return {
        'strat_ret' : test_ret,
        'bnh_ret'   : bnh,
        'price'     : test_price,
        'ema_short' : ema_s[SPLIT:],
        'ema_long'  : ema_l[SPLIT:],
        'position'  : test_pos,
        'strat_m'   : compute_metrics(test_ret),
        'bnh_m'     : compute_metrics(bnh),
    }


# ============================================================
# CELL 6 — Backtest All 5 Stocks + Summary Table
# ============================================================
results = {t: backtest(prices[t]) for t in TICKERS}

rows = []
for t in TICKERS:
    for label, key in [('EMA Crossover', 'strat_m'), ('Buy & Hold', 'bnh_m')]:
        m = results[t][key]
        rows.append({
            'Stock'        : NAMES[t],
            'Strategy'     : label,
            'Total Return' : f"{m['total_return']:+.2%}",
            'Sharpe Ratio' : f"{m['sharpe']:.3f}",
            'Max Drawdown' : f"{m['max_drawdown']:.2%}",
        })

summary = pd.DataFrame(rows).set_index(['Stock', 'Strategy'])
print(summary.to_string())


# ============================================================
# CELL 7 — Sensitivity Analysis
# ============================================================
SHORT_GRID = [10, 20, 50]
LONG_GRID  = [50, 100, 200]

combo_sharpes = {}
for s in SHORT_GRID:
    for l in LONG_GRID:
        if s >= l:
            continue
        sharpes = [backtest(prices[t], short=s, long_=l)['strat_m']['sharpe']
                   for t in TICKERS]
        combo_sharpes[(s, l)] = np.nanmean(sharpes)

sharpe_grid = pd.DataFrame(
    index   = [f'Short={s}' for s in SHORT_GRID],
    columns = [f'Long={l}'  for l in LONG_GRID],
    dtype   = float,
)
for s in SHORT_GRID:
    for l in LONG_GRID:
        key = (s, l)
        sharpe_grid.loc[f'Short={s}', f'Long={l}'] = (
            combo_sharpes[key] if key in combo_sharpes else np.nan
        )

print("Average Sharpe Ratio across 5 stocks (test period):")
print(sharpe_grid.round(3).to_string())


# ============================================================
# CELL 8 — Visualisations
# ============================================================

# ── Figure 1: Equity Curves ─────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, t in enumerate(TICKERS):
    ax  = axes[i]
    res = results[t]

    strat_eq = (1 + res['strat_ret']).cumprod()
    bnh_eq   = (1 + res['bnh_ret']).cumprod()

    ax.plot(strat_eq, label='EMA Crossover', color='#1976D2', lw=1.8)
    ax.plot(bnh_eq,   label='Buy & Hold',    color='#E53935', lw=1.8, ls='--')
    ax.set_title(NAMES[t], fontsize=12, fontweight='bold')
    ax.set_ylabel('Portfolio Value (£1 start)')
    ax.set_xlabel('Date')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

axes[-1].set_visible(False)
fig.suptitle(
    'EMA Crossover vs Buy & Hold — Test Period (Jul 2022 – Dec 2024)',
    fontsize=14, fontweight='bold',
)
plt.tight_layout()
plt.savefig('equity_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Figure 2: Signal Plot (Shell) ──────────────────────────
res_sh = results['SHEL.L']
pos_ch = res_sh['position'].diff()
buys   = pos_ch[pos_ch == 1.0].index
sells  = pos_ch[pos_ch == -1.0].index

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]},
)

ax1.plot(res_sh['price'],     label='Price',               color='#212121', lw=1.0, alpha=0.85)
ax1.plot(res_sh['ema_short'], label=f'EMA-{SHORT_SPAN}',   color='#1976D2', lw=1.6)
ax1.plot(res_sh['ema_long'],  label=f'EMA-{LONG_SPAN}',    color='#FF8F00', lw=1.6)
ax1.scatter(buys,  res_sh['price'].reindex(buys),
            marker='^', color='#2E7D32', s=90, zorder=5, label='Buy')
ax1.scatter(sells, res_sh['price'].reindex(sells),
            marker='v', color='#C62828', s=90, zorder=5, label='Sell')
ax1.set_title('Shell (SHEL.L) — EMA Crossover Signals, Test Period',
              fontsize=13, fontweight='bold')
ax1.set_ylabel('Price (GBp)')
ax1.legend(fontsize=9)

ax2.fill_between(res_sh['position'].index, res_sh['position'], 0,
                 where=(res_sh['position'] > 0.5),
                 color='#4CAF50', alpha=0.35, label='In market')
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['Out', 'In'])
ax2.set_ylabel('Position')
ax2.set_xlabel('Date')
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('signal_plot_shell.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Figure 3: Sensitivity Heatmap ──────────────────────────
mask = sharpe_grid.isnull()

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(
    sharpe_grid.astype(float),
    annot=True, fmt='.3f',
    cmap='RdYlGn',
    mask=mask,
    linewidths=0.5,
    cbar_kws={'label': 'Average Sharpe Ratio'},
    ax=ax,
)
ax.set_title(
    'Sensitivity Analysis — Average Sharpe Ratio\n'
    'EMA Parameter Grid  ×  5 FTSE 100 Stocks  (Test Period)',
    fontsize=12, fontweight='bold',
)
ax.set_xlabel('Long EMA Period',  fontsize=11)
ax.set_ylabel('Short EMA Period', fontsize=11)
plt.tight_layout()
plt.savefig('sensitivity_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
