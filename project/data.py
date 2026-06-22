"""data.py — data acquisition, cleaning, and train/test splitting."""

import os
import time

import pandas as pd
import yfinance as yf

# Local CSV cache path — avoids re-downloading on every run and handles
# Yahoo Finance rate limits gracefully.
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data_cache", "prices.csv")


def fetch_data(
    tickers: list[str],
    start: str,
    end: str,
    max_retries: int = 5,
    retry_delay: float = 15.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download adjusted daily close prices for a list of tickers.

    Adjusted close is used instead of raw close because it accounts for
    corporate actions (dividends, stock splits), ensuring the return series
    reflects the actual economic gain a holder would have received.  Using
    raw close would introduce artificial price jumps on ex-dividend dates that
    would trigger false trading signals.

    A local CSV cache is maintained so repeated runs do not re-hit the Yahoo
    Finance API.  Pass ``use_cache=False`` to force a fresh download.

    Retries are included because Yahoo Finance enforces rate limits; a brief
    pause between attempts is sufficient in practice.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance ticker symbols (e.g. ["SHEL.L", "AZN.L"]).
    start : str
        Start date in "YYYY-MM-DD" format (inclusive).
    end : str
        End date in "YYYY-MM-DD" format (inclusive).
    max_retries : int
        Number of download attempts before raising.
    retry_delay : float
        Seconds to wait between retries.
    use_cache : bool
        If True and a cache file exists, load from disk instead of downloading.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with one column per ticker containing
        adjusted close prices.
    """
    if use_cache and os.path.exists(_CACHE_PATH):
        print(f"[data] Loading from cache: {_CACHE_PATH}")
        prices = pd.read_csv(_CACHE_PATH, index_col=0, parse_dates=True)
        # Ensure all requested tickers are present in cache
        missing = [t for t in tickers if t not in prices.columns]
        if not missing:
            return prices[tickers]
        print(f"[data] Cache missing tickers {missing} — re-downloading all.")

    frames = {}
    for ticker in tickers:
        series = None
        for attempt in range(1, max_retries + 1):
            try:
                t    = yf.Ticker(ticker)
                hist = t.history(start=start, end=end, auto_adjust=True)
                if not hist.empty:
                    series = hist["Close"]
                    series.index = pd.to_datetime(series.index).tz_localize(None)
                    print(f"[data] {ticker}: {len(series)} rows downloaded.")
                    break
            except Exception as exc:
                print(f"[data] {ticker} attempt {attempt}/{max_retries} failed: {exc}")

            if attempt < max_retries:
                print(f"[data] Retrying {ticker} in {retry_delay}s …")
                time.sleep(retry_delay)

        if series is None:
            raise RuntimeError(
                f"Failed to download {ticker} after {max_retries} attempts."
            )
        frames[ticker] = series

    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index)

    # Persist to cache for future runs
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    prices.to_csv(_CACHE_PATH)
    print(f"[data] Prices cached to {_CACHE_PATH}")

    return prices


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Verify data integrity and forward-fill any missing values.

    Missing values arise from public holidays, exchange closures, or
    occasional data-provider gaps.  Forward-filling carries the last known
    price forward, which is the standard convention for equity price series
    (position value is unchanged on a non-trading day).

    Parameters
    ----------
    df : pd.DataFrame
        Raw price DataFrame as returned by ``fetch_data``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with no NaN values.
    """
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print("[data] NaN values detected before cleaning:")
        print(nan_counts[nan_counts > 0])
        df = df.ffill()   # forward-fill first
        df = df.bfill()   # back-fill handles any leading NaNs at series start
        print("[data] Forward/back-fill applied.")
    else:
        print("[data] No missing values found.")

    return df


def split_data(
    df: pd.DataFrame, split_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the price DataFrame into train and test periods.

    The split date separates the in-sample (training) period used for
    strategy development from the out-of-sample (test) period used for
    performance evaluation.  Keeping these periods separate prevents
    information leakage: parameters are never tuned on the test set.

    Parameters
    ----------
    df : pd.DataFrame
        Full price DataFrame indexed by date.
    split_date : str
        First date of the test period in "YYYY-MM-DD" format.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) where train ends the day before split_date
        and test starts on split_date.
    """
    split_ts = pd.Timestamp(split_date)
    train = df[df.index < split_ts]
    test  = df[df.index >= split_ts]

    print(
        f"[data] Train: {train.index[0].date()} → {train.index[-1].date()} "
        f"({len(train)} trading days)"
    )
    print(
        f"[data] Test : {test.index[0].date()} → {test.index[-1].date()} "
        f"({len(test)} trading days)"
    )

    return train, test
