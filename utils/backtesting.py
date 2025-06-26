"""Utility functions for running simple and dynamic backtests."""

from typing import Callable, Optional, Tuple, Union
from collections import OrderedDict
import os

import math
import pandas as pd
import numpy as np
import yfinance as yf  # For ETF liquidity data fetching
from utils.data_retrieval import get_risk_free_rate

# In-memory LRU cache for ETF volume data keyed by (ticker, start_date, end_date)
_ETF_VOLUME_CACHE_MAX_SIZE = 10
_etf_volume_cache: "OrderedDict[tuple[str, str, str], pd.DataFrame]" = OrderedDict()


def _enforce_cache_limit(cache_dir: Optional[str] = None) -> None:
    """Evict the least recently used entries when the cache exceeds the limit."""

    while len(_etf_volume_cache) > _ETF_VOLUME_CACHE_MAX_SIZE:
        old_key, old_df = _etf_volume_cache.popitem(last=False)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            filename = f"{old_key[0]}_{old_key[1]}_{old_key[2]}.pkl"
            old_df.to_pickle(os.path.join(cache_dir, filename))
    if cache_dir:
        _cleanup_disk_cache(cache_dir)


def _cleanup_disk_cache(cache_dir: str) -> None:
    """Remove oldest cache files on disk when exceeding the cache limit."""

    try:
        files = [
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith(".pkl")
        ]
    except FileNotFoundError:
        return

    if len(files) <= _ETF_VOLUME_CACHE_MAX_SIZE:
        return

    files.sort(key=os.path.getmtime, reverse=True)
    for path in files[_ETF_VOLUME_CACHE_MAX_SIZE:]:
        try:
            os.remove(path)
        except OSError:
            pass


def get_cached_etf_data(
    etf_ticker: str,
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Retrieve ETF data using yfinance with LRU caching and optional persistence."""

    key = (etf_ticker, str(start_date), str(end_date))

    if key in _etf_volume_cache:
        # Mark as recently used
        _etf_volume_cache.move_to_end(key)
        return _etf_volume_cache[key]

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        filename = f"{etf_ticker}_{str(start_date)}_{str(end_date)}.pkl"
        path = os.path.join(cache_dir, filename)
        if os.path.exists(path):
            data = pd.read_pickle(path)
            _etf_volume_cache[key] = data
            _etf_volume_cache.move_to_end(key)
            _enforce_cache_limit(cache_dir)
            return data

    data = yf.download(etf_ticker, start=start_date, end=end_date, group_by='column')
    data.reset_index(inplace=True)

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):

        def flatten(col):
            return col[1] if isinstance(col, tuple) and len(col) > 1 else col

        data.columns = [flatten(col) for col in data.columns]

    _etf_volume_cache[key] = data
    _etf_volume_cache.move_to_end(key)
    _enforce_cache_limit(cache_dir)

    return data


def full_invested_strategy(df: pd.DataFrame) -> float:
    """Return a constant allocation weight of ``1.0``.

    Parameters
    ----------
    df : pd.DataFrame
        Price data used by the strategy. The contents are ignored.

    Returns
    -------
    float
        Constant weight indicating the portfolio is fully invested.
    """

    # Return 1 for every row so broadcasting works in ``simple_backtest``.
    return 1.0

def simple_backtest(
    prices_df: pd.DataFrame,
    allocation_strategy: Callable[[pd.DataFrame], Union[float, pd.Series]],
) -> Tuple[float, float, pd.Series]:
    """Run a naive or user-defined backtest.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Historical prices containing a ``Date`` column and either ``Close`` or
        ``Adj Close`` prices.
    allocation_strategy : Callable[[pd.DataFrame], float | pd.Series]
        Function that returns either a constant weight or a series of weights
        when provided with the price DataFrame.

    Returns
    -------
    Tuple[float, float, pd.Series]
        ``(cumulative_return, alpha, cumulative_series)`` where ``cumulative_series``
        contains the cumulative strategy returns over time.
    """
    # Ensure a proper Date column exists and sort by date
    if 'Date' not in prices_df.columns:
        prices_df.reset_index(inplace=True)
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    prices_df.sort_values(by='Date', inplace=True)
    
    # Determine the correct price column
    if 'Close' in prices_df.columns:
        price_col = 'Close'
    elif 'Adj Close' in prices_df.columns:
        price_col = 'Adj Close'
    elif 'close' in prices_df.columns:
        price_col = 'close'
    elif 'adj close' in prices_df.columns:
        price_col = 'adj close'
    else:
        raise ValueError("No valid price column found in the price data.")
    
    # Drop rows with missing prices
    prices_df = prices_df.dropna(subset=[price_col])
    
    # Calculate daily returns and fill NaN for first row
    prices_df['returns'] = prices_df[price_col].pct_change().fillna(0)
    
    # Apply the allocation strategy to get a series of daily allocations
    allocation_series = allocation_strategy(prices_df)
    prices_df['strategy_returns'] = prices_df['returns'] * allocation_series
    
    cumulative_series = (prices_df['strategy_returns'] + 1).cumprod()
    if cumulative_series.empty:
        raise ValueError("Cumulative series is empty. Check your data and date range.")
    
    cumulative_return = cumulative_series.iloc[-1] - 1
    alpha = cumulative_return - prices_df['returns'].mean()
    
    return cumulative_return, alpha, cumulative_series


def dynamic_market_timing_strategy_advanced(
    df: pd.DataFrame, etf_ticker: Optional[str] = None
) -> pd.Series:
    """Generate allocation weights using price momentum, volatility and liquidity.

    This vectorised version uses rolling windows instead of explicit loops.
    ``df`` should contain ``Date``, ``Close`` and optionally ``Volume`` columns
    with daily ``returns`` already calculated.
    """

    lookback = 20

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    if "Close" in df.columns:
        price_col = "Close"
    elif "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "close" in df.columns:
        price_col = "close"
    elif "adj close" in df.columns:
        price_col = "adj close"
    else:
        raise ValueError("No valid price column found in the price data.")

    if "returns" not in df.columns:
        df["returns"] = df[price_col].pct_change().fillna(0)

    # ---------------------------------------------------------------
    # Momentum signal
    momentum = df[price_col] / df[price_col].shift(lookback) - 1
    x = (50 * momentum).clip(-700, 700)
    momentum_signal = 1 / (1 + np.exp(-x))

    # ---------------------------------------------------------------
    # Volatility scaling
    vol = df["returns"].rolling(lookback).std()
    target_vol = 0.02
    # Avoid divide-by-zero warnings by applying the ratio only where vol is
    # greater than the target. Remaining values stay at ``1``.
    volatility_signal = np.ones_like(vol, dtype=float)
    mask = vol > target_vol
    volatility_signal[mask] = target_vol / vol[mask]
    volatility_signal = np.minimum(volatility_signal, 1)

    # ---------------------------------------------------------------
    # Liquidity signal
    if "Volume" in df.columns and not df["Volume"].isnull().all():
        avg_volume = df["Volume"].rolling(lookback).mean()
        ratio = df["Volume"] / avg_volume
    elif etf_ticker:
        start_date = df["Date"].min()
        end_date = df["Date"].max()
        etf_data = get_cached_etf_data(etf_ticker, start_date, end_date)
        etf_data["Date"] = pd.to_datetime(etf_data["Date"])
        etf_data.sort_values("Date", inplace=True)
        etf_series = (
            etf_data.set_index("Date")["Volume"].reindex(df["Date"], method="ffill")
        )
        etf_series.index = df.index
        avg_volume = etf_series.rolling(lookback).mean()
        ratio = etf_series / avg_volume
    else:
        ratio = pd.Series(1.0, index=df.index)

    ratio = ratio.fillna(1)
    liquidity_signal = np.where(ratio >= 0.8, ratio, ratio / 0.8)
    liquidity_signal = np.minimum(liquidity_signal, 1)

    # ---------------------------------------------------------------
    # Combine signals
    allocation = momentum_signal * volatility_signal * liquidity_signal
    allocation = allocation.clip(0, 1)

    # Set the initial lookback period to fully invested
    allocation.iloc[:lookback] = 1.0

    return allocation


def compute_metrics(
    prices_df: pd.DataFrame,
    naive_returns: pd.Series,
    strategy_daily_returns: pd.Series,
    naive_series: pd.Series,
    strategy_series: pd.Series,
    start_date: str,
    end_date: str,
    fred_api_key: Optional[str] = None,
) -> dict:
    """Calculate performance metrics for naive and active strategies.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame containing a ``Date`` column and daily ``returns`` for the
        benchmark index.
    naive_returns : pd.Series
        Daily returns of the benchmark strategy.
    strategy_daily_returns : pd.Series
        Daily returns of the active strategy.
    naive_series : pd.Series
        Cumulative return series for the benchmark strategy.
    strategy_series : pd.Series
        Cumulative return series for the active strategy.
    start_date, end_date : str
        Date range used to fetch the risk‑free rate if ``fred_api_key`` is
        provided.
    fred_api_key : str, optional
        API key for FRED. If not supplied, a zero risk‑free rate is assumed.

    Returns
    -------
    dict
        Dictionary containing metrics for both strategies.
    """

    if fred_api_key:
        risk_free_df = get_risk_free_rate(fred_api_key, start_date, end_date)
    else:
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        risk_free_df = pd.DataFrame({"Date": date_range, "daily_rate": 0})

    prices_df = prices_df.copy()
    prices_df.sort_values(by="Date", inplace=True)
    risk_free_df.sort_values(by="Date", inplace=True)
    merged_df = pd.merge_asof(prices_df, risk_free_df, on="Date", direction="backward")
    merged_df["risk_free"] = merged_df["daily_rate"].ffill()

    # --- Naive strategy metrics ---
    merged_df["naive_excess"] = naive_returns - merged_df["risk_free"]
    naive_vol_excess = np.std(merged_df["naive_excess"]) * np.sqrt(252)
    naive_avg_excess = np.mean(merged_df["naive_excess"]) * 252
    naive_sharpe = naive_avg_excess / (naive_vol_excess if naive_vol_excess != 0 else 1)
    naive_downside = (
        np.std(merged_df["naive_excess"][merged_df["naive_excess"] < 0]) * np.sqrt(252)
    )
    if np.isnan(naive_downside) or naive_downside == 0:
        naive_sortino = naive_avg_excess
    else:
        naive_sortino = naive_avg_excess / naive_downside
    naive_drawdown = (naive_series / naive_series.cummax() - 1).min()

    # --- Active strategy metrics ---
    merged_df["strategy_excess"] = strategy_daily_returns - merged_df["risk_free"]
    strategy_vol_excess = np.std(merged_df["strategy_excess"]) * np.sqrt(252)
    strategy_avg_excess = np.mean(merged_df["strategy_excess"]) * 252
    strategy_sharpe = strategy_avg_excess / (
        strategy_vol_excess if strategy_vol_excess != 0 else 1
    )
    strategy_downside = (
        np.std(merged_df["strategy_excess"][merged_df["strategy_excess"] < 0]) * np.sqrt(252)
    )
    if np.isnan(strategy_downside) or strategy_downside == 0:
        strategy_sortino = strategy_avg_excess
    else:
        strategy_sortino = strategy_avg_excess / strategy_downside
    strategy_drawdown = (strategy_series / strategy_series.cummax() - 1).min()

    aligned = pd.concat(
        [strategy_daily_returns.rename("strategy"), naive_returns.rename("naive")],
        axis=1,
    ).dropna()
    naive_var = aligned["naive"].var()
    beta = aligned["strategy"].cov(aligned["naive"]) / naive_var if naive_var > 1e-8 else 0
    jensens_alpha = strategy_avg_excess - beta * naive_avg_excess
    strategy_treynor = strategy_avg_excess / (beta if beta != 0 else 1)

    return {
        "naive_sharpe": naive_sharpe,
        "naive_sortino": naive_sortino,
        "naive_vol_excess": naive_vol_excess,
        "naive_drawdown": naive_drawdown,
        "naive_avg_excess": naive_avg_excess,
        "strategy_sharpe": strategy_sharpe,
        "strategy_sortino": strategy_sortino,
        "strategy_beta": beta,
        "strategy_jensens_alpha": jensens_alpha,
        "strategy_treynor": strategy_treynor,
        "strategy_vol_excess": strategy_vol_excess,
        "strategy_drawdown": strategy_drawdown,
        "strategy_avg_excess": strategy_avg_excess,
    }


def dynamic_market_timing_strategy_macro(df, macro_df, etf_ticker=None):
    """Market timing strategy using price and macroeconomic signals."""

    lookback = 20

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    if "Close" in df.columns:
        price_col = "Close"
    elif "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "close" in df.columns:
        price_col = "close"
    elif "adj close" in df.columns:
        price_col = "adj close"
    else:
        raise ValueError("No valid price column found in the price data.")

    if "returns" not in df.columns:
        df["returns"] = df[price_col].pct_change().fillna(0)

    macro_df = macro_df.copy()
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    macro_df.sort_values("date", inplace=True)

    # ---------------------------------------------------------------
    # Momentum signal
    momentum = df[price_col] / df[price_col].shift(lookback) - 1
    momentum_signal = np.tanh(10 * momentum)

    # ---------------------------------------------------------------
    # Macro signal
    macro_series = macro_df.set_index("date")["value"].reindex(df["Date"], method="ffill")
    macro_series.index = df.index
    rolling_avg = macro_series.rolling(lookback).mean()
    rolling_std = macro_series.rolling(lookback).std().replace(0, 1)
    macro_z = (macro_series - rolling_avg) / rolling_std
    macro_signal = np.tanh(macro_z).fillna(0)

    combined_signal = 0.5 * momentum_signal + 0.5 * macro_signal

    # ---------------------------------------------------------------
    # Volatility scaling
    vol = df["returns"].rolling(lookback).std()
    target_vol = 0.02
    vol_scaling = np.ones_like(vol, dtype=float)
    mask = vol > target_vol
    vol_scaling[mask] = target_vol / vol[mask]

    # ---------------------------------------------------------------
    # Liquidity signal
    if "Volume" in df.columns and not df["Volume"].isnull().all():
        avg_volume = df["Volume"].rolling(lookback).mean()
        liquidity_ratio = df["Volume"] / avg_volume
    else:
        liquidity_ratio = pd.Series(1.0, index=df.index)

    liquidity_ratio = liquidity_ratio.fillna(1)
    liquidity_signal = np.where(liquidity_ratio >= 0.8, liquidity_ratio, liquidity_ratio / 0.8)
    liquidity_signal = np.minimum(liquidity_signal, 1)

    allocation = combined_signal * vol_scaling * liquidity_signal
    allocation = allocation.clip(-1, 1)

    allocation.iloc[:lookback] = 1.0

    return allocation



def dynamic_macro_strategy(
    df: pd.DataFrame,
    macro_df: pd.DataFrame,
    etf_ticker: Optional[str] = None,
) -> pd.Series:
    """Allocate purely based on macroeconomic signals.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with a ``Date`` column.
    macro_df : pd.DataFrame
        Macroeconomic values with ``date`` and ``value`` columns.
    etf_ticker : str, optional
        Kept for API compatibility; ignored in this implementation.

    Returns
    -------
    pd.Series
        Series of allocations indexed like ``df``.
    """

    lookback = 20
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    
    macro_df = macro_df.copy()
    macro_df['date'] = pd.to_datetime(macro_df['date'])
    macro_df.sort_values(by='date', inplace=True)
    
    allocations: list[float] = []
    n = len(df)
    
    for i in range(n):
        if i < lookback:
            allocations.append(1.0)
        else:
            current_date = df['Date'].iloc[i]
            macro_values = macro_df[macro_df['date'] <= current_date]['value']
            if macro_values.empty or len(macro_values) < lookback:
                allocation = 1.0
            else:
                current_macro = macro_values.iloc[-1]
                macro_window = macro_df[macro_df['date'] <= current_date].tail(lookback)
                rolling_avg = macro_window['value'].mean()
                rolling_std = macro_window['value'].std()
                macro_z = (current_macro - rolling_avg) / (rolling_std if rolling_std != 0 else 1)
                allocation = math.tanh(macro_z)
                allocation = max(-1, min(allocation, 1))
            allocations.append(allocation)
    return pd.Series(allocations, index=df.index)
