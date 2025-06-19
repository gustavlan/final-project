"""Utility functions for running simple and dynamic backtests."""

from typing import Callable, Optional, Tuple, Union

import math
import pandas as pd
import numpy as np
import yfinance as yf  # For ETF liquidity data fetching

# In-memory cache for ETF volume data keyed by (ticker, start_date, end_date)
_etf_volume_cache = {}


def get_cached_etf_data(etf_ticker, start_date, end_date):
    """Retrieve ETF data using yfinance with simple in-memory caching."""
    key = (etf_ticker, str(start_date), str(end_date))
    if key not in _etf_volume_cache:
        data = yf.download(etf_ticker, start=start_date, end=end_date, group_by='column')
        data.reset_index(inplace=True)
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            def flatten(col):
                return col[1] if isinstance(col, tuple) and len(col) > 1 else col

            data.columns = [flatten(col) for col in data.columns]
        _etf_volume_cache[key] = data
    return _etf_volume_cache[key]


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
) -> float:
    """Calculate an allocation weight using momentum, volatility and liquidity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``Date``, ``Close`` and ``returns`` columns.
    etf_ticker : str, optional
        ETF symbol to fetch volume data from if index volume is missing.

    Returns
    -------
    float
        Allocation weight between 0 and 1.
    """
    lookback = 20
    if len(df) < lookback + 1:
        return 1  # Not enough data; default to fully invested.
    
    # ---- Momentum Signal ----
    momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-(lookback + 1)] - 1)
    momentum_signal = 1 / (1 + math.exp(-50 * momentum))
    
    # ---- Volatility Signal ----
    vol = df['returns'].iloc[-lookback:].std()
    target_vol = 0.02  # Target daily volatility (e.g., 2%)
    volatility_signal = target_vol / vol if vol > target_vol else 1
    volatility_signal = min(volatility_signal, 1)
    
    # ---- Liquidity Signal ----
    if 'Volume' in df.columns and not df['Volume'].isnull().all():
        recent_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].iloc[-lookback:].mean()
        liquidity_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        liquidity_signal = liquidity_ratio if liquidity_ratio >= 0.8 else liquidity_ratio / 0.8
        liquidity_signal = min(liquidity_signal, 1)
    else:
        # If index volume data is missing and an ETF ticker is provided, fetch ETF data.
        if etf_ticker:
            start_date = df['Date'].min()
            end_date = df['Date'].max()
            etf_data = get_cached_etf_data(etf_ticker, start_date, end_date)
            if 'Volume' in etf_data.columns and not etf_data['Volume'].isnull().all():
                sub_df = etf_data[(etf_data['Date'] >= start_date) & (etf_data['Date'] <= end_date)]
                recent_volume = sub_df['Volume'].iloc[-1]
                avg_volume = sub_df['Volume'].iloc[-lookback:].mean()
                liquidity_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                liquidity_signal = liquidity_ratio if liquidity_ratio >= 0.8 else liquidity_ratio / 0.8
                liquidity_signal = min(liquidity_signal, 1)
            else:
                liquidity_signal = 1
        else:
            liquidity_signal = 1
    
    allocation = momentum_signal * volatility_signal * liquidity_signal
    allocation = max(0, min(allocation, 1))
    return allocation


def dynamic_market_timing_strategy_macro(df, macro_df, etf_ticker=None):
    """Vectorised market timing strategy using price and macro signals."""
    lookback = 20

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    macro_df = macro_df.copy()
    macro_df['date'] = pd.to_datetime(macro_df['date'])
    macro_df.sort_values(by='date', inplace=True)
    
    allocations: list[float] = []
    n = len(df)
    
    for i in range(n):
        # Until we have enough data, default to fully long
        if i < lookback:
            allocations.append(1.0)
        else:
            window = df.iloc[i - lookback : i + 1].copy()
            # Ensure we have daily returns computed
            window['returns'] = window['Close'].pct_change().fillna(0)
            
            # --- Momentum Signal ---
            # Use the change from the first day in the window to the current day
            momentum = window['Close'].iloc[-1] / window['Close'].iloc[0] - 1
            momentum_signal = math.tanh(10 * momentum)  # scales to roughly -1 to 1
            
            # --- Macro Signal ---
            # Get macro data up to the current date
            current_date = window['Date'].iloc[-1]
            macro_values = macro_df[macro_df['date'] <= current_date]['value']
            if macro_values.empty or len(macro_values) < lookback:
                macro_signal = 0.0  # neutral if not enough macro data
            else:
                current_macro = macro_values.iloc[-1]
                # Use the last 'lookback' days from the macro series
                macro_window = macro_df[macro_df['date'] <= current_date].tail(lookback)
                rolling_avg = macro_window['value'].mean()
                rolling_std = macro_window['value'].std()
                macro_z = (rolling_avg - current_macro) / (rolling_std if rolling_std != 0 else 1)
                macro_signal = math.tanh(macro_z)
            
            # --- Combine Signals ---
            combined_signal = 0.5 * momentum_signal + 0.5 * macro_signal
            
            # --- Volatility Scaling ---
            vol = window['returns'].std()
            target_vol = 0.02  # Target daily volatility
            vol_scaling = target_vol / vol if vol > target_vol else 1
            
            # --- Liquidity Signal ---
            if 'Volume' in window.columns and not window['Volume'].isnull().all():
                recent_volume = window['Volume'].iloc[-1]
                avg_volume = window['Volume'].mean()
                liquidity_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                liquidity_signal = liquidity_ratio if liquidity_ratio >= 0.8 else liquidity_ratio / 0.8
                liquidity_signal = min(liquidity_signal, 1)
            else:
                liquidity_signal = 1
            
            # --- Final Allocation ---
            allocation = combined_signal * vol_scaling * liquidity_signal
            # Clamp allocation between -1 (fully short) and 1 (fully long)
            allocation = max(-1, min(allocation, 1))
            allocations.append(allocation)
            
    # Return a series that aligns with the input DataFrame's index
    return pd.Series(allocations, index=df.index)



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
                macro_z = (rolling_avg - current_macro) / (rolling_std if rolling_std != 0 else 1)
                allocation = math.tanh(macro_z)
                allocation = max(-1, min(allocation, 1))
            allocations.append(allocation)
    return pd.Series(allocations, index=df.index)
