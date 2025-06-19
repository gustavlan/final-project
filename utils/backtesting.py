import pandas as pd
import numpy as np
import math
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


def full_invested_strategy(df: pd.DataFrame):
    """Simple allocation strategy that is always fully invested."""
    # Return 1 for every row so broadcasting works in ``simple_backtest``.
    return 1

def simple_backtest(prices_df, allocation_strategy):
    """
    Naive buy & hold or dynamic backtest without macro data.
    
    Parameters:
        prices_df (DataFrame): Historical price data with a valid price column and Date column.
        allocation_strategy (function): Function that returns an allocation weight given the DataFrame.
    
    Returns:
        cumulative_return (float): The cumulative return of the strategy.
        alpha (float): The difference between cumulative return and average daily return.
        cumulative_series (Series): The cumulative returns over time.
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


def dynamic_market_timing_strategy_advanced(df, etf_ticker=None):
    """
    Advanced market timing strategy using index data and ETF liquidity proxy.
    
    Parameters:
        df (DataFrame): Historical price data; must contain at least 'Close' and 'returns' columns.
        etf_ticker (str): Optional ETF ticker for liquidity proxy.
        
    Returns:
        allocation (float): Allocation weight between 0 and 1.
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

    macro_df.sort_values('date', inplace=True)

    # --- Price based calculations ---
    df['returns'] = df['Close'].pct_change().fillna(0)
    momentum = df['Close'].pct_change(periods=lookback)
    momentum_signal = np.tanh(10 * momentum)

    vol = df['returns'].rolling(lookback).std()
    target_vol = 0.02
    vol_scaling = target_vol / vol
    vol_scaling[vol <= target_vol] = 1

    # --- Macro data alignment and signals ---
    macro_series = pd.merge_asof(
        df[['Date']], macro_df[['date', 'value']], left_on='Date', right_on='date', direction='backward'
    )['value']

    macro_mean = macro_series.rolling(lookback).mean()
    macro_std = macro_series.rolling(lookback).std()
    macro_z = (macro_mean - macro_series) / macro_std.replace(0, np.nan)
    macro_signal = np.tanh(macro_z)

    # Replace initial NaNs with neutral values
    momentum_signal = momentum_signal.fillna(0)
    macro_signal = macro_signal.fillna(0)
    vol_scaling = vol_scaling.fillna(1)

    # --- Liquidity ---
    if 'Volume' in df.columns and not df['Volume'].isnull().all():
        avg_volume = df['Volume'].rolling(lookback).mean()
        liquidity_ratio = df['Volume'] / avg_volume
        liquidity_signal = np.where(liquidity_ratio >= 0.8, liquidity_ratio, liquidity_ratio / 0.8)
        liquidity_signal = np.minimum(liquidity_signal, 1)
        liquidity_signal = pd.Series(liquidity_signal, index=df.index).fillna(1)
    else:
        liquidity_signal = pd.Series(1, index=df.index)

    combined_signal = 0.5 * momentum_signal + 0.5 * macro_signal
    allocation = combined_signal * vol_scaling * liquidity_signal
    allocation = allocation.clip(-1, 1)

    # Until we have enough data, default to fully invested
    allocation.iloc[:lookback] = 1

    return pd.Series(allocation, index=df.index)


def dynamic_macro_strategy(df, macro_df, etf_ticker=None):
    lookback = 20
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    
    macro_df = macro_df.copy()
    macro_df['date'] = pd.to_datetime(macro_df['date'])
    macro_df.sort_values(by='date', inplace=True)
    
    allocations = []
    n = len(df)
    
    for i in range(n):
        if i < lookback:
            allocations.append(1)
        else:
            current_date = df['Date'].iloc[i]
            macro_values = macro_df[macro_df['date'] <= current_date]['value']
            if macro_values.empty or len(macro_values) < lookback:
                allocation = 1
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
