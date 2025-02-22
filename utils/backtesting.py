import pandas as pd
import numpy as np
import math
import yfinance as yf  # For ETF liquidity data fetching

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
    
    return cumulative_return, alpha, cumulative_series, prices_df['strategy_returns']


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
            etf_data = yf.download(etf_ticker, start=start_date, end=end_date, group_by='column')
            etf_data.reset_index(inplace=True)
            # Flatten ETF columns if they are MultiIndex.
            if isinstance(etf_data.columns, pd.MultiIndex):
                def flatten_etf(col):
                    return col[1] if isinstance(col, tuple) and len(col) > 1 else col
                etf_data.columns = [flatten_etf(col) for col in etf_data.columns]
            if 'Volume' in etf_data.columns and not etf_data['Volume'].isnull().all():
                recent_volume = etf_data['Volume'].iloc[-1]
                avg_volume = etf_data['Volume'].iloc[-lookback:].mean()
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
    lookback = 20
    # Work on copies and ensure date ordering
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    
    macro_df = macro_df.copy()
    macro_df['date'] = pd.to_datetime(macro_df['date'])
    macro_df.sort_values(by='date', inplace=True)
    
    allocations = []
    n = len(df)
    
    for i in range(n):
        # Until we have enough data, default to fully long
        if i < lookback:
            allocations.append(1)
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
                macro_signal = 0  # neutral if not enough macro data
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
