import pandas as pd
import numpy as np
import math
import yfinance as yf  # Added for ETF liquidity fetching

def simple_backtest(prices_df):
    """
    Naive buy & hold backtest (fully invested).
    
    Parameters:
        prices_df (DataFrame): Historical price data with a valid price column and Date column.
    
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
    
    # Naive strategy: allocation always 1 (fully invested)
    prices_df['strategy_returns'] = prices_df['returns'] * 1
    
    cumulative_series = (prices_df['strategy_returns'] + 1).cumprod()
    if cumulative_series.empty:
        raise ValueError("Cumulative series is empty. Check your data and date range.")
    
    cumulative_return = cumulative_series.iloc[-1] - 1
    alpha = cumulative_return - prices_df['returns'].mean()
    
    return cumulative_return, alpha, cumulative_series


def dynamic_market_timing_strategy_advanced(df, etf_ticker=None):
    """
    Advanced market timing strategy using index data, with ETF liquidity proxy if needed.
    
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
    # If index volume data is available, use it.
    if 'Volume' in df.columns and not df['Volume'].isnull().all():
        recent_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].iloc[-lookback:].mean()
        liquidity_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        liquidity_signal = liquidity_ratio if liquidity_ratio >= 0.8 else liquidity_ratio / 0.8
        liquidity_signal = min(liquidity_signal, 1)
    else:
        # If not available and an ETF ticker is provided, fetch ETF volume data.
        if etf_ticker:
            start_date = df['Date'].min()
            end_date = df['Date'].max()
            etf_data = yf.download(etf_ticker, start=start_date, end=end_date, group_by='column')
            etf_data.reset_index(inplace=True)
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
    """
    Advanced market timing strategy that incorporates macroeconomic data from FRED, with ETF liquidity proxy.
    
    Parameters:
        df (DataFrame): Historical price data; must contain at least 'Close' and 'returns' columns.
        macro_df (DataFrame): Macroeconomic data with columns 'date' and 'value'.
        etf_ticker (str): Optional ETF ticker for liquidity proxy.
        
    Returns:
        allocation (float): Allocation weight between 0 and 1.
    """
    lookback = 20
    if len(df) < lookback + 1 or macro_df.empty:
        return 1
    
    # ---- Momentum Signal ----
    momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-(lookback + 1)] - 1)
    momentum_signal = 1 / (1 + math.exp(-50 * momentum))
    
    # ---- Volatility Signal ----
    vol = df['returns'].iloc[-lookback:].std()
    target_vol = 0.02
    volatility_signal = target_vol / vol if vol > target_vol else 1
    volatility_signal = min(volatility_signal, 1)
    
    # ---- Macro Signal ----
    macro_df['date'] = pd.to_datetime(macro_df['date'])
    macro_df.sort_values(by='date', inplace=True)
    latest_macro_value = macro_df['value'].iloc[-1]
    historical_macro_avg = macro_df['value'].rolling(window=lookback).mean().iloc[-1]
    macro_signal = 1 if latest_macro_value < historical_macro_avg else historical_macro_avg / latest_macro_value
    macro_signal = max(0, min(macro_signal, 1))
    
    # ---- Liquidity Signal ---- (using ETF data if necessary)
    if 'Volume' in df.columns and not df['Volume'].isnull().all():
        recent_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].iloc[-lookback:].mean()
        liquidity_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        liquidity_signal = liquidity_ratio if liquidity_ratio >= 0.8 else liquidity_ratio / 0.8
        liquidity_signal = min(liquidity_signal, 1)
    else:
        if etf_ticker:
            start_date = df['Date'].min()
            end_date = df['Date'].max()
            etf_data = yf.download(etf_ticker, start=start_date, end=end_date, group_by='column')
            etf_data.reset_index(inplace=True)
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
    
    allocation = momentum_signal * volatility_signal * liquidity_signal * macro_signal
    allocation = max(0, min(allocation, 1))
    return allocation


def dynamic_macro_strategy(df, macro_df, etf_ticker=None):
    """
    A market timing strategy that uses only macroeconomic data from FRED as a signal.
    
    Parameters:
        df (DataFrame): Historical price data (used mainly for date alignment).
        macro_df (DataFrame): Macroeconomic data with columns 'date' and 'value'.
        etf_ticker (str): Optional ETF ticker for liquidity proxy (unused here).
    
    Returns:
        allocation (float): Allocation weight between 0 and 1, based solely on macro data.
    """
    lookback = 20
    if macro_df.empty or len(macro_df) < lookback:
        return 1
    
    macro_df['date'] = pd.to_datetime(macro_df['date'])
    macro_df.sort_values(by='date', inplace=True)
    latest_macro = macro_df['value'].iloc[-1]
    historical_avg = macro_df['value'].rolling(window=lookback).mean().iloc[-1]
    if latest_macro <= historical_avg:
        macro_signal = 1
    else:
        macro_signal = historical_avg / latest_macro
    macro_signal = max(0, min(macro_signal, 1))
    return macro_signal
