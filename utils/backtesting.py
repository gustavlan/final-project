import pandas as pd
import numpy as np
import math

def simple_backtest(prices_df, allocation_strategy):
    """
    Backtesting engine without macro data.

    Parameters:
        prices_df (DataFrame): DataFrame expected to have a price column.
        allocation_strategy (function): Function returning allocation weight(s) given prices_df.

    Returns:
        cumulative_return (float): The cumulative return of the strategy.
        alpha (float): A dummy alpha calculation.
        cumulative_series (Series): The cumulative returns over time.
    """
    # Flatten the DataFrame columns if they are tuples
    def flatten_col(col):
        return col[0] if isinstance(col, tuple) else col

    prices_df.columns = [flatten_col(col) for col in prices_df.columns]
    print("Flattened Price DataFrame columns:", prices_df.columns.tolist())

    # Ensure a proper Date column exists
    if 'Date' not in prices_df.columns:
        prices_df.reset_index(inplace=True)
        if 'Date' not in prices_df.columns:
            raise ValueError("Price data must have a 'Date' column.")
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
    print("Price DataFrame shape after dropping NaNs:", prices_df.shape)

    # Calculate daily returns and fill NaN (first row becomes 0)
    prices_df['returns'] = prices_df[price_col].pct_change().fillna(0)
    print("Returns head:", prices_df['returns'].head())

    # Apply the allocation strategy (if scalar, broadcasting works automatically)
    allocation = allocation_strategy(prices_df)
    prices_df['strategy_returns'] = prices_df['returns'] * allocation
    print("Strategy returns head:", prices_df['strategy_returns'].head())

    # Compute cumulative returns
    cumulative_series = (prices_df['strategy_returns'] + 1).cumprod()
    print("Cumulative series head:", cumulative_series.head())

    if cumulative_series.empty:
        raise ValueError("Cumulative series is empty. Check your data and date range.")

    cumulative_return = cumulative_series.iloc[-1] - 1
    # Dummy alpha calculation: difference between cumulative return and average daily return
    alpha = cumulative_return - prices_df['returns'].mean()

    return cumulative_return, alpha, cumulative_series

def simple_backtest_with_macro(prices_df, macro_df, allocation_strategy):
    """
    Backtesting engine that incorporates macroeconomic data.

    Parameters:
        prices_df (DataFrame): Historical price data with a 'Date' column.
        macro_df (DataFrame): Macroeconomic data with columns 'date' and 'value'.
        allocation_strategy (function): Function that returns an allocation weight given the merged DataFrame.

    Returns:
        cumulative_return (float): The cumulative return of the strategy.
        alpha (float): A dummy alpha calculation.
        cumulative_series (Series): The cumulative returns over time.
    """
    # Flatten price DataFrame columns
    def flatten_col(col):
        return col[0] if isinstance(col, tuple) else col

    prices_df.columns = [flatten_col(col) for col in prices_df.columns]
    print("Flattened Price DataFrame columns:", prices_df.columns.tolist())

    if 'Date' not in prices_df.columns:
        prices_df.reset_index(inplace=True)
        if 'Date' not in prices_df.columns:
            raise ValueError("Price data must have a 'Date' column.")
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    prices_df.sort_values(by='Date', inplace=True)

    # Determine price column
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

    prices_df = prices_df.dropna(subset=[price_col])
    print("Price DataFrame shape after dropping NaNs:", prices_df.shape)

    # Clean macro data (convert column names to lowercase)
    macro_df.columns = [col.lower() if isinstance(col, str) else col for col in macro_df.columns]
    if 'date' not in macro_df.columns or 'value' not in macro_df.columns:
        raise ValueError("Macro data must have 'date' and 'value' columns.")
    macro_df['date'] = pd.to_datetime(macro_df['date'])
    macro_df.sort_values(by='date', inplace=True)
    print("Macro DataFrame shape:", macro_df.shape)

    # Merge macro data with price data using merge_asof
    merged_df = pd.merge_asof(prices_df, macro_df, left_on='Date', right_on='date', direction='backward')
    print("Merged DataFrame shape:", merged_df.shape)

    # Calculate returns on the merged DataFrame
    merged_df['returns'] = merged_df[price_col].pct_change().fillna(0)
    print("Merged returns head:\n", merged_df['returns'].head())

    # Apply the allocation strategy (can use macro info)
    allocation = allocation_strategy(merged_df)
    merged_df['strategy_returns'] = merged_df['returns'] * allocation
    print("Merged strategy returns head:\n", merged_df['strategy_returns'].head())

    cumulative_series = (merged_df['strategy_returns'] + 1).cumprod()
    print("Merged cumulative series head:\n", cumulative_series.head())

    if cumulative_series.empty:
        raise ValueError("Cumulative series is empty. Check your data and date range.")

    cumulative_return = cumulative_series.iloc[-1] - 1
    alpha = cumulative_return - merged_df['returns'].mean()

    return cumulative_return, alpha, cumulative_series

# Example allocation strategy: always fully invested.
def full_invested_strategy(df):
    return 1

import math

def dynamic_market_timing_strategy_advanced(df, etf_ticker=None):
    """
    A more advanced market timing strategy that combines momentum, volatility, and liquidity.
    
    Parameters:
        df (DataFrame): Must contain at least 'Close' and 'returns' columns.
        etf_ticker (str): Optional ETF ticker to use for liquidity proxy.
    
    Returns:
        allocation (float): Allocation weight between 0 and 1.
    """
    lookback = 20  # Lookback period in days for computing signals.
    if len(df) < lookback + 1:
        return 1  # Not enough data: default to fully invested.
    
    # ---- Momentum Signal ----
    momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-(lookback + 1)] - 1)
    momentum_signal = 1 / (1 + math.exp(-50 * momentum))
    
    # ---- Volatility Signal ----
    vol = df['returns'].iloc[-lookback:].std()
    target_vol = 0.02  # Target daily volatility (e.g., 2%).
    volatility_signal = target_vol / vol if vol > target_vol else 1
    volatility_signal = min(volatility_signal, 1)
    
    # ---- Liquidity Signal ----
    # If 'Volume' is available, use it; otherwise, if an ETF ticker is provided, you could fetch ETF volume.
    if 'Volume' in df.columns and not df['Volume'].isnull().all():
        recent_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].iloc[-lookback:].mean()
        liquidity_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        liquidity_signal = liquidity_ratio if liquidity_ratio >= 0.8 else liquidity_ratio / 0.8
        liquidity_signal = min(liquidity_signal, 1)
    else:
        # If volume data is not available for the index, we can either:
        # a) Use a proxy ETF's volume (this would require additional data fetching), or
        # b) Assume a neutral liquidity signal.
        liquidity_signal = 1

    # ---- Combine Signals ----
    allocation = momentum_signal * volatility_signal * liquidity_signal
    allocation = max(0, min(allocation, 1))
    
    return allocation
