import pandas as pd
import numpy as np

def simple_backtest(prices_df, allocation_strategy):
    """
    Backtesting engine.

    Parameters:
        prices_df (DataFrame): DataFrame expected to have a price column.
        allocation_strategy (function): Function returning allocation weight(s) given prices_df.

    Returns:
        cumulative_return (float): The cumulative return of the strategy.
        alpha (float): A dummy alpha calculation.
    """
    # Flatten the DataFrame columns if they are tuples
    def flatten_col(col):
        return col[0] if isinstance(col, tuple) else col

    prices_df.columns = [flatten_col(col) for col in prices_df.columns]
    print("Flattened DataFrame columns:", prices_df.columns.tolist())

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
        raise ValueError("No valid price column found in the DataFrame.")

    # Ensure DataFrame is sorted by date if a 'Date' column exists
    if 'Date' in prices_df.columns:
        prices_df.sort_values(by='Date', inplace=True)
    else:
        prices_df.sort_index(inplace=True)

    # Drop rows where the price column is NaN
    prices_df = prices_df.dropna(subset=[price_col])
    print("DataFrame shape after dropping NaNs:", prices_df.shape)

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

    return cumulative_return, alpha

# Example allocation strategy: always fully invested
def full_invested_strategy(prices_df):
    return 1
