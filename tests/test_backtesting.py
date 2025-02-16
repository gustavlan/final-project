import pandas as pd
import numpy as np
import pytest
from utils.backtesting import simple_backtest, full_invested_strategy

def test_simple_backtest():
    # Dummy DataFrame with controlled price data.
    data = {
        'Date': pd.date_range(start='2021-01-01', periods=5, freq='D'),
        'Close': [100, 102, 101, 103, 105]
    }
    df = pd.DataFrame(data)
    
    # Run the backtest using the fully invested strategy.
    cumulative_return, alpha, cumulative_series = simple_backtest(df.copy(), full_invested_strategy)
    
    # Daily returns are calculated as (price_today / price_yesterday) - 1:
    # Day 1: NaN replaced with 0, then:
    # Day 2: (102/100 - 1) = 0.02
    # Day 3: (101/102 - 1) ≈ -0.0098, etc.
    # For cumulative return, calculate (1 + return).cumprod() - 1
    expected_returns = df['Close'].pct_change().fillna(0)
    expected_cum_return = (expected_returns + 1).cumprod().iloc[-1] - 1
    
    # Test to compare the values (allowing small differences)
    np.testing.assert_almost_equal(cumulative_return, expected_cum_return, decimal=2)
