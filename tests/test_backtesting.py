import os
import sys
import pandas as pd
import numpy as np
import pytest
from utils.backtesting import (
    simple_backtest,
    full_invested_strategy,
    dynamic_market_timing_strategy_macro,
    dynamic_market_timing_strategy_advanced,
    _etf_volume_cache,
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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

    
def test_dynamic_market_timing_strategy_macro_basic():
    """Ensure the macro strategy returns a valid allocation series."""
    dates = pd.date_range(start="2021-01-01", periods=30, freq="D")
    prices = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 130, 30), "Volume": 1000})
    macro = pd.DataFrame({"date": dates, "value": np.linspace(1, 2, 30)})

    alloc = dynamic_market_timing_strategy_macro(prices, macro)

    assert isinstance(alloc, pd.Series)
    assert len(alloc) == len(prices)
    # First lookback values should be 1
    assert (alloc.iloc[:20] == 1).all()
    # Allocation bounds
    assert (alloc <= 1).all() and (alloc >= -1).all()

    
def test_etf_volume_caching(monkeypatch):
    """Ensure ETF data is fetched only once for repeated calls."""
    import pandas as pd

    # Create minimal price data without a Volume column to trigger ETF fetch
    dates = pd.date_range(start='2022-01-01', periods=30, freq='D')
    df = pd.DataFrame({'Date': dates, 'Close': range(100, 130)})
    df['returns'] = df['Close'].pct_change().fillna(0)

    call_count = {'n': 0}

    def dummy_download(ticker, start, end, group_by='column'):
        call_count['n'] += 1
        vol_df = pd.DataFrame({'Volume': [100000] * len(dates)}, index=dates)
        vol_df.index.name = 'Date'
        return vol_df

    monkeypatch.setattr('utils.backtesting.yf.download', dummy_download)

    dynamic_market_timing_strategy_advanced(df.copy(), etf_ticker='SPY')
    dynamic_market_timing_strategy_advanced(df.copy(), etf_ticker='SPY')

    assert call_count['n'] == 1
    assert ('SPY', str(df['Date'].min()), str(df['Date'].max())) in _etf_volume_cache
