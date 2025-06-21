import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import pytest
from utils.backtesting import (
    simple_backtest,
    full_invested_strategy,
    dynamic_market_timing_strategy_macro,
    dynamic_market_timing_strategy_advanced,
    _etf_volume_cache,
    get_cached_etf_data,
)


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

    
def test_etf_volume_caching(mock_yfinance):
    """Ensure ETF data is fetched only once for repeated calls."""

    # Create minimal price data without a Volume column to trigger ETF fetch
    dates = pd.date_range(start='2022-01-01', periods=30, freq='D')
    df = pd.DataFrame({'Date': dates, 'Close': range(100, 130)})
    df['returns'] = df['Close'].pct_change().fillna(0)

    # Clear cache to ensure clean test state
    _etf_volume_cache.clear()

    dynamic_market_timing_strategy_advanced(df.copy(), etf_ticker='SPY')
    dynamic_market_timing_strategy_advanced(df.copy(), etf_ticker='SPY')

    assert mock_yfinance['n'] == 1
    assert ('SPY', str(df['Date'].min()), str(df['Date'].max())) in _etf_volume_cache


def test_dynamic_advanced_returns_series():
    """Advanced strategy should return a Series of allocations."""
    dates = pd.date_range(start="2021-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 130, 30), "Volume": 1000})
    df['returns'] = df['Close'].pct_change().fillna(0)

    alloc = dynamic_market_timing_strategy_advanced(df)

    assert isinstance(alloc, pd.Series)
    assert len(alloc) == len(df)
    assert (alloc.iloc[:20] == 1).all()
    assert (alloc >= 0).all() and (alloc <= 1).all()


def test_simple_backtest_with_dynamic_series():
    """Ensure simple_backtest works when the strategy returns a Series."""
    dates = pd.date_range(start="2021-01-01", periods=40, freq="D")
    df = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 140, 40), "Volume": 1000})

    cumulative_return, alpha, cumulative_series = simple_backtest(df.copy(), dynamic_market_timing_strategy_advanced)

    df['returns'] = df['Close'].pct_change().fillna(0)
    alloc = dynamic_market_timing_strategy_advanced(df)
    expected_series = (df['returns'] * alloc + 1).cumprod()

    pd.testing.assert_series_equal(
        cumulative_series.reset_index(drop=True),
        expected_series.reset_index(drop=True),
        check_names=False,
    )


def test_etf_cache_eviction(monkeypatch, mock_yfinance):
    """Cache should evict least recently used items when size limit is exceeded."""
    monkeypatch.setattr('utils.backtesting._ETF_VOLUME_CACHE_MAX_SIZE', 2)
    _etf_volume_cache.clear()

    start, end = '2022-01-01', '2022-01-10'
    get_cached_etf_data('AAA', start, end)
    get_cached_etf_data('BBB', start, end)
    assert len(_etf_volume_cache) == 2

    # This call should trigger eviction of the oldest key ('AAA')
    get_cached_etf_data('CCC', start, end)
    assert len(_etf_volume_cache) == 2
    assert ('AAA', start, end) not in _etf_volume_cache
