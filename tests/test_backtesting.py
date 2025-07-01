"""Unit tests for the backtesting utilities."""

import pandas as pd
import numpy as np
import pytest
from utils.backtesting import (
    simple_backtest,
    full_invested_strategy,
    dynamic_market_timing_strategy_macro,
    dynamic_market_timing_strategy_advanced,
    dynamic_macro_strategy,
    ExecutionModel,
    _etf_volume_cache,
    _enforce_cache_limit,
    _ETF_VOLUME_CACHE_MAX_SIZE,
)


def test_simple_backtest():
    # Dummy DataFrame with controlled price data.
    data = {
        "Date": pd.date_range(start="2021-01-01", periods=5, freq="D"),
        "Close": [100, 102, 101, 103, 105],
    }
    df = pd.DataFrame(data)

    # Run the backtest using the fully invested strategy.
    cumulative_return, alpha, cumulative_series = simple_backtest(df.copy(), full_invested_strategy)

    # Daily returns are calculated as (price_today / price_yesterday) - 1:
    # Day 1: NaN replaced with 0, then:
    # Day 2: (102/100 - 1) = 0.02
    # Day 3: (101/102 - 1) ≈ -0.0098, etc.
    # For cumulative return, calculate (1 + return).cumprod() - 1
    expected_returns = df["Close"].pct_change().fillna(0)
    expected_cum_return = (expected_returns + 1).cumprod().iloc[-1] - 1

    # Test to compare the values (allowing small differences)
    np.testing.assert_almost_equal(cumulative_return, expected_cum_return, decimal=2)


def test_simple_backtest_with_execution_costs():
    dates = pd.date_range(start="2021-01-01", periods=4, freq="D")
    df = pd.DataFrame({"Date": dates, "Close": [100, 101, 102, 103]})

    def strategy(_df: pd.DataFrame) -> pd.Series:
        return pd.Series([0, 1, 1, 0], index=_df.index)

    model = ExecutionModel(bid_ask_spread=0.01, commission=0.001)

    cumulative_return, alpha, series = simple_backtest(df.copy(), strategy, execution_model=model)

    df["returns"] = df["Close"].pct_change().fillna(0)
    alloc = strategy(df)
    trade_size = alloc.diff().abs().fillna(alloc.iloc[0])
    cost = trade_size * model.bid_ask_spread + (trade_size > 0) * model.commission
    expected_series = (df["returns"] * alloc - cost + 1).cumprod()

    pd.testing.assert_series_equal(
        series.reset_index(drop=True),
        expected_series.reset_index(drop=True),
        check_names=False,
    )


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
    dates = pd.date_range(start="2022-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Date": dates, "Close": range(100, 130)})
    df["returns"] = df["Close"].pct_change().fillna(0)

    # Clear cache to ensure clean test state
    _etf_volume_cache.clear()

    dynamic_market_timing_strategy_advanced(df.copy(), etf_ticker="SPY")
    dynamic_market_timing_strategy_advanced(df.copy(), etf_ticker="SPY")

    assert mock_yfinance["n"] == 1
    assert ("SPY", str(df["Date"].min()), str(df["Date"].max())) in _etf_volume_cache


def test_disk_cache_cleanup(tmp_path):
    """Ensure disk cache is cleaned when exceeding the size limit."""

    _etf_volume_cache.clear()
    cache_dir = tmp_path / "cache"

    for i in range(_ETF_VOLUME_CACHE_MAX_SIZE + 5):
        key = (f"T{i}", "2020-01-01", "2020-01-02")
        df = pd.DataFrame({"Date": [pd.Timestamp("2020-01-01")], "Close": [1], "Volume": [1]})
        _etf_volume_cache[key] = df

    _enforce_cache_limit(str(cache_dir))

    cache_files = list(cache_dir.glob("*.pkl"))
    assert len(cache_files) <= _ETF_VOLUME_CACHE_MAX_SIZE


def test_dynamic_advanced_returns_series():
    """Advanced strategy should return a Series of allocations."""
    dates = pd.date_range(start="2021-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 130, 30), "Volume": 1000})
    df["returns"] = df["Close"].pct_change().fillna(0)

    alloc = dynamic_market_timing_strategy_advanced(df)

    assert isinstance(alloc, pd.Series)
    assert len(alloc) == len(df)
    assert (alloc.iloc[:20] == 1).all()
    assert (alloc >= 0).all() and (alloc <= 1).all()


def test_simple_backtest_with_dynamic_series():
    """Ensure simple_backtest works when the strategy returns a Series."""
    dates = pd.date_range(start="2021-01-01", periods=40, freq="D")
    df = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 140, 40), "Volume": 1000})

    cumulative_return, alpha, cumulative_series = simple_backtest(
        df.copy(), dynamic_market_timing_strategy_advanced
    )

    df["returns"] = df["Close"].pct_change().fillna(0)
    alloc = dynamic_market_timing_strategy_advanced(df)
    expected_series = (df["returns"] * alloc + 1).cumprod()

    pd.testing.assert_series_equal(
        cumulative_series.reset_index(drop=True),
        expected_series.reset_index(drop=True),
        check_names=False,
    )


def test_momentum_signal_no_overflow():
    """Extreme price changes should not cause overflow in momentum signal."""
    dates = pd.date_range(start="2021-01-01", periods=25, freq="D")
    up = pd.DataFrame({"Date": dates, "Close": [1] * 20 + [1000] * 5, "Volume": 1000})
    up["returns"] = up["Close"].pct_change().fillna(0)
    alloc_up = dynamic_market_timing_strategy_advanced(up)
    assert np.isfinite(alloc_up.iloc[-1])

    down = pd.DataFrame({"Date": dates, "Close": [1000] * 20 + [1] * 5, "Volume": 1000})
    down["returns"] = down["Close"].pct_change().fillna(0)
    alloc_down = dynamic_market_timing_strategy_advanced(down)
    assert np.isfinite(alloc_down.iloc[-1])


def test_macro_allocation_positive_trend():
    """Macro-only strategy should allocate positively when macro trend is up."""
    dates = pd.date_range(start="2021-01-01", periods=40, freq="D")
    prices = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 140, 40)})
    macro = pd.DataFrame({"date": dates, "value": np.linspace(1, 2, 40)})

    alloc = dynamic_macro_strategy(prices, macro)

    # After the lookback period, the allocation should be positive for an
    # upward-trending macro series.
    assert alloc.iloc[-1] > 0


def test_zero_volatility_allocation():
    """Strategy should not raise errors when volatility is zero."""
    dates = pd.date_range(start="2021-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Date": dates, "Close": 100, "Volume": 1000})
    df["returns"] = df["Close"].pct_change().fillna(0)

    alloc = dynamic_market_timing_strategy_advanced(df)

    assert np.all(np.isfinite(alloc))
    # After the lookback period the allocation should equal the momentum signal
    assert alloc.iloc[20] == pytest.approx(0.5, abs=1e-6)
