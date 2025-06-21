# Helper functions to run backtests and compute metrics
from __future__ import annotations

import os
import pandas as pd
import numpy as np

from utils.data_retrieval import (
    get_yahoo_data,
    get_fred_data,
    get_risk_free_rate,
)
from utils.backtesting import (
    simple_backtest,
    dynamic_market_timing_strategy_advanced,
    dynamic_market_timing_strategy_macro,
    dynamic_macro_strategy,
)
from utils.visualizations import create_return_plot


# Mapping from index ticker to ETF ticker used as a liquidity proxy
ETF_MAPPING = {
    '^GSPC': 'SPY',
    '^DJI': 'DIA',
    '^IXIC': 'QQQ',
    '^FTSE': 'ISF',
    '^N225': 'EWJ',
    '^HSI': '2800.HK',
    '^GDAXI': 'DAXY',
    '^FCHI': 'EWU',
    '^STOXX50E': 'FEZ',
    '^BSESN': 'INDA',
}


def fetch_price_data(symbol: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
    """Download price data and determine the usable price column."""
    prices_df = get_yahoo_data(symbol, start_date, end_date)
    prices_df.reset_index(inplace=True)
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])

    # Flatten multi-index columns if present
    def flatten(col):
        return col[1] if isinstance(col, tuple) and len(col) > 1 else col

    prices_df.columns = [flatten(c) for c in prices_df.columns]

    # Handle edge case where columns are like ['', '^GSPC', '^GSPC', ...]
    cols = prices_df.columns.tolist()
    if cols and cols[0] == '' and all(c == symbol for c in cols[1:]):
        if len(cols) == 7:
            prices_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        elif len(cols) == 6:
            prices_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close']

    if 'Close' in prices_df.columns:
        price_col = 'Close'
    elif 'Adj Close' in prices_df.columns:
        price_col = 'Adj Close'
    elif 'close' in prices_df.columns:
        price_col = 'close'
    elif 'adj close' in prices_df.columns:
        price_col = 'adj close'
    else:
        raise ValueError(
            "No valid price column found. Available columns: " + str(prices_df.columns.tolist())
        )

    return prices_df, price_col


def compute_naive(prices_df: pd.DataFrame, price_col: str, symbol: str, start: str, end: str):
    """Return naive cumulative series and related values."""
    returns = prices_df[price_col].pct_change().fillna(0)
    cumulative = (returns + 1).cumprod()
    if cumulative.empty:
        raise ValueError(f"No price data available for symbol {symbol} in the requested date range {start} to {end}")
    total_return = float(cumulative.iloc[-1] - 1)
    alpha = float(total_return - returns.mean())
    return returns, cumulative, total_return, alpha


def apply_strategy(
    prices_df: pd.DataFrame,
    price_col: str,
    strategy_method: str,
    start_date: str,
    end_date: str,
    symbol: str,
    naive_returns: pd.Series,
    naive_series: pd.Series,
    naive_return: float,
    naive_alpha: float,
):
    """Run the selected strategy and return results."""
    etf_ticker = ETF_MAPPING.get(symbol)

    if strategy_method == 'naive':
        series = naive_series
        daily = naive_returns
        ret = naive_return
        alpha = naive_alpha
    elif strategy_method == 'advanced':
        ret, alpha, series = simple_backtest(
            prices_df, lambda df: dynamic_market_timing_strategy_advanced(df, etf_ticker)
        )
        ret = float(ret)
        alpha = float(alpha)
        daily = series.pct_change().fillna(0)
    elif strategy_method == 'macro':
        fred_api_key = os.getenv('FRED_API_KEY')
        if not fred_api_key:
            raise RuntimeError('FRED API key not set')
        macro_df = get_fred_data(fred_api_key, 'DGS3MO', start_date, end_date)
        ret, alpha, series = simple_backtest(
            prices_df,
            lambda df: dynamic_market_timing_strategy_macro(df, macro_df, etf_ticker),
        )
        ret = float(ret)
        alpha = float(alpha)
        daily = series.pct_change().fillna(0)
    elif strategy_method == 'macro_only':
        fred_api_key = os.getenv('FRED_API_KEY')
        if not fred_api_key:
            raise RuntimeError('FRED API key not set')
        macro_df = get_fred_data(fred_api_key, 'DGS3MO', start_date, end_date)
        allocation = dynamic_macro_strategy(prices_df, macro_df, etf_ticker)
        prices_df['returns'] = prices_df[price_col].pct_change().fillna(0)
        series = (prices_df['returns'] * allocation + 1).cumprod()
        ret = float(series.iloc[-1] - 1)
        alpha = float(ret - prices_df['returns'].mean())
        daily = prices_df['returns'] * allocation
    else:
        series = naive_series
        daily = naive_returns
        ret = naive_return
        alpha = naive_alpha

    if strategy_method == 'advanced':
        label = 'Advanced Market Timing Strategy'
    elif strategy_method == 'macro':
        label = 'Macro Market Timing Strategy'
    elif strategy_method == 'macro_only':
        label = 'Macro-Only Strategy'
    else:
        label = 'Naive Buy & Hold Strategy'

    return series, daily, ret, alpha, label


def performance_metrics(
    prices_df: pd.DataFrame,
    naive_returns: pd.Series,
    naive_series: pd.Series,
    strategy_daily: pd.Series,
    strategy_series: pd.Series,
    start_date: str,
    end_date: str,
):
    """Calculate risk-adjusted metrics for naive and strategy returns."""
    fred_api_key = os.getenv('FRED_API_KEY')
    if fred_api_key:
        risk_free_df = get_risk_free_rate(fred_api_key, start_date, end_date)
    else:
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        risk_free_df = pd.DataFrame({'Date': date_range, 'daily_rate': 0})

    prices_df.sort_values('Date', inplace=True)
    risk_free_df.sort_values('Date', inplace=True)
    merged = pd.merge_asof(prices_df, risk_free_df, on='Date', direction='backward')
    merged['risk_free'] = merged['daily_rate'].ffill()

    merged['naive_excess'] = naive_returns - merged['risk_free']
    naive_vol_excess = np.std(merged['naive_excess']) * np.sqrt(252)
    naive_avg_excess = np.mean(merged['naive_excess']) * 252
    naive_sharpe = naive_avg_excess / (naive_vol_excess if naive_vol_excess != 0 else 1)
    naive_downside = np.std(merged['naive_excess'][merged['naive_excess'] < 0]) * np.sqrt(252)
    naive_sortino = naive_avg_excess / (naive_downside if naive_downside != 0 else 1)
    naive_drawdown = (naive_series / naive_series.cummax() - 1).min()

    merged['strategy_excess'] = strategy_daily - merged['risk_free']
    strategy_vol_excess = np.std(merged['strategy_excess']) * np.sqrt(252)
    strategy_avg_excess = np.mean(merged['strategy_excess']) * 252
    strategy_sharpe = strategy_avg_excess / (strategy_vol_excess if strategy_vol_excess != 0 else 1)
    strategy_downside = np.std(merged['strategy_excess'][merged['strategy_excess'] < 0]) * np.sqrt(252)
    strategy_sortino = strategy_avg_excess / (strategy_downside if strategy_downside != 0 else 1)
    strategy_drawdown = (strategy_series / strategy_series.cummax() - 1).min()

    aligned = pd.concat([
        strategy_daily.rename('strategy'),
        naive_returns.rename('naive'),
    ], axis=1).dropna()
    naive_var = aligned['naive'].var()
    beta = aligned['strategy'].cov(aligned['naive']) / (naive_var if naive_var != 0 else 1)
    jensens_alpha = strategy_avg_excess - beta * naive_avg_excess
    strategy_treynor = strategy_avg_excess / (beta if beta != 0 else 1)

    return {
        'naive_vol_excess': naive_vol_excess,
        'naive_avg_excess': naive_avg_excess,
        'naive_sharpe': naive_sharpe,
        'naive_sortino': naive_sortino,
        'naive_drawdown': naive_drawdown,
        'strategy_vol_excess': strategy_vol_excess,
        'strategy_avg_excess': strategy_avg_excess,
        'strategy_sharpe': strategy_sharpe,
        'strategy_sortino': strategy_sortino,
        'strategy_drawdown': strategy_drawdown,
        'strategy_beta': beta,
        'strategy_jensens_alpha': jensens_alpha,
        'strategy_treynor': strategy_treynor,
    }


def run_backtest(symbol: str, start: str, end: str, method: str) -> dict:
    """Execute full backtest workflow and return metrics."""
    prices_df, price_col = fetch_price_data(symbol, start, end)
    naive_returns, naive_series, naive_return, naive_alpha = compute_naive(prices_df, price_col, symbol, start, end)
    strat_series, strat_daily, strat_return, strat_alpha, label = apply_strategy(
        prices_df,
        price_col,
        method,
        start,
        end,
        symbol,
        naive_returns,
        naive_series,
        naive_return,
        naive_alpha,
    )
    metrics = performance_metrics(
        prices_df,
        naive_returns,
        naive_series,
        strat_daily,
        strat_series,
        start,
        end,
    )
    plot_html = create_return_plot(prices_df['Date'], naive_series, strat_series)

    result = {
        'naive_return': naive_return,
        'naive_alpha': naive_alpha,
        'strategy_return': strat_return,
        'strategy_alpha': strat_alpha,
        'strategy_label': label,
        'strategy_method': method,
        'plot_html': plot_html,
        **metrics,
    }
    return result
