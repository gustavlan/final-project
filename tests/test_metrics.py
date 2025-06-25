import pandas as pd
import numpy as np
from utils.backtesting import compute_metrics


def test_compute_metrics_basic():
    dates = pd.date_range(start="2021-01-01", periods=5, freq="D")
    returns = pd.Series([0.0, 0.02, -0.01, 0.03, -0.02])
    prices_df = pd.DataFrame({"Date": dates, "returns": returns})
    naive_returns = prices_df["returns"]
    strategy_daily_returns = naive_returns * 2
    naive_series = (naive_returns + 1).cumprod()
    strategy_series = (strategy_daily_returns + 1).cumprod()

    metrics = compute_metrics(
        prices_df,
        naive_returns,
        strategy_daily_returns,
        naive_series,
        strategy_series,
        "2021-01-01",
        "2021-01-05",
    )

    assert np.isclose(metrics["strategy_beta"], 2.0)
    assert np.isclose(metrics["strategy_jensens_alpha"], 0.0)
    assert np.isclose(metrics["strategy_treynor"], metrics["naive_avg_excess"])


def test_compute_metrics_zero_variance():
    dates = pd.date_range(start="2021-01-01", periods=5, freq="D")
    returns = pd.Series([0.0] * 5)
    prices_df = pd.DataFrame({"Date": dates, "returns": returns})
    naive_returns = prices_df["returns"]
    strategy_daily_returns = naive_returns.copy()
    naive_series = (naive_returns + 1).cumprod()
    strategy_series = naive_series.copy()

    metrics = compute_metrics(
        prices_df,
        naive_returns,
        strategy_daily_returns,
        naive_series,
        strategy_series,
        "2021-01-01",
        "2021-01-05",
    )

    assert metrics["strategy_beta"] == 0

def test_sortino_no_negative_returns():
    dates = pd.date_range(start="2021-01-01", periods=5, freq="D")
    returns = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02])
    prices_df = pd.DataFrame({"Date": dates, "returns": returns})
    naive_returns = prices_df["returns"]
    strategy_daily_returns = naive_returns.copy()
    naive_series = (naive_returns + 1).cumprod()
    strategy_series = naive_series.copy()

    metrics = compute_metrics(
        prices_df,
        naive_returns,
        strategy_daily_returns,
        naive_series,
        strategy_series,
        "2021-01-01",
        "2021-01-05",
    )

    assert not np.isnan(metrics["naive_sortino"])
    assert not np.isnan(metrics["strategy_sortino"])

