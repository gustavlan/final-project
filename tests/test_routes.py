"""Integration tests for the Flask routes."""

import os
import pytest
from app import create_app
from config import TestingConfig
from extensions import db


@pytest.fixture
def create_client():
    def _create_client():
        app = create_app(TestingConfig)
        app.config["TESTING"] = True
        with app.app_context():
            db.create_all()
        return app.test_client()

    return _create_client


def test_home_route(create_client):
    client = create_client()
    # Test that the home page loads correctly.
    response = client.get("/")
    assert response.status_code == 200
    assert b"Asset Allocation Backtester" in response.data  # Check for known content


def test_backtest_route_simple(create_client, mock_yfinance, mock_fred):

    # Post form data to the /backtest route with the naive strategy.
    client = create_client()
    response = client.post(
        "/backtest",
        data={
            "symbol": "DUMMY",
            "start_date": "2021-01-01",
            "end_date": "2021-01-05",
            "strategy_method": "naive",
        },
    )
    # Check that the response is OK and contains expected elements.
    assert response.status_code == 200
    assert b"Cumulative Return" in response.data


def test_beta_zero_flat_prices(create_client, mock_yfinance, mock_fred):
    """Beta should be zero when benchmark returns have zero variance."""

    client = create_client()
    response = client.post(
        "/backtest",
        data={
            "symbol": "DUMMY",
            "start_date": "2021-01-01",
            "end_date": "2021-01-05",
            "strategy_method": "naive",
        },
    )

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(response.data, "html.parser")
    beta_values = [
        float(cells[1].text)
        for row in soup.find_all("tr")
        for cells in [row.find_all("td")]
        if len(cells) == 2 and cells[0].text.strip() == "Beta"
    ]
    assert len(beta_values) >= 2
    # Second beta corresponds to the strategy metrics
    assert beta_values[1] == 0.0


def test_yahoo_error_returns_message(create_client, mock_yfinance_error, mock_fred):
    client = create_client()
    response = client.post(
        "/backtest",
        data={
            "symbol": "DUMMY",
            "start_date": "2021-01-01",
            "end_date": "2021-01-05",
            "strategy_method": "naive",
        },
    )
    assert response.status_code == 502
    assert b"Yahoo Finance" in response.data


def test_fred_error_returns_message(create_client, mock_yfinance, mock_fred_error):
    client = create_client()
    os.environ["FRED_API_KEY"] = "dummy"
    response = client.post(
        "/backtest",
        data={
            "symbol": "DUMMY",
            "start_date": "2021-01-01",
            "end_date": "2021-01-05",
            "strategy_method": "macro",
        },
    )
    assert response.status_code == 502
    assert b"FRED" in response.data


def test_missing_risk_free_data_returns_message(create_client, mock_yfinance):
    os.environ.pop("FRED_API_KEY", None)
    os.environ.pop("RISK_FREE_RATE", None)
    client = create_client()
    response = client.post(
        "/backtest",
        data={
            "symbol": "DUMMY",
            "start_date": "2021-01-01",
            "end_date": "2021-01-05",
            "strategy_method": "naive",
        },
    )
    assert response.status_code == 500
    assert b"Risk-free rate" in response.data
    os.environ.setdefault("RISK_FREE_RATE", "0")
