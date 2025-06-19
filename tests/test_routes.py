import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import create_app
from config import TestingConfig
from extensions import db

@pytest.fixture
def create_client():
    def _create_client():
        app = create_app(TestingConfig)
        app.config['TESTING'] = True
        with app.app_context():
            db.create_all()
        return app.test_client()

    return _create_client

def test_home_route(create_client):
    client = create_client()
    # Test that the home page loads correctly.
    response = client.get('/')
    assert response.status_code == 200
    assert b'Asset Allocation Backtester' in response.data  # Check for known content

def test_backtest_route_simple(create_client, monkeypatch):
    # Create a dummy version of get_yahoo_data to return a controlled DataFrame
    import pandas as pd
    def dummy_get_yahoo_data(symbol, start_date, end_date):
        data = {
            'Date': pd.date_range(start=start_date, periods=5, freq='D'),
            'Close': [100, 102, 101, 103, 105]
        }
        return pd.DataFrame(data)
    
    monkeypatch.setattr('utils.data_retrieval.get_yahoo_data', dummy_get_yahoo_data)
    
    # Post form data to the /backtest route with 'simple' backtest method.
    client = create_client()
    response = client.post('/backtest', data={
        'symbol': 'DUMMY',
        'start_date': '2021-01-01',
        'end_date': '2021-01-05',
        'backtest_method': 'simple'
    })
    # Check that the response is OK and contains expected elements.
    assert response.status_code == 200
    assert b'Cumulative Return' in response.data
