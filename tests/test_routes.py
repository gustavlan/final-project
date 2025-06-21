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

def test_backtest_route_simple(create_client, mock_yfinance, mock_fred):
    
    # Post form data to the /backtest route with the naive strategy.
    client = create_client()
    response = client.post('/backtest', data={
        'symbol': 'DUMMY',
        'start_date': '2021-01-01',
        'end_date': '2021-01-05',
        'strategy_method': 'naive'
    })
    # Check that the response is OK and contains expected elements.
    assert response.status_code == 200
    assert b'Cumulative Return' in response.data
