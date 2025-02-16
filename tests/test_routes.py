import pytest
from app import app  # Import your Flask app

@pytest.fixture
def client():
    # Set up the Flask test client using your Testing configuration
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    # Test that the home page loads correctly.
    response = client.get('/')
    assert response.status_code == 200
    assert b'Asset Allocation Backtester' in response.data  # Check for known content

def test_backtest_route_simple(client, monkeypatch):
    # Create a dummy version of get_yahoo_data to return a controlled DataFrame
    import pandas as pd
    def dummy_get_yahoo_data(symbol, start_date, end_date):
        data = {
            'Date': pd.date_range(start=start_date, periods=5, freq='D'),
            'Close': [100, 102, 101, 103, 105]
        }
        return pd.DataFrame(data)
    
    monkeypatch.setattr('app.get_yahoo_data', dummy_get_yahoo_data)
    
    # Post form data to the /backtest route with 'simple' backtest method.
    response = client.post('/backtest', data={
        'symbol': 'DUMMY',
        'start_date': '2021-01-01',
        'end_date': '2021-01-05',
        'backtest_method': 'simple'
    })
    # Check that the response is OK and contains expected elements.
    assert response.status_code == 200
    assert b'Cumulative Return' in response.data
