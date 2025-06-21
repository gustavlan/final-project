import os
import pandas as pd
import pytest

# Ensure SECRET_KEY is set for tests that import the app module
os.environ.setdefault("SECRET_KEY", "test-secret-key")

@pytest.fixture
def mock_yfinance(monkeypatch):
    call_count = {'n': 0}

    def dummy_download(ticker, start=None, end=None, group_by='column'):
        call_count['n'] += 1
        dates = pd.date_range(start=start, end=end, freq='D')
        data = pd.DataFrame(
            {
                'Close': [100] * len(dates),
                'Volume': [1000] * len(dates),
            },
            index=dates,
        )
        data.index.name = 'Date'
        return data

    monkeypatch.setattr('yfinance.download', dummy_download)
    monkeypatch.setattr('utils.data_retrieval.yf.download', dummy_download)
    monkeypatch.setattr('utils.backtesting.yf.download', dummy_download)
    yield call_count


@pytest.fixture
def mock_fred(monkeypatch):
    class DummyFred:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def get_series(self, series_id, observation_start=None, observation_end=None):
            dates = pd.date_range(start=observation_start, end=observation_end, freq='D')
            return pd.Series([1.0] * len(dates), index=dates)

    monkeypatch.setattr('fredapi.Fred', DummyFred)
    monkeypatch.setattr('utils.data_retrieval.Fred', DummyFred)
    yield DummyFred
