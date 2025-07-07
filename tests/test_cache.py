import pandas as pd
import pytest

from app import _cache_path, _save_cache


def test_cache_path_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr("app.CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        _cache_path("yahoo", "../evil", "2020-01-01", "2020-01-02")


def test_save_cache_valid_symbol(tmp_path, monkeypatch):
    monkeypatch.setattr("app.CACHE_DIR", str(tmp_path))
    df = pd.DataFrame({"A": [1]})
    _save_cache(df, "yahoo", "SPY", "2020-01-01", "2020-01-02")
    expected = tmp_path / "yahoo_SPY_2020-01-01_2020-01-02.pkl"
    assert expected.exists()
