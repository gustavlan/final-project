import os
import pytest
from config import Config


def test_validate_missing_risk_free(monkeypatch):
    old = Config.RISK_FREE_RATE
    monkeypatch.delenv("RISK_FREE_RATE", raising=False)
    Config.RISK_FREE_RATE = os.environ.get("RISK_FREE_RATE")
    Config.validate()
    assert Config.RISK_FREE_RATE is None
    Config.RISK_FREE_RATE = old


def test_validate_invalid_risk_free(monkeypatch):
    old = Config.RISK_FREE_RATE
    monkeypatch.setenv("RISK_FREE_RATE", "abc")
    Config.RISK_FREE_RATE = os.environ.get("RISK_FREE_RATE")
    with pytest.raises(RuntimeError, match="RISK_FREE_RATE environment variable"):
        Config.validate()
    Config.RISK_FREE_RATE = old
