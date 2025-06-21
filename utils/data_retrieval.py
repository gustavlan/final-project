"""Helpers for fetching price and macroeconomic data."""

import logging
import pandas as pd
import yfinance as yf
from fredapi import Fred

logger = logging.getLogger(__name__)

def get_yahoo_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical price data from Yahoo Finance."""
    try:
        data = yf.download(symbol, start=start_date, end=end_date, group_by="column")
    except Exception as exc:
        logger.error("Failed to download data from Yahoo Finance for %s: %s", symbol, exc)
        raise RuntimeError(f"Yahoo Finance download failed for {symbol}") from exc
    return data

def get_fred_data(
    api_key: str, series_id: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Retrieve a FRED series as a DataFrame with ``date`` and ``value``."""
    fred = Fred(api_key=api_key)
    try:
        data = fred.get_series(
            series_id, observation_start=start_date, observation_end=end_date
        )
    except Exception as exc:
        logger.error(
            "Failed to retrieve FRED series %s between %s and %s: %s",
            series_id,
            start_date,
            end_date,
            exc,
        )
        raise RuntimeError(f"FRED data retrieval failed for series {series_id}") from exc
    df = (
        pd.DataFrame(data, columns=["value"])
        .reset_index()
        .rename(columns={"index": "date"})
    )
    return df

def get_risk_free_rate(api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Return a daily risk-free rate series from FRED.

    Parameters
    ----------
    api_key : str
        FRED API key.
    start_date, end_date : str
        Date range to retrieve.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Date`` and ``daily_rate`` columns representing the
        3-month Treasury yield converted to a daily rate.
    """
    series_id = "DGS3MO"
    df = get_fred_data(api_key, series_id, start_date, end_date)
    # Convert the annual percentage rate to a daily decimal rate.
    # (Divide by 100 to convert percent to decimal and by 252 for daily rate)
    df['daily_rate'] = df['value'] / 100 / 252

    # Rename the date column for consistency
    df.rename(columns={'date': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Create a complete date range from start_date to end_date.
    full_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = df.reindex(full_range)
    
    # Forward fill missing daily_rate values.
    df['daily_rate'] = df['daily_rate'].fillna(method='ffill')
    df = df.reset_index().rename(columns={'index': 'Date'})

    return df[['Date', 'daily_rate']]
