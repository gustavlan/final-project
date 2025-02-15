import yfinance as yf
from fredapi import Fred
import pandas as pd

def get_yahoo_data(symbol, start_date, end_date):
    """Retrieve historical data for a given symbol from Yahoo Finance."""
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def get_fred_data(api_key, series_id, start_date, end_date):
    """Retrieve macroeconomic data from FRED."""
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    df = pd.DataFrame(data, columns=['value']).reset_index().rename(columns={'index': 'date'})
    return df
