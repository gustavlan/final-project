import yfinance as yf
from fredapi import Fred
import pandas as pd

def get_yahoo_data(symbol, start_date, end_date):
    """Retrieve historical data for a given symbol from Yahoo Finance, grouping by column names."""
    data = yf.download(symbol, start=start_date, end=end_date, group_by='column')
    return data

def get_fred_data(api_key, series_id, start_date, end_date):
    """Retrieve macroeconomic data from FRED."""
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    df = pd.DataFrame(data, columns=['value']).reset_index().rename(columns={'index': 'date'})
    return df

def get_risk_free_rate(api_key, start_date, end_date):
    """
    Retrieve the 3-Month Treasury yield from FRED and convert it to a daily rate.
    Assumes the yield is annualized in percentage terms.
    """
    series_id = "DGS3MO"
    df = get_fred_data(api_key, series_id, start_date, end_date)
    # Convert the annual percentage rate to a daily decimal rate.
    # (Divide by 100 to convert percent to decimal and by 252 for daily rate)
    df['daily_rate'] = df['value'] / 100 / 252
    # Rename the date column for consistency
    df.rename(columns={'date': 'Date'}, inplace=True)
    
    return df[['Date', 'daily_rate']]
