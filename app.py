from flask import Flask, render_template, request
from extensions import db
from config import DevelopmentConfig
import pandas as pd
import os
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
db.init_app(app)

# --- Logging Setup ---
if not app.debug and not app.config['LOG_TO_STDOUT']:
    log_dir = os.path.dirname(app.config['LOG_FILE'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_handler = RotatingFileHandler(app.config['LOG_FILE'], maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(getattr(logging, app.config['LOG_LEVEL'].upper()))
    app.logger.addHandler(file_handler)

if app.config['LOG_TO_STDOUT']:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, app.config['LOG_LEVEL'].upper()))
    app.logger.addHandler(stream_handler)

app.logger.setLevel(getattr(logging, app.config['LOG_LEVEL'].upper()))
app.logger.info('Final Project startup')

# Define a mapping from index tickers to ETF tickers.
ETF_MAPPING = {
    '^GSPC': 'SPY',
    '^DJI':  'DIA',
    '^IXIC': 'QQQ',
    '^FTSE': 'ISF',
    '^N225': 'EWJ',
    '^HSI':  '2800.HK',
    '^GDAXI': 'DAXY',
    '^FCHI': 'EWU',
    '^STOXX50E': 'FEZ',
    '^BSESN': 'INDA'
}

# Import models after initializing the app
from models import Index, HistoricalPrice, MacroData, Strategy, BacktestResult
from utils.data_retrieval import get_yahoo_data, get_fred_data
from utils.backtesting import (
    simple_backtest,
    dynamic_market_timing_strategy_advanced,
    dynamic_market_timing_strategy_macro,
    dynamic_macro_strategy
)
from utils.visualizations import create_return_plot

@app.route('/')
def home():
    app.logger.info("Home page accessed")
    return render_template('index.html')

@app.route('/backtest', methods=['POST'])
def backtest():
    symbol = request.form.get('symbol')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    # Read the strategy selection; valid options: 'naive', 'advanced', 'macro', 'macro_only'
    strategy_method = request.form.get('strategy_method', 'naive')

    # Retrieve price data for the index
    prices_df = get_yahoo_data(symbol, start_date, end_date)
    prices_df.reset_index(inplace=True)
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    
    # Flatten columns if they are MultiIndex
    def flatten_col(col):
        return col[1] if isinstance(col, tuple) and len(col) > 1 else col
    prices_df.columns = [flatten_col(col) for col in prices_df.columns]
    app.logger.info("Flattened Price DataFrame columns: " + str(prices_df.columns.tolist()))
    
    # If columns appear as ['', '^STOXX50E', '^STOXX50E', '^STOXX50E', '^STOXX50E', '^STOXX50E', '^STOXX50E'], rename manually:
    cols = prices_df.columns.tolist()
    if cols[0] == '' and all(c == symbol for c in cols[1:]): 
        if len(cols) == 7: 
            prices_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] 
        elif len(cols) == 6: 
            prices_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close']
    
    # Determine a valid price column
    if 'Close' in prices_df.columns:
        price_col = 'Close'
    elif 'Adj Close' in prices_df.columns:
        price_col = 'Adj Close'
    elif 'close' in prices_df.columns:
        price_col = 'close'
    elif 'adj close' in prices_df.columns:
        price_col = 'adj close'
    else:
        raise ValueError("No valid price column found. Available columns: " + str(prices_df.columns.tolist()))
    
    # Compute naive (Buy & Hold) cumulative returns:
    naive_returns = prices_df[price_col].pct_change().fillna(0)
    naive_series = (naive_returns + 1).cumprod()
    # Check if we have data; if not, raise error.
    if naive_series.empty:
        raise ValueError("No price data available for symbol " + symbol + " in the requested date range " + start_date + " to " + end_date)
    naive_return = float(naive_series.iloc[-1] - 1)
    naive_alpha = float(naive_return - naive_returns.mean())
    
    # Determine ETF ticker for liquidity proxy (if available)
    etf_ticker = ETF_MAPPING.get(symbol, None)
    
    # Choose strategy based on user selection:
    if strategy_method == 'naive':
        # Use naive buy & hold (fully invested)
        strategy_return, strategy_alpha, strategy_series = naive_return, naive_alpha, naive_series
    elif strategy_method == 'advanced':
        # Advanced market timing strategy using index data signals (plus ETF liquidity)
        strategy_return, strategy_alpha, strategy_series = simple_backtest(
            prices_df,
            lambda df: dynamic_market_timing_strategy_advanced(df, etf_ticker)
        )
        strategy_return = float(strategy_return)
        strategy_alpha = float(strategy_alpha)
    elif strategy_method == 'macro':
        # Advanced market timing strategy incorporating macro data along with index signals
        fred_api_key = os.getenv("FRED_API_KEY")
        if not fred_api_key:
            return "FRED API key not set", 500
        default_series_id = "DGS3MO"  # Example FRED series (e.g., 3-Month Treasury Rate)
        macro_df = get_fred_data(fred_api_key, default_series_id, start_date, end_date)
        strategy_return, strategy_alpha, strategy_series = simple_backtest(
            prices_df,
            lambda df: dynamic_market_timing_strategy_macro(df, macro_df, etf_ticker)
        )
        strategy_return = float(strategy_return)
        strategy_alpha = float(strategy_alpha)
    elif strategy_method == 'macro_only':
        # Strategy using only macro signals as a signal (allocations from -1 to 1)
        fred_api_key = os.getenv("FRED_API_KEY")
        if not fred_api_key:
            return "FRED API key not set", 500
        default_series_id = "DGS3MO"
        macro_df = get_fred_data(fred_api_key, default_series_id, start_date, end_date)
        allocation = dynamic_macro_strategy(prices_df, macro_df, etf_ticker)
        prices_df['returns'] = prices_df[price_col].pct_change().fillna(0)
        strategy_series = (prices_df['returns'] * allocation + 1).cumprod()
        strategy_return = float(strategy_series.iloc[-1] - 1)
        strategy_alpha = float(strategy_return - prices_df['returns'].mean())
    else:
        # Default to naive if unrecognized
        strategy_return, strategy_alpha, strategy_series = naive_return, naive_alpha, naive_series

    # Persist backtest results to the database (using dummy strategy and index IDs)
    result = BacktestResult(
        strategy_id=1,
        index_id=1,
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date),
        returns=strategy_return,
        alpha=strategy_alpha
    )
    db.session.add(result)
    db.session.commit()
    
    # Generate interactive Plotly chart with both naive and strategy series
    plot_dates = prices_df['Date']
    plot_html = create_return_plot(plot_dates, naive_series, strategy_series)
    
    # Set labels based on the selected strategy
    if strategy_method == "advanced":
        strategy_label = "Advanced Market Timing Strategy"
    elif strategy_method == "macro":
        strategy_label = "Macro Market Timing Strategy"
    elif strategy_method == "macro_only":
        strategy_label = "Macro-Only Strategy"
    else:
        strategy_label = "Naive Buy & Hold Strategy"
    
    return render_template(
        'results.html',
        strategy_method=strategy_method,
        strategy_label=strategy_label,
        naive_return=naive_return,
        naive_alpha=naive_alpha,
        strategy_return=strategy_return,
        strategy_alpha=strategy_alpha,
        plot_html=plot_html
    )

if __name__ == '__main__':
    app.run(debug=True)
