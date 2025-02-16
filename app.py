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
from utils.backtesting import simple_backtest, simple_backtest_with_macro, full_invested_strategy, dynamic_market_timing_strategy_advanced
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
    # Read the strategy selection
    strategy_method = request.form.get('strategy_method', 'naive')

    # Retrieve price data...
    prices_df = get_yahoo_data(symbol, start_date, end_date)
    prices_df.reset_index(inplace=True)
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    def flatten_col(col):
        return col[1] if isinstance(col, tuple) and len(col) > 1 else col
    prices_df.columns = [flatten_col(col) for col in prices_df.columns]
    app.logger.info("Flattened Price DataFrame columns: " + str(prices_df.columns.tolist()))
    
    cols = prices_df.columns.tolist()
    if len(cols) == 6 and cols[0] == '' and all(c == symbol for c in cols[1:]):
        prices_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close']
        app.logger.info("Renamed columns to: " + str(prices_df.columns.tolist()))
    
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
    
    # Naive strategy (Buy & Hold)
    naive_returns = prices_df[price_col].pct_change().fillna(0)
    naive_series = (naive_returns + 1).cumprod()
    naive_return = float(naive_series.iloc[-1] - 1)
    naive_alpha = float(naive_return - naive_returns.mean())
    
    # Determine ETF ticker if needed
    etf_ticker = ETF_MAPPING.get(symbol, None)
    
    # Choose the strategy based on user selection
    if strategy_method == 'naive':
        # For naive, the strategy is just buy & hold (identical to the above)
        strategy_return, strategy_alpha, strategy_series = naive_return, naive_alpha, naive_series
    elif strategy_method == 'full_invested':
        # Full invested constant strategy; effectively, it always returns 1
        strategy_return, strategy_alpha, strategy_series = simple_backtest(prices_df, full_invested_strategy)
        strategy_return = float(strategy_return)
        strategy_alpha = float(strategy_alpha)
    elif strategy_method == 'advanced':
        # Advanced market timing strategy using dynamic_market_timing_strategy_advanced
        # (Optionally, macro data could be used; for now, we'll assume simple mode)
        strategy_return, strategy_alpha, strategy_series = simple_backtest(
            prices_df,
            lambda df: dynamic_market_timing_strategy_advanced(df, etf_ticker)
        )
        strategy_return = float(strategy_return)
        strategy_alpha = float(strategy_alpha)
    else:
        # Default to naive if unrecognized
        strategy_return, strategy_alpha, strategy_series = naive_return, naive_alpha, naive_series

    # Persist results and generate chart (as before)...
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

    plot_dates = prices_df['Date']
    plot_html = create_return_plot(plot_dates, naive_series, strategy_series)

    return render_template(
        'results.html',
        naive_return=naive_return,
        naive_alpha=naive_alpha,
        strategy_return=strategy_return,
        strategy_alpha=strategy_alpha,
        plot_html=plot_html
    )


if __name__ == '__main__':
    app.run(debug=True)
