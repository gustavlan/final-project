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

# Configure SQLite database
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///final_project.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Logging Setup ---
if not app.debug and not app.config['LOG_TO_STDOUT']:
    # Ensure the logs folder exists
    log_dir = os.path.dirname(app.config['LOG_FILE'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure the file handler with rotation
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

# Import models after initializing the app
from models import Index, HistoricalPrice, MacroData, Strategy, BacktestResult
from utils.data_retrieval import get_yahoo_data, get_fred_data
from utils.backtesting import simple_backtest, simple_backtest_with_macro, full_invested_strategy
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
    method = request.form.get('backtest_method', 'simple')

    # Retrieve price data
    prices_df = get_yahoo_data(symbol, start_date, end_date)
    prices_df.reset_index(inplace=True)
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])

    if method == 'macro':
        # Retrieve macro data using FRED API
        fred_api_key = os.getenv("FRED_API_KEY")
        if not fred_api_key:
            return "FRED API key not set", 500
        default_series_id = "DGS3MO"  # Example series (3-Month Treasury Rate)
        macro_df = get_fred_data(fred_api_key, default_series_id, start_date, end_date)
        cumulative_return, alpha, cumulative_series = simple_backtest_with_macro(prices_df, macro_df, full_invested_strategy)
        # Use the Date column from the merged data (assumed to be the same as prices_df here)
        plot_dates = prices_df['Date']
    else:
        cumulative_return, alpha, cumulative_series = simple_backtest(prices_df, full_invested_strategy)
        plot_dates = prices_df['Date']

    # Persist backtest results to the database, with dummy values (1)
    result = BacktestResult(
        strategy_id=1,
        index_id=1,
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date),
        returns=cumulative_return,
        alpha=alpha
    )
    db.session.add(result)
    db.session.commit()

    # Generate interactive Plotly chart
    plot_html = create_return_plot(plot_dates, cumulative_series)
    return render_template('results.html', cumulative_return=cumulative_return, alpha=alpha, plot_html=plot_html)

if __name__ == '__main__':
    app.run(debug=True)
