from flask import Flask, render_template, request
from extensions import db
from config import DevelopmentConfig, TestingConfig, ProductionConfig
import click
import pandas as pd
import numpy as np
import os
import logging
from logging.handlers import RotatingFileHandler
from utils.data_retrieval import get_risk_free_rate
from utils.validation import validate_date_range


def create_app(config_class=None):
    """Create and configure the Flask application."""

    if config_class is None:
        env = os.getenv("FLASK_ENV", "development").lower()
        if env == "production":
            config_class = ProductionConfig
        elif env == "testing":
            config_class = TestingConfig
        else:
            config_class = DevelopmentConfig

    app = Flask(__name__)
    app.config.from_object(config_class)
    config_class.validate()
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

    with app.app_context():
        db.create_all()

    register_routes(app)
    register_cli(app)

    return app


def register_cli(app):
    @app.cli.command('init-db')
    def init_db_command():
        """Initialize the database."""
        db.create_all()
        click.echo('Initialized the database.')


def register_routes(app):
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
        dynamic_macro_strategy,
        compute_metrics,
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
        try:
            validate_date_range(start_date, end_date)
        except ValueError as exc:
            return str(exc), 400
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
        if cols and cols[0] == '' and all(c == symbol for c in cols[1:]):
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
            raise ValueError(
                "No price data available for symbol "
                + symbol
                + " in the requested date range "
                + start_date
                + " to "
                + end_date
            )
        naive_return = float(naive_series.iloc[-1] - 1)
    
        # Determine ETF ticker for liquidity proxy (if available)
        etf_ticker = ETF_MAPPING.get(symbol, None)
    
        # Choose strategy based on user selection:
        if strategy_method == 'naive':
            # Use naive buy & hold (fully invested)
            strategy_return, strategy_series = naive_return, naive_series
            strategy_daily_returns = naive_returns
            strategy_alpha = 0.0
        elif strategy_method == 'advanced':
            # Advanced market timing strategy using index data signals (plus ETF liquidity)
            strategy_return, _, strategy_series = simple_backtest(
                prices_df, lambda df: dynamic_market_timing_strategy_advanced(df, etf_ticker)
            )
            strategy_return = float(strategy_return)
            strategy_daily_returns = strategy_series.pct_change().fillna(0)
        elif strategy_method == 'macro':
            # Advanced market timing strategy incorporating macro data along with index signals
            fred_api_key = os.getenv("FRED_API_KEY")
            if not fred_api_key:
                return "FRED API key not set", 500
            default_series_id = "DGS3MO"  # FRED series (e.g., 3-Month Treasury Rate)
            macro_df = get_fred_data(fred_api_key, default_series_id, start_date, end_date)
            strategy_return, _, strategy_series = simple_backtest(
                prices_df,
                lambda df: dynamic_market_timing_strategy_macro(df, macro_df, etf_ticker)
            )
            strategy_return = float(strategy_return)
            strategy_daily_returns = strategy_series.pct_change().fillna(0)
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
            strategy_daily_returns = prices_df['returns'] * allocation
        else:
            # Default to naive if unrecognized
            strategy_return, strategy_series = naive_return, naive_series
            strategy_daily_returns = naive_returns
            strategy_alpha = 0.0

        metrics = compute_metrics(
            prices_df,
            naive_returns,
            strategy_daily_returns,
            naive_series,
            strategy_series,
            start_date,
            end_date,
            os.getenv("FRED_API_KEY"),
        )
        naive_vol_excess = metrics["naive_vol_excess"]
        naive_avg_excess = metrics["naive_avg_excess"]
        naive_sharpe = metrics["naive_sharpe"]
        naive_sortino = metrics["naive_sortino"]
        naive_drawdown = metrics["naive_drawdown"]
        strategy_vol_excess = metrics["strategy_vol_excess"]
        strategy_avg_excess = metrics["strategy_avg_excess"]
        strategy_sharpe = metrics["strategy_sharpe"]
        strategy_sortino = metrics["strategy_sortino"]
        strategy_beta = metrics["strategy_beta"]
        jensens_alpha = metrics["strategy_jensens_alpha"]
        strategy_treynor = metrics["strategy_treynor"]
        strategy_drawdown = metrics["strategy_drawdown"]

        # Persist backtest results with the actual strategy and index IDs
        strategy = Strategy.query.filter_by(name=strategy_method).first()
        if not strategy:
            strategy = Strategy(name=strategy_method)
            db.session.add(strategy)
            db.session.commit()

        index = Index.query.filter_by(symbol=symbol).first()
        if not index:
            index = Index(name=symbol, symbol=symbol)
            db.session.add(index)
            db.session.commit()

        result = BacktestResult(
            strategy_id=strategy.id,
            index_id=index.id,
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
            plot_html=plot_html,
            naive_sharpe=naive_sharpe,
            naive_sortino=naive_sortino,
            naive_beta=1,
            naive_treynor=naive_avg_excess,
            naive_vol_excess=naive_vol_excess,
            naive_drawdown=naive_drawdown,
            strategy_sharpe=strategy_sharpe,
            strategy_sortino=strategy_sortino,
            strategy_beta=strategy_beta,
            strategy_treynor=strategy_treynor,
            strategy_vol_excess=strategy_vol_excess,
            strategy_drawdown=strategy_drawdown,
            naive_avg_excess=naive_avg_excess,
            strategy_avg_excess=strategy_avg_excess
        )

app = create_app()

if __name__ == '__main__':
    # Rely on the FLASK_ENV environment variable for debug configuration
    app.run()
