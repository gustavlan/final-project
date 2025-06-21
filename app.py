from flask import Flask, render_template, request
from extensions import db
from config import DevelopmentConfig, TestingConfig, ProductionConfig
import click
import pandas as pd
import os
import logging
from logging.handlers import RotatingFileHandler
from utils.backtest_service import run_backtest


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
    # Import models after initializing the app
    from models import BacktestResult

    @app.route('/')
    def home():
        app.logger.info("Home page accessed")
        return render_template('index.html')

    @app.route('/backtest', methods=['POST'])
    def backtest():
        symbol = request.form.get('symbol')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        strategy_method = request.form.get('strategy_method', 'naive')

        try:
            results = run_backtest(symbol, start_date, end_date, strategy_method)
        except RuntimeError as exc:
            return str(exc), 500

        # Persist summary results
        result = BacktestResult(
            strategy_id=1,
            index_id=1,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            returns=results['strategy_return'],
            alpha=results['strategy_alpha'],
        )
        db.session.add(result)
        db.session.commit()

        return render_template(
            'results.html',
            **results,
            naive_beta=1,
            naive_jensens_alpha=0,
            naive_treynor=results['naive_avg_excess'],
        )

app = create_app()

if __name__ == '__main__':
    # Rely on the FLASK_ENV environment variable for debug configuration
    app.run()
