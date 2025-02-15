from flask import Flask, render_template, request
from extensions import db  # Import db from the new file
import pandas as pd
import os

app = Flask(__name__)

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///final_project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with the app
db.init_app(app)

# Import models after initializing db to avoid circular imports
from models import Index, HistoricalPrice, MacroData, Strategy, BacktestResult

# Import utility functions
from utils.data_retrieval import get_yahoo_data
from utils.backtesting import simple_backtest, full_invested_strategy
from utils.visualizations import create_return_plot

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/backtest', methods=['POST'])
def backtest():
    symbol = request.form.get('symbol')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    
    # Retrieve historical price data from Yahoo Finance
    prices_df = get_yahoo_data(symbol, start_date, end_date)
    
    # Reset the index to convert the Date index to a column
    prices_df.reset_index(inplace=True)
    
    # Convert the Date column to datetime if it's not already
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    
    # Calculate daily returns and replace NaN (from the first row) with 0
    prices_df['returns'] = prices_df['Close'].pct_change().fillna(0)
    
    # Compute cumulative returns
    cumulative_returns = (prices_df['returns'] + 1).cumprod()
    
    # Run the backtesting engine with the full_invested_strategy
    cumulative_return, alpha = simple_backtest(prices_df, full_invested_strategy)
    
    # Generate the interactive Plotly chart cumulative returns
    plot_html = create_return_plot(prices_df['Date'], cumulative_returns)
    
    return render_template('results.html', cumulative_return=cumulative_return, alpha=alpha, plot_html=plot_html)

if __name__ == '__main__':
    app.run(debug=True)
