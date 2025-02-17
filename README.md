# CS50 Final Project: Asset Allocation Backtester

This project is a web application for backtesting asset allocation strategies using historical index data and macroeconomic indicators. It allows users to compare a naive buy & hold strategy with more advanced market timing strategies that incorporate signals from momentum, volatility, and liquidity (using ETFs as proxies for liquidity).

## Features

- **Index Selection:**  
  Choose from a list of common indices (e.g., S&P 500, Dow Jones, NASDAQ, etc.). The app automatically maps each index to a commonly traded ETF for liquidity data. Currently only some of the biggest common indices are availible.
  
- **Strategy Comparison:**  
  Compare multiple strategies:
  - **Naive Buy & Hold:** Simple cumulative return calculation.
  - **Advanced Market Timing:** Uses a dynamic strategy combining momentum, volatility, and liquidity signals.
  
- **Interactive Visualization:**  
  Uses Plotly to generate interactive charts that display the cumulative returns of both the naive and the selected market timing strategies.
  
- **API Integration:**  
  - Retrieves historical index data from Yahoo Finance.
  - Retrieves macroeconomic data from FRED (if using the macro-based strategy).
  
- **Data Persistence:**  
  Backtest results are stored in a SQLite database using SQLAlchemy.
  
- **Testing:**  
  The project includes unit tests and integration tests using pytest to ensure code quality and robustness.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/cs50_final_project.git
   cd cs50_final_project
   ```

2. **Set Up a Virtual Environment:**

   Create and activate a virtual environment:
   
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**

   Install the required packages using pip:
   
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**

   Set up necessary environment variables. For example, set your FRED API key in your shell profile or manually before running the app:

   ```bash
   export FRED_API_KEY="your_fred_api_key_here"
   ```

## Usage

1. **Run the Application:**

   With your virtual environment activated, start the Flask server:
   
   ```bash
   python app.py
   ```
   
2. **Access the App:**

   Open your web browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

3. **Backtest a Strategy:**

   - Select an index from the dropdown (e.g., S&P 500, Dow Jones, etc.).
   - Enter the start and end dates.
   - Choose a strategy from the "Select Strategy" dropdown:
     - **Naive Buy & Hold**
     - **Advanced Market Timing**
   - Click **Run Backtest**.
   - View the output, which includes cumulative return and alpha metrics for both the naive strategy and the chosen market timing strategy, along with an interactive Plotly graph comparing the two.

## Example Output

Below is an example of what the results screen might look like:

![Example Graph](path/to/example_graph.png)

- **Naive Strategy (Buy & Hold):**  
  - Cumulative Return: 0.2475  
  - Alpha: 0.2466  

- **Market Timing Strategy:**  
  - Cumulative Return: 0.2730  
  - Alpha: 0.2620  

## Running Tests

The project uses pytest for unit and integration tests. To run the tests:

1. **Activate your virtual environment:**

   ```bash
   source venv/bin/activate
   ```

2. **Run pytest:**

   ```bash
   python3 -m pytest
   ```

This will execute all tests located in the `tests/` directory. The tests include checks for key utility functions (backtesting, data retrieval, visualization) as well as integration tests for the Flask routes.

## Directory Structure

```
cs50_final_project/
├── app.py                # Main Flask application
├── config.py             # Configuration settings
├── extensions.py         # SQLAlchemy instance
├── models.py             # Database models
├── requirements.txt      # Python package dependencies
├── final_project.db      # SQLite database file (generated)
├── static/               # Static files (CSS, JavaScript, images)
├── templates/            # HTML templates (index.html, results.html, etc.)
├── tests/                # Unit and integration tests
│   ├── test_backtesting.py
│   └── test_routes.py
├── utils/                # Utility modules
│   ├── backtesting.py
│   ├── data_retrieval.py
│   └── visualizations.py
├── venv/                 # Virtual environment directory
└── README.md             # This file
```

## License

[MIT License](LICENSE)

---
