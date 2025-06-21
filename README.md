# CS50 Final Project: Asset Allocation Backtester

#### Description:

This project is a web application for backtesting asset allocation strategies using historical index data and macroeconomic indicators. It allows users to compare a naive buy & hold strategy with more advanced market timing strategies that incorporate signals from momentum, volatility, liquidity, and macroeconomic data.

## Features

- **Index Selection:**  
  Choose from a list of common indices (e.g., S&P 500, Dow Jones, NASDAQ, etc.). The app automatically maps each index to a commonly traded ETF for liquidity data.

- **Strategy Comparison:**  
  Compare multiple strategies:
  - **Naive Buy & Hold:**  
    Simply holds the asset over the backtest period and calculates cumulative returns.
  - **Advanced Market Timing:**  
    Dynamically adjusts exposure using a combination of momentum, volatility, and liquidity signals.
  - **Macro Market Timing:**  
    Enhances the advanced strategy by incorporating macroeconomic indicators.
  - **Macro-Only Strategy:**  
    Determines allocation solely based on macroeconomic signals.

- **Interactive Visualization:**  
  Uses Plotly to generate interactive charts displaying cumulative returns for both the naive strategy and the selected market timing strategy.

- **API Integration:**  
  - Retrieves historical index data from Yahoo Finance.
  - Retrieves macroeconomic data from FRED (if using the macro-based strategy).

- **Data Persistence:**
  Backtest results are stored in a SQLite database using SQLAlchemy.

- **ETF Volume Caching:**
  Volume data retrieved from yfinance is cached with an in-memory LRU cache.
  Entries beyond the size limit are optionally written to disk so repeated
  strategy calls avoid unnecessary network requests without unbounded memory
  usage.

- **Testing:**  
  Includes unit and integration tests (using pytest) to ensure code quality and robustness.

## Methodology

### Naive Buy & Hold
- **Concept:**  
  Simply holding the asset for the entire period.
- **Calculations:**  
  - **Daily Return:**  
    ```
    r_t = (P_t / P_{t-1}) - 1
    ```
  - **Cumulative Return:**  
    ```
    CR = (Product of (1 + r_t) for t = 1 to T) - 1
    ```
  - **Alpha (simplified):**  
    The difference between the cumulative return and the average daily return.

### Advanced Market Timing Strategy
This strategy adjusts exposure daily based on three signals:
  
1. **Momentum Signal**  
   - **Calculation:**  
     ```
     Momentum = (P_t / P_{t-n}) - 1
     ```
   - **Transformation:**  
     A logistic transformation is applied to convert momentum into a weight between 0 and 1:
     ```
     S_m = 1 / (1 + exp(-50 * Momentum))
     ```
   - **Purpose:**  
     Captures recent price trends and translates them into an allocation weight.
  
2. **Volatility Signal**  
   - **Calculation:**  
     Compute the standard deviation (σ) of daily returns over a lookback period.
   - **Scaling:**  
     ```
     S_v = min(Target_Volatility / σ, 1)
     ```
     (For example, if the target daily volatility is 2%, then Target_Volatility = 0.02)
   - **Purpose:**  
     Adjusts the allocation to manage periods of high volatility.

3. **Liquidity Signal**  
   - **Calculation:**  
     Ratio of the most recent volume to the average volume over the lookback period.
   - **Adjustment:**  
     If the ratio is below a threshold (e.g., 0.8), it is scaled to fall between 0 and 1.
   - **Purpose:**  
     Ensures that the asset is liquid enough for trading.

- **Final Allocation:**  
  Multiply the three signals and clamp the result between 0 and 1:
  ```
  Allocation = clamp(S_m * S_v * S_l, 0, 1)
  ```

### Macro Market Timing Strategy
This approach builds on the advanced strategy by incorporating macroeconomic data:

1. **Modified Momentum Signal:**  
   ```
   S_m = tanh(10 * Momentum)
   ```
2. **Macro Signal:**  
   - **Calculation:**  
     First, compute the z-score of the current macro indicator:
     ```
     z = (Rolling_Average - Current_Macro) / Standard_Deviation_of_Macro
     ```
   - **Transformation:**  
     ```
     S_macro = tanh(z)
     ```
3. **Combined Signal:**  
   Average the momentum and macro signals:
   ```
   S_c = 0.5 * S_m + 0.5 * S_macro
   ```
4. **Adjustments:**  
   Scale the combined signal using the same volatility (S_v) and liquidity (S_l) signals.
5. **Final Allocation:**  
   Here, the allocation can be negative (indicating a potential short position):
   ```
   Allocation = clamp(S_c * S_v * S_l, -1, 1)
   ```

### Macro-Only Strategy
- **Concept:**  
  Uses only macroeconomic indicators to determine allocation.
- **Calculation:**  
  Similar to the macro signal above:
  ```
  Allocation = clamp(tanh(z), -1, 1)
  ```
  where `z` is computed as:
  ```
  z = (Rolling_Average - Current_Macro) / Standard_Deviation_of_Macro
  ```

### Risk-Adjusted Performance Metrics
In addition to return calculations, the backtester computes several risk-adjusted metrics:

- **Sharpe Ratio:**  
  ```
  Sharpe Ratio = (R_p - R_f) / σ_p
  ```
  where R_p is the portfolio return, R_f is the risk-free rate, and σ_p is the standard deviation of portfolio returns.

- **Treynor Ratio:**  
  ```
  Treynor Ratio = (R_p - R_f) / Beta
  ```
  This measures the excess return per unit of systematic risk (Beta).

- **Jensen's Alpha:**  
  ```
  Jensen's Alpha = R_p - [R_f + Beta * (R_m - R_f)]
  ```
  where R_m is the market return. A positive value indicates performance above what is predicted by CAPM.

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

   The application reads several configuration values from the environment.
   Below is a list of the variables and their purpose:

   | Variable        | Description                                                    | Default                      |
   |-----------------|----------------------------------------------------------------|------------------------------|
   | `SECRET_KEY`    | Flask session secret key (**required**).                       | none                         |
   | `DATABASE_URL`  | SQLAlchemy database URI.                                       | `sqlite:///final_project.db` |
   | `LOG_TO_STDOUT` | If set, logs are written to STDOUT instead of a file.          | not set                      |
   | `LOG_LEVEL`     | Logging verbosity level.                                       | `INFO`                       |
   | `FRED_API_KEY`  | API key for retrieving macro data from FRED (optional).        | not set                      |
   | `FLASK_ENV`     | Selects the configuration: use `development` for local debugging (default), `production` for deployment, or `testing` during tests. | `development`               |

   Example of exporting these variables in your shell:

   ```bash
   export SECRET_KEY="change_me"
   # Optional settings
   export DATABASE_URL="sqlite:///custom.db"
   export FRED_API_KEY="your_fred_api_key_here"
   ```

## Usage

1. **Run the Application:**

   Ensure the `SECRET_KEY` environment variable is defined. Use
   `FLASK_ENV=development` for local debugging or `FLASK_ENV=production`
   for deployment. If not specified, the app defaults to the development
   configuration. Then start the server:

   ```bash
   flask run
   ```

   You can also start it directly with Python:

   ```bash
   python app.py
   ```
   
2. **Access the App:**

   Open your web browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

3. **Backtest a Strategy:**

   - Select an index (e.g., S&P 500, Dow Jones, etc.).
   - Enter the start and end dates.
   - Choose a strategy from the "Select Strategy" dropdown:
     - Naive Buy & Hold
     - Advanced Market Timing
     - Macro Market Timing
     - Macro-Only Strategy
   - Click **Run Backtest**.
   - View the output, which includes cumulative returns and risk-adjusted metrics (such as Jensen's Alpha and the Treynor Ratio), along with an interactive Plotly chart comparing the strategies.

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

This will execute all tests located in the `tests/` directory.

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
