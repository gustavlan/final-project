Below is a proposed updated version of your README.md that includes a new “Methodology” section. This section outlines the math behind each strategy and explains how risk‐adjusted performance metrics (such as Treynor Ratio and Jensen’s Alpha) are computed. You can further customize it as needed.

---

```markdown
# CS50 Final Project: Asset Allocation Backtester

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

- **Testing:**  
  Includes unit and integration tests (using pytest) to ensure code quality and robustness.

## Methodology

### Naive Buy & Hold
- **Concept:**  
  Simply holding the asset for the entire period.
- **Calculations:**  
  - **Daily Return:**  
    \[
    r_t = \frac{P_t}{P_{t-1}} - 1
    \]
  - **Cumulative Return:**  
    \[
    CR = \prod_{t=1}^{T}(1 + r_t) - 1
    \]

### Advanced Market Timing Strategy
This strategy adjusts exposure daily based on three signals:
  
1. **Momentum Signal**  
   - **Calculation:**  
     \[
     \text{Momentum} = \frac{P_t}{P_{t-n}} - 1
     \]
   - **Transformation:**  
     A logistic transformation is applied:  
     \[
     S_m = \frac{1}{1 + e^{-50 \times \text{Momentum}}}
     \]
   - **Purpose:**  
     Captures recent price trends and transforms them into a weight between 0 and 1.
  
2. **Volatility Signal**  
   - **Calculation:**  
     Compute the standard deviation (\(\sigma\)) of daily returns over a lookback period.
   - **Scaling:**  
     \[
     S_v = \min\left(\frac{T}{\sigma}, 1\right)
     \]
     where \(T\) is a target daily volatility (e.g., 2%).
   - **Purpose:**  
     Adjusts the allocation to control for periods of high volatility.

3. **Liquidity Signal**  
   - **Calculation:**  
     Ratio of the most recent volume to the average volume over the lookback period.
   - **Adjustment:**  
     If the ratio is below a threshold (e.g., 0.8), it is scaled to fall between 0 and 1.
   - **Purpose:**  
     Ensures that the asset is liquid enough for trading.

- **Final Allocation:**  
  The three signals are multiplied together and clamped between 0 and 1:
  \[
  A = \text{clamp}(S_m \times S_v \times S_l, 0, 1)
  \]

### Macro Market Timing Strategy
This approach builds on the advanced strategy by incorporating macroeconomic data:

1. **Momentum Signal (Modified):**  
   \[
   S_m = \tanh(10 \times \text{Momentum})
   \]
2. **Macro Signal:**  
   - **Calculation:**  
     Calculate the deviation of the current macro indicator from its rolling average:
     \[
     z = \frac{\text{Rolling Average} - \text{Current Macro}}{\sigma_{\text{macro}}}
     \]
   - **Transformation:**  
     \[
     S_{\text{macro}} = \tanh(z)
     \]
3. **Combined Signal:**  
   Average the momentum and macro signals:
   \[
   S_c = 0.5 \times S_m + 0.5 \times S_{\text{macro}}
   \]
4. **Adjustments:**  
   The combined signal is then scaled using the same volatility and liquidity signals as above.
5. **Final Allocation:**  
   \[
   A = \text{clamp}(S_c \times S_v \times S_l, -1, 1)
   \]
   Note: Here the allocation can be negative, indicating a potential short position.

### Macro-Only Strategy
- **Concept:**  
  Uses only macroeconomic indicators to determine allocation.
- **Calculation:**  
  Similar to the macro signal above:
  \[
  A = \text{clamp}(\tanh(z), -1, 1)
  \]
  where \(z\) is defined as in the macro signal calculation.

### Risk-Adjusted Performance Metrics
In addition to return calculations, the backtester computes several risk-adjusted metrics:

- **Sharpe Ratio:**  
  \[
  \text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}
  \]
  where \(R_p\) is the portfolio return, \(R_f\) is the risk-free rate, and \(\sigma_p\) is the standard deviation of portfolio returns.

- **Treynor Ratio:**  
  \[
  \text{Treynor Ratio} = \frac{R_p - R_f}{\beta}
  \]
  This measures the excess return per unit of systematic risk (beta).

- **Jensen's Alpha:**  
  \[
  \alpha = R_p - \left[ R_f + \beta (R_m - R_f) \right]
  \]
  Here, \(R_m\) is the market return. A positive alpha indicates performance above what is predicted by the Capital Asset Pricing Model (CAPM).

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

   Set up necessary environment variables. For example, set your FRED API key:
   
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

   - Select an index (e.g., S&P 500, Dow Jones, etc.).
   - Enter the start and end dates.
   - Choose a strategy from the "Select Strategy" dropdown:
     - Naive Buy & Hold
     - Advanced Market Timing
     - Macro Market Timing
     - Macro-Only Strategy
   - Click **Run Backtest**.
   - View the output, which includes cumulative returns and risk-adjusted metrics like Jensen's Alpha and the Treynor Ratio, along with an interactive Plotly chart comparing the strategies.

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
```

---

This updated README now documents the new strategies and provides the underlying mathematical rationale for how each method computes its signals and allocations, as well as the risk-adjusted performance metrics used in your backtests. Feel free to adjust the descriptions and formulas to best match your implementation and desired level of detail.
