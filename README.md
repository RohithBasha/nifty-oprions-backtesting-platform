# NIFTY Options Backtester

A professional-grade options backtesting platform for Indian markets (NIFTY, BANKNIFTY, SENSEX).

## Features

- **Multiple Strategies**: Short/Long Straddle, Strangle, Iron Condor, Bull Call Spread, Bear Put Spread
- **Custom Strategy Builder**: Create multi-leg strategies with configurable parameters
- **Strategy Comparison**: Compare multiple strategies side-by-side with performance metrics
- **Trade Simulator**: Replay historical trading days with intraday charts
- **Risk Management**: Stop Loss, Target, Trailing SL, Time-based exit, Gap filters
- **Advanced Analytics**: Monte Carlo simulation, PDF export, Sharpe ratio, drawdown analysis

## Requirements

```
streamlit
pandas
numpy
plotly
duckdb
reportlab (optional, for PDF export)
kaleido (optional, for chart images)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RohithBasha/nifty-options-backtester.git
cd nifty-options-backtester
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data file (`nifty_data.duckdb`) in the project directory

4. Run the application:
```bash
streamlit run nifty_options_backtest_v3.py
```

## Data Format

The application expects a DuckDB database with two tables:
- `spot_data`: NIFTY spot prices with columns (date, time/datetime, open, high, low, close)
- `options_data`: Option chain data with columns (date, time/datetime, strike, option_type, expiry, open, high, low, close)

## Usage

### Backtest Mode
Configure strategy parameters in the sidebar and run backtests on historical data.

### Compare Mode  
Select two strategies (or build custom ones) and compare their performance.

### Simulator Mode
Replay a specific trading day with intraday price action.

## Screenshots

Coming soon...

## License

MIT License

## Author

Rohith Basha - [@RohithBasha](https://github.com/RohithBasha)
