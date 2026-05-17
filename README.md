# Streamlit Finance Dashboards

Two interactive finance dashboards built with Streamlit. Each app demonstrates a different domain — portfolio analytics and options pricing — with separated analytics modules, full test suites, and interactive Plotly visualizations.

## Projects

### [Portfolio Analytics Dashboard](./portfolio-dashboard/)

Multi-asset portfolio construction and analysis with real market data from Yahoo Finance.

- **Performance** — cumulative returns vs benchmark, rolling returns
- **Risk Analysis** — drawdown chart, return distribution, VaR/CVaR, rolling volatility
- **Correlations** — heatmap and rolling pairwise correlation
- **Monte Carlo** — forward simulation with terminal wealth distribution
- **Efficient Frontier** — mean-variance optimization (Max Sharpe, Min Variance), Capital Market Line
- **Factor Analysis** — Fama-French 3-factor regression, alpha/beta decomposition, return attribution

25 unit tests · `analytics.py` separated from UI · scipy SLSQP optimization · statsmodels OLS regression

### [Options Pricing & Greeks Dashboard](./options-dashboard/)

Black-Scholes European options pricing with full Greeks visualization and implied volatility solver.

- **Greeks vs Spot** — price curve with intrinsic value overlay, delta/gamma/theta/vega profiles
- **Greeks Surface** — interactive 3D surfaces across spot × volatility
- **Time Decay** — theta acceleration near expiry, multi-DTE curve comparison
- **Put-Call Parity** — no-arbitrage verification across strikes
- **Scenario Analysis** — P&L heatmap under simultaneous spot and vol shocks
- **IV Calculator** — backs out implied volatility from market prices via Brent's method

40 unit tests · `options_analytics.py` separated from UI · scipy root-finding for IV

## Quick Start

```bash
# Portfolio dashboard
cd portfolio-dashboard
pip install -r requirements.txt
streamlit run app.py

# Options dashboard
cd options-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

Streamlit · Plotly · NumPy · Pandas · SciPy · statsmodels · yfinance · pytest

## Architecture

Both projects follow the same pattern: a pure analytics module with no framework dependency (`analytics.py` / `options_analytics.py`) and a Streamlit app that handles layout and charting (`app.py`). This separation makes the math testable, reusable in notebooks or scripts, and reviewable independently from the UI.

---


