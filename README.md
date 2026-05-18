# Options Pricing & Greeks Dashboard

A Streamlit-powered Black-Scholes options analyzer.

## Features

| Tab | What it does |
|-----|-------------|
| **Greeks vs Spot** | Option price + all 4 key Greeks plotted across spot prices, with strike/spot markers |
| **Greeks Surface** | 3D surface plots of price & Greeks across spot × volatility (interactive rotation) |
| **Time Decay** | Price & theta vs days to expiration, multi-DTE overlay comparison |
| **Put-Call Parity** | Verifies C − P = S − PV(K) across all strikes, flags any deviations |
| **Scenario Analysis** | P&L heatmap under simultaneous spot & vol shocks, shocked Greeks display |
| **Live Options Chain** | Real market data — candlestick chart with SMAs, historical vol, volume, IV surface, skew curves, term structure |

### Key Metrics (top bar)
- Option Price (Black-Scholes)
- Delta (Δ), Gamma (Γ), Theta (Θ), Vega (ν), Rho (ρ)
- Moneyness indicator (ITM / ATM / OTM)

### Live Options Chain Tab
- Candlestick chart with 9/20/100/200-day simple moving averages
- Historical (realized) volatility at 20/60/120-day windows
- Volume chart with 20-day average and up/down day coloring
- 3D implied volatility surface across strikes and expirations
- Volatility skew curves for calls and puts by expiration
- ATM implied vol term structure
- Chain summary statistics

### Implied Volatility Calculator
Enter an observed market price in the sidebar and click "Compute IV" to back out the implied volatility using Brent's method.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project Structure

```
options-dashboard/
├── app.py                     # Streamlit UI — layout, charts, interactivity
├── options_analytics.py       # Pure analytics (BS pricing, Greeks, IV solver)
├── test_options_analytics.py  # 40 unit tests (pytest)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

### Architecture: Separated Analytics

All math lives in `options_analytics.py` with zero Streamlit dependency, so it can be:
- **Tested** directly with pytest (40 tests covering pricing, Greeks, parity, IV)
- **Reused** in notebooks, trading scripts, or other apps
- **Reviewed** independently from the UI code

## Finance & Math Concepts

| Concept | Implementation | Why it matters |
|---------|---------------|----------------|
| **Black-Scholes model** | `bs_call`, `bs_put` | Foundation of options pricing theory |
| **d1, d2** | `d1()`, `d2()` | Core intermediate values driving all BS formulas |
| **Delta (Δ)** | `delta()` | Hedge ratio — how many shares to hold to delta-hedge |
| **Gamma (Γ)** | `gamma()` | Convexity — how quickly delta changes (rebalancing frequency) |
| **Theta (Θ)** | `theta()` | Time decay — the cost of holding an option position |
| **Vega (ν)** | `vega()` | Vol sensitivity — critical for vol trading strategies |
| **Rho (ρ)** | `rho()` | Rate sensitivity — matters for long-dated options |
| **Implied Volatility** | `implied_volatility()` via Brent's method | Market's expectation of future vol, extracted from prices |
| **Put-Call Parity** | `put_call_parity_check()` | No-arbitrage relationship — foundational constraint |
| **Scenario/Stress Testing** | Spot × vol shock grid | How traders assess risk under extreme moves |
| **Volatility Surface** | Live IV across strikes × expirations | Where theory meets reality — the shape reveals market expectations |
| **Volatility Skew** | IV across strikes at fixed expiration | Downside protection is expensive — puts have higher IV than calls |
| **Term Structure** | ATM IV across expirations | Shows whether the market expects near-term or long-term uncertainty |
| **Realized vs Implied Vol** | Rolling HV vs chain IV | The gap between these drives most options trading P&L |

## Model Assumptions & Limitations

Black-Scholes makes several simplifying assumptions that don't hold in real markets. Understanding these limitations is as important as understanding the model itself:

- **Constant volatility** — BS assumes vol doesn't change. In reality, vol is stochastic and varies across strikes (skew) and expirations (term structure). The vol surface in the Live tab shows exactly where this assumption breaks down.
- **Log-normal returns** — BS assumes stock returns follow a normal distribution. Real markets have fat tails — extreme moves happen far more often than the model predicts.
- **No dividends** — the implementation doesn't account for dividends. For dividend-paying stocks, this means call prices are slightly overstated and put prices slightly understated.
- **European exercise only** — BS prices European options (exercise only at expiry). Most US-listed equity options are American (exercise any time). For calls on non-dividend stocks the prices are identical, but American puts can be worth more than European puts.
- **Continuous trading** — BS assumes you can hedge continuously. In practice, you rebalance discretely, which introduces hedging error.
- **No transaction costs** — real delta hedging incurs commissions, bid-ask spreads, and market impact.
- **Constant risk-free rate** — rates are assumed fixed over the option's life.

These are standard, well-known limitations. The model remains the industry foundation because it provides a common language and useful first approximation. More advanced models (Heston, SABR, local vol) address some of these issues.

## Testing

```bash
pytest test_options_analytics.py -v
```

40 tests covering:
- Black-Scholes pricing (boundary cases, monotonicity, expiration payoffs)
- Put-call parity (exact equality across multiple strikes)
- All 5 Greeks (ranges, peaks, relationships, edge cases)
- Implied volatility (round-trip recovery across vol range, failure cases)
- Grid computation (output shapes, monotonicity)

## Ideas for Extension

- **American options** — binomial tree or finite difference pricing
- **Strategy builder** — multi-leg P&L diagrams (spreads, strangles, condors)
- **Real-time data** — connect to a live options feed (CBOE, Interactive Brokers)
- **Greeks hedging simulator** — simulate delta-hedging P&L over historical data
- **Heston model** — stochastic volatility for more realistic pricing
- **SABR model** — industry standard for interest rate options
- **Auto-fill from market** — type a ticker, auto-populate spot and IV in the sidebar

## Tech Stack

- **Streamlit** — UI framework
- **NumPy** — numerical computing
- **Pandas** — data manipulation
- **SciPy** — `norm` for Black-Scholes CDFs, `brentq` for IV root-finding
- **Plotly** — interactive 2D and 3D charting
- **yfinance** — real-time market and options chain data
- **pytest** — unit testing
