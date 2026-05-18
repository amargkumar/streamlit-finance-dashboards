#Options Pricing & Greeks Dashboard

A Streamlit-powered Black-Scholes options analyzer.

## Features

| Tab | What it does |
|-----|-------------|
| **Greeks vs Spot** | Option price + all 4 key Greeks plotted across spot prices, with strike/spot markers |
| **Greeks Surface** | 3D surface plots of price & Greeks across spot × volatility (interactive rotation) |
| **Time Decay** | Price & theta vs days to expiration, multi-DTE overlay comparison |
| **Put-Call Parity** | Verifies C − P = S − PV(K) across all strikes, flags any deviations |
| **Scenario Analysis** | P&L heatmap under simultaneous spot & vol shocks, shocked Greeks display |

### Key Metrics (top bar)
- Option Price (Black-Scholes)
- Delta (Δ), Gamma (Γ), Theta (Θ), Vega (ν), Rho (ρ)
- Moneyness indicator (ITM / ATM / OTM)

### Implied Volatility Calculator
Enter an observed market price in the sidebar and click "Compute IV" to back out the implied volatility using Brent's method.

## Quick Start

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd options-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

# 4. Run tests
pytest test_options_analytics.py -v
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

## Testing

```bash
pytest test_options_analytics.py -v
```

40 tests covering:
- Black-Scholes pricing (boundary cases, monotonicity, expiration)
- Put-call parity (exact equality, multiple strikes)
- All 5 Greeks (ranges, peaks, relationships, edge cases)
- Implied volatility (round-trip recovery, failure cases)
- Grid computation (shapes, monotonicity)

## Ideas for Extension

- **American options** — binomial tree or finite difference pricing
- **Volatility smile/surface** — fetch real options chains and plot IV surface
- **Strategy builder** — multi-leg P&L diagrams (spreads, strangles, condors)
- **Real-time data** — connect to a live options feed
- **Greeks hedging simulator** — simulate delta-hedging P&L over time
- **Jump-diffusion model** — Merton's model for fat-tailed distributions

## Tech Stack

- **Streamlit** — UI framework
- **NumPy** — numerical computing
- **SciPy** — `norm` for Black-Scholes CDFs, `brentq` for IV root-finding
- **Plotly** — interactive 2D and 3D charting
- **pytest** — unit testing

---
