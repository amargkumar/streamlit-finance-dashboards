"""
Options Pricing & Greeks Dashboard
Built with Streamlit — 

Features:
- Black-Scholes pricing for European calls & puts
- All 5 Greeks: Delta, Gamma, Theta, Vega, Rho
- Greeks surface plots (spot × volatility)
- Greeks vs spot price curves
- Implied volatility calculator (Newton/Brent)
- Time decay visualization
- Put-call parity verification
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict

from options_analytics import (
    bs_call, bs_put, bs_price,
    delta, gamma, theta, vega, rho, all_greeks,
    implied_volatility, put_call_parity_check,
    compute_greeks_surface, compute_greeks_vs_spot, compute_price_vs_time,
)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Options Pricing & Greeks",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700&display=swap');

    .stApp { font-family: 'Outfit', sans-serif; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #0d1117 0%, #161b22 100%);
        border: 1px solid rgba(88,166,255,0.12);
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #7d8590 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'Space Mono', monospace !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace !important;
        color: #e6edf3 !important;
        font-size: 1.4rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #010409 0%, #0d1117 100%);
        border-right: 1px solid rgba(88,166,255,0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e6edf3 !important;
        font-family: 'Space Mono', monospace !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 18px;
        font-weight: 500;
        font-family: 'Outfit', sans-serif;
    }

    /* Header */
    .dashboard-header {
        padding: 0.5rem 0;
        margin-bottom: 0.8rem;
    }
    .dashboard-header h1 {
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 1.8rem;
        color: #e6edf3;
        margin: 0;
    }
    .dashboard-header p {
        color: #7d8590;
        font-size: 0.9rem;
        margin: 4px 0 0 0;
    }

    /* Greek badge */
    .greek-badge {
        display: inline-block;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        padding: 3px 10px;
        border-radius: 4px;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Chart Style
# ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit, sans-serif", color="#7d8590"),
    title_font=dict(color="#e6edf3", size=15, family="Space Mono, monospace"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7d8590")),
    xaxis=dict(gridcolor="rgba(88,166,255,0.06)", zerolinecolor="rgba(88,166,255,0.1)"),
    yaxis=dict(gridcolor="rgba(88,166,255,0.06)", zerolinecolor="rgba(88,166,255,0.1)"),
    margin=dict(l=40, r=20, t=50, b=40),
)

COLORS = {
    "price": "#58a6ff",
    "delta": "#3fb950",
    "gamma": "#d29922",
    "theta": "#f85149",
    "vega": "#bc8cff",
    "rho": "#79c0ff",
    "call": "#3fb950",
    "put": "#f85149",
    "accent": "#58a6ff",
    "muted": "#7d8590",
}


def styled_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ──────────────────────────────────────────────
# Sidebar — Option Parameters
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Option Parameters")

    presets = {
        "Custom": {},
        "ATM Call (S=K=100)": {"S": 100.0, "K": 100.0, "T_days": 90, "r": 5.0, "sigma": 20.0, "opt": "call"},
        "Deep ITM Put": {"S": 100.0, "K": 130.0, "T_days": 60, "r": 5.0, "sigma": 25.0, "opt": "put"},
        "Short-dated OTM Call": {"S": 100.0, "K": 115.0, "T_days": 14, "r": 5.0, "sigma": 35.0, "opt": "call"},
        "LEAPS Call (1yr)": {"S": 100.0, "K": 100.0, "T_days": 365, "r": 5.0, "sigma": 22.0, "opt": "call"},
        "High Vol Put": {"S": 100.0, "K": 100.0, "T_days": 45, "r": 5.0, "sigma": 60.0, "opt": "put"},
    }
    preset_choice = st.selectbox("Quick Presets", list(presets.keys()))

    st.markdown("---")

    if preset_choice != "Custom":
        p = presets[preset_choice]
        def_S, def_K = p["S"], p["K"]
        def_T, def_r, def_sigma = p["T_days"], p["r"], p["sigma"]
        def_opt_idx = 0 if p["opt"] == "call" else 1
    else:
        def_S, def_K = 100.0, 100.0
        def_T, def_r, def_sigma = 90, 5.0, 20.0
        def_opt_idx = 0

    option_type = st.radio("Option Type", ["call", "put"], index=def_opt_idx, horizontal=True)

    st.markdown("### Underlying")
    S_input = st.number_input("Spot Price ($)", min_value=1.0, max_value=10000.0, value=def_S, step=1.0)

    st.markdown("### Contract")
    K_input = st.number_input("Strike Price ($)", min_value=1.0, max_value=10000.0, value=def_K, step=1.0)
    T_days = st.slider("Days to Expiration", 1, 730, def_T)
    T_input = T_days / 365.0

    st.markdown("### Market")
    r_input = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=def_r, step=0.25) / 100
    sigma_input = st.number_input("Volatility (%)", min_value=1.0, max_value=200.0, value=def_sigma, step=1.0) / 100

    st.markdown("---")
    st.markdown("### Implied Vol Calculator")
    market_price_input = st.number_input(
        "Market Price ($)", min_value=0.01, max_value=5000.0, value=10.0, step=0.1,
        help="Enter observed market price to back out implied volatility"
    )

    calc_iv = st.button("🔍 Compute IV", use_container_width=True)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(
    '<div class="dashboard-header">'
    "<h1>⚡ Options Pricing & Greeks</h1>"
    "<p>Black-Scholes pricing, Greeks visualization, and implied volatility</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Compute current option values
# ──────────────────────────────────────────────
price = bs_price(S_input, K_input, T_input, r_input, sigma_input, option_type)
greeks = all_greeks(S_input, K_input, T_input, r_input, sigma_input, option_type)

# Intrinsic & time value
if option_type == "call":
    intrinsic = max(S_input - K_input, 0)
else:
    intrinsic = max(K_input - S_input, 0)
time_value = price - intrinsic

# Put-call parity
call_p = bs_call(S_input, K_input, T_input, r_input, sigma_input)
put_p = bs_put(S_input, K_input, T_input, r_input, sigma_input)
parity = put_call_parity_check(call_p, put_p, S_input, K_input, T_input, r_input)

# Moneyness label
if option_type == "call":
    moneyness = "ITM" if S_input > K_input else ("ATM" if abs(S_input - K_input) < 0.5 else "OTM")
else:
    moneyness = "ITM" if S_input < K_input else ("ATM" if abs(S_input - K_input) < 0.5 else "OTM")

# ──────────────────────────────────────────────
# KPI Row
# ──────────────────────────────────────────────
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
type_label = "Call" if option_type == "call" else "Put"
k1.metric(f"{type_label} Price", f"${price:.4f}")
k2.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
k3.metric("Gamma (Γ)", f"{greeks['gamma']:.4f}")
k4.metric("Theta (Θ)", f"{greeks['theta']:.4f}")
k5.metric("Vega (ν)", f"{greeks['vega']:.4f}")
k6.metric("Rho (ρ)", f"{greeks['rho']:.4f}")
k7.metric("Moneyness", moneyness)

# IV result (if computed)
if calc_iv:
    iv_result = implied_volatility(market_price_input, S_input, K_input, T_input, r_input, option_type)
    if iv_result is not None:
        st.success(f"**Implied Volatility: {iv_result:.2%}** — backed out from market price "
                   f"\\${market_price_input:.2f} for a {type_label} with S=\\${S_input:.0f}, K=\\${K_input:.0f}, T={T_days}d")
    else:
        st.error("Could not compute IV — the market price may be below intrinsic value or the option is expired.")


# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab_greeks, tab_surface, tab_decay, tab_parity, tab_scenarios, tab_live = st.tabs(
    ["📊 Greeks vs Spot", "🌊 Greeks Surface", "⏳ Time Decay", "⚖️ Put-Call Parity",
     "🎛️ Scenario Analysis", "📡 Live Options Chain"]
)


# ── TAB 1: Greeks vs Spot ──
with tab_greeks:
    spot_min = S_input * 0.5
    spot_max = S_input * 1.5

    data = compute_greeks_vs_spot(
        K=K_input, T=T_input, r=r_input, sigma=sigma_input,
        option_type=option_type, spot_min=spot_min, spot_max=spot_max, n_points=300,
    )

    # Price chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=data["spots"], y=data["price"],
        line=dict(color=COLORS["price"], width=2.5),
        name=f"{type_label} Price",
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
    ))

    # Intrinsic value overlay
    if option_type == "call":
        intrinsic_vals = np.maximum(data["spots"] - K_input, 0)
    else:
        intrinsic_vals = np.maximum(K_input - data["spots"], 0)
    fig_price.add_trace(go.Scatter(
        x=data["spots"], y=intrinsic_vals,
        line=dict(color=COLORS["muted"], width=1.5, dash="dash"),
        name="Intrinsic Value",
    ))

    # Current spot marker
    fig_price.add_vline(x=S_input, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                        annotation_text=f"S={S_input:.0f}")
    fig_price.add_vline(x=K_input, line_dash="dot", line_color="rgba(248,81,73,0.4)",
                        annotation_text=f"K={K_input:.0f}")
    fig_price.update_layout(
        title=f"{type_label} Price vs Spot — K={K_input:.0f}, T={T_days}d, σ={sigma_input:.0%}",
        xaxis_title="Spot Price ($)",
        yaxis_title="Option Price ($)",
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(styled_fig(fig_price), use_container_width=True)

    # Greeks subplots (2×2)
    greeks_data = [
        ("Delta (Δ)", data["delta"], COLORS["delta"],
         "Sensitivity of option price to $1 move in underlying"),
        ("Gamma (Γ)", data["gamma"], COLORS["gamma"],
         "Rate of change of delta — convexity of the option"),
        ("Theta (Θ/day)", data["theta"], COLORS["theta"],
         "Daily time decay — how much value bleeds per day"),
        ("Vega (per 1% vol)", data["vega"], COLORS["vega"],
         "Sensitivity to a 1 percentage point change in implied vol"),
    ]

    col1, col2 = st.columns(2)
    for i, (name, values, color, desc) in enumerate(greeks_data):
        with col1 if i % 2 == 0 else col2:
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                x=data["spots"], y=values,
                line=dict(color=color, width=2),
                fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)",
                name=name,
            ))
            fig_g.add_vline(x=S_input, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig_g.add_vline(x=K_input, line_dash="dot", line_color="rgba(248,81,73,0.3)")
            fig_g.update_layout(title=name, xaxis_title="Spot Price ($)", height=320, hovermode="x unified")
            st.plotly_chart(styled_fig(fig_g), use_container_width=True)
            st.caption(desc)


# ── TAB 2: Greeks Surface ──
with tab_surface:
    st.markdown("Surface plots showing how each Greek varies across **spot price** (x) and **volatility** (y).")

    surface_data = compute_greeks_surface(
        S=S_input, K=K_input, T=T_input, r=r_input, sigma=sigma_input,
        option_type=option_type, n_points=40,
    )

    surface_metrics = [
        ("Option Price", "price", "Teal"),
        ("Delta (Δ)", "delta", "Greens"),
        ("Gamma (Γ)", "gamma", "YlOrRd"),
        ("Theta (Θ)", "theta", "RdBu"),
        ("Vega (ν)", "vega", "Purp"),
    ]

    # Price surface (full width)
    fig_surf = go.Figure(data=[go.Surface(
        x=surface_data["spots"],
        y=surface_data["vols"] * 100,
        z=surface_data["price"],
        colorscale="Teal",
        colorbar=dict(title="Price ($)", tickfont=dict(color="#7d8590")),
        opacity=0.9,
    )])
    fig_surf.update_layout(
        title=f"{type_label} Price Surface — K={K_input:.0f}, T={T_days}d",
        scene=dict(
            xaxis_title="Spot Price ($)",
            yaxis_title="Volatility (%)",
            zaxis_title="Option Price ($)",
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(88,166,255,0.1)"),
            yaxis=dict(gridcolor="rgba(88,166,255,0.1)"),
            zaxis=dict(gridcolor="rgba(88,166,255,0.1)"),
        ),
        height=500,
    )
    st.plotly_chart(styled_fig(fig_surf), use_container_width=True)

    # Greek surfaces (2×2)
    greek_surfaces = [
        ("Delta Surface", "delta", "Greens"),
        ("Gamma Surface", "gamma", "YlOrRd"),
        ("Theta Surface", "theta", "RdBu_r"),
        ("Vega Surface", "vega", "Purples"),
    ]

    sc1, sc2 = st.columns(2)
    for i, (title, key, colorscale) in enumerate(greek_surfaces):
        with sc1 if i % 2 == 0 else sc2:
            fig_gs = go.Figure(data=[go.Surface(
                x=surface_data["spots"],
                y=surface_data["vols"] * 100,
                z=surface_data[key],
                colorscale=colorscale,
                opacity=0.85,
                colorbar=dict(tickfont=dict(color="#7d8590", size=10)),
            )])
            fig_gs.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="Spot ($)",
                    yaxis_title="Vol (%)",
                    zaxis_title=key.capitalize(),
                    bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="rgba(88,166,255,0.1)"),
                    yaxis=dict(gridcolor="rgba(88,166,255,0.1)"),
                    zaxis=dict(gridcolor="rgba(88,166,255,0.1)"),
                ),
                height=420,
            )
            st.plotly_chart(styled_fig(fig_gs), use_container_width=True)


# ── TAB 3: Time Decay ──
with tab_decay:
    st.markdown("How the option price and theta evolve as expiration approaches.")

    max_T_display = max(T_input * 1.5, 0.25)
    time_data = compute_price_vs_time(
        S=S_input, K=K_input, r=r_input, sigma=sigma_input,
        option_type=option_type, max_T=max_T_display, n_points=300,
    )

    td_col1, td_col2 = st.columns(2)

    with td_col1:
        fig_td = go.Figure()
        fig_td.add_trace(go.Scatter(
            x=time_data["times_days"], y=time_data["price"],
            line=dict(color=COLORS["price"], width=2.5),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
            name="Option Price",
        ))
        # Intrinsic value line
        fig_td.add_hline(y=intrinsic, line_dash="dash", line_color=COLORS["muted"],
                         annotation_text=f"Intrinsic: ${intrinsic:.2f}")
        # Current DTE marker
        fig_td.add_vline(x=T_days, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                         annotation_text=f"Now ({T_days}d)")
        fig_td.update_layout(
            title="Price vs Days to Expiration",
            xaxis_title="Days to Expiration",
            yaxis_title="Option Price ($)",
            height=420,
            xaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(styled_fig(fig_td), use_container_width=True)

    with td_col2:
        fig_theta_t = go.Figure()
        fig_theta_t.add_trace(go.Scatter(
            x=time_data["times_days"], y=time_data["theta"],
            line=dict(color=COLORS["theta"], width=2.5),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.08)",
            name="Theta ($/day)",
        ))
        fig_theta_t.add_vline(x=T_days, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                              annotation_text=f"Now ({T_days}d)")
        fig_theta_t.update_layout(
            title="Theta (Daily Decay) vs Days to Expiration",
            xaxis_title="Days to Expiration",
            yaxis_title="Theta ($/day)",
            height=420,
            xaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(styled_fig(fig_theta_t), use_container_width=True)

    # Time value breakdown
    st.markdown("#### Value Breakdown")
    vb1, vb2, vb3 = st.columns(3)
    vb1.metric("Total Price", f"${price:.4f}")
    vb2.metric("Intrinsic Value", f"${intrinsic:.4f}")
    vb3.metric("Time Value", f"${time_value:.4f}")

    # Multi-DTE comparison
    st.markdown("#### Price Curves at Different Expirations")
    fig_multi = go.Figure()
    dte_list = [7, 30, 60, 90, 180, 365]
    dte_colors = ["#f85149", "#d29922", "#e3b341", "#3fb950", "#58a6ff", "#bc8cff"]

    for dte, color in zip(dte_list, dte_colors):
        t_yr = dte / 365.0
        curve_data = compute_greeks_vs_spot(
            K=K_input, T=t_yr, r=r_input, sigma=sigma_input,
            option_type=option_type, spot_min=S_input * 0.6, spot_max=S_input * 1.4, n_points=150,
        )
        fig_multi.add_trace(go.Scatter(
            x=curve_data["spots"], y=curve_data["price"],
            line=dict(color=color, width=1.8),
            name=f"{dte}d",
        ))

    # Intrinsic line
    spot_range = np.linspace(S_input * 0.6, S_input * 1.4, 150)
    if option_type == "call":
        intrinsic_line = np.maximum(spot_range - K_input, 0)
    else:
        intrinsic_line = np.maximum(K_input - spot_range, 0)
    fig_multi.add_trace(go.Scatter(
        x=spot_range, y=intrinsic_line,
        line=dict(color="#7d8590", width=2, dash="dash"),
        name="Intrinsic (T=0)",
    ))

    fig_multi.add_vline(x=K_input, line_dash="dot", line_color="rgba(248,81,73,0.3)",
                        annotation_text=f"K={K_input:.0f}")
    fig_multi.update_layout(
        title="Option Price by Time to Expiration",
        xaxis_title="Spot Price ($)",
        yaxis_title="Option Price ($)",
        height=420,
        hovermode="x unified",
    )
    st.plotly_chart(styled_fig(fig_multi), use_container_width=True)


# ── TAB 4: Put-Call Parity ──
with tab_parity:
    st.markdown("""
    > **Put-Call Parity**: C − P = S − K·e^(−rT)
    >
    > For European options, the call-put price difference must equal the forward value
    > of the underlying minus the present value of the strike. Any deviation represents
    > a risk-free arbitrage opportunity (in theory).
    """)

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Call Price", f"${call_p:.4f}")
    pc2.metric("Put Price", f"${put_p:.4f}")
    pc3.metric("C − P", f"${parity['call_minus_put']:.4f}")

    pc4, pc5, pc6 = st.columns(3)
    pv_strike = K_input * np.exp(-r_input * T_input)
    pc4.metric("S (Spot)", f"${S_input:.2f}")
    pc5.metric("PV(K)", f"${pv_strike:.4f}")
    pc6.metric("S − PV(K)", f"${parity['spot_minus_pv_strike']:.4f}")

    if parity["parity_holds"]:
        st.success(f"✅ **Parity holds** — deviation: ${parity['deviation']:.6f}")
    else:
        st.error(f"❌ **Parity violation** — deviation: ${parity['deviation']:.4f} (arbitrage opportunity!)")

    # Parity across strikes
    st.markdown("#### Parity Verification Across Strikes")
    strikes = np.linspace(K_input * 0.6, K_input * 1.4, 100)
    deviations = []
    calls_arr = []
    puts_arr = []
    for k in strikes:
        c = bs_call(S_input, k, T_input, r_input, sigma_input)
        p = bs_put(S_input, k, T_input, r_input, sigma_input)
        calls_arr.append(c)
        puts_arr.append(p)
        pc = put_call_parity_check(c, p, S_input, k, T_input, r_input)
        deviations.append(pc["deviation"])

    fig_parity = go.Figure()

    # Call and put prices
    fig_parity.add_trace(go.Scatter(
        x=strikes, y=calls_arr,
        line=dict(color=COLORS["call"], width=2),
        name="Call Price",
    ))
    fig_parity.add_trace(go.Scatter(
        x=strikes, y=puts_arr,
        line=dict(color=COLORS["put"], width=2),
        name="Put Price",
    ))

    # C - P
    c_minus_p = np.array(calls_arr) - np.array(puts_arr)
    fig_parity.add_trace(go.Scatter(
        x=strikes, y=c_minus_p,
        line=dict(color=COLORS["accent"], width=2, dash="dash"),
        name="C − P",
    ))

    # Theoretical line S - PV(K)
    theoretical = S_input - strikes * np.exp(-r_input * T_input)
    fig_parity.add_trace(go.Scatter(
        x=strikes, y=theoretical,
        line=dict(color=COLORS["muted"], width=2, dash="dot"),
        name="S − PV(K)",
    ))

    fig_parity.update_layout(
        title="Call & Put Prices + Put-Call Parity Across Strikes",
        xaxis_title="Strike Price ($)",
        yaxis_title="Price ($)",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(styled_fig(fig_parity), use_container_width=True)


# ── TAB 5: Scenario Analysis ──
with tab_scenarios:
    st.markdown("See how the option price changes under different spot and vol scenarios simultaneously.")

    sc_col1, sc_col2 = st.columns([1, 3])

    with sc_col1:
        spot_shock_range = st.slider("Spot Shock Range (%)", 5, 50, 20)
        vol_shock_range = st.slider("Vol Shock Range (pts)", 5, 40, 15)

    with sc_col2:
        # Build scenario table
        spot_shocks = np.arange(-spot_shock_range, spot_shock_range + 1, 5)
        vol_shocks = np.arange(-vol_shock_range, vol_shock_range + 1, 5)

        scenario_grid = np.zeros((len(vol_shocks), len(spot_shocks)))
        pnl_grid = np.zeros((len(vol_shocks), len(spot_shocks)))

        for i, dv in enumerate(vol_shocks):
            for j, ds in enumerate(spot_shocks):
                new_S = S_input * (1 + ds / 100)
                new_sigma = max(sigma_input + dv / 100, 0.01)
                new_price = bs_price(new_S, K_input, T_input, r_input, new_sigma, option_type)
                scenario_grid[i, j] = new_price
                pnl_grid[i, j] = new_price - price

        fig_scenario = go.Figure(data=go.Heatmap(
            x=[f"{s:+d}%" for s in spot_shocks],
            y=[f"{v:+d}pp" for v in vol_shocks],
            z=pnl_grid,
            colorscale=[
                [0, "#f85149"],
                [0.5, "#0d1117"],
                [1, "#3fb950"],
            ],
            zmid=0,
            text=[[f"${v:.2f}" for v in row] for row in pnl_grid],
            texttemplate="%{text}",
            textfont=dict(size=11, color="#e6edf3", family="Space Mono"),
            colorbar=dict(title="P&L ($)", tickfont=dict(color="#7d8590")),
            hovertemplate="Spot: %{x}<br>Vol: %{y}<br>P&L: $%{z:.2f}<extra></extra>",
        ))
        fig_scenario.update_layout(
            title=f"P&L Heatmap — {type_label} @ S={S_input:.0f}, K={K_input:.0f}",
            xaxis_title="Spot Price Shock",
            yaxis_title="Volatility Shock",
            height=480,
        )
        st.plotly_chart(styled_fig(fig_scenario), use_container_width=True)

    # Greeks under shock
    st.markdown("#### Greeks Under Current Scenario")
    st.markdown("Select a specific shock to see how all Greeks change:")

    shock_col1, shock_col2 = st.columns(2)
    with shock_col1:
        spot_pct = st.slider("Spot shock (%)", -30, 30, 0, key="spot_shock")
    with shock_col2:
        vol_pts = st.slider("Vol shock (pp)", -20, 20, 0, key="vol_shock")

    shocked_S = S_input * (1 + spot_pct / 100)
    shocked_sigma = max(sigma_input + vol_pts / 100, 0.01)
    shocked_price = bs_price(shocked_S, K_input, T_input, r_input, shocked_sigma, option_type)
    shocked_greeks = all_greeks(shocked_S, K_input, T_input, r_input, shocked_sigma, option_type)

    sg1, sg2, sg3, sg4, sg5, sg6 = st.columns(6)
    sg1.metric("Price", f"${shocked_price:.4f}", delta=f"${shocked_price - price:.4f}")
    sg2.metric("Delta", f"{shocked_greeks['delta']:.4f}", delta=f"{shocked_greeks['delta'] - greeks['delta']:.4f}")
    sg3.metric("Gamma", f"{shocked_greeks['gamma']:.4f}", delta=f"{shocked_greeks['gamma'] - greeks['gamma']:.4f}")
    sg4.metric("Theta", f"{shocked_greeks['theta']:.4f}", delta=f"{shocked_greeks['theta'] - greeks['theta']:.4f}")
    sg5.metric("Vega", f"{shocked_greeks['vega']:.4f}", delta=f"{shocked_greeks['vega'] - greeks['vega']:.4f}")
    sg6.metric("Rho", f"{shocked_greeks['rho']:.4f}", delta=f"{shocked_greeks['rho'] - greeks['rho']:.4f}")


# ── TAB 6: Live Options Chain ──
with tab_live:
    st.markdown("""
    > Pull **real options chain data** from Yahoo Finance, compute implied volatility for every contract,
    > and visualize the **volatility surface** — how IV varies across strikes and expirations.
    """)

    live_col1, live_col2 = st.columns([1, 3])

    with live_col1:
        live_ticker = st.text_input("Ticker", value="AAPL", key="live_ticker").upper().strip()
        fetch_btn = st.button("📡 Fetch Options Chain", use_container_width=True)

    if fetch_btn or st.session_state.get("live_data_loaded"):
        with st.spinner(f"Fetching options data for {live_ticker}..."):
            try:
                ticker_obj = yf.Ticker(live_ticker)
                spot_price = ticker_obj.history(period="1d")["Close"].iloc[-1]
                expirations = ticker_obj.options

                if len(expirations) == 0:
                    st.error(f"No options data found for {live_ticker}.")
                else:
                    st.session_state["live_data_loaded"] = True

                    with live_col2:
                        st.metric(f"{live_ticker} Spot Price", f"${spot_price:.2f}")

                    # ── Stock Price Chart ──
                    st.markdown(f"#### {live_ticker} Price History")
                    period_choice = st.radio(
                        "Period", ["1M", "3M", "6M", "1Y", "2Y"], index=2, horizontal=True,
                        key="price_period"
                    )
                    period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y"}
                    hist = ticker_obj.history(period=period_map[period_choice])

                    if len(hist) > 0:
                        fig_stock = go.Figure()
                        fig_stock.add_trace(go.Candlestick(
                            x=hist.index,
                            open=hist["Open"],
                            high=hist["High"],
                            low=hist["Low"],
                            close=hist["Close"],
                            increasing_line_color="#3fb950",
                            decreasing_line_color="#f85149",
                            name="Price",
                        ))

                        # Simple Moving Averages
                        sma_config = [
                            (9, "#ffd700", "dash"),
                            (20, "#58a6ff", "solid"),
                            (100, "#bc8cff", "solid"),
                            (200, "#f85149", "dot"),
                        ]
                        for window, color, dash in sma_config:
                            if len(hist) >= window:
                                sma = hist["Close"].rolling(window=window).mean()
                                fig_stock.add_trace(go.Scatter(
                                    x=hist.index,
                                    y=sma,
                                    mode="lines",
                                    line=dict(color=color, width=1.5, dash=dash),
                                    name=f"SMA {window}",
                                ))

                        # Add volume as bar chart on secondary axis
                        fig_stock.add_trace(go.Bar(
                            x=hist.index,
                            y=hist["Volume"],
                            marker_color="rgba(88,166,255,0.15)",
                            name="Volume",
                            yaxis="y2",
                        ))
                        fig_stock.update_layout(
                            title=f"{live_ticker} — {period_choice}",
                            yaxis_title="Price ($)",
                            yaxis2=dict(
                                title="Volume",
                                overlaying="y",
                                side="right",
                                showgrid=False,
                                range=[0, hist["Volume"].max() * 4],
                                tickfont=dict(color="#7d8590"),
                            ),
                            xaxis_rangeslider_visible=False,
                            height=420,
                            showlegend=True,
                            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                        )
                        st.plotly_chart(styled_fig(fig_stock), use_container_width=True)

                        # Key stats row
                        price_change = hist["Close"].iloc[-1] - hist["Close"].iloc[0]
                        price_change_pct = price_change / hist["Close"].iloc[0]
                        high_52 = hist["High"].max()
                        low_52 = hist["Low"].min()
                        avg_vol = hist["Volume"].mean()

                        ps1, ps2, ps3, ps4 = st.columns(4)
                        ps1.metric("Period Return", f"{price_change_pct:.2%}", delta=f"${price_change:.2f}")
                        ps2.metric("Period High", f"${high_52:.2f}")
                        ps3.metric("Period Low", f"${low_52:.2f}")
                        ps4.metric("Avg Daily Volume", f"{avg_vol:,.0f}")

                        # ── Historical Volatility & Volume Charts ──
                        hv_col1, hv_col2 = st.columns(2)

                        with hv_col1:
                            # Historical (realized) volatility
                            daily_returns = hist["Close"].pct_change().dropna()
                            fig_hv = go.Figure()

                            for window, color, label in [
                                (20, "#ffd700", "20d HV"),
                                (60, "#58a6ff", "60d HV"),
                                (120, "#bc8cff", "120d HV"),
                            ]:
                                if len(daily_returns) >= window:
                                    hv = daily_returns.rolling(window=window).std() * np.sqrt(252) * 100
                                    fig_hv.add_trace(go.Scatter(
                                        x=hv.index, y=hv,
                                        mode="lines",
                                        line=dict(color=color, width=1.8),
                                        name=label,
                                    ))

                            fig_hv.update_layout(
                                title="Historical (Realized) Volatility",
                                xaxis_title="Date",
                                yaxis_title="Annualized Vol (%)",
                                height=350,
                                hovermode="x unified",
                            )
                            st.plotly_chart(styled_fig(fig_hv), use_container_width=True)
                            st.caption("Rolling realized vol — compare this to implied vol to spot opportunities.")

                        with hv_col2:
                            # Volume chart with moving average
                            fig_vol = go.Figure()
                            fig_vol.add_trace(go.Bar(
                                x=hist.index, y=hist["Volume"],
                                marker_color=np.where(
                                    hist["Close"] >= hist["Open"],
                                    "rgba(63,185,80,0.4)",
                                    "rgba(248,81,73,0.4)"
                                ),
                                name="Daily Volume",
                            ))

                            if len(hist) >= 20:
                                vol_sma = hist["Volume"].rolling(window=20).mean()
                                fig_vol.add_trace(go.Scatter(
                                    x=hist.index, y=vol_sma,
                                    mode="lines",
                                    line=dict(color="#58a6ff", width=2),
                                    name="20d Avg Volume",
                                ))

                            fig_vol.update_layout(
                                title="Trading Volume",
                                xaxis_title="Date",
                                yaxis_title="Volume",
                                height=350,
                                hovermode="x unified",
                            )
                            st.plotly_chart(styled_fig(fig_vol), use_container_width=True)
                            st.caption("Volume spikes often precede or coincide with big option activity.")

                    st.markdown("---")

                    # Collect IV data across expirations
                    iv_records = []
                    max_expiries = min(len(expirations), 8)  # limit to keep it fast

                    progress = st.progress(0, text="Computing implied volatilities...")

                    for idx, exp_date in enumerate(expirations[:max_expiries]):
                        chain = ticker_obj.option_chain(exp_date)
                        exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
                        T_live = max((exp_dt - datetime.now()).days / 365.0, 0.001)

                        for _, row in chain.calls.iterrows():
                            strike = row["strike"]
                            mid = (row.get("bid", 0) + row.get("ask", 0)) / 2
                            if mid > 0.01 and 0.5 * spot_price < strike < 1.5 * spot_price:
                                iv = implied_volatility(mid, spot_price, strike, T_live, r_input, "call")
                                if iv is not None and 0.01 < iv < 3.0:
                                    intrinsic_val = max(spot_price - strike, 0)
                                    time_val = mid - intrinsic_val
                                    g = all_greeks(spot_price, strike, T_live, r_input, iv, "call")
                                    iv_records.append({
                                        "expiration": exp_date,
                                        "days_to_exp": max((exp_dt - datetime.now()).days, 1),
                                        "strike": strike,
                                        "moneyness": strike / spot_price,
                                        "mid_price": mid,
                                        "intrinsic": intrinsic_val,
                                        "time_value": max(time_val, 0),
                                        "iv": iv,
                                        "delta": g["delta"],
                                        "gamma": g["gamma"],
                                        "theta": g["theta"],
                                        "vega": g["vega"],
                                        "volume": row.get("volume", 0) or 0,
                                        "open_interest": row.get("openInterest", 0) or 0,
                                        "type": "call",
                                    })

                        for _, row in chain.puts.iterrows():
                            strike = row["strike"]
                            mid = (row.get("bid", 0) + row.get("ask", 0)) / 2
                            if mid > 0.01 and 0.5 * spot_price < strike < 1.5 * spot_price:
                                iv = implied_volatility(mid, spot_price, strike, T_live, r_input, "put")
                                if iv is not None and 0.01 < iv < 3.0:
                                    intrinsic_val = max(strike - spot_price, 0)
                                    time_val = mid - intrinsic_val
                                    g = all_greeks(spot_price, strike, T_live, r_input, iv, "put")
                                    iv_records.append({
                                        "expiration": exp_date,
                                        "days_to_exp": max((exp_dt - datetime.now()).days, 1),
                                        "strike": strike,
                                        "moneyness": strike / spot_price,
                                        "mid_price": mid,
                                        "intrinsic": intrinsic_val,
                                        "time_value": max(time_val, 0),
                                        "iv": iv,
                                        "delta": g["delta"],
                                        "gamma": g["gamma"],
                                        "theta": g["theta"],
                                        "vega": g["vega"],
                                        "volume": row.get("volume", 0) or 0,
                                        "open_interest": row.get("openInterest", 0) or 0,
                                        "type": "put",
                                    })

                        progress.progress((idx + 1) / max_expiries, text=f"Processing {exp_date}...")

                    progress.empty()

                    if len(iv_records) == 0:
                        st.warning("No valid IV data computed. The market may be closed or bid/ask spreads too wide.")
                    else:
                        iv_df = pd.DataFrame(iv_records)

                        # ── Volatility Surface (3D) ──
                        st.markdown("#### Implied Volatility Surface")

                        # Pivot for surface: calls only for cleaner surface
                        calls_df = iv_df[iv_df["type"] == "call"].copy()

                        if len(calls_df) > 10:
                            # Create pivot for surface
                            pivot = calls_df.pivot_table(
                                values="iv", index="days_to_exp", columns="strike", aggfunc="mean"
                            )
                            pivot = pivot.dropna(axis=1, thresh=2).dropna(axis=0, thresh=2)
                            pivot = pivot.interpolate(axis=1).interpolate(axis=0)

                            fig_vol_surf = go.Figure(data=[go.Surface(
                                x=pivot.columns.values,  # strikes
                                y=pivot.index.values,     # DTE
                                z=pivot.values * 100,     # IV as percentage
                                colorscale="Viridis",
                                colorbar=dict(title="IV (%)", tickfont=dict(color="#7d8590")),
                                opacity=0.9,
                            )])
                            fig_vol_surf.update_layout(
                                title=f"{live_ticker} Implied Volatility Surface (Calls)",
                                scene=dict(
                                    xaxis_title="Strike ($)",
                                    yaxis_title="Days to Expiration",
                                    zaxis_title="Implied Vol (%)",
                                    bgcolor="rgba(0,0,0,0)",
                                    xaxis=dict(gridcolor="rgba(88,166,255,0.1)"),
                                    yaxis=dict(gridcolor="rgba(88,166,255,0.1)"),
                                    zaxis=dict(gridcolor="rgba(88,166,255,0.1)"),
                                ),
                                height=550,
                            )
                            st.plotly_chart(styled_fig(fig_vol_surf), use_container_width=True)
                        else:
                            st.info("Not enough data points for a surface plot. Showing skew curves instead.")

                        # ── Volatility Skew (2D) ──
                        st.markdown("#### Volatility Skew by Expiration")

                        skew_col1, skew_col2 = st.columns(2)

                        with skew_col1:
                            fig_skew_call = go.Figure()
                            skew_colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149",
                                           "#bc8cff", "#79c0ff", "#e3b341", "#ff7b72"]
                            for i, exp in enumerate(sorted(calls_df["expiration"].unique())):
                                exp_data = calls_df[calls_df["expiration"] == exp].sort_values("strike")
                                dte = exp_data["days_to_exp"].iloc[0]
                                fig_skew_call.add_trace(go.Scatter(
                                    x=exp_data["strike"], y=exp_data["iv"] * 100,
                                    mode="lines+markers",
                                    marker=dict(size=4),
                                    line=dict(color=skew_colors[i % len(skew_colors)], width=1.8),
                                    name=f"{exp} ({dte}d)",
                                ))
                            fig_skew_call.add_vline(x=spot_price, line_dash="dot",
                                                    line_color="rgba(255,255,255,0.3)",
                                                    annotation_text=f"Spot: ${spot_price:.0f}")
                            fig_skew_call.update_layout(
                                title="Call IV Skew",
                                xaxis_title="Strike ($)",
                                yaxis_title="Implied Vol (%)",
                                height=420,
                                hovermode="x unified",
                            )
                            st.plotly_chart(styled_fig(fig_skew_call), use_container_width=True)

                        with skew_col2:
                            puts_df = iv_df[iv_df["type"] == "put"].copy()
                            fig_skew_put = go.Figure()
                            for i, exp in enumerate(sorted(puts_df["expiration"].unique())):
                                exp_data = puts_df[puts_df["expiration"] == exp].sort_values("strike")
                                dte = exp_data["days_to_exp"].iloc[0]
                                fig_skew_put.add_trace(go.Scatter(
                                    x=exp_data["strike"], y=exp_data["iv"] * 100,
                                    mode="lines+markers",
                                    marker=dict(size=4),
                                    line=dict(color=skew_colors[i % len(skew_colors)], width=1.8),
                                    name=f"{exp} ({dte}d)",
                                ))
                            fig_skew_put.add_vline(x=spot_price, line_dash="dot",
                                                   line_color="rgba(255,255,255,0.3)",
                                                   annotation_text=f"Spot: ${spot_price:.0f}")
                            fig_skew_put.update_layout(
                                title="Put IV Skew",
                                xaxis_title="Strike ($)",
                                yaxis_title="Implied Vol (%)",
                                height=420,
                                hovermode="x unified",
                            )
                            st.plotly_chart(styled_fig(fig_skew_put), use_container_width=True)

                        # ── Term Structure ──
                        st.markdown("#### ATM Volatility Term Structure")
                        atm_data = iv_df[
                            (iv_df["moneyness"] > 0.95) & (iv_df["moneyness"] < 1.05)
                        ].groupby(["days_to_exp", "type"])["iv"].mean().reset_index()

                        fig_term = go.Figure()
                        for opt_type, color in [("call", COLORS["call"]), ("put", COLORS["put"])]:
                            type_data = atm_data[atm_data["type"] == opt_type].sort_values("days_to_exp")
                            if len(type_data) > 0:
                                fig_term.add_trace(go.Scatter(
                                    x=type_data["days_to_exp"], y=type_data["iv"] * 100,
                                    mode="lines+markers",
                                    line=dict(color=color, width=2.5),
                                    marker=dict(size=8),
                                    name=f"ATM {opt_type.capitalize()} IV",
                                ))
                        fig_term.update_layout(
                            title=f"{live_ticker} ATM Implied Vol Term Structure",
                            xaxis_title="Days to Expiration",
                            yaxis_title="Implied Vol (%)",
                            height=400,
                        )
                        st.plotly_chart(styled_fig(fig_term), use_container_width=True)

                        # ── Summary Stats ──
                        st.markdown("#### Chain Summary")
                        sum1, sum2, sum3, sum4 = st.columns(4)
                        sum1.metric("Contracts Analyzed", f"{len(iv_df):,}")
                        sum2.metric("Expirations", f"{iv_df['expiration'].nunique()}")
                        avg_iv = iv_df["iv"].mean()
                        sum3.metric("Avg IV (all)", f"{avg_iv:.1%}")
                        atm_avg = iv_df[
                            (iv_df["moneyness"] > 0.97) & (iv_df["moneyness"] < 1.03)
                        ]["iv"].mean()
                        sum4.metric("ATM IV", f"{atm_avg:.1%}" if not pd.isna(atm_avg) else "N/A")

                        # ── Options Chain Table ──
                        st.markdown("#### Options Chain")

                        chain_tab_calls, chain_tab_puts = st.tabs(["📗 Calls", "📕 Puts"])

                        display_df = iv_df.copy()
                        display_df["strike"] = display_df["strike"].map("${:.2f}".format)
                        display_df["mid_price"] = display_df["mid_price"].map("${:.2f}".format)
                        display_df["intrinsic"] = display_df["intrinsic"].map("${:.2f}".format)
                        display_df["time_value"] = display_df["time_value"].map("${:.2f}".format)
                        display_df["iv"] = display_df["iv"].map("{:.1%}".format)
                        display_df["delta"] = display_df["delta"].map("{:.4f}".format)
                        display_df["gamma"] = display_df["gamma"].map("{:.4f}".format)
                        display_df["theta"] = display_df["theta"].map("{:.4f}".format)
                        display_df["vega"] = display_df["vega"].map("{:.4f}".format)
                        display_df["volume"] = display_df["volume"].fillna(0).astype(int)
                        display_df["open_interest"] = display_df["open_interest"].fillna(0).astype(int)
                        display_df = display_df.rename(columns={
                            "expiration": "Expiration",
                            "days_to_exp": "DTE",
                            "strike": "Strike",
                            "mid_price": "Mid",
                            "intrinsic": "Intrinsic",
                            "time_value": "Time Val",
                            "iv": "IV",
                            "delta": "Δ",
                            "gamma": "Γ",
                            "theta": "Θ",
                            "vega": "ν",
                            "volume": "Vol",
                            "open_interest": "OI",
                            "type": "Type",
                            "moneyness": "Moneyness",
                        })
                        display_cols = ["Expiration", "DTE", "Strike", "Mid", "Intrinsic", "Time Val",
                                        "IV", "Δ", "Γ", "Θ", "ν", "Vol", "OI"]

                        with chain_tab_calls:
                            calls_table = display_df[display_df["Type"] == "call"][display_cols].sort_values(
                                ["Expiration", "Strike"]
                            )
                            st.dataframe(calls_table, hide_index=True, use_container_width=True, height=400)

                        with chain_tab_puts:
                            puts_table = display_df[display_df["Type"] == "put"][display_cols].sort_values(
                                ["Expiration", "Strike"]
                            )
                            st.dataframe(puts_table, hide_index=True, use_container_width=True, height=400)

            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.info("Make sure the ticker is valid and markets have been open recently.")


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with Streamlit • Black-Scholes model (European options)"
)
