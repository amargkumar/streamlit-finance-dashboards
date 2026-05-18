"""
options_analytics.py — Black-Scholes pricing, Greeks, and implied volatility.

Pure functions with no Streamlit dependency — importable for testing and reuse.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Dict, Tuple, Optional


# ──────────────────────────────────────────────
# Black-Scholes Pricing
# ──────────────────────────────────────────────
def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 in the Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 in the Black-Scholes formula."""
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call option price.

    Args:
        S: Current spot price of the underlying.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized, continuous).
        sigma: Volatility of the underlying (annualized).

    Returns:
        Theoretical call option price.
    """
    if T <= 0:
        return max(S - K, 0.0)
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    return S * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put option price.

    Args:
        S: Current spot price of the underlying.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized, continuous).
        sigma: Volatility of the underlying (annualized).

    Returns:
        Theoretical put option price.
    """
    if T <= 0:
        return max(K - S, 0.0)
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes price for call or put."""
    if option_type == "call":
        return bs_call(S, K, T, r, sigma)
    return bs_put(S, K, T, r, sigma)


# ──────────────────────────────────────────────
# Greeks
# ──────────────────────────────────────────────
def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Option delta — sensitivity of price to underlying price."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    _d1 = d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(_d1)
    return norm.cdf(_d1) - 1


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Option gamma — rate of change of delta. Same for calls and puts."""
    if T <= 0 or sigma <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma)
    return norm.pdf(_d1) / (S * sigma * np.sqrt(T))


def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Option theta — sensitivity of price to time (per calendar day).

    Returns a negative value (options lose value as time passes).
    Expressed per calendar day (divide annual by 365).
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    common = -(S * norm.pdf(_d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        result = common - r * K * np.exp(-r * T) * norm.cdf(_d2)
    else:
        result = common + r * K * np.exp(-r * T) * norm.cdf(-_d2)
    return result / 365  # per calendar day


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Option vega — sensitivity of price to volatility.

    Returns change in price per 1% change in vol (i.e., divided by 100).
    Same for calls and puts.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(_d1) * np.sqrt(T) / 100


def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Option rho — sensitivity of price to interest rate.

    Returns change in price per 1% change in rate (divided by 100).
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    _d2 = d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(_d2) / 100
    return -K * T * np.exp(-r * T) * norm.cdf(-_d2) / 100


def all_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Dict[str, float]:
    """Compute all Greeks at once."""
    return {
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option_type),
        "vega": vega(S, K, T, r, sigma),
        "rho": rho(S, K, T, r, sigma, option_type),
    }


# ──────────────────────────────────────────────
# Implied Volatility
# ──────────────────────────────────────────────
def implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float,
    option_type: str = "call", tol: float = 1e-8, max_iter: int = 100,
) -> Optional[float]:
    """Compute implied volatility using Brent's method.

    Args:
        market_price: Observed market price of the option.
        S: Spot price.
        K: Strike price.
        T: Time to expiration (years).
        r: Risk-free rate.
        option_type: 'call' or 'put'.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        Implied volatility (annualized), or None if it cannot converge.
    """
    if T <= 0:
        return None

    # Check intrinsic value bounds
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if market_price < intrinsic:
        return None

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        iv = brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
        return iv
    except (ValueError, RuntimeError):
        return None


# ──────────────────────────────────────────────
# Surface / Grid Computation
# ──────────────────────────────────────────────
def compute_greeks_surface(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call",
    spot_range: Tuple[float, float] = (0.7, 1.3),
    vol_range: Tuple[float, float] = (0.05, 0.80),
    n_points: int = 50,
) -> Dict[str, np.ndarray]:
    """Compute Greeks across a grid of spot prices and volatilities.

    Args:
        S: Center spot price.
        K: Strike price.
        T: Time to expiration.
        r: Risk-free rate.
        sigma: Center volatility (used for reference but grid covers vol_range).
        option_type: 'call' or 'put'.
        spot_range: (low_mult, high_mult) — multipliers of S for the spot axis.
        vol_range: (low_vol, high_vol) — range of volatilities.
        n_points: Grid resolution per axis.

    Returns:
        Dict with 'spots', 'vols', and 2D arrays for each Greek and price.
    """
    spots = np.linspace(S * spot_range[0], S * spot_range[1], n_points)
    vols = np.linspace(vol_range[0], vol_range[1], n_points)

    price_grid = np.zeros((n_points, n_points))
    delta_grid = np.zeros((n_points, n_points))
    gamma_grid = np.zeros((n_points, n_points))
    theta_grid = np.zeros((n_points, n_points))
    vega_grid = np.zeros((n_points, n_points))

    for i, v in enumerate(vols):
        for j, s in enumerate(spots):
            price_grid[i, j] = bs_price(s, K, T, r, v, option_type)
            delta_grid[i, j] = delta(s, K, T, r, v, option_type)
            gamma_grid[i, j] = gamma(s, K, T, r, v)
            theta_grid[i, j] = theta(s, K, T, r, v, option_type)
            vega_grid[i, j] = vega(s, K, T, r, v)

    return {
        "spots": spots,
        "vols": vols,
        "price": price_grid,
        "delta": delta_grid,
        "gamma": gamma_grid,
        "theta": theta_grid,
        "vega": vega_grid,
    }


def compute_greeks_vs_spot(
    K: float, T: float, r: float, sigma: float, option_type: str = "call",
    spot_min: float = 50.0, spot_max: float = 200.0, n_points: int = 200,
) -> Dict[str, np.ndarray]:
    """Compute option price and Greeks across a range of spot prices.

    Returns dict with 'spots', 'price', 'delta', 'gamma', 'theta', 'vega', 'rho'.
    """
    spots = np.linspace(spot_min, spot_max, n_points)
    prices = np.array([bs_price(s, K, T, r, sigma, option_type) for s in spots])
    deltas = np.array([delta(s, K, T, r, sigma, option_type) for s in spots])
    gammas = np.array([gamma(s, K, T, r, sigma) for s in spots])
    thetas = np.array([theta(s, K, T, r, sigma, option_type) for s in spots])
    vegas = np.array([vega(s, K, T, r, sigma) for s in spots])
    rhos = np.array([rho(s, K, T, r, sigma, option_type) for s in spots])

    return {
        "spots": spots,
        "price": prices,
        "delta": deltas,
        "gamma": gammas,
        "theta": thetas,
        "vega": vegas,
        "rho": rhos,
    }


def compute_price_vs_time(
    S: float, K: float, r: float, sigma: float, option_type: str = "call",
    max_T: float = 1.0, n_points: int = 200,
) -> Dict[str, np.ndarray]:
    """Compute option price and theta across time to expiration.

    Returns dict with 'times' (in days), 'price', 'theta'.
    """
    times_years = np.linspace(max_T, 0.001, n_points)
    times_days = times_years * 365

    prices = np.array([bs_price(S, K, t, r, sigma, option_type) for t in times_years])
    thetas = np.array([theta(S, K, t, r, sigma, option_type) for t in times_years])

    return {
        "times_days": times_days,
        "times_years": times_years,
        "price": prices,
        "theta": thetas,
    }


# ──────────────────────────────────────────────
# Put-Call Parity
# ──────────────────────────────────────────────
def put_call_parity_check(
    call_price: float, put_price: float, S: float, K: float, T: float, r: float
) -> Dict[str, float]:
    """Check put-call parity: C - P = S - K*e^(-rT).

    Returns the LHS, RHS, and deviation.
    """
    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)
    return {
        "call_minus_put": lhs,
        "spot_minus_pv_strike": rhs,
        "deviation": lhs - rhs,
        "parity_holds": abs(lhs - rhs) < 0.01,
    }
