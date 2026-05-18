"""
Unit tests for options analytics functions.

Run with:  pytest test_options_analytics.py -v
"""

import numpy as np
import pandas as pd
import pytest

from options_analytics import (
    d1, d2, bs_call, bs_put, bs_price,
    delta, gamma, theta, vega, rho, all_greeks,
    implied_volatility, put_call_parity_check,
    compute_greeks_vs_spot,
)


# ──────────────────────────────────────────────
# Reference values from Hull "Options, Futures, and Other Derivatives"
# S=100, K=100, T=1, r=5%, sigma=20%
# ──────────────────────────────────────────────
S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


class TestBlackScholesPricing:
    def test_atm_call_price(self):
        """ATM call with known parameters should match textbook value."""
        price = bs_call(S, K, T, R, SIGMA)
        # Expected ~10.45 (Hull reference)
        assert 10.0 < price < 11.0

    def test_atm_put_price(self):
        """ATM put should be less than call (positive rates, ATM)."""
        call = bs_call(S, K, T, R, SIGMA)
        put = bs_put(S, K, T, R, SIGMA)
        assert put < call

    def test_deep_itm_call(self):
        """Deep ITM call should be close to intrinsic value."""
        price = bs_call(150, 100, 0.01, R, SIGMA)
        assert abs(price - 50.0) < 1.0

    def test_deep_otm_call(self):
        """Deep OTM call should be near zero."""
        price = bs_call(50, 100, 0.1, R, SIGMA)
        assert price < 0.01

    def test_expired_call(self):
        """Expired call should return intrinsic value."""
        assert bs_call(110, 100, 0, R, SIGMA) == 10.0
        assert bs_call(90, 100, 0, R, SIGMA) == 0.0

    def test_expired_put(self):
        """Expired put should return intrinsic value."""
        assert bs_put(90, 100, 0, R, SIGMA) == 10.0
        assert bs_put(110, 100, 0, R, SIGMA) == 0.0

    def test_call_increases_with_spot(self):
        """Call price should increase with spot price."""
        p1 = bs_call(95, K, T, R, SIGMA)
        p2 = bs_call(105, K, T, R, SIGMA)
        assert p2 > p1

    def test_put_increases_with_strike(self):
        """Put price should increase as spot decreases."""
        p1 = bs_put(105, K, T, R, SIGMA)
        p2 = bs_put(95, K, T, R, SIGMA)
        assert p2 > p1

    def test_price_increases_with_vol(self):
        """Both call and put should increase with volatility."""
        c_low = bs_call(S, K, T, R, 0.10)
        c_high = bs_call(S, K, T, R, 0.40)
        assert c_high > c_low

        p_low = bs_put(S, K, T, R, 0.10)
        p_high = bs_put(S, K, T, R, 0.40)
        assert p_high > p_low

    def test_price_increases_with_time(self):
        """Options should be worth more with more time to expiration."""
        c_short = bs_call(S, K, 0.25, R, SIGMA)
        c_long = bs_call(S, K, 2.0, R, SIGMA)
        assert c_long > c_short


class TestPutCallParity:
    def test_parity_holds(self):
        """Put-call parity: C - P = S - K*e^(-rT)."""
        call = bs_call(S, K, T, R, SIGMA)
        put = bs_put(S, K, T, R, SIGMA)
        lhs = call - put
        rhs = S - K * np.exp(-R * T)
        assert abs(lhs - rhs) < 1e-10

    def test_parity_various_strikes(self):
        """Parity should hold for various strikes."""
        for strike in [80, 90, 100, 110, 120]:
            call = bs_call(S, strike, T, R, SIGMA)
            put = bs_put(S, strike, T, R, SIGMA)
            lhs = call - put
            rhs = S - strike * np.exp(-R * T)
            assert abs(lhs - rhs) < 1e-10

    def test_parity_checker(self):
        """The parity check function should confirm parity."""
        call = bs_call(S, K, T, R, SIGMA)
        put = bs_put(S, K, T, R, SIGMA)
        result = put_call_parity_check(call, put, S, K, T, R)
        assert result["parity_holds"]
        assert abs(result["deviation"]) < 0.01


class TestDelta:
    def test_call_delta_range(self):
        """Call delta should be between 0 and 1."""
        d = delta(S, K, T, R, SIGMA, "call")
        assert 0 < d < 1

    def test_put_delta_range(self):
        """Put delta should be between -1 and 0."""
        d = delta(S, K, T, R, SIGMA, "put")
        assert -1 < d < 0

    def test_call_put_delta_relationship(self):
        """Call delta - Put delta = 1."""
        dc = delta(S, K, T, R, SIGMA, "call")
        dp = delta(S, K, T, R, SIGMA, "put")
        assert abs((dc - dp) - 1.0) < 1e-10

    def test_deep_itm_call_delta(self):
        """Deep ITM call delta should be close to 1."""
        d = delta(200, 100, T, R, SIGMA, "call")
        assert d > 0.99

    def test_deep_otm_call_delta(self):
        """Deep OTM call delta should be close to 0."""
        d = delta(50, 100, T, R, SIGMA, "call")
        assert d < 0.01

    def test_atm_call_delta_near_half(self):
        """ATM call delta should be near 0.5 (slightly above due to drift)."""
        d = delta(S, K, T, R, SIGMA, "call")
        assert 0.45 < d < 0.65


class TestGamma:
    def test_gamma_positive(self):
        """Gamma should always be positive."""
        g = gamma(S, K, T, R, SIGMA)
        assert g > 0

    def test_gamma_same_for_call_put(self):
        """Gamma is the same for calls and puts (it doesn't depend on type)."""
        # gamma function doesn't take option_type — this is by design
        g = gamma(S, K, T, R, SIGMA)
        assert g > 0

    def test_gamma_peaks_atm(self):
        """Gamma should be highest near ATM."""
        g_atm = gamma(100, 100, T, R, SIGMA)
        g_itm = gamma(120, 100, T, R, SIGMA)
        g_otm = gamma(80, 100, T, R, SIGMA)
        assert g_atm > g_itm
        assert g_atm > g_otm

    def test_gamma_increases_near_expiry(self):
        """ATM gamma should increase as expiration approaches."""
        g_far = gamma(S, K, 1.0, R, SIGMA)
        g_near = gamma(S, K, 0.05, R, SIGMA)
        assert g_near > g_far


class TestTheta:
    def test_call_theta_negative(self):
        """Call theta should be negative (time decay)."""
        t = theta(S, K, T, R, SIGMA, "call")
        assert t < 0

    def test_put_theta_negative(self):
        """Put theta is usually negative for ATM."""
        t = theta(S, K, T, R, SIGMA, "put")
        assert t < 0

    def test_theta_accelerates_near_expiry(self):
        """ATM theta magnitude should increase near expiration."""
        t_far = abs(theta(S, K, 1.0, R, SIGMA, "call"))
        t_near = abs(theta(S, K, 0.05, R, SIGMA, "call"))
        assert t_near > t_far


class TestVega:
    def test_vega_positive(self):
        """Vega should be positive."""
        v = vega(S, K, T, R, SIGMA)
        assert v > 0

    def test_vega_peaks_atm(self):
        """Vega should be highest near ATM."""
        v_atm = vega(100, 100, T, R, SIGMA)
        v_itm = vega(130, 100, T, R, SIGMA)
        v_otm = vega(70, 100, T, R, SIGMA)
        assert v_atm > v_itm
        assert v_atm > v_otm

    def test_vega_increases_with_time(self):
        """Vega should increase with time to expiration."""
        v_short = vega(S, K, 0.1, R, SIGMA)
        v_long = vega(S, K, 2.0, R, SIGMA)
        assert v_long > v_short


class TestRho:
    def test_call_rho_positive(self):
        """Call rho should be positive."""
        r_val = rho(S, K, T, R, SIGMA, "call")
        assert r_val > 0

    def test_put_rho_negative(self):
        """Put rho should be negative."""
        r_val = rho(S, K, T, R, SIGMA, "put")
        assert r_val < 0


class TestImpliedVolatility:
    def test_roundtrip_call(self):
        """Price a call, then recover IV from the price."""
        price = bs_call(S, K, T, R, 0.30)
        iv = implied_volatility(price, S, K, T, R, "call")
        assert iv is not None
        assert abs(iv - 0.30) < 1e-6

    def test_roundtrip_put(self):
        """Price a put, then recover IV from the price."""
        price = bs_put(S, K, T, R, 0.25)
        iv = implied_volatility(price, S, K, T, R, "put")
        assert iv is not None
        assert abs(iv - 0.25) < 1e-6

    def test_roundtrip_various_vols(self):
        """IV recovery should work across a range of volatilities."""
        for target_vol in [0.05, 0.15, 0.30, 0.50, 0.80, 1.5]:
            price = bs_call(S, K, T, R, target_vol)
            iv = implied_volatility(price, S, K, T, R, "call")
            assert iv is not None
            assert abs(iv - target_vol) < 1e-4, f"Failed for vol={target_vol}"

    def test_below_intrinsic_returns_none(self):
        """Price below intrinsic should return None."""
        iv = implied_volatility(0.5, 110, 100, T, R, "call")  # intrinsic = 10
        assert iv is None

    def test_expired_returns_none(self):
        """Expired option should return None."""
        iv = implied_volatility(5.0, S, K, 0.0, R, "call")
        assert iv is None


class TestGreeksVsSpot:
    def test_output_shape(self):
        """compute_greeks_vs_spot should return matching array lengths."""
        result = compute_greeks_vs_spot(K=100, T=0.5, r=0.05, sigma=0.20, n_points=50)
        assert len(result["spots"]) == 50
        assert len(result["price"]) == 50
        assert len(result["delta"]) == 50
        assert len(result["gamma"]) == 50

    def test_call_price_monotonic(self):
        """Call price should increase monotonically with spot."""
        result = compute_greeks_vs_spot(K=100, T=0.5, r=0.05, sigma=0.20,
                                        option_type="call", n_points=100)
        diffs = np.diff(result["price"])
        assert all(diffs >= -1e-10)


class TestAllGreeks:
    def test_returns_all_keys(self):
        """all_greeks should return all 5 Greeks."""
        g = all_greeks(S, K, T, R, SIGMA, "call")
        assert set(g.keys()) == {"delta", "gamma", "theta", "vega", "rho"}

    def test_consistency_with_individual(self):
        """all_greeks values should match individual function calls."""
        g = all_greeks(S, K, T, R, SIGMA, "call")
        assert abs(g["delta"] - delta(S, K, T, R, SIGMA, "call")) < 1e-10
        assert abs(g["gamma"] - gamma(S, K, T, R, SIGMA)) < 1e-10
        assert abs(g["theta"] - theta(S, K, T, R, SIGMA, "call")) < 1e-10
        assert abs(g["vega"] - vega(S, K, T, R, SIGMA)) < 1e-10
        assert abs(g["rho"] - rho(S, K, T, R, SIGMA, "call")) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
