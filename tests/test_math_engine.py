"""
tests/test_math_engine.py
--------------------------
Validation tests for core/math_engine.py (OrbiTrade v.1).

Test strategy:
    1) Comparison against known reference values (classic Hull textbook
       example: S=100, K=100, T=1, r=0.05, sigma=0.20).
    2) Cross-validation via mathematical identities (put-call parity,
       delta bounds, vega always positive, etc.).
    3) Implied volatility round-trip test (price -> IV -> price).
    4) Technical indicators tested against hand-computed small examples.
    5) Edge/error cases for Kelly and VaR.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.math_engine import (
    black_scholes_price,
    calculate_greeks,
    implied_volatility,
    calculate_rsi,
    calculate_atr,
    calculate_bollinger_bands,
    kelly_criterion,
    parametric_var,
    historical_var,
)


# ---------------------------------------------------------------------------
# BLACK-SCHOLES PRICING
# ---------------------------------------------------------------------------

class TestBlackScholesPrice:
    # Reference: Hull, "Options, Futures and Other Derivatives" classic example
    # S=100, K=100, T=1, r=0.05, sigma=0.20 -> Call ~= 10.4506, Put ~= 5.5735
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    def test_call_price_matches_reference(self):
        price = black_scholes_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert price == pytest.approx(10.4506, abs=1e-3)

    def test_put_price_matches_reference(self):
        price = black_scholes_price(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert price == pytest.approx(5.5735, abs=1e-3)

    def test_put_call_parity(self):
        # C - P = S - K * e^(-rT)
        call = black_scholes_price(self.S, self.K, self.T, self.r, self.sigma, "call")
        put = black_scholes_price(self.S, self.K, self.T, self.r, self.sigma, "put")
        rhs = self.S - self.K * math.exp(-self.r * self.T)
        assert (call - put) == pytest.approx(rhs, abs=1e-8)

    def test_deep_itm_call_converges_to_intrinsic(self):
        # Very low volatility + deep ITM -> price should converge to intrinsic value
        price = black_scholes_price(200.0, 100.0, 1.0, 0.05, 0.01, "call")
        intrinsic = 200.0 - 100.0 * math.exp(-0.05 * 1.0)
        assert price == pytest.approx(intrinsic, abs=0.5)

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            black_scholes_price(self.S, self.K, self.T, self.r, self.sigma, "invalid")

    def test_zero_or_negative_T_raises(self):
        with pytest.raises(ValueError):
            black_scholes_price(self.S, self.K, 0.0, self.r, self.sigma, "call")

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            black_scholes_price(self.S, self.K, self.T, self.r, -0.1, "call")


# ---------------------------------------------------------------------------
# GREEKS
# ---------------------------------------------------------------------------

class TestGreeks:
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    def test_call_delta_between_0_and_1(self):
        g = calculate_greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert 0.0 <= g.delta <= 1.0

    def test_put_delta_between_minus1_and_0(self):
        g = calculate_greeks(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert -1.0 <= g.delta <= 0.0

    def test_call_put_delta_relationship(self):
        # Delta_call - Delta_put = 1
        g_call = calculate_greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        g_put = calculate_greeks(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert (g_call.delta - g_put.delta) == pytest.approx(1.0, abs=1e-8)

    def test_gamma_is_positive_and_equal_for_call_and_put(self):
        g_call = calculate_greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        g_put = calculate_greeks(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert g_call.gamma > 0
        assert g_call.gamma == pytest.approx(g_put.gamma, rel=1e-8)

    def test_vega_is_positive_and_equal_for_call_and_put(self):
        g_call = calculate_greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        g_put = calculate_greeks(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert g_call.vega > 0
        assert g_call.vega == pytest.approx(g_put.vega, rel=1e-8)

    def test_atm_call_delta_near_half(self):
        # ATM (S=K) call delta is generally around 0.5-0.6 depending on r and T
        g = calculate_greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert 0.5 <= g.delta <= 0.65

    def test_greeks_numerically_via_finite_difference(self):
        # Cross-check delta via finite-difference (price difference / dS)
        eps = 0.01
        price_up = black_scholes_price(self.S + eps, self.K, self.T, self.r, self.sigma, "call")
        price_down = black_scholes_price(self.S - eps, self.K, self.T, self.r, self.sigma, "call")
        numeric_delta = (price_up - price_down) / (2 * eps)

        analytic_delta = calculate_greeks(self.S, self.K, self.T, self.r, self.sigma, "call").delta
        assert numeric_delta == pytest.approx(analytic_delta, abs=1e-3)


# ---------------------------------------------------------------------------
# IMPLIED VOLATILITY
# ---------------------------------------------------------------------------

class TestImpliedVolatility:
    def test_round_trip_recovers_original_sigma(self):
        S, K, T, r, true_sigma = 100.0, 105.0, 0.5, 0.03, 0.25
        market_price = black_scholes_price(S, K, T, r, true_sigma, "call")

        recovered_sigma = implied_volatility(market_price, S, K, T, r, "call")
        assert recovered_sigma == pytest.approx(true_sigma, abs=1e-4)

    def test_round_trip_for_put(self):
        S, K, T, r, true_sigma = 50.0, 45.0, 0.25, 0.02, 0.35
        market_price = black_scholes_price(S, K, T, r, true_sigma, "put")

        recovered_sigma = implied_volatility(market_price, S, K, T, r, "put")
        assert recovered_sigma == pytest.approx(true_sigma, abs=1e-4)

    def test_unreachable_price_raises(self):
        # A price below the intrinsic value / outside arbitrage-free bounds -> unsolvable
        with pytest.raises(ValueError):
            implied_volatility(-5.0, 100.0, 100.0, 1.0, 0.05, "call")


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_all_gains_gives_rsi_100(self):
        # Steadily increasing series -> no losses -> RSI = 100
        closes = [100 + i for i in range(20)]
        rsi = calculate_rsi(closes, period=14)
        assert rsi[14] == pytest.approx(100.0)

    def test_all_losses_gives_rsi_0(self):
        closes = [100 - i for i in range(20)]
        rsi = calculate_rsi(closes, period=14)
        assert rsi[14] == pytest.approx(0.0)

    def test_rsi_bounded_between_0_and_100(self):
        rng = np.random.default_rng(42)
        closes = 100 + np.cumsum(rng.normal(0, 1, 100))
        rsi = calculate_rsi(closes, period=14)
        valid = rsi[~np.isnan(rsi)]
        assert np.all((valid >= 0) & (valid <= 100))

    def test_insufficient_data_returns_nan(self):
        closes = [100, 101, 102]
        rsi = calculate_rsi(closes, period=14)
        assert np.all(np.isnan(rsi))


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestATR:
    def test_atr_matches_manual_calculation_flat_range(self):
        # A constant True Range (high-low = 2, no gaps) means ATR should
        # converge to 2 over time.
        n = 30
        closes = [100.0] * n
        highs = [101.0] * n
        lows = [99.0] * n

        atr = calculate_atr(highs, lows, closes, period=14)
        assert atr[14] == pytest.approx(2.0, abs=1e-8)
        assert atr[-1] == pytest.approx(2.0, abs=1e-8)

    def test_atr_is_never_negative(self):
        rng = np.random.default_rng(7)
        closes = 100 + np.cumsum(rng.normal(0, 1, 50))
        highs = closes + np.abs(rng.normal(0.5, 0.2, 50))
        lows = closes - np.abs(rng.normal(0.5, 0.2, 50))

        atr = calculate_atr(highs, lows, closes, period=14)
        valid = atr[~np.isnan(atr)]
        assert np.all(valid >= 0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            calculate_atr([1, 2, 3], [1, 2], [1, 2, 3])


# ---------------------------------------------------------------------------
# BOLLINGER BANDS
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_bands_straddle_middle_correctly(self):
        closes = list(np.linspace(100, 120, 40))
        middle, upper, lower = calculate_bollinger_bands(closes, period=20, num_std=2.0)

        valid_idx = ~np.isnan(middle)
        assert np.all(upper[valid_idx] >= middle[valid_idx])
        assert np.all(lower[valid_idx] <= middle[valid_idx])

    def test_constant_series_gives_zero_width_bands(self):
        closes = [100.0] * 30
        middle, upper, lower = calculate_bollinger_bands(closes, period=20, num_std=2.0)
        valid_idx = ~np.isnan(middle)
        assert np.allclose(upper[valid_idx], middle[valid_idx])
        assert np.allclose(lower[valid_idx], middle[valid_idx])


# ---------------------------------------------------------------------------
# KELLY CRITERION
# ---------------------------------------------------------------------------

class TestKellyCriterion:
    def test_classic_example(self):
        # W=0.6, R=2 -> f* = 0.6 - 0.4/2 = 0.4
        f_star = kelly_criterion(win_prob=0.6, win_loss_ratio=2.0)
        assert f_star == pytest.approx(0.4)

    def test_half_kelly_fraction(self):
        f_star = kelly_criterion(win_prob=0.6, win_loss_ratio=2.0, fraction=0.5)
        assert f_star == pytest.approx(0.2)

    def test_negative_edge_gives_negative_fraction(self):
        # Low win probability and bad R -> should not trade (negative kelly)
        f_star = kelly_criterion(win_prob=0.3, win_loss_ratio=1.0)
        assert f_star < 0

    def test_invalid_win_prob_raises(self):
        with pytest.raises(ValueError):
            kelly_criterion(win_prob=1.5, win_loss_ratio=2.0)

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            kelly_criterion(win_prob=0.5, win_loss_ratio=0.0)


# ---------------------------------------------------------------------------
# VaR
# ---------------------------------------------------------------------------

class TestVaR:
    def test_parametric_var_is_positive(self):
        var = parametric_var(portfolio_value=100_000, mu=0.0005, sigma=0.02, confidence=0.95)
        assert var > 0

    def test_parametric_var_scales_with_horizon(self):
        var_1d = parametric_var(portfolio_value=100_000, mu=0.0, sigma=0.02, confidence=0.95, horizon_days=1)
        var_4d = parametric_var(portfolio_value=100_000, mu=0.0, sigma=0.02, confidence=0.95, horizon_days=4)
        # Volatility scales with sqrt(t) -> 4-day VaR should be ~2x the 1-day VaR
        assert var_4d == pytest.approx(var_1d * 2, rel=1e-6)

    def test_historical_var_matches_percentile(self):
        returns = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, -0.10]
        var = historical_var(returns, portfolio_value=10_000, confidence=0.95)
        expected_cutoff = np.percentile(returns, 5)
        assert var == pytest.approx(-10_000 * expected_cutoff, abs=1e-6)

    def test_historical_var_empty_raises(self):
        with pytest.raises(ValueError):
            historical_var([], portfolio_value=10_000)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
