"""
core/math_engine.py
--------------------
OrbiTrade v.1 - Deterministic mathematics layer.

Instead of letting an LLM perform (and potentially hallucinate) numerical
calculations, every quantitative computation lives here in pure
Python/NumPy/SciPy. Agents call these functions as tools and only reason
qualitatively over the numeric results they return.

Scope:
    1) Black-Scholes option pricing (call/put)
    2) Greeks: Delta, Gamma, Vega, Theta, Rho
    3) Implied Volatility (Brent's method root-finding)
    4) Technical indicators: RSI, ATR, Bollinger Bands
    5) Risk / position sizing: Kelly Criterion, Parametric VaR, Historical VaR

Notes:
    - All "T" (time to expiry) values are in years (e.g. 30 days -> 30/365).
    - All rates (r, sigma) are annualized decimals (e.g. 5% -> 0.05).
    - Functions currently operate on single float inputs; they can be
      vectorized later to run across a full option chain in bulk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

OptionType = Literal["call", "put"]


# ---------------------------------------------------------------------------
# 1) BLACK-SCHOLES CORE
# ---------------------------------------------------------------------------

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Computes the Black-Scholes d1 and d2 terms."""
    if T <= 0:
        raise ValueError("Time to expiry (T) must be greater than zero.")
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be greater than zero.")
    if S <= 0 or K <= 0:
        raise ValueError("Spot (S) and strike (K) must be greater than zero.")

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType = "call"
) -> float:
    """
    Theoretical Black-Scholes price for a European-style option.

    Args:
        S: Spot (underlying) price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free interest rate (annualized, decimal)
        sigma: Annualized volatility (decimal, e.g. 0.20 = 20%)
        option_type: "call" or "put"

    Returns:
        Theoretical option price (float)
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float   # per 1% (0.01) change in volatility
    theta: float  # daily time decay (per 1 day)
    rho: float    # per 1% (0.01) change in interest rate


def calculate_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType = "call"
) -> Greeks:
    """
    Computes the option's risk sensitivities (Greeks).

    Vega and Rho are normalized per 1% (0.01) change; Theta is normalized
    to DAILY time decay (annual/365) so the Debate/Risk agent can read a
    directly usable "daily theta cost" figure.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    sqrt_T = math.sqrt(T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T * 0.01  # per 1% vol change

    if option_type == "call":
        delta = norm.cdf(d1)
        theta_annual = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * math.exp(-r * T) * norm.cdf(d2)
        )
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) * 0.01
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
        theta_annual = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T) + r * K * math.exp(-r * T) * norm.cdf(-d2)
        )
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) * 0.01
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    theta_daily = theta_annual / 365.0

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta_daily, rho=rho)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    sigma_bounds: tuple[float, float] = (1e-4, 5.0),
) -> float:
    """
    Solves for implied volatility from a given market_price using Brent's
    method. Brent is preferred over Newton-Raphson because it doesn't
    require a derivative and converges more reliably (Newton-Raphson can
    diverge in deep ITM/OTM regions where vega is near zero).

    Raises:
        ValueError: If no root exists within sigma_bounds (i.e. market_price
                    lies outside the arbitrage-free price bounds).
    """
    lo, hi = sigma_bounds

    def objective(sigma: float) -> float:
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            "Implied volatility could not be found within the given bounds; "
            "market_price may lie outside the arbitrage-free bounds."
        )

    return brentq(objective, lo, hi, xtol=1e-8, maxiter=200)


# ---------------------------------------------------------------------------
# 2) TECHNICAL INDICATORS
# ---------------------------------------------------------------------------

def calculate_rsi(closes: Sequence[float], period: int = 14) -> np.ndarray:
    """
    Relative Strength Index (RSI) using Wilder's original smoothing method.

    Args:
        closes: Sequence of closing prices (oldest to newest)
        period: RSI period (default 14)

    Returns:
        Numpy array the same length as closes; the first `period` elements
        are NaN (not enough data yet).
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    rsi = np.full(n, np.nan)

    if n <= period:
        return rsi

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    rsi[period] = _rsi_from_avgs(avg_gain, avg_loss)

    # Wilder's exponential smoothing (smoothed moving average)
    for i in range(period + 1, n):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rsi[i] = _rsi_from_avgs(avg_gain, avg_loss)

    return rsi


def calculate_atr(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14
) -> np.ndarray:
    """
    Average True Range (ATR) using Wilder's exponential smoothing method.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)

    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows, and closes must be the same length.")

    atr = np.full(n, np.nan)
    if n <= period:
        return atr

    prev_close = closes[:-1]
    tr = np.empty(n - 1)
    tr[0] = highs[1] - lows[1]  # no previous close for the first bar, just H-L
    for i in range(1, n - 1):
        tr[i] = max(
            highs[i + 1] - lows[i + 1],
            abs(highs[i + 1] - prev_close[i]),
            abs(lows[i + 1] - prev_close[i]),
        )

    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period

    return atr


def calculate_bollinger_bands(
    closes: Sequence[float], period: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands: middle band (SMA), upper band, and lower band.

    Returns:
        (middle_band, upper_band, lower_band) - all the same length as
        closes; the first `period-1` elements are NaN.
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        mean = window.mean()
        std = window.std(ddof=0)
        middle[i] = mean
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std

    return middle, upper, lower


# ---------------------------------------------------------------------------
# 3) RISK / POSITION SIZING
# ---------------------------------------------------------------------------

def kelly_criterion(win_prob: float, win_loss_ratio: float, fraction: float = 1.0) -> float:
    """
    Computes the optimal bet/position size fraction using the Kelly Criterion.

    f* = W - (1 - W) / R

    Args:
        win_prob: Probability of winning (0-1), W
        win_loss_ratio: Average win / average loss ratio, R
        fraction: "Fractional Kelly" multiplier (e.g. 0.5 = Half-Kelly).
                  Full Kelly is aggressive, so 0.25-0.5 is common in practice.

    Returns:
        The fraction of the portfolio (0-1, negative means don't trade)
        that should be allocated to this trade.
    """
    if not (0 <= win_prob <= 1):
        raise ValueError("win_prob must be between 0 and 1.")
    if win_loss_ratio <= 0:
        raise ValueError("win_loss_ratio must be greater than zero.")

    f_star = win_prob - (1 - win_prob) / win_loss_ratio
    return f_star * fraction


def parametric_var(
    portfolio_value: float, mu: float, sigma: float, confidence: float = 0.95, horizon_days: int = 1
) -> float:
    """
    Parametric (variance-covariance) Value at Risk.

    Args:
        portfolio_value: Portfolio value (currency)
        mu: Expected daily return (decimal, e.g. 0.001)
        sigma: Daily return standard deviation (decimal)
        confidence: Confidence level (default 95%)
        horizon_days: Number of days the VaR is computed over

    Returns:
        A positive number representing the expected maximum loss (currency).
        i.e. "with 95% confidence, over N days you won't lose more than this."
    """
    z = norm.ppf(1 - confidence)  # negative value, e.g. ~ -1.645 for 95%
    scaled_mu = mu * horizon_days
    scaled_sigma = sigma * math.sqrt(horizon_days)
    var = -(portfolio_value * (scaled_mu + z * scaled_sigma))
    return max(var, 0.0)


def historical_var(returns: Sequence[float], portfolio_value: float, confidence: float = 0.95) -> float:
    """
    Value at Risk via historical simulation: uses the empirical percentile
    of past returns without assuming any distribution.

    Args:
        returns: Historical periodic returns (decimal, e.g. [-0.01, 0.02, ...])
        portfolio_value: Portfolio value
        confidence: Confidence level

    Returns:
        A positive number representing the expected maximum loss (currency).
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        raise ValueError("returns must not be empty.")

    percentile = (1 - confidence) * 100
    cutoff_return = np.percentile(returns, percentile)
    return max(-portfolio_value * cutoff_return, 0.0)
