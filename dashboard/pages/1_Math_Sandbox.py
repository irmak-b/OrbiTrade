"""
dashboard/pages/1_Math_Sandbox.py
-----------------------------------
OrbiTrade - Math Engine Sandbox. Streamlit'in native multi-page özelliği
sayesinde bu dosya dashboard/app.py ile AYNI deployment / AYNI URL altında,
sol menüde ikinci sekme olarak otomatik görünür. Ayrı bir "streamlit run"
veya ayrı bir Cloud deployment'ına gerek yok.

Bu dosya eski dashboard/app_2.py ile dashboard/app.py'deki tab_sandbox
bloğunun birleşimidir; içerik değişmedi, sadece konumu değişti.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> proje kökü

from core.math_engine import (
    black_scholes_price,
    calculate_greeks,
    implied_volatility,
    calculate_rsi,
    calculate_atr,
    calculate_bollinger_bands,
    kelly_criterion,
    parametric_var,
)

st.set_page_config(page_title="OrbiTrade - Math Engine Sandbox", layout="wide", page_icon="🧮")

st.title("Deterministic Option Pricing (Black-Scholes & Greeks)")
st.caption(
    "Zero LLM hallucinations. Used by the Data Agent to price every candidate "
    "option contract before the Quant Agent ever sees it -- 35/35 tests passed."
)

tab_options, tab_technical, tab_risk = st.tabs(
    ["Option Pricing & Greeks", "Technical Indicators", "Kelly & VaR"]
)

# ---------------------------------------------------------------------------
# TAB 1: Black-Scholes & Greeks
# ---------------------------------------------------------------------------
with tab_options:
    col_inputs, col_outputs = st.columns([1, 2])

    with col_inputs:
        st.subheader("Inputs")
        S = st.slider("Spot price (S)", 10.0, 500.0, 150.0, step=1.0)
        K = st.slider("Strike (K)", 10.0, 500.0, 155.0, step=1.0)
        days = st.slider("Days to expiry", 1, 365, 30)
        r = st.slider("Risk-free rate (%)", 0.0, 15.0, 5.0, step=0.25) / 100
        sigma = st.slider("Implied Volatility (%)", 1.0, 150.0, 25.0, step=1.0) / 100
        option_type = st.radio("Option type", ["call", "put"], horizontal=True)
        T = days / 365.0

    price = black_scholes_price(S, K, T, r, sigma, option_type)
    greeks = calculate_greeks(S, K, T, r, sigma, option_type)

    with col_outputs:
        st.subheader("Results")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Theoretical price", f"${price:.2f}")
        m2.metric("Delta", f"{greeks.delta:.3f}")
        m3.metric("Gamma", f"{greeks.gamma:.4f}")
        m4.metric("Vega (per 1% vol)", f"{greeks.vega:.3f}")
        m5.metric("Theta (daily)", f"{greeks.theta:.3f}")

        spot_range = np.linspace(max(1, S * 0.5), S * 1.5, 80)
        prices = [black_scholes_price(s, K, T, r, sigma, option_type) for s in spot_range]
        deltas = [calculate_greeks(s, K, T, r, sigma, option_type).delta for s in spot_range]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=prices, name="Option price", yaxis="y1"))
        fig.add_trace(go.Scatter(x=spot_range, y=deltas, name="Delta", yaxis="y2", line=dict(dash="dot")))
        fig.add_vline(x=S, line_dash="dash", line_color="gray", annotation_text="Current spot")
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Delta", overlaying="y", side="right", range=[-1, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Implied Volatility Solver")
    market_price_input = st.number_input(
        "Enter a market price to solve for IV", min_value=0.01, value=round(price, 2), step=0.01
    )
    try:
        iv = implied_volatility(market_price_input, S, K, T, r, option_type)
        st.success(f"Implied volatility: **{iv * 100:.2f}%**")
    except ValueError as e:
        st.error(str(e))


# ---------------------------------------------------------------------------
# TAB 2: Technical Indicators (synthetic data demo)
# ---------------------------------------------------------------------------
with tab_technical:
    st.subheader("Technical indicators (synthetic price series)")
    st.caption("Demo series -- real Alpaca bars are used live in the Agent Terminal page.")

    n_bars = st.slider("Number of bars", 60, 300, 120)
    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0.05, 1.2, n_bars))
    highs = closes + np.abs(rng.normal(0.6, 0.3, n_bars))
    lows = closes - np.abs(rng.normal(0.6, 0.3, n_bars))
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_bars, freq="D")

    rsi = calculate_rsi(closes, period=14)
    atr = calculate_atr(highs, lows, closes, period=14)
    mid, upper, lower = calculate_bollinger_bands(closes, period=20, num_std=2.0)

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=dates, y=closes, name="Close", line=dict(color="#2a78d6")))
    fig_price.add_trace(go.Scatter(x=dates, y=upper, name="Bollinger upper", line=dict(color="gray", dash="dot")))
    fig_price.add_trace(go.Scatter(x=dates, y=lower, name="Bollinger lower", line=dict(color="gray", dash="dot")))
    fig_price.add_trace(go.Scatter(x=dates, y=mid, name="SMA(20)", line=dict(color="#eb6834")))
    fig_price.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Price + Bollinger Bands")
    st.plotly_chart(fig_price, use_container_width=True)

    col_rsi, col_atr = st.columns(2)
    with col_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=dates, y=rsi, line=dict(color="#eb6834")))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
        fig_rsi.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), title="RSI(14)", yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col_atr:
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(x=dates, y=atr, line=dict(color="#6250d6")))
        fig_atr.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), title="ATR(14)")
        st.plotly_chart(fig_atr, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3: Kelly & VaR
# ---------------------------------------------------------------------------
with tab_risk:
    col_kelly, col_var = st.columns(2)

    with col_kelly:
        st.subheader("Kelly criterion")
        win_prob = st.slider("Win probability", 0.0, 1.0, 0.6, step=0.01)
        win_loss_ratio = st.slider("Win / loss ratio (R)", 0.1, 5.0, 2.0, step=0.1)
        fraction = st.slider("Fractional Kelly multiplier", 0.1, 1.0, 0.5, step=0.1)

        f_star = kelly_criterion(win_prob, win_loss_ratio, fraction)
        st.metric("Suggested position size", f"{f_star * 100:.1f}%")
        if f_star < 0:
            st.warning("Negative Kelly: there is no statistical edge under these conditions; trading is not recommended.")

    with col_var:
        st.subheader("Value at Risk")
        portfolio_value = st.number_input("Portfolio value ($)", min_value=1000, value=100_000, step=1000)
        mu = st.slider("Expected daily return (%)", -1.0, 1.0, 0.05, step=0.01) / 100
        vol = st.slider("Daily volatility (%)", 0.1, 10.0, 2.0, step=0.1) / 100
        confidence = st.select_slider("Confidence level", options=[0.90, 0.95, 0.99], value=0.95)
        horizon = st.slider("Horizon (days)", 1, 20, 1)

        var = parametric_var(portfolio_value, mu, vol, confidence, horizon)
        st.metric(f"{confidence*100:.0f}% confidence, {horizon}-day VaR", f"${var:,.2f}")

st.divider()
st.caption("core/math_engine.py — Black-Scholes, Greeks, RSI, ATR, Bollinger, Kelly, VaR — 35/35 tests passed.")