"""
dashboard/app.py
-----------------
OrbiTrade v3 - Streamlit Command Terminal.
This is the main interactive interface for the hackathon demo. 

Tab 1: Live Agent Terminal - Runs the full options trading pipeline 
(Data -> Quant -> Risk -> Execution -> Reflection) against the Alpaca MCP.
Tab 2: Math Engine Sandbox - Interactive visualizer for the deterministic
Black-Scholes, Greeks, and Kelly sizing engine to prove the math layer.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestration.graph import run_pipeline
from core.math_engine import (
    black_scholes_price,
    calculate_greeks,
    implied_volatility,
    kelly_criterion,
)

st.set_page_config(page_title="OrbiTrade V3 Terminal", layout="wide", page_icon="🌌")

st.title("Autonomous Options Terminal")
st.caption("Multi-Agent LLM Pipeline + Deterministic Math Engine (Alpaca MCP Integration)")

tab_live, tab_sandbox = st.tabs(["🔴 Live Agent Pipeline", "🧮 Math Engine Sandbox"])

# ---------------------------------------------------------------------------
# TAB 1: LIVE AGENT PIPELINE
# ---------------------------------------------------------------------------
with tab_live:
    col_ctrl, col_status = st.columns([1, 3])
    
    with col_ctrl:
        st.subheader("Mission Control")
        symbol = st.text_input("Target Ticker (e.g., AAPL, SPY)", value="AAPL").upper()
        debate_rounds = st.slider("Max Debate Rounds (Quant vs Risk)", 1, 3, 2)
        
        execute_btn = st.button("Initialize Sequence", type="primary", use_container_width=True)

    if execute_btn:
        with st.spinner(f"Executing OrbiTrade multi-agent pipeline for {symbol}..."):
            try:
                # Trigger the LangGraph pipeline
                final_state = run_pipeline(symbol, max_debate_rounds=debate_rounds)
                status = final_state.get("status", "error")
                
                with col_status:
                    if status == "error":
                        st.error(f"Pipeline Error: {final_state.get('error', 'Unknown failure')}")
                    else:
                        st.success(f"Pipeline Completed. Final Status: **{status.upper()}**")
                        
                if status != "error":
                    # --- Section 1: Market Data & Options Chain ---
                    mkt_state = final_state.get("market_state")
                    if mkt_state:
                        with st.expander("📊 Data Agent: Market State & Priced Option Chain", expanded=False):
                            st.metric("Spot Price", f"${mkt_state.latest_price:.2f}")
                            if mkt_state.option_chain:
                                df_chain = pd.DataFrame([c.model_dump() for c in mkt_state.option_chain])
                                st.dataframe(df_chain, use_container_width=True)
                            else:
                                st.warning("No valid option contracts found or priced.")
                    
                    # --- Section 2: Agent Debate ---
                    col_q, col_r = st.columns(2)
                    with col_q:
                        st.markdown("### 🧠 Quant Agent (Qwen)")
                        thesis = final_state.get("thesis")
                        if thesis:
                            if thesis.selected_contract:
                                st.info(
                                    f"**Action:** {thesis.recommended_action} @ Strike "
                                    f"{thesis.selected_contract.strike} ({thesis.selected_contract.contract_symbol})"
                                )
                            else:
                                st.info(f"**Action:** {thesis.recommended_action} (no contract selected)")
                            st.write(f"**Bias:** {thesis.bias} (Confidence: {thesis.confidence_score:.2f})")
                            st.write(f"**Thesis:** {thesis.thesis}")
                        else:
                            st.warning("No thesis generated.")
                            
                    with col_r:
                        st.markdown("### ⚖️ Risk Agent (DeepSeek)")
                        verdict = final_state.get("verdict")
                        if verdict:
                            color = "green" if verdict.is_approved else "red"
                            st.markdown(f"**Verdict:** :{color}[{'APPROVED' if verdict.is_approved else 'VETOED'}]")
                            st.write(f"**Win Prob:** {verdict.win_probability:.2f} | **W/L Ratio:** {verdict.win_loss_ratio:.2f}")
                            if not verdict.is_approved:
                                st.error(f"**Veto Reason:** {verdict.veto_reason}")
                            st.write(f"**Counter Thesis:** {verdict.counter_thesis}")
                            st.caption(f"Debate Rounds Used: {final_state.get('debate_round', 0)}")
                        else:
                            st.warning("No risk verdict generated.")
                            
                    # --- Section 3: Execution & Reflection ---
                    st.divider()
                    col_exec, col_ref = st.columns(2)
                    
                    with col_exec:
                        st.markdown("### ⚡ Execution Agent")
                        exec_res = final_state.get("execution_result")
                        if exec_res:
                            if exec_res.order_submitted:
                                st.success(f"Order Placed! ID: {exec_res.order_id}")
                                st.write(f"**Action:** {exec_res.side.upper()} {exec_res.qty} contracts")
                                st.write(f"**Kelly Fraction Applied:** {exec_res.kelly_fraction_used:.3f}")
                            else:
                                st.warning(f"Order Skipped: {exec_res.rejection_reason}")
                        else:
                            st.info("Execution node bypassed.")
                            
                    with col_ref:
                        st.markdown("### 🪞 Self-Reflection (Memory)")
                        # In V3, the system saves the outcome to memory. 
                        # We can read the latest lesson from the DB for display.
                        try:
                            from core.memory import TradeMemory
                            db = TradeMemory()
                            lessons = [m["lesson"] for m in db.get_all_memories() if m.get("symbol") == symbol]
                            if lessons:
                                st.success(f"**Latest Lesson Stored:** {lessons[-1]}")
                            else:
                                st.write("No reflection data found for this symbol.")
                        except Exception as e:
                            st.write("Memory module unavailable.")

            except Exception as e:
                st.error(f"Critical Application Failure: {e}")

# ---------------------------------------------------------------------------
# TAB 2: MATH ENGINE SANDBOX (Deterministic Proof)
# ---------------------------------------------------------------------------
with tab_sandbox:
    st.markdown("### Deterministic Option Pricing (Black-Scholes & Greeks)")
    st.caption("Used by the Data Agent to prevent LLM hallucination of derivatives pricing.")
    
    col_inputs, col_outputs = st.columns([1, 2])

    with col_inputs:
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
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Theoretical Price", f"${price:.2f}")
        m2.metric("Delta", f"{greeks.delta:.3f}")
        m3.metric("Gamma", f"{greeks.gamma:.4f}")
        m4.metric("Vega (per 1%)", f"{greeks.vega:.3f}")
        m5.metric("Theta (daily)", f"{greeks.theta:.3f}")

        # Price/delta curve across a range of spot prices
        spot_range = np.linspace(max(1, S * 0.5), S * 1.5, 80)
        prices = [black_scholes_price(s, K, T, r, sigma, option_type) for s in spot_range]
        deltas = [calculate_greeks(s, K, T, r, sigma, option_type).delta for s in spot_range]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=prices, name="Option price", yaxis="y1"))
        fig.add_trace(go.Scatter(x=spot_range, y=deltas, name="Delta", yaxis="y2", line=dict(dash="dot")))
        fig.add_vline(x=S, line_dash="dash", line_color="gray", annotation_text="Current spot")
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Delta", overlaying="y", side="right", range=[-1, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)