"""
dashboard/app.py
-----------------
OrbiTrade V3 - Live Agent Terminal (ana sayfa).
Bu artık multi-page Streamlit app'in giriş noktası. İkinci sayfa
(Math Engine Sandbox) dashboard/pages/1_Math_Sandbox.py içinde --
Streamlit onu otomatik olarak sol menüye ekler, tek deployment /
tek URL altında iki sayfa olur.

Run with:
    streamlit run dashboard/app.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- Streamlit Cloud secrets -> os.environ köprüsü -------------------------
# mcp_client.py / llm_factory.py os.getenv() ile okuyor; Streamlit Cloud'da
# .env dosyası olmadığı için secrets'ı burada enjekte etmemiz gerekiyor.
try:
    for _key, _val in st.secrets.items():
        os.environ.setdefault(_key, str(_val))
except Exception:
    pass  # yerelde .env zaten mcp_client.py tarafından yükleniyor

from orchestration.graph import run_pipeline

st.set_page_config(page_title="OrbiTrade V3 Terminal", layout="wide", page_icon="🌌")

st.title("Autonomous Options Terminal")
st.caption("Multi-Agent LLM Pipeline + Deterministic Math Engine (Alpaca MCP Integration)")

# ---------------------------------------------------------------------------
# LIVE AGENT PIPELINE
# ---------------------------------------------------------------------------
col_ctrl, col_status = st.columns([1, 3])

with col_ctrl:
    st.subheader("Mission Control")
    symbol = st.text_input("Target Ticker (e.g., AAPL, SPY)", value="AAPL").upper()
    debate_rounds = st.slider("Max Debate Rounds (Quant vs Risk)", 1, 3, 2)

    execute_btn = st.button("Initialize Sequence", type="primary", use_container_width=True)

if execute_btn:
    with st.spinner(f"Executing OrbiTrade multi-agent pipeline for {symbol}..."):
        try:
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
                    try:
                        from core.memory import TradeMemory
                        db = TradeMemory()
                        lessons = [m["lesson"] for m in db.get_all_memories() if m.get("symbol") == symbol]
                        if lessons:
                            st.success(f"**Latest Lesson Stored:** {lessons[-1]}")
                        else:
                            st.write("No reflection data found for this symbol.")
                    except Exception:
                        st.write("Memory module unavailable.")

        except Exception as e:
            st.error(f"Critical Application Failure: {e}")
