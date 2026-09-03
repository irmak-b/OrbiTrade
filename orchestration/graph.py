"""
orchestration/graph.py
-------------------------
LangGraph StateGraph wiring the OrbiTrade agents into a single, autonomous,
self-improving pipeline:

    Data -> Quant (reads past lessons) -> [HOLD? -> Finalize]
                                              |
                                     Risk (Devil's Advocate)
                                              |
                            approved? ----+---- not approved?
                               |                    |
                           Execution      debate_round < max? -> back to Quant
                               |                    |
                          Reflection               Finalize (rejected)
                               |
                           Finalize

Design goal (per project spec): the compiled graph is exposed through two
plain, isolated functions -- run_pipeline() and stream_pipeline() -- that
take a single input (a symbol) and return/yield state. Neither function
depends on anything beyond LangGraph itself, so this module can be called
with one line from a terminal script (cli.py), a Streamlit app, or an API
endpoint.

Safety design: every node is wrapped so that an exception anywhere in the
pipeline (a malformed LLM response, a network failure, an unexpected
schema mismatch) is caught, recorded in state["error"], and routed
straight to finalize with status="error". No exception can skip past a
node and accidentally reach execution_node with incomplete data -- an
error always means "no trade", never "trade with default values". This
mirrors execution_agent.py's own safety ceiling: two independent layers
both have to agree before an order is ever placed.

The Quant <-> Risk debate loop is bounded by max_debate_rounds (counted as
the number of Risk Agent evaluations, not Quant Agent calls) so a
disagreement between the two agents can never loop forever.

Day 4 (Reflection): reflection_node runs only after a real order attempt
(execution_node), turning that decision cycle into one durable lesson via
agents/quant_agent.py::generate_reflection, persisted through
core/memory.py. quant_node reads those lessons back in on every call
(including the very first one for a symbol) so each run of the pipeline is
informed by the desk's own trading history. Reflection is strictly
best-effort: a failure there is recorded in state["reflection_error"] but
never changes `status` or rolls back the trade that already happened.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, StateGraph

from agents.data_agent import fetch_market_state
from agents.execution_agent import execute
from agents.quant_agent import generate_reflection, generate_thesis
from agents.risk_agent import evaluate
from core import memory
from core.state import AgentState, initial_state

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def data_node(state: AgentState) -> dict:
    try:
        market_state = fetch_market_state(state["symbol"])
        return {"market_state": market_state}
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure here means "no trade"
        import traceback
        traceback.print_exc()  # TEMP DEBUG: prints exact file/line to the console
        return {"status": "error", "error": f"data_agent failed: {exc}"}


def quant_node(state: AgentState) -> dict:
    try:
        prior_verdict = state.get("verdict")
        counter_thesis = prior_verdict.counter_thesis if prior_verdict else None

        market_state = state["market_state"]
        lessons = memory.TradeMemory().get_lessons_as_context(state["symbol"])
        if lessons and market_state.lessons != lessons:
            market_state = market_state.model_copy(update={"lessons": lessons})

        thesis = generate_thesis(market_state, counter_thesis=counter_thesis, lessons_context=lessons)
        return {"thesis": thesis, "market_state": market_state}
    except Exception as exc:
        return {"status": "error", "error": f"quant_agent failed: {exc}"}


def risk_node(state: AgentState) -> dict:
    try:
        verdict = evaluate(state["market_state"], state["thesis"])
        return {"verdict": verdict, "debate_round": state.get("debate_round", 0) + 1}
    except Exception as exc:
        return {"status": "error", "error": f"risk_agent failed: {exc}"}


def execution_node(state: AgentState) -> dict:
    try:
        result = execute(state["thesis"], state["verdict"])
        status = "approved" if result.order_submitted else "rejected"
        return {"execution_result": result, "status": status}
    except Exception as exc:
        return {"status": "error", "error": f"execution_agent failed: {exc}"}


def reflection_node(state: AgentState) -> dict:
    """Day-4 self-reflection: only runs after execution_node has attempted
    an order (approved or not). Best-effort -- never raises past this node."""
    execution_result = state.get("execution_result")
    if execution_result is None:
        return {}

    try:
        lesson = generate_reflection(
            state["market_state"], state["thesis"], state["verdict"], execution_result
        )
        memory.TradeMemory().save_trade_reflection(
            {
                "symbol": state["symbol"],
                "strategy": state["thesis"].recommended_action if state.get("thesis") else None,
                "contract_symbol": execution_result.contract_symbol,
                "pnl": None,  # realized PnL is only known once the position is later closed
            },
            lesson,
        )
        return {"reflection": lesson}
    except Exception as exc:  # noqa: BLE001 - reflection failing must never undo the trade above
        return {"reflection_error": f"reflection failed: {exc}"}


def finalize_node(state: AgentState) -> dict:
    """Assigns a terminal status for paths that reach END without ever
    hitting execution_node (HOLD thesis, or max debate rounds exhausted)."""
    if state.get("status") == "error":
        return {}
    if state.get("execution_result") is not None:
        return {}  # execution_node already set the final status
    if state.get("thesis") is not None and state["thesis"].recommended_action == "HOLD":
        return {"status": "no_trade"}
    if state.get("verdict") is not None and not state["verdict"].is_approved:
        return {"status": "rejected"}
    return {"status": "error", "error": "Pipeline reached END in an unexpected state."}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def route_after_data(state: AgentState) -> Literal["quant", "end"]:
    if state.get("status") == "error":
        return "end"
    return "quant"


def route_after_quant(state: AgentState) -> Literal["risk", "end"]:
    if state.get("status") == "error":
        return "end"
    if state["thesis"].recommended_action == "HOLD":
        return "end"
    return "risk"


def route_after_risk(state: AgentState) -> Literal["execution", "quant", "end"]:
    if state.get("status") == "error":
        return "end"
    if state["verdict"].is_approved:
        return "execution"
    if state["debate_round"] < state["max_debate_rounds"]:
        return "quant"
    return "end"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("data", data_node)
    workflow.add_node("quant", quant_node)
    workflow.add_node("risk", risk_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("data")

    workflow.add_conditional_edges("data", route_after_data, {"quant": "quant", "end": "finalize"})
    workflow.add_conditional_edges("quant", route_after_quant, {"risk": "risk", "end": "finalize"})
    workflow.add_conditional_edges(
        "risk", route_after_risk, {"execution": "execution", "quant": "quant", "end": "finalize"}
    )
    workflow.add_edge("execution", "reflection")
    workflow.add_edge("reflection", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


app = build_graph()


# ---------------------------------------------------------------------------
# Isolated entry points -- call these from anywhere with one line
# ---------------------------------------------------------------------------


def run_pipeline(symbol: str, max_debate_rounds: int = 2) -> AgentState:
    """
    Runs the full OrbiTrade pipeline for one symbol and returns the final
    state. This is the single function a CLI command, a Streamlit button,
    or an API endpoint should call -- it has no other dependency.

        state = run_pipeline("AAPL")
        print(state["status"], state.get("execution_result"))
    """
    return app.invoke(initial_state(symbol, max_debate_rounds))


def stream_pipeline(symbol: str, max_debate_rounds: int = 2) -> Iterator[dict]:
    """
    Same pipeline, but yields the state delta after every node completes,
    so a caller can show live progress (a Streamlit spinner per stage, a
    streaming API response, a verbose CLI trace) instead of waiting for the
    whole run to finish.

        for step in stream_pipeline("AAPL"):
            print(step)
    """
    yield from app.stream(initial_state(symbol, max_debate_rounds))


if __name__ == "__main__":
    import json
    import sys as _sys

    symbol = _sys.argv[1] if len(_sys.argv) > 1 else "AAPL"
    final_state = run_pipeline(symbol)
    print(
        json.dumps(
            {
                "symbol": final_state["symbol"],
                "status": final_state["status"],
                "error": final_state.get("error"),
                "debate_round": final_state.get("debate_round"),
                "thesis": final_state["thesis"].model_dump() if final_state.get("thesis") else None,
                "verdict": final_state["verdict"].model_dump() if final_state.get("verdict") else None,
                "execution_result": (
                    final_state["execution_result"].model_dump()
                    if final_state.get("execution_result")
                    else None
                ),
                "reflection": final_state.get("reflection"),
                "reflection_error": final_state.get("reflection_error"),
            },
            indent=2,
            default=str,
        )
    )
