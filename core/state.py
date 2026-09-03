"""
core/state.py
----------------
Shared LangGraph state for the OrbiTrade pipeline. Every node in
orchestration/graph.py reads and writes this single TypedDict -- it's the
multi-agent "memory" the Quant <-> Risk debate loop runs on top of.

`status` is the authoritative summary field a caller (CLI, Streamlit, an
API endpoint) should read first:
    "pending"   -> still running (only seen mid-stream, never on the final state)
    "no_trade"  -> Quant Agent recommended HOLD; pipeline ended cleanly, no debate needed
    "approved"  -> Risk Agent approved the trade; execution_result is set
    "rejected"  -> Risk Agent vetoed the trade after all debate rounds were used
    "error"     -> an exception occurred somewhere in the pipeline; no order was placed

Day 4 adds `reflection` (the one-sentence lesson generated after
execution_node, see agents/quant_agent.py::generate_reflection) and
`reflection_error` (set instead, if reflection itself failed -- this is
always best-effort and never changes `status`, since a failed reflection
must never retroactively invalidate a trade that already happened).
"""

from __future__ import annotations

from typing import Optional, TypedDict

from core.schemas import ExecutionResult, MarketState, QuantThesis, RiskVerdict


class AgentState(TypedDict, total=False):
    symbol: str
    market_state: Optional[MarketState]
    thesis: Optional[QuantThesis]
    verdict: Optional[RiskVerdict]
    execution_result: Optional[ExecutionResult]
    debate_round: int
    max_debate_rounds: int
    status: str
    error: Optional[str]
    reflection: Optional[str]
    reflection_error: Optional[str]


def initial_state(symbol: str, max_debate_rounds: int = 2) -> AgentState:
    """Builds a fresh AgentState for a new pipeline run."""
    return AgentState(
        symbol=symbol,
        market_state=None,
        thesis=None,
        verdict=None,
        execution_result=None,
        debate_round=0,
        max_debate_rounds=max_debate_rounds,
        status="pending",
        error=None,
        reflection=None,
        reflection_error=None,
    )
