"""
tests/test_graph.py
----------------------
Tests the LangGraph wiring in orchestration/graph.py: routing logic, the
bounded Quant<->Risk debate loop, the HOLD shortcut, the Day-4 reflection
step, and the fail-safe behavior when a node raises. All agent functions
(fetch_market_state, generate_thesis, evaluate, execute, generate_reflection)
and core.memory.TradeMemory are monkeypatched with counting/stub
implementations -- no real LLM calls, no real Alpaca/MCP calls, no network,
no real trade_memory.json file touched.

These tests exercise orchestration.graph.run_pipeline() end to end at the
*graph* level, which is the right level to catch routing bugs: e.g. an
off-by-one in the debate-round cap, or a path that could reach
execution_node despite an unapproved verdict, or reflection_node running
(and touching real disk) on a HOLD/rejected path where nothing was executed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orchestration.graph as graph
from core.schemas import ExecutionResult, MarketState, OptionContract, QuantThesis, RiskVerdict


def _market_state(symbol="AAPL") -> MarketState:
    return MarketState(symbol=symbol, latest_price=150.0, closes=[150.0] * 20)


def _contract() -> OptionContract:
    return OptionContract(
        contract_symbol="AAPL260116C00150000",
        underlying_symbol="AAPL",
        strike=150.0,
        expiration="2026-01-16",
        option_type="call",
        mid_price=5.0,
    )


def _thesis(action="BUY_CALL") -> QuantThesis:
    selected = None if action == "HOLD" else _contract()
    return QuantThesis(
        symbol="AAPL",
        bias="BULLISH" if action == "BUY_CALL" else ("BEARISH" if action == "BUY_PUT" else "NEUTRAL"),
        confidence_score=0.7,
        recommended_action=action,
        thesis="stub thesis",
        key_metrics={},
        selected_contract=selected,
    )


def _verdict(approved: bool) -> RiskVerdict:
    return RiskVerdict(
        symbol="AAPL",
        is_approved=approved,
        counter_thesis="stub counter-thesis",
        win_probability=0.6,
        win_loss_ratio=2.0,
        veto_reason=None if approved else "stub veto",
    )


class _CallCounter:
    def __init__(self):
        self.count = 0
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.count += 1
        self.calls.append((args, kwargs))


class _FakeTradeMemory:
    """Stub for core.memory.TradeMemory -- never touches real disk."""

    saved: list = []

    def get_lessons_as_context(self, symbol, limit=3):
        return None

    def save_trade_reflection(self, trade_data, reflection_lesson):
        _FakeTradeMemory.saved.append((trade_data, reflection_lesson))


@pytest.fixture(autouse=True)
def _patch_data_agent_and_memory(monkeypatch):
    """Every test gets a working data_node and a disk-free memory store for free."""
    monkeypatch.setattr(graph, "fetch_market_state", lambda symbol, **kw: _market_state(symbol))
    monkeypatch.setattr(graph, "memory", type("M", (), {"TradeMemory": _FakeTradeMemory}))
    _FakeTradeMemory.saved = []


class TestHoldShortcut:
    def test_hold_thesis_skips_risk_execution_and_reflection_entirely(self, monkeypatch):
        risk_calls = _CallCounter()
        exec_calls = _CallCounter()
        reflect_calls = _CallCounter()
        monkeypatch.setattr(graph, "generate_thesis", lambda *a, **kw: _thesis("HOLD"))
        monkeypatch.setattr(graph, "evaluate", risk_calls)
        monkeypatch.setattr(graph, "execute", exec_calls)
        monkeypatch.setattr(graph, "generate_reflection", reflect_calls)

        state = graph.run_pipeline("AAPL")

        assert state["status"] == "no_trade"
        assert risk_calls.count == 0
        assert exec_calls.count == 0
        assert reflect_calls.count == 0
        assert state.get("execution_result") is None
        assert _FakeTradeMemory.saved == []


class TestApprovedPath:
    def test_approved_on_first_pass_reaches_execution_and_reflection(self, monkeypatch):
        quant_calls = _CallCounter()

        def fake_generate_thesis(market_state, counter_thesis=None, lessons_context=None):
            quant_calls.count += 1
            return _thesis("BUY_CALL")

        def fake_execute(thesis, verdict):
            return ExecutionResult(
                symbol="AAPL", instrument="option", contract_symbol="AAPL260116C00150000",
                order_submitted=True, order_id="abc", qty=1, side="buy",
            )

        def fake_reflect(market_state, thesis, verdict, execution_result):
            return "Lesson: the setup worked as expected."

        monkeypatch.setattr(graph, "generate_thesis", fake_generate_thesis)
        monkeypatch.setattr(graph, "evaluate", lambda *a, **kw: _verdict(approved=True))
        monkeypatch.setattr(graph, "execute", fake_execute)
        monkeypatch.setattr(graph, "generate_reflection", fake_reflect)

        state = graph.run_pipeline("AAPL")

        assert state["status"] == "approved"
        assert quant_calls.count == 1
        assert state["debate_round"] == 1
        assert state["execution_result"].order_submitted is True
        assert state["reflection"] == "Lesson: the setup worked as expected."
        assert len(_FakeTradeMemory.saved) == 1
        assert _FakeTradeMemory.saved[0][0]["symbol"] == "AAPL"

    def test_reflection_failure_is_best_effort_and_never_changes_status(self, monkeypatch):
        def fake_execute(thesis, verdict):
            return ExecutionResult(
                symbol="AAPL", instrument="option", contract_symbol="AAPL260116C00150000",
                order_submitted=True, order_id="abc", qty=1, side="buy",
            )

        def fake_reflect_boom(*a, **kw):
            raise RuntimeError("reflection LLM timed out")

        monkeypatch.setattr(graph, "generate_thesis", lambda *a, **kw: _thesis("BUY_CALL"))
        monkeypatch.setattr(graph, "evaluate", lambda *a, **kw: _verdict(approved=True))
        monkeypatch.setattr(graph, "execute", fake_execute)
        monkeypatch.setattr(graph, "generate_reflection", fake_reflect_boom)

        state = graph.run_pipeline("AAPL")

        assert state["status"] == "approved"  # the trade itself is unaffected
        assert "reflection failed" in state.get("reflection_error", "")
        assert state.get("reflection") is None


class TestDebateLoop:
    def test_rejected_trade_loops_back_to_quant_then_stops_at_cap(self, monkeypatch):
        """Risk always vetoes -> the loop must run exactly max_debate_rounds
        times and then end with status='rejected', never reaching execution
        or reflection."""
        quant_calls = _CallCounter()
        risk_calls = _CallCounter()
        exec_calls = _CallCounter()
        reflect_calls = _CallCounter()

        def fake_generate_thesis(market_state, counter_thesis=None, lessons_context=None):
            quant_calls.count += 1
            return _thesis("BUY_CALL")

        def fake_evaluate(market_state, thesis):
            risk_calls.count += 1
            return _verdict(approved=False)

        monkeypatch.setattr(graph, "generate_thesis", fake_generate_thesis)
        monkeypatch.setattr(graph, "evaluate", fake_evaluate)
        monkeypatch.setattr(graph, "execute", exec_calls)
        monkeypatch.setattr(graph, "generate_reflection", reflect_calls)

        state = graph.run_pipeline("AAPL", max_debate_rounds=2)

        assert state["status"] == "rejected"
        assert state["debate_round"] == 2  # exactly the cap, never exceeded
        assert risk_calls.count == 2
        assert quant_calls.count == 2  # initial pass + one retry
        assert exec_calls.count == 0
        assert reflect_calls.count == 0
        assert state.get("execution_result") is None

    def test_quant_changes_mind_after_veto_and_gets_approved(self, monkeypatch):
        """First pass gets vetoed; on reconsideration (counter_thesis passed
        in) the Quant Agent should be seen receiving that counter_thesis,
        and a subsequent approval should reach execution."""
        received_counter_theses = []
        risk_call_count = _CallCounter()

        def fake_generate_thesis(market_state, counter_thesis=None, lessons_context=None):
            received_counter_theses.append(counter_thesis)
            return _thesis("BUY_CALL")

        def fake_evaluate(market_state, thesis):
            risk_call_count.count += 1
            # Vetoed on the first evaluation, approved on the second.
            return _verdict(approved=risk_call_count.count >= 2)

        def fake_execute(thesis, verdict):
            return ExecutionResult(
                symbol="AAPL", instrument="option", contract_symbol="AAPL260116C00150000",
                order_submitted=True, order_id="xyz", qty=2, side="buy",
            )

        monkeypatch.setattr(graph, "generate_thesis", fake_generate_thesis)
        monkeypatch.setattr(graph, "evaluate", fake_evaluate)
        monkeypatch.setattr(graph, "execute", fake_execute)
        monkeypatch.setattr(graph, "generate_reflection", lambda *a, **kw: "stub lesson")

        state = graph.run_pipeline("AAPL", max_debate_rounds=3)

        assert state["status"] == "approved"
        assert received_counter_theses[0] is None  # first pass sees no prior objection
        assert received_counter_theses[1] == "stub counter-thesis"  # retry sees the veto reason
        assert state["debate_round"] == 2


class TestFailSafeOnError:
    def test_data_agent_failure_never_reaches_quant_or_risk(self, monkeypatch):
        quant_calls = _CallCounter()
        risk_calls = _CallCounter()
        exec_calls = _CallCounter()

        def boom(symbol, **kw):
            raise RuntimeError("Alpaca MCP call failed")

        monkeypatch.setattr(graph, "fetch_market_state", boom)
        monkeypatch.setattr(graph, "generate_thesis", quant_calls)
        monkeypatch.setattr(graph, "evaluate", risk_calls)
        monkeypatch.setattr(graph, "execute", exec_calls)

        state = graph.run_pipeline("AAPL")

        assert state["status"] == "error"
        assert "data_agent failed" in state["error"]
        assert quant_calls.count == 0
        assert risk_calls.count == 0
        assert exec_calls.count == 0

    def test_quant_agent_failure_never_reaches_risk_or_execution(self, monkeypatch):
        risk_calls = _CallCounter()
        exec_calls = _CallCounter()

        def boom(*a, **kw):
            raise ValueError("malformed JSON from the LLM")

        monkeypatch.setattr(graph, "generate_thesis", boom)
        monkeypatch.setattr(graph, "evaluate", risk_calls)
        monkeypatch.setattr(graph, "execute", exec_calls)

        state = graph.run_pipeline("AAPL")

        assert state["status"] == "error"
        assert "quant_agent failed" in state["error"]
        assert risk_calls.count == 0
        assert exec_calls.count == 0

    def test_risk_agent_failure_never_reaches_execution(self, monkeypatch):
        exec_calls = _CallCounter()

        def boom(*a, **kw):
            raise RuntimeError("risk LLM timed out")

        monkeypatch.setattr(graph, "generate_thesis", lambda *a, **kw: _thesis("BUY_CALL"))
        monkeypatch.setattr(graph, "evaluate", boom)
        monkeypatch.setattr(graph, "execute", exec_calls)

        state = graph.run_pipeline("AAPL")

        assert state["status"] == "error"
        assert "risk_agent failed" in state["error"]
        assert exec_calls.count == 0


class TestStreamPipeline:
    def test_stream_yields_one_chunk_per_node(self, monkeypatch):
        monkeypatch.setattr(graph, "generate_thesis", lambda *a, **kw: _thesis("HOLD"))
        monkeypatch.setattr(graph, "evaluate", lambda *a, **kw: (_ for _ in ()).throw(AssertionError))
        monkeypatch.setattr(graph, "execute", lambda *a, **kw: (_ for _ in ()).throw(AssertionError))

        chunks = list(graph.stream_pipeline("AAPL"))
        node_names = [list(chunk.keys())[0] for chunk in chunks]

        assert "data" in node_names
        assert "quant" in node_names
        assert "finalize" in node_names
        assert "risk" not in node_names  # HOLD shortcut must skip it
        assert "reflection" not in node_names  # HOLD shortcut must skip it too
