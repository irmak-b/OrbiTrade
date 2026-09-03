"""
tests/test_agents_logic.py
-----------------------------
Tests for the deterministic, network-free logic inside the agents:

    - agents/data_agent.py   -> build_market_state() (pure function over
      synthetic bars, no Alpaca call)
    - agents/risk_agent.py   -> check_greek_risk() (pure threshold filter,
      now applied to a selected option contract's Greeks)
    - agents/execution_agent.py -> calculate_option_order_qty() (pure
      whole-contract sizing math) and execute() (with alpaca_tools
      monkeypatched, so no network/LLM calls are made)

The LLM-calling branches (quant_agent.generate_thesis / generate_reflection,
risk_agent.evaluate's debate path) require a live FEATHERLESS_API_KEY and
network access, so they are intentionally NOT exercised here -- only
import/syntax is checked elsewhere. This file locks down the deterministic
backbone the LLMs sit on top of.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.data_agent import build_market_state
from agents.execution_agent import calculate_option_order_qty, execute
from agents.risk_agent import check_greek_risk
from core.schemas import ExecutionResult, OptionContract, OptionGreeksSnapshot, QuantThesis, RiskVerdict


# ---------------------------------------------------------------------------
# data_agent.build_market_state
# ---------------------------------------------------------------------------

class TestBuildMarketState:
    def test_computes_all_technicals_with_enough_bars(self):
        rng = np.random.default_rng(1)
        closes = list(100 + np.cumsum(rng.normal(0, 1, 60)))
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]

        state = build_market_state("AAPL", closes, highs, lows, latest_price=closes[-1])

        assert state.symbol == "AAPL"
        assert state.technicals.rsi is not None
        assert 0 <= state.technicals.rsi <= 100
        assert state.technicals.atr is not None
        assert state.technicals.atr >= 0
        assert state.technicals.bollinger_mid is not None
        assert state.technicals.bollinger_lower <= state.technicals.bollinger_mid <= state.technicals.bollinger_upper
        assert state.option_chain == []  # build_market_state is pure -- chain must be pre-priced and passed in

    def test_insufficient_bars_leaves_technicals_none(self):
        closes = [100.0, 101.0, 102.0]
        state = build_market_state("AAPL", closes, closes, closes, latest_price=102.0)
        assert state.technicals.rsi is None
        assert state.technicals.atr is None
        assert state.technicals.bollinger_mid is None

    def test_news_items_are_parsed(self):
        closes = [100.0] * 20
        news = [{"headline": "AAPL beats earnings", "source": "Reuters", "published_at": "2026-08-01"}]
        state = build_market_state("AAPL", closes, closes, closes, latest_price=100.0, news=news)
        assert len(state.news) == 1
        assert state.news[0].headline == "AAPL beats earnings"

    def test_pre_priced_option_chain_passes_through_unchanged(self):
        contract = OptionContract(
            contract_symbol="AAPL260116C00150000",
            underlying_symbol="AAPL",
            strike=150.0,
            expiration="2026-01-16",
            option_type="call",
            mid_price=5.0,
        )
        closes = [100.0] * 20
        state = build_market_state("AAPL", closes, closes, closes, latest_price=100.0, option_chain=[contract])
        assert state.option_chain == [contract]


# ---------------------------------------------------------------------------
# risk_agent.check_greek_risk
# ---------------------------------------------------------------------------

class TestCheckGreekRisk:
    def test_none_greeks_returns_none(self):
        assert check_greek_risk(None) is None

    def test_within_thresholds_returns_none(self):
        greeks = OptionGreeksSnapshot(delta=0.5, gamma=0.02, vega=0.1, theta=-0.05)
        assert check_greek_risk(greeks) is None

    def test_excessive_vega_triggers_veto(self):
        greeks = OptionGreeksSnapshot(vega=0.9)
        reason = check_greek_risk(greeks, max_abs_vega=0.5)
        assert reason is not None
        assert "Vega" in reason

    def test_excessive_theta_triggers_veto(self):
        greeks = OptionGreeksSnapshot(theta=-0.8)
        reason = check_greek_risk(greeks, max_abs_theta=0.5)
        assert reason is not None
        assert "Theta" in reason


# ---------------------------------------------------------------------------
# execution_agent.calculate_option_order_qty
# ---------------------------------------------------------------------------

class TestCalculateOptionOrderQty:
    def test_basic_sizing_whole_contracts_only(self):
        # cash=10_000, kelly=0.2 (uncapped), contract_price=5.00 -> cost/contract = $500
        # dollar_amount = 2000 -> 4 whole contracts
        qty = calculate_option_order_qty(cash=10_000, contract_price=5.0, kelly_fraction=0.2, max_position_pct=0.5)
        assert qty == 4

    def test_kelly_capped_by_max_position_pct(self):
        # Kelly suggests 40%, but the hard cap is 10% -> only 10% of cash used
        qty_capped = calculate_option_order_qty(
            cash=10_000, contract_price=2.0, kelly_fraction=0.4, max_position_pct=0.10
        )
        qty_uncapped = calculate_option_order_qty(
            cash=10_000, contract_price=2.0, kelly_fraction=0.4, max_position_pct=1.0
        )
        # 10% of 10_000 = 1000 -> cost/contract = 200 -> 5 contracts
        # 40% of 10_000 = 4000 -> cost/contract = 200 -> 20 contracts
        assert qty_capped == 5
        assert qty_uncapped == 20
        assert qty_capped < qty_uncapped

    def test_negative_kelly_gives_zero_qty(self):
        qty = calculate_option_order_qty(cash=10_000, contract_price=5.0, kelly_fraction=-0.1, max_position_pct=0.5)
        assert qty == 0

    def test_zero_price_gives_zero_qty(self):
        qty = calculate_option_order_qty(cash=10_000, contract_price=0, kelly_fraction=0.2, max_position_pct=0.5)
        assert qty == 0

    def test_rounds_down_to_whole_contracts_never_up(self):
        # dollar_amount = 999, cost/contract = 500 -> 1.998 contracts -> must floor to 1, never 2
        qty = calculate_option_order_qty(cash=999, contract_price=5.0, kelly_fraction=1.0, max_position_pct=1.0)
        assert qty == 1


# ---------------------------------------------------------------------------
# execution_agent.execute (alpaca_tools monkeypatched -- no network/LLM)
# ---------------------------------------------------------------------------

def _contract(strike=150.0, mid_price=5.0, option_type="call", symbol="AAPL260116C00150000") -> OptionContract:
    return OptionContract(
        contract_symbol=symbol,
        underlying_symbol="AAPL",
        strike=strike,
        expiration="2026-01-16",
        option_type=option_type,
        mid_price=mid_price,
        greeks=OptionGreeksSnapshot(delta=0.5, gamma=0.02, vega=0.1, theta=-0.05),
    )


def _thesis(action="BUY_CALL", confidence=0.7, contract: OptionContract | None = None) -> QuantThesis:
    selected = None if action == "HOLD" else (contract or _contract())
    return QuantThesis(
        symbol="AAPL",
        bias="BULLISH" if action == "BUY_CALL" else ("BEARISH" if action == "BUY_PUT" else "NEUTRAL"),
        confidence_score=confidence,
        recommended_action=action,
        thesis="Synthetic thesis for testing.",
        key_metrics={"rsi": 55.0},
        selected_contract=selected,
    )


def _verdict(approved=True, win_prob=0.6, win_loss_ratio=2.0, veto_reason=None) -> RiskVerdict:
    return RiskVerdict(
        symbol="AAPL",
        is_approved=approved,
        counter_thesis="Synthetic counter-thesis for testing.",
        win_probability=win_prob,
        win_loss_ratio=win_loss_ratio,
        veto_reason=veto_reason,
    )


def _open_clock():
    return {"is_open": True, "next_open": None, "next_close": None}


class TestExecute:
    def test_unapproved_trade_is_rejected_without_touching_alpaca(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("alpaca_tools should not be called for an unapproved trade")

        monkeypatch.setattr("agents.execution_agent.alpaca_tools.get_clock", _boom)
        monkeypatch.setattr("agents.execution_agent.alpaca_tools.get_account", _boom)
        monkeypatch.setattr("agents.execution_agent.alpaca_tools.place_option_order", _boom)

        result = execute(_thesis(), _verdict(approved=False, veto_reason="Too risky"))

        assert isinstance(result, ExecutionResult)
        assert result.order_submitted is False
        assert result.rejection_reason == "Too risky"

    def test_hold_action_never_places_an_order(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("alpaca_tools should not be called for a HOLD thesis")

        monkeypatch.setattr("agents.execution_agent.alpaca_tools.get_clock", _boom)
        monkeypatch.setattr("agents.execution_agent.alpaca_tools.get_account", _boom)
        monkeypatch.setattr("agents.execution_agent.alpaca_tools.place_option_order", _boom)

        result = execute(_thesis(action="HOLD"), _verdict())
        assert result.order_submitted is False
        assert "HOLD" in result.rejection_reason

    def test_closed_market_blocks_order(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("no order should be attempted while the market is closed")

        monkeypatch.setattr(
            "agents.execution_agent.alpaca_tools.get_clock",
            lambda: {"is_open": False, "next_open": "2026-09-03T13:30:00Z", "next_close": None},
        )
        monkeypatch.setattr("agents.execution_agent.alpaca_tools.get_account", _boom)
        monkeypatch.setattr("agents.execution_agent.alpaca_tools.place_option_order", _boom)

        result = execute(_thesis(), _verdict())
        assert result.order_submitted is False
        assert "closed" in result.rejection_reason.lower()

    def test_clock_failure_is_fail_safe_not_a_crash(self, monkeypatch):
        def _clock_boom():
            raise RuntimeError("MCP session dropped")

        monkeypatch.setattr("agents.execution_agent.alpaca_tools.get_clock", _clock_boom)

        result = execute(_thesis(), _verdict())  # must not raise
        assert result.order_submitted is False
        assert "clock" in result.rejection_reason.lower()

    def test_negative_kelly_blocks_order(self, monkeypatch):
        monkeypatch.setattr("agents.execution_agent.alpaca_tools.get_clock", _open_clock)
        result = execute(_thesis(), _verdict(win_prob=0.2, win_loss_ratio=0.5))
        assert result.order_submitted is False
        assert "Kelly" in result.rejection_reason

    def test_approved_buy_places_order_within_position_cap(self, monkeypatch):
        monkeypatch.setattr("agents.execution_agent.alpaca_tools.get_clock", _open_clock)
        monkeypatch.setattr("agents.execution_agent.alpaca_tools.get_account", lambda: {"cash": 100_000})
        monkeypatch.setattr("agents.execution_agent.alpaca_tools.get_positions", lambda: [])
        placed_orders = []

        def fake_place_option_order(contract_symbol, qty, side="buy"):
            placed_orders.append((contract_symbol, qty, side))
            return {"order_id": "test-order-1", "contract_symbol": contract_symbol, "qty": qty, "side": side, "status": "accepted"}

        monkeypatch.setattr("agents.execution_agent.alpaca_tools.place_option_order", fake_place_option_order)

        result = execute(_thesis(), _verdict(win_prob=0.6, win_loss_ratio=2.0))

        assert result.order_submitted is True
        assert result.order_id == "test-order-1"
        assert result.instrument == "option"
        assert result.contract_symbol == "AAPL260116C00150000"
        assert result.side == "buy"
        assert len(placed_orders) == 1

        # cash*Kelly is capped at MAX_POSITION_SIZE_PCT (0.10 default) of cash
        import config
        max_dollar_amount = 100_000 * config.MAX_POSITION_SIZE_PCT
        assert result.qty * 5.0 * 100 <= max_dollar_amount  # qty contracts * mid_price * 100/contract

    def test_duplicate_buy_position_is_skipped(self, monkeypatch):
        monkeypatch.setattr("agents.execution_agent.alpaca_tools.get_clock", _open_clock)
        monkeypatch.setattr("agents.execution_agent.alpaca_tools.get_account", lambda: {"cash": 100_000})
        monkeypatch.setattr(
            "agents.execution_agent.alpaca_tools.get_positions",
            lambda: [{"symbol": "AAPL260116C00150000", "qty": 2, "avg_entry_price": 4.5, "market_value": 900.0}],
        )

        def _boom(*args, **kwargs):
            raise AssertionError("place_option_order should not be called for a duplicate buy")

        monkeypatch.setattr("agents.execution_agent.alpaca_tools.place_option_order", _boom)

        result = execute(_thesis(), _verdict())
        assert result.order_submitted is False
        assert "Existing position" in result.rejection_reason
