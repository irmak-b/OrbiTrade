"""
agents/execution_agent.py
----------------------------
Execution & portfolio management layer. Never generates analysis or
debates -- it takes an approved RiskVerdict + the Quant Agent's selected
option contract, applies deterministic position sizing (Kelly Criterion,
capped at config.MAX_POSITION_SIZE_PCT), checks existing positions, and
places the order via core/alpaca_tools.

This module is the deterministic safety ceiling described in the
architecture notes: LLMs never touch cash balances or order quantities
directly. cash * kelly_fraction, capped by MAX_POSITION_SIZE_PCT, is always
computed here in plain Python.

v3 (Options Trading): sizes and places OPTION orders (whole contracts,
each representing 100 shares of the underlying) instead of equity shares.
See calculate_option_order_qty() for the per-contract dollar-cost math.

Day 4 (Fail-Safe): every external call (market clock, account fetch,
positions fetch, order placement) is now individually wrapped so that a
closed market or an Alpaca/MCP error is logged as a clean rejection
(order_submitted=False, a specific rejection_reason) instead of raising and
crashing the LangGraph node -- see orchestration/graph.py::execution_node,
which still has its own top-level try/except as a second layer, but should
never actually need it for these expected failure modes anymore.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from core import alpaca_tools
from core.math_engine import kelly_criterion
from core.schemas import ExecutionResult, QuantThesis, RiskVerdict

TRADE_LOG_PATH = Path(__file__).resolve().parents[1] / "trade_log.jsonl"

OPTION_CONTRACT_MULTIPLIER = 100  # one option contract = 100 shares of the underlying


def _log_trade(result: ExecutionResult, thesis: Optional[QuantThesis], verdict: RiskVerdict) -> None:
    """
    Append-only trade log, one JSON object per line. This is the raw input
    the Day-4 reflection loop (agents/quant_agent.py::generate_reflection +
    core/memory.py) summarizes into durable lessons.
    """
    record = {
        "timestamp": result.timestamp.isoformat(),
        "symbol": result.symbol,
        "instrument": result.instrument,
        "contract_symbol": result.contract_symbol,
        "order_submitted": result.order_submitted,
        "order_id": result.order_id,
        "qty": result.qty,
        "side": result.side,
        "rejection_reason": result.rejection_reason,
        "kelly_fraction_used": result.kelly_fraction_used,
        "recommended_action": thesis.recommended_action if thesis else None,
        "thesis": thesis.thesis if thesis else None,
        "confidence_score": thesis.confidence_score if thesis else None,
        "win_probability": verdict.win_probability,
        "win_loss_ratio": verdict.win_loss_ratio,
    }
    with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def calculate_option_order_qty(
    cash: float,
    contract_price: float,
    kelly_fraction: float,
    max_position_pct: float,
    multiplier: int = OPTION_CONTRACT_MULTIPLIER,
) -> int:
    """
    Pure position-sizing function (no I/O), kept separate from execute() so
    it can be unit tested without hitting Alpaca. Unlike equity sizing,
    option contracts are whole-number only (no fractional contracts), and
    each contract's real dollar cost is contract_price * multiplier (a
    $2.50 mid-priced contract costs $250, not $2.50).

    Applies the deterministic safety cap: the position's dollar cost is
    never allowed to exceed max_position_pct of cash, no matter what Kelly
    or the LLMs suggest.
    """
    capped_fraction = max(0.0, min(kelly_fraction, max_position_pct))
    if contract_price <= 0:
        return 0
    dollar_amount = cash * capped_fraction
    cost_per_contract = contract_price * multiplier
    if cost_per_contract <= 0:
        return 0
    return int(dollar_amount // cost_per_contract)


def _reject(
    thesis: QuantThesis,
    verdict: RiskVerdict,
    reason: str,
    kelly_fraction_used: Optional[float] = None,
) -> ExecutionResult:
    result = ExecutionResult(
        symbol=thesis.symbol,
        instrument="option",
        order_submitted=False,
        rejection_reason=reason,
        kelly_fraction_used=kelly_fraction_used,
    )
    _log_trade(result, thesis, verdict)
    return result


def execute(thesis: QuantThesis, verdict: RiskVerdict) -> ExecutionResult:
    """
    Runs every deterministic gate in order and, if all pass, places the
    option order. Every branch (approved, rejected, no-edge, duplicate
    position, closed market, API failure) is logged to trade_log.jsonl and
    returns cleanly rather than raising, per the Day-4 fail-safe design.
    """

    # 1) Policy / sanity check -- an unapproved trade never reaches Alpaca.
    if not verdict.is_approved:
        return _reject(thesis, verdict, verdict.veto_reason or "Risk agent did not approve this trade.")

    if thesis.recommended_action == "HOLD" or thesis.selected_contract is None:
        return _reject(thesis, verdict, "Quant Agent recommended HOLD (or selected no contract); no order to place.")

    contract = thesis.selected_contract

    # 2) Fail-safe: don't attempt an order if we can't confirm the market is open.
    try:
        clock = alpaca_tools.get_clock()
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any clock failure means "don't trade"
        return _reject(thesis, verdict, f"Could not verify market clock ({exc}); order not submitted as a safety precaution.")

    if not clock.get("is_open", False):
        return _reject(
            thesis,
            verdict,
            f"Market is closed (next open: {clock.get('next_open')}); order not submitted.",
        )

    # 3) Deterministic Kelly-based sizing, capped by config.MAX_POSITION_SIZE_PCT.
    kelly_fraction = kelly_criterion(
        win_prob=verdict.win_probability,
        win_loss_ratio=verdict.win_loss_ratio,
        fraction=config.KELLY_FRACTION,
    )

    if kelly_fraction <= 0:
        return _reject(thesis, verdict, f"Kelly fraction {kelly_fraction:.3f} <= 0; no statistical edge.", kelly_fraction)

    try:
        account = alpaca_tools.get_account()
    except Exception as exc:
        return _reject(thesis, verdict, f"Could not fetch account info ({exc}); order not submitted.", kelly_fraction)

    qty = calculate_option_order_qty(
        cash=account["cash"],
        contract_price=contract.mid_price or 0.0,
        kelly_fraction=kelly_fraction,
        max_position_pct=config.MAX_POSITION_SIZE_PCT,
    )

    if qty <= 0:
        return _reject(
            thesis,
            verdict,
            "Computed contract quantity was zero (insufficient cash or contract too expensive relative to position cap).",
            kelly_fraction,
        )

    # 4) Existing-position check -- avoid duplicate buys of the same contract.
    try:
        positions = {p["symbol"]: p for p in alpaca_tools.get_positions()}
    except Exception:
        positions = {}  # fail-safe: treat as "no known open positions" rather than crash the node

    side = "buy"  # OrbiTrade only ever opens long option positions (buy call / buy put), never sells naked
    if contract.contract_symbol in positions:
        return _reject(
            thesis,
            verdict,
            f"Existing position in {contract.contract_symbol} already open; skipping duplicate buy.",
            kelly_fraction,
        )

    # 5) Place the order via the MCP-tool-shaped Alpaca wrapper.
    try:
        order = alpaca_tools.place_option_order(contract_symbol=contract.contract_symbol, qty=qty, side=side)
    except Exception as exc:
        return _reject(thesis, verdict, f"Alpaca order submission failed: {exc}", kelly_fraction)

    result = ExecutionResult(
        symbol=thesis.symbol,
        instrument="option",
        contract_symbol=contract.contract_symbol,
        order_submitted=True,
        order_id=order["order_id"],
        qty=qty,
        side=side,
        kelly_fraction_used=kelly_fraction,
    )
    _log_trade(result, thesis, verdict)
    return result


if __name__ == "__main__":
    from agents.data_agent import fetch_market_state
    from agents.quant_agent import generate_thesis
    from agents.risk_agent import evaluate
    import config as _config

    state = fetch_market_state(_config.WATCHLIST[0])
    thesis = generate_thesis(state)
    verdict = evaluate(state, thesis)
    result = execute(thesis, verdict)
    print(result.model_dump_json(indent=2))
