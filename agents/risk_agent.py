"""
agents/risk_agent.py
-----------------------
Risk / Devil's Advocate agent. Challenges the Quant Agent's thesis, runs a
deterministic Greek/volatility exposure check on the *selected* option
contract, and estimates the win probability / win-loss ratio that feed the
Kelly Criterion sizing used by execution_agent.py. This agent has final
veto power (is_approved).

Two layers of defense, in order:
    1) check_greek_risk() -- a pure, deterministic threshold filter against
       thesis.selected_contract.greeks. No LLM call is needed to veto an
       obviously excessive Vega/Theta exposure.
    2) evaluate() -- if the deterministic filter passes, a Devil's Advocate
       LLM debate produces the final RiskVerdict.

v3 (Options Trading): check_greek_risk() now reads Greeks off the Quant
Agent's selected_contract (there is no longer a market_state.greeks -- see
core/schemas.py). A HOLD thesis has no selected_contract and skips the
Greek check entirely (nothing to size or veto).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
from json_repair import repair_json 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.llm_factory import get_agent_llm
from core.schemas import MarketState, OptionGreeksSnapshot, QuantThesis, RiskVerdict

SYSTEM_PROMPT = """You are the Devil's Advocate risk officer on a systematic options
trading desk.

You will receive: (1) a deterministic market-state JSON, and (2) an
options thesis produced by the Quant Agent, including the exact contract
(strike, expiration, option_type) it selected and that contract's Greeks.
Your job is to challenge that thesis, not rubber-stamp it.

If the thesis is BUY_CALL, argue the downside case (news already priced
in, upcoming earnings, macro headwinds, IV crush after an event). If it is
BUY_PUT, argue the upside case that could get missed. Always weigh:
  - Theta: how much daily time-decay cost is being accepted, and whether
    the thesis's timeline realistically beats it;
  - Vega: how exposed the position is to a volatility crush (e.g. right
    after an earnings date);
  - Delta: whether the position's directional exposure actually matches
    the stated confidence level.
Flag any of these explicitly in your counter_thesis when they matter.

You must also estimate, based on the strength of the evidence, a realistic
win probability (0-1) and win/loss payoff ratio (>0) for this trade -- these
feed a Kelly Criterion position-sizing formula downstream, so keep them
conservative and evidence-based rather than promotional.

Respond with ONLY a single JSON object matching this schema, nothing else,
no markdown fences:
{
  "is_approved": true | false,
  "counter_thesis": "<the strongest opposing case you can build>",
  "win_probability": <float between 0.0 and 1.0>,
  "win_loss_ratio": <float greater than 0.0>,
  "veto_reason": "<string, required and non-null if is_approved is false, else null>"
}
"""

DEFAULT_MAX_ABS_VEGA = 0.5   # price-move units per 1% vol change
DEFAULT_MAX_ABS_THETA = 0.5  # price-move units of decay per day

def _parse_json_response(raw: str) -> dict:
    """Parses the LLM's JSON output. It first attempts the standard `json.loads`;
    if that fails (e.g., due to common LLM errors like unescaped quotes or
    line breaks within free-text fields), it tries to repair the output
    using `json_repair`. If both attempts fail, it re-raises the error,
    including the first 500 characters of the raw output for diagnostic purposes."""
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        try:
            repaired = repair_json(cleaned)
            return json.loads(repaired)
        except Exception:
            raise ValueError(
                f"LLM response was not valid JSON and could not be repaired: {exc}\n"
                f"First 500 characters of raw response:\n{cleaned[:500]}"
            ) from exc

def check_greek_risk(
    greeks: Optional[OptionGreeksSnapshot],
    max_abs_vega: float = DEFAULT_MAX_ABS_VEGA,
    max_abs_theta: float = DEFAULT_MAX_ABS_THETA,
) -> Optional[str]:
    """
    Deterministic pre-screen (no LLM involved) for excessive Greek exposure
    on the selected contract.

    Returns a human-readable veto reason if a hard threshold is breached,
    otherwise None. Kept as a standalone pure function so it's trivial to
    unit test and to tune thresholds without touching the LLM prompt.
    """
    if greeks is None:
        return None
    if greeks.vega is not None and abs(greeks.vega) > max_abs_vega:
        return f"Vega exposure {greeks.vega:.3f} exceeds the {max_abs_vega} threshold (volatility-crush risk)."
    if greeks.theta is not None and abs(greeks.theta) > max_abs_theta:
        return f"Theta decay {greeks.theta:.3f}/day exceeds the {max_abs_theta} threshold (time-decay risk)."
    return None


def _strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def evaluate(market_state: MarketState, thesis: QuantThesis) -> RiskVerdict:
    """Runs the deterministic Greek filter first (on the selected
    contract's Greeks), then (if it passes) the Devil's Advocate LLM debate
    (DeepSeek-V3 via Featherless)."""

    selected_greeks = thesis.selected_contract.greeks if thesis.selected_contract else None
    hard_veto = check_greek_risk(selected_greeks)
    if hard_veto:
        return RiskVerdict(
            symbol=market_state.symbol,
            is_approved=False,
            counter_thesis="Deterministic Greek-risk filter triggered before the debate step.",
            win_probability=0.0,
            win_loss_ratio=0.01,
            veto_reason=hard_veto,
        )

    llm = get_agent_llm("risk")
    payload = json.dumps(
        {
            "market_state": json.loads(market_state.model_dump_json()),
            "quant_thesis": json.loads(thesis.model_dump_json()),
        },
        indent=2,
    )

    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{payload}\n\nProduce your verdict as JSON."},
        ]
    )

    data = _parse_json_response(response.content)
    data["symbol"] = market_state.symbol
    return RiskVerdict(**data)


if __name__ == "__main__":
    from agents.data_agent import fetch_market_state
    from agents.quant_agent import generate_thesis
    import config

    state = fetch_market_state(config.WATCHLIST[0])
    thesis = generate_thesis(state)
    verdict = evaluate(state, thesis)
    print(verdict.model_dump_json(indent=2))
