"""
agents/quant_agent.py
-----------------------
Quant Agent = a senior *options* trader, NOT a decision maker. It reads the
deterministic MarketState produced by data_agent.py (technicals, and a
short pre-priced option_chain with Greeks/IV for each candidate contract)
and produces a structured thesis (QuantThesis) for the Risk/Debate agent to
challenge. It is explicitly forbidden from inventing its own numbers -- it
may only reason over the numbers it's handed, and it must select a contract
strictly from the option_chain it was given (never invent a strike/
expiration/contract symbol).

v3 (Options Trading): recommended_action is now BUY_CALL / BUY_PUT / HOLD
(no bare-equity BUY/SELL) -- see core/schemas.py::QuantThesis.

Day 4 (Reflection): this module also owns generate_reflection(), called
by orchestration/graph.py's reflection_node right after a trade is
executed. It turns the (market_state, thesis, verdict, execution_result)
tuple into one concrete, reusable lesson, which core/memory.py persists
and which future calls to generate_thesis() re-inject via
`lessons_context` -- this is the "learns from its own trades" loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from json_repair import repair_json 

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.llm_factory import get_agent_llm
from core.schemas import ExecutionResult, MarketState, QuantThesis, RiskVerdict

SYSTEM_PROMPT = """You are a senior options trader on a systematic trading desk.

You do not trade the underlying equity directly. You will be given a
deterministic market-state JSON package containing:
- technical indicators (RSI, ATR, Bollinger Bands), pre-computed by a
  Python math engine;
- `option_chain`: a short list of REAL, currently tradable option contracts
  on this underlying, each already priced with its live mid-price, implied
  volatility, and Greeks (Delta, Gamma, Vega, Theta, Rho), all pre-computed
  via Black-Scholes -- you never compute or estimate these yourself;
- recent news headlines, when available;
- `lessons`, when present: dated takeaways from this desk's own past
  trades on this exact symbol -- weigh these the way a trader would weigh
  their own trading journal, not as instructions to blindly follow.

Hard rules:
- NEVER perform or invent your own arithmetic, and NEVER invent a strike,
  expiration, contract symbol, price, or Greek that isn't literally present
  in the option_chain you were given.
- You must think in terms of DIRECTION *and* TIME *and* VOLATILITY, not
  direction alone: a bullish view with only 15 days left and high Vega
  exposure may be a worse trade than a similar view further out, or no
  trade at all. Say so explicitly in your thesis when it applies.
- If recommended_action is BUY_CALL or BUY_PUT, you MUST set
  selected_contract to one of the entries from option_chain, copied
  exactly (contract_symbol, strike, expiration, option_type). Never
  fabricate a contract.
- If option_chain is empty, or none of the available contracts fit a
  sound risk/reward profile, recommended_action MUST be "HOLD" -- there is
  no equity fallback in this system.
- Your job is to build a thesis, not to place a trade.

Respond with ONLY a single JSON object matching this schema, nothing else,
no markdown fences:
{
  "bias": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence_score": <float between 0.0 and 1.0>,
  "recommended_action": "BUY_CALL" | "BUY_PUT" | "HOLD",
  "thesis": "<detailed reasoning grounded strictly in the provided numbers, covering direction, time (days_to_expiry / Theta), and volatility (IV / Vega)>",
  "key_metrics": {"<metric_name>": <value>, ...},
  "selected_contract": <one full object copied verbatim from option_chain, or null if recommended_action is "HOLD">
}
"""

REFLECTION_SYSTEM_PROMPT = """You are the same options trader, now writing a single, highly specific
lesson for your own future reference after just completing one decision
cycle (thesis -> risk debate -> execution attempt).

Write exactly ONE sentence: concrete, falsifiable, and specific to the
setup that just occurred (e.g. mention the option type, moneyness/strike
relationship, days to expiry, or the IV/Greek level that mattered, and
what the risk desk's response revealed) -- not a generic platitude like
"be careful with risk." If the trade was rejected or held, the lesson
should be about what pattern to watch for next time, not about an outcome
that hasn't happened yet.

Respond with ONLY the lesson sentence, no quotes, no markdown, no preamble.
"""


def _strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    return raw.strip()
def _parse_json_response(raw: str) -> dict:
    """LLM'in JSON çıktısını parse eder. Önce standart json.loads dener;
    başarısız olursa (örn. serbest-metin alanı içinde kaçışsız tırnak/
    satır sonu gibi yaygın LLM hatalarından dolayı) json_repair ile
    onarmayı dener. İkisi de başarısız olursa, teşhis için ham çıktının
    ilk 500 karakterini hataya ekleyerek yeniden fırlatır."""
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        try:
            repaired = repair_json(cleaned)
            return json.loads(repaired)
        except Exception:
            raise ValueError(
                f"LLM yanıtı geçerli JSON değildi ve onarılamadı: {exc}\n"
                f"Ham yanıtın ilk 500 karakteri:\n{cleaned[:500]}"
            ) from exc

def generate_thesis(
    market_state: MarketState,
    counter_thesis: str | None = None,
    lessons_context: str | None = None,
) -> QuantThesis:
    """Calls the quant LLM (Qwen2.5-72B via Featherless) and parses its
    structured output into a QuantThesis.

    Args:
        market_state: The deterministic market data package, including the
            pre-priced option_chain the model must choose from.
        counter_thesis: If this is a reconsideration after a Risk Agent veto
            (LangGraph debate loop), the Risk Agent's counter_thesis is
            passed in here so the Quant Agent can revise its thesis in
            light of the objection -- it does not see this on the first pass.
        lessons_context: Day-4 self-reflection input. Recent lessons this
            desk has recorded for this exact symbol (core/memory.py), or
            None if there is no history yet.
    """
    llm = get_agent_llm("quant")
    payload = market_state.model_dump_json(indent=2)

    user_content = f"Market state:\n{payload}\n\n"
    if lessons_context:
        user_content += (
            f"Lessons from this desk's past trades on {market_state.symbol}:\n"
            f"{lessons_context}\n\n"
        )
    user_content += "Produce your thesis as JSON."
    if counter_thesis:
        user_content += (
            "\n\nNote: a previous version of this thesis was challenged by the "
            f"risk desk with the following objection:\n\"{counter_thesis}\"\n"
            "Revise your bias, confidence_score, recommended_action, and/or "
            "selected_contract if this objection changes your read of the "
            "deterministic data -- do not simply repeat your previous answer "
            "if the objection has merit."
        )

    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )

    data = _parse_json_response(response.content)
    data["symbol"] = market_state.symbol
    return QuantThesis(**data)


def generate_reflection(
    market_state: MarketState,
    thesis: QuantThesis,
    verdict: RiskVerdict,
    execution_result: ExecutionResult,
) -> str:
    """
    Day-4 self-reflection: summarizes one completed decision cycle into a
    single reusable lesson sentence. Called by
    orchestration/graph.py::reflection_node right after execution_node,
    and persisted via core/memory.py::TradeMemory.save_trade_reflection so
    the *next* generate_thesis() call on this symbol sees it.

    Deliberately does not require a closed-trade PnL: the lesson is about
    the quality of the setup and the risk desk's reaction to it, which is
    known immediately, rather than the eventual (unknown, out-of-scope-for-
    this-pipeline) profit/loss outcome.
    """
    llm = get_agent_llm("quant")
    payload = json.dumps(
        {
            "symbol": market_state.symbol,
            "technicals": json.loads(market_state.technicals.model_dump_json()),
            "thesis": json.loads(thesis.model_dump_json()),
            "verdict": json.loads(verdict.model_dump_json()),
            "execution_result": json.loads(execution_result.model_dump_json()),
        },
        indent=2,
        default=str,
    )

    response = llm.invoke(
        [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"{payload}\n\nWrite the lesson sentence."},
        ]
    )

    return _strip_code_fences(response.content).strip()


if __name__ == "__main__":
    from agents.data_agent import fetch_market_state
    import config

    state = fetch_market_state(config.WATCHLIST[0])
    thesis = generate_thesis(state)
    print(thesis.model_dump_json(indent=2))
