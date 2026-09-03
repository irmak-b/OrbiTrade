"""
core/schemas.py
-----------------
Shared Pydantic data contracts that flow between agents. Every agent reads
and writes one of these models instead of passing around loose dicts, so
the pipeline (data -> quant -> risk -> execution) has a typed, validated
interface at every hop -- and so LangGraph's state (Day 3) can be built
directly on top of these models.

v3 (Options Trading): OrbiTrade's hackathon track requires every strategy
to trade *options*, not the underlying equity. The schema changes here are
deliberately concentrated in three places:
  - MarketState.option_chain: a short, pre-priced list of candidate option
    contracts (computed by core/math_engine.py, never invented by an LLM)
    that the Quant Agent must choose from.
  - QuantThesis.recommended_action is now BUY_CALL | BUY_PUT | HOLD (no more
    raw BUY/SELL of the underlying), and QuantThesis.selected_contract pins
    down exactly which contract from the chain was chosen.
  - ExecutionResult now carries `instrument` + `contract_symbol` so the
    trade log / memory layer always knows it was an options trade.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

OptionType = Literal["call", "put"]


# ---------------------------------------------------------------------------
# Data Agent output
# ---------------------------------------------------------------------------

class TechnicalSnapshot(BaseModel):
    """Latest-bar technical indicator readings, computed by core/math_engine.py."""

    rsi: Optional[float] = None
    atr: Optional[float] = None
    bollinger_mid: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None


class OptionGreeksSnapshot(BaseModel):
    """Option Greeks + IV, computed by core/math_engine.py. All optional -- a
    contract we couldn't price (e.g. no quote available) simply omits this."""

    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    implied_volatility: Optional[float] = None


class OptionContract(BaseModel):
    """
    One candidate (or chosen) option contract. Populated by data_agent.py
    from Alpaca's option chain + math_engine's Black-Scholes/Greeks -- an
    LLM only ever *selects* one of these by contract_symbol, it never
    invents strike/expiration/pricing numbers itself.
    """

    contract_symbol: str  # Alpaca/OCC contract symbol, e.g. "AAPL260116C00150000"
    underlying_symbol: str
    strike: float
    expiration: str  # ISO date, e.g. "2026-01-16"
    option_type: OptionType
    days_to_expiry: Optional[int] = None
    mid_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    greeks: Optional[OptionGreeksSnapshot] = None


class NewsHeadline(BaseModel):
    headline: str
    source: Optional[str] = None
    published_at: Optional[str] = None


class MarketState(BaseModel):
    """
    Data Agent output. This is the single source of truth handed to the
    Quant Agent -- it never sees raw Alpaca responses, only this package.
    """

    symbol: str
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latest_price: float
    closes: list[float] = Field(default_factory=list)
    highs: list[float] = Field(default_factory=list)
    lows: list[float] = Field(default_factory=list)
    technicals: TechnicalSnapshot = Field(default_factory=TechnicalSnapshot)
    option_chain: list[OptionContract] = Field(default_factory=list)
    news: list[NewsHeadline] = Field(default_factory=list)
    lessons: Optional[str] = None  # Day-4: relevant past lessons for this symbol, from core/memory.py


# ---------------------------------------------------------------------------
# Quant Agent output
# ---------------------------------------------------------------------------

class QuantThesis(BaseModel):
    """
    Quant Agent output. The Quant Agent is an analyst, not a decision maker:
    it interprets MarketState and hands this thesis to the Risk Agent for
    challenge, it does not trigger execution directly.

    recommended_action is options-only per the hackathon's core requirement
    ("all strategies must incorporate options trading"): BUY_CALL / BUY_PUT
    to open a long option position, or HOLD to skip this cycle. There is no
    raw-equity BUY/SELL action anymore.
    """

    symbol: str
    bias: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    recommended_action: Literal["BUY_CALL", "BUY_PUT", "HOLD"]
    thesis: str
    key_metrics: dict[str, float] = Field(default_factory=dict)
    selected_contract: Optional[OptionContract] = None

    @model_validator(mode="after")
    def _contract_required_unless_hold(self) -> "QuantThesis":
        if self.recommended_action != "HOLD" and self.selected_contract is None:
            raise ValueError(
                f"recommended_action={self.recommended_action!r} requires a selected_contract "
                "(options trading is mandatory -- there is no bare-equity fallback)."
            )
        return self


# ---------------------------------------------------------------------------
# Risk / Devil's Advocate Agent output
# ---------------------------------------------------------------------------

class RiskVerdict(BaseModel):
    """
    Risk Agent output. Holds final veto power over a trade (is_approved) and
    supplies the win_probability / win_loss_ratio inputs that
    execution_agent.py feeds into the Kelly Criterion for position sizing.
    """

    symbol: str
    is_approved: bool
    counter_thesis: str
    win_probability: float = Field(ge=0.0, le=1.0)
    win_loss_ratio: float = Field(gt=0.0)
    veto_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Execution Agent output
# ---------------------------------------------------------------------------

class ExecutionResult(BaseModel):
    """Execution Agent output. Logged verbatim to trade_log.jsonl, and (Day 4)
    summarized into a lesson by agents/quant_agent.py::generate_reflection
    and stored in core/memory.py for future decision cycles."""

    symbol: str
    instrument: Literal["option", "stock"] = "option"
    contract_symbol: Optional[str] = None
    order_submitted: bool
    order_id: Optional[str] = None
    qty: Optional[float] = None
    side: Optional[Literal["buy", "sell"]] = None
    rejection_reason: Optional[str] = None
    kelly_fraction_used: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
