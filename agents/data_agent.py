"""
agents/data_agent.py
----------------------
Fetches raw market data from Alpaca (via core/alpaca_tools, an MCP-tool-
shaped wrapper) and turns it into a clean MarketState package by running it
through the deterministic math engine. This is the entry point of the
OrbiTrade pipeline: symbol in, MarketState out.

v3 (Options Trading): in addition to equity bars/quote/news, this module
now fetches Alpaca's live option chain for the symbol, prices each
candidate contract (implied volatility solved from its live mid-price,
then Greeks from Black-Scholes) via core/math_engine.py, and attaches a
short, pre-priced candidate list to MarketState.option_chain. The Quant
Agent (agents/quant_agent.py) then *selects* one of these contracts -- it
never invents a strike, expiration, or Greek value itself.

The computation logic is split out into `build_market_state()` (pure,
network-free) so it can be unit tested with synthetic bars -- only
`fetch_market_state()` actually talks to Alpaca.

IMPORTANT -- no silent fallbacks: fetch_market_state() deliberately does
NOT catch exceptions and substitute a fake price/empty state. An earlier
version did this (price=100.0, closes=[], option_chain=[] on any error),
which silently hid real bugs (a bad MCP tool name, a malformed response,
missing credentials) behind a MarketState that *looked* valid but wasn't
-- the Quant Agent would then confidently produce a HOLD thesis from fake
data, forever, with no visible error anywhere. Real failures should
surface: orchestration/graph.py's data_node already wraps this call in its
own try/except and records the real exception message in state["error"],
which is what should be debugged, not papered over here. Use
scripts/debug_mcp_responses.py or ORBITRADE_DEBUG=1 (see
core/alpaca_tools.py) to see exactly what Alpaca/MCP returned.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from core import alpaca_tools
from core.math_engine import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_greeks,
    calculate_rsi,
    implied_volatility,
)
from core.schemas import MarketState, NewsHeadline, OptionContract, OptionGreeksSnapshot, TechnicalSnapshot


def _price_option_chain(
    symbol: str,
    spot_price: float,
    raw_contracts: list[dict],
    risk_free_rate: float,
    max_candidates: int,
) -> list[OptionContract]:
    """
    Prices each raw Alpaca option-contract dict (strike/expiration/type) by
    pulling its live quote and solving implied volatility + Greeks via
    core/math_engine.py. A single bad/illiquid contract (missing quote, no
    arbitrage-free IV solution, malformed fields, etc.) is skipped -- with
    a printed reason -- rather than aborting the whole chain; this is the
    fail-safe that keeps one flaky contract from blocking every other
    candidate. If NOTHING gets priced, the reasons printed here are the
    first place to look.
    """
    priced: list[OptionContract] = []
    skipped = 0
    today = date.today()

    for raw in raw_contracts:
        try:
            strike = float(raw["strike_price"])
            expiration = raw["expiration_date"]
            option_type = raw["type"]
            contract_symbol = raw["symbol"]
        except (KeyError, TypeError, ValueError) as exc:
            skipped += 1
            print(f"⚠️  data_agent: skipping malformed contract entry ({exc}): {raw!r}")
            continue

        try:
            exp_date = datetime.fromisoformat(expiration).date()
            days_to_expiry = (exp_date - today).days
            if days_to_expiry <= 0:
                skipped += 1
                continue

            quote = alpaca_tools.get_option_latest_quote(contract_symbol)
            bid, ask = quote.get("bid"), quote.get("ask")
            if not bid or not ask or bid <= 0 or ask <= 0:
                skipped += 1
                print(f"⚠️  data_agent: no live two-sided quote for {contract_symbol} (bid={bid}, ask={ask}); skipping.")
                continue
            mid_price = round((bid + ask) / 2, 4)

            T = days_to_expiry / 365.0
            iv = implied_volatility(mid_price, spot_price, strike, T, risk_free_rate, option_type)
            greeks = calculate_greeks(spot_price, strike, T, risk_free_rate, iv, option_type)

            priced.append(
                OptionContract(
                    contract_symbol=contract_symbol,
                    underlying_symbol=symbol,
                    strike=strike,
                    expiration=expiration,
                    option_type=option_type,
                    days_to_expiry=days_to_expiry,
                    mid_price=mid_price,
                    bid=bid,
                    ask=ask,
                    open_interest=raw.get("open_interest"),
                    volume=raw.get("volume") or raw.get("daily_volume"),
                    greeks=OptionGreeksSnapshot(
                        delta=round(greeks.delta, 4),
                        gamma=round(greeks.gamma, 4),
                        vega=round(greeks.vega, 4),
                        theta=round(greeks.theta, 4),
                        rho=round(greeks.rho, 4),
                        implied_volatility=round(iv, 4),
                    ),
                )
            )
        except (ValueError, ZeroDivisionError, KeyError, TypeError) as exc:
            skipped += 1
            print(f"⚠️  data_agent: could not price {contract_symbol} ({exc}); skipping.")
            continue

    if not priced and raw_contracts:
        print(
            f"⚠️  data_agent: fetched {len(raw_contracts)} raw contracts for {symbol} "
            f"but priced 0 of them ({skipped} skipped) -- see the warnings above for why."
        )

    # Nearest-the-money first: the Quant Agent gets the most decision-relevant
    # contracts up front rather than a huge, unsorted chain.
    priced.sort(key=lambda oc: abs(oc.strike - spot_price))
    return priced[:max_candidates]


def build_market_state(
    symbol: str,
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    latest_price: float,
    news: Optional[list[dict]] = None,
    option_chain: Optional[list[OptionContract]] = None,
) -> MarketState:
    """
    Pure computation step: runs the deterministic math engine over a bar
    series and assembles a MarketState. No network calls, fully testable.
    `option_chain` is accepted pre-priced (or omitted -- defaults to []) so
    this function stays network-free; pricing itself happens in
    _price_option_chain(), called only from fetch_market_state().
    """
    technicals = TechnicalSnapshot()

    if len(closes) > config.RSI_PERIOD:
        rsi_series = calculate_rsi(closes, period=config.RSI_PERIOD)
        if len(rsi_series):
            technicals.rsi = float(rsi_series[-1])

    if len(closes) > config.ATR_PERIOD:
        atr_series = calculate_atr(highs, lows, closes, period=config.ATR_PERIOD)
        if len(atr_series):
            technicals.atr = float(atr_series[-1])

    if len(closes) >= config.BOLLINGER_PERIOD:
        mid, upper, lower = calculate_bollinger_bands(
            closes, period=config.BOLLINGER_PERIOD, num_std=config.BOLLINGER_NUM_STD
        )
        if len(mid):
            technicals.bollinger_mid = float(mid[-1])
            technicals.bollinger_upper = float(upper[-1])
            technicals.bollinger_lower = float(lower[-1])

    news_items = [NewsHeadline(**item) for item in (news or [])]

    return MarketState(
        symbol=symbol,
        latest_price=latest_price,
        closes=list(closes),
        highs=list(highs),
        lows=list(lows),
        technicals=technicals,
        option_chain=option_chain or [],
        news=news_items,
    )


def fetch_market_state(symbol: str, lookback_days: int = 30) -> MarketState:
    """
    Full pipeline step: pulls bars/quote/news AND the live option chain
    from Alpaca, prices the chain, and builds a MarketState. This is what
    orchestration/graph.py's data_node calls.

    Deliberately raises on failure instead of returning a fabricated
    MarketState -- see the module docstring for why. orchestration/graph.py's
    data_node catches this and records it as state["error"]; cli.py then
    prints it as clean JSON instead of a raw traceback.
    """
    print(f"📊 Fetching market data for {symbol}...")

    bars = alpaca_tools.get_stock_bars(symbol, lookback_days=lookback_days)
    quote = alpaca_tools.get_latest_quote(symbol)
    news = alpaca_tools.get_news(symbol, limit=5)

    spot_price = quote.get("price")
    if not spot_price or spot_price <= 0:
        raise ValueError(
            f"get_latest_quote({symbol!r}) returned no usable price (bid={quote.get('bid')}, "
            f"ask={quote.get('ask')}). Run scripts/debug_mcp_responses.py to inspect the raw "
            "MCP response and confirm the quote tool's response shape."
        )

    raw_contracts = alpaca_tools.get_option_chain(
        symbol,
        min_days_to_expiry=config.OPTION_MIN_DAYS_TO_EXPIRY,
        max_days_to_expiry=config.OPTION_MAX_DAYS_TO_EXPIRY,
    )
    option_chain = _price_option_chain(
        symbol,
        spot_price=spot_price,
        raw_contracts=raw_contracts,
        risk_free_rate=config.RISK_FREE_RATE,
        max_candidates=config.OPTION_MAX_CANDIDATES,
    )

    if not option_chain:
        print(
            f"⚠️  data_agent: no priced option contracts for {symbol} -- the Quant Agent "
            "will be forced to HOLD this cycle (there is no equity fallback)."
        )

    return build_market_state(
        symbol=symbol,
        closes=bars["closes"],
        highs=bars["highs"],
        lows=bars["lows"],
        latest_price=spot_price,
        news=news,
        option_chain=option_chain,
    )


if __name__ == "__main__":
    state = fetch_market_state(config.WATCHLIST[0])
    print(state.model_dump_json(indent=2))
