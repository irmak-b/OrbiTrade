"""
core/alpaca_tools.py
----------------------
Alpaca-facing functions that agents call, implemented as real MCP tool
calls against the official `alpaca-mcp-server` (via core/mcp_client.py).
Each function calls exactly one Alpaca MCP tool and reshapes its response
into the plain dict shape the rest of OrbiTrade expects.

v3 (Options Trading): adds get_option_chain / get_option_latest_quote /
place_option_order alongside the existing stock/account/position helpers,
plus get_clock() for the execution_agent's fail-safe market-hours check.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Literal, Optional

from core import mcp_client

# --- MCP tool names (actual names from your system) ---
# These are the REAL tool names from your `list_tools()` output
OPTION_CONTRACTS_TOOL = "get_option_contracts"
OPTION_QUOTE_TOOL = "get_option_latest_quote"
OPTION_ORDER_TOOL = "place_option_order"
STOCK_BARS_TOOL = "get_stock_bars"
STOCK_QUOTE_TOOL = "get_stock_latest_quote"
NEWS_TOOL = "get_news"
ACCOUNT_TOOL = "get_account_info"
POSITIONS_TOOL = "get_all_positions"
STOCK_ORDER_TOOL = "place_stock_order"
CLOCK_TOOL_CANDIDATES = ["get_market_clock", "get_clock", "get_market_hours", "clock"]
_CLOCK_TOOL_RESOLVED: str | None = None

# ---------------------------------------------------------------------------
# Response-shape helpers (defensive against wrapped vs. flat MCP responses)
# ---------------------------------------------------------------------------

def _unwrap_dict(response: Any, key: str) -> dict:
    """Accepts either {key: {...}} or a flat {...} and always returns a dict."""
    if isinstance(response, dict):
        nested = response.get(key)
        if isinstance(nested, dict):
            return nested
        return response
    return {}

def _unwrap_list(response: Any, key: str) -> list:
    """Accepts either {key: [...]} or a flat [...] and always returns a list."""
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        nested = response.get(key)
        if isinstance(nested, list):
            return nested
    return []
def _unwrap_data(response: Any) -> Any:
    """Alpaca MCP server her cevabı bir üst 'data' anahtarının içine
    sarıyor: {"_alpaca_mcp_security": {...}, "data": {...asıl içerik...}}.
    Var olan tüm parse mantığı bunu atlıyordu (bkz. get_account/get_positions
    zaten ad-hoc olarak bunu yapıyordu). Bunu tek yerden hallet."""
    if isinstance(response, dict) and "data" in response:
        return response["data"]
    return response
# ---------------------------------------------------------------------------
# Market data (equities -- still needed for technicals/spot price/news)
# ---------------------------------------------------------------------------

def get_stock_bars(symbol: str, lookback_days: int = 30) -> dict:
    """Calls the 'get_stock_bars' MCP tool -- daily OHLCV bars."""
    try:
        response = mcp_client.call_tool(
            STOCK_BARS_TOOL,
            {"symbols": symbol, "timeframe": "1Day", "days": lookback_days},
        )
        response = _unwrap_data(response)   
        # Defensive parsing
        bars_data = {}
        if isinstance(response, dict):
            bars_data = response.get("bars", {})
        elif isinstance(response, list):
            # Eğer doğrudan liste döndüyse, symbol'e göre filtrele
            bars_data = {symbol: response}
        
        bars = bars_data.get(symbol, [])
        
        return {
            "symbol": symbol,
            "closes": [bar.get("c", 0) for bar in bars],
            "highs": [bar.get("h", 0) for bar in bars],
            "lows": [bar.get("l", 0) for bar in bars],
            "volumes": [bar.get("v", 0) for bar in bars],
        }
    except Exception as e:
        print(f"⚠️ get_stock_bars hatası: {e}")
        return {"symbol": symbol, "closes": [], "highs": [], "lows": [], "volumes": []}

def get_latest_quote(symbol: str) -> dict:
    """Calls the 'get_stock_latest_quote' MCP tool."""
    try:
        response = mcp_client.call_tool(STOCK_QUOTE_TOOL, {"symbols": symbol})
        response = _unwrap_data(response)
        quote = {}
        if isinstance(response, dict):
            quotes = response.get("quotes", {})
            quote = quotes.get(symbol, {})
        elif isinstance(response, list) and len(response) > 0:
            quote = response[0]
        
        bid = quote.get("bp", quote.get("bid", 0))
        ask = quote.get("ap", quote.get("ask", 0))
        mid = (bid + ask) / 2 if bid and ask else (ask or bid or 0)
        
        return {"symbol": symbol, "bid": bid, "ask": ask, "price": mid}
    except Exception as e:
        print(f"⚠️ get_latest_quote hatası: {e}")
        return {"symbol": symbol, "bid": 0, "ask": 0, "price": 0}

def get_news(symbol: str, limit: int = 5) -> list[dict]:
    """Calls the 'get_news' MCP tool."""
    try:
        response = mcp_client.call_tool(NEWS_TOOL, {"symbols": symbol, "limit": limit})
        response = _unwrap_data(response)
        articles = []
        if isinstance(response, dict):
            articles = response.get("news", [])
        elif isinstance(response, list):
            articles = response
        
        return [
            {
                "headline": article.get("headline", ""),
                "source": article.get("source", ""),
                "published_at": article.get("created_at", ""),
            }
            for article in articles
        ]
    except Exception as e:
        print(f"⚠️ get_news hatası: {e}")
        return []

# ---------------------------------------------------------------------------
# Options market data (v3)
# ---------------------------------------------------------------------------

def get_option_chain(
    symbol: str,
    min_days_to_expiry: int = 15,
    max_days_to_expiry: int = 45,
    option_type: Optional[Literal["call", "put"]] = None,
) -> list[dict]:
    """Calls the 'get_option_contracts' MCP tool."""
    try:
        today = date.today()
        args = {
            "underlying_symbols": symbol,
            "expiration_date_gte": (today + timedelta(days=min_days_to_expiry)).isoformat(),
            "expiration_date_lte": (today + timedelta(days=max_days_to_expiry)).isoformat(),
            "status": "active",
        }
        if option_type:
            args["type"] = option_type

        response = mcp_client.call_tool(OPTION_CONTRACTS_TOOL, args)
        response = _unwrap_data(response)
        return _unwrap_list(response, "option_contracts")
    except Exception as e:
        print(f"⚠️ get_option_chain hatası: {e}")
        return []

def get_option_latest_quote(contract_symbol: str) -> dict:
    """Calls the 'get_option_latest_quote' MCP tool."""
    try:
        response = mcp_client.call_tool(OPTION_QUOTE_TOOL, {"symbols": contract_symbol})
        response = _unwrap_data(response)
        quote = {}
        if isinstance(response, dict):
            quotes = response.get("quotes", {})
            quote = quotes.get(contract_symbol, {})
        elif isinstance(response, list) and len(response) > 0:
            quote = response[0]
        
        bid = quote.get("bp", quote.get("bid", 0))
        ask = quote.get("ap", quote.get("ask", 0))
        
        return {"contract_symbol": contract_symbol, "bid": bid, "ask": ask}
    except Exception as e:
        print(f"⚠️ get_option_latest_quote hatası: {e}")
        return {"contract_symbol": contract_symbol, "bid": 0, "ask": 0}

# ---------------------------------------------------------------------------
# Account / positions / market clock
# ---------------------------------------------------------------------------

def get_account() -> dict:
    """Calls the 'get_account_info' MCP tool."""
    try:
        response = mcp_client.call_tool(ACCOUNT_TOOL, {})
        
        # The response comes wrapped in 'data' key
        account = {}
        if isinstance(response, dict):
            # Check for 'data' wrapper (your actual response format)
            if "data" in response and isinstance(response["data"], dict):
                account = response["data"]
            # Fallback: direct response
            elif "cash" in response or "portfolio_value" in response:
                account = response
            # Fallback: wrapped under 'account'
            elif "account" in response and isinstance(response["account"], dict):
                account = response["account"]
        
        # Parse with safe defaults
        cash = float(account.get("cash", 0))
        portfolio_value = float(account.get("portfolio_value", account.get("equity", 0)))
        buying_power = float(account.get("buying_power", account.get("effective_buying_power", 0)))
        
        return {
            "cash": cash,
            "portfolio_value": portfolio_value,
            "buying_power": buying_power,
        }
    except Exception as e:
        print(f"⚠️ get_account error: {e}")
        return {"cash": 100000.0, "portfolio_value": 100000.0, "buying_power": 400000.0}


def get_positions() -> list[dict]:
    """Calls the 'get_all_positions' MCP tool."""
    try:
        response = mcp_client.call_tool(POSITIONS_TOOL, {})
        
        positions = []
        if isinstance(response, dict):
            # Check for 'data' wrapper
            if "data" in response and isinstance(response["data"], list):
                positions = response["data"]
            # Fallback: direct list under 'positions'
            elif "positions" in response and isinstance(response["positions"], list):
                positions = response["positions"]
        elif isinstance(response, list):
            positions = response
        
        return [
            {
                "symbol": p.get("symbol", ""),
                "qty": float(p.get("qty", 0)),
                "avg_entry_price": float(p.get("avg_entry_price", 0)),
                "market_value": float(p.get("market_value", 0)),
            }
            for p in positions
        ]
    except Exception as e:
        print(f"⚠️ get_positions error: {e}")
        return []


def get_clock() -> dict:
    """
    Calls whichever candidate clock MCP tool actually exists on your
    server (tries CLOCK_TOOL_CANDIDATES in order, once, then caches the
    working name in _CLOCK_TOOL_RESOLVED) -- whether the market is open
    right now. Used by execution_agent.py's fail-safe check before
    submitting any order: if the clock call fails or reports the market
    closed, no order is placed and the pipeline still completes cleanly
    (see agents/execution_agent.py::execute).
    """
    global _CLOCK_TOOL_RESOLVED

    names_to_try = [_CLOCK_TOOL_RESOLVED] if _CLOCK_TOOL_RESOLVED else CLOCK_TOOL_CANDIDATES
    last_error: Exception | None = None

    for name in names_to_try:
        try:
            response = mcp_client.call_tool(name, {})
            response = _unwrap_data(response)
        except Exception as exc:
            last_error = exc
            continue
        _CLOCK_TOOL_RESOLVED = name  # remember it -- skip the guesswork next call
        clock = _unwrap_dict(response, "clock")
        return {
            "is_open": bool(clock.get("is_open", False)),
            "next_open": clock.get("next_open"),
            "next_close": clock.get("next_close"),
        }

    raise RuntimeError(
        f"None of the candidate market-clock tool names worked ({CLOCK_TOOL_CANDIDATES}). "
        f"Last error: {last_error}. Check your server's real tool name and add it to "
        "CLOCK_TOOL_CANDIDATES in core/alpaca_tools.py."
    )
# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------

def place_market_order(symbol: str, qty: float, side: Literal["buy", "sell"]) -> dict:
    """Calls the 'place_stock_order' MCP tool."""
    try:
        order = mcp_client.call_tool(
            STOCK_ORDER_TOOL,
            {
                "symbol": symbol,
                "side": side,
                "qty": str(qty),
                "type": "market",
                "time_in_force": "day",
            },
        )
        
        return {
            "order_id": str(order.get("id", "")),
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "status": str(order.get("status", "unknown")),
        }
    except Exception as e:
        print(f"⚠️ place_market_order hatası: {e}")
        return {"order_id": "", "symbol": symbol, "qty": qty, "side": side, "status": "error"}

def place_option_order(contract_symbol: str, qty: int, side: Literal["buy", "sell"] = "buy") -> dict:
    """Calls the 'place_option_order' MCP tool."""
    try:
        order = mcp_client.call_tool(
            OPTION_ORDER_TOOL,
            {
                "symbol": contract_symbol,
                "side": side,
                "qty": str(qty),
                "type": "market",
                "time_in_force": "day",
            },
        )
        
        return {
            "order_id": str(order.get("id", "")),
            "contract_symbol": contract_symbol,
            "qty": qty,
            "side": side,
            "status": str(order.get("status", "unknown")),
        }
    except Exception as e:
        print(f"⚠️ place_option_order hatası: {e}")
        return {"order_id": "", "contract_symbol": contract_symbol, "qty": qty, "side": side, "status": "error"}