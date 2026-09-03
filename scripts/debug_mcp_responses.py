"""
scripts/debug_mcp_responses.py
---------------------------------
Prints the RAW, unmodified response of a few key MCP tool calls -- before
any of core/alpaca_tools.py's unwrapping logic touches them. Run this once
and paste the output back so we can fix the *actual* response shape
instead of guessing at it again.

Run with:
    python scripts/debug_mcp_responses.py AAPL
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: F401 -- loads .env via config.py's load_dotenv()
from core import mcp_client

symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"


def dump(label: str, tool_name: str, args: dict):
    print(f"\n{'=' * 70}\n{label}  ->  call_tool({tool_name!r}, {args!r})\n{'=' * 70}")
    try:
        response = mcp_client.call_tool(tool_name, args)
        print(f"type(response) = {type(response)}")
        print(json.dumps(response, indent=2, default=str))
    except Exception as exc:
        print(f"RAISED: {type(exc).__name__}: {exc}")


print("Available tools on this MCP server:")
try:
    print(mcp_client.list_tools())
except Exception as exc:
    print(f"list_tools() failed: {exc}")

dump("get_account_info", "get_account_info", {})
dump("get_stock_latest_quote", "get_stock_latest_quote", {"symbols": symbol})
dump("get_stock_bars", "get_stock_bars", {"symbols": symbol, "timeframe": "1Day", "days": 5})
dump(
    "get_option_contracts",
    "get_option_contracts",
    {
        "underlying_symbols": symbol,
        "expiration_date_gte": "2026-09-15",
        "expiration_date_lte": "2026-10-15",
        "status": "active",
    },
)
dump("get_market_clock", "get_market_clock", {})
