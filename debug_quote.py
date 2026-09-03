"""
debug_quote.py
----------------
Copy it to the project's root directory (the same folder as cli.py) and run it:

    python debug_quote.py

This script connects to the real alpaca-mcp-server and:
  1) lists all available tools,
  2) fetches raw responses from get_stock_latest_quote and get_stock_bars
and prints them. We will use the actual field names here to fix
the get_latest_quote function in core/alpaca_tools.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import mcp_client

print("=" * 60)
print("1) Full tool names on the server:")
print("=" * 60)
try:
    tools = mcp_client.list_tools()
    for t in tools:
        print(" -", t)
except Exception as e:
    print("list_tools() error:", e)

print()
print("=" * 60)
print("2) get_stock_latest_quote('AAPL') raw response:")
print("=" * 60)
try:
    raw = mcp_client.call_tool("get_stock_latest_quote", {"symbols": "AAPL"})
    print(json.dumps(raw, indent=2, default=str))
    print()
    print("Python type:", type(raw))
except Exception as e:
    print("get_stock_latest_quote() error:", e)

print()
print("=" * 60)
print("3) get_stock_bars('AAPL') raw response (first 2 bars):")
print("=" * 60)
try:
    raw_bars = mcp_client.call_tool("get_stock_bars", {"symbols": "AAPL", "timeframe": "1Day", "days": 5})
    print(json.dumps(raw_bars, indent=2, default=str)[:2000])
except Exception as e:
    print("get_stock_bars() error:", e)
