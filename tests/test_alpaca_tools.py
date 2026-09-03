"""
tests/test_alpaca_tools.py
-----------------------------
Verifies that core/alpaca_tools.py calls the *correct* Alpaca MCP tool
names with the *correct* argument shapes, and parses realistic Alpaca REST
API response bodies correctly -- both the flat shape and the key-wrapped
shape different alpaca-mcp-server versions have shipped (see
_unwrap_dict/_unwrap_list in core/alpaca_tools.py). core.mcp_client.call_tool
is monkeypatched with a recording stub -- no subprocess, no network, no
real credentials.

This is the most important safety test in the suite: a wrong tool name or
a wrong parameter (e.g. sending qty as an int instead of a string, or
mixing up buy/sell) at this layer would translate directly into a wrong
order against a real brokerage account.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.alpaca_tools as alpaca_tools


class _RecordingStub:
    """Records every call_tool invocation and returns a pre-canned response."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return self.response


class TestGetStockBars:
    def test_calls_correct_tool_and_args(self, monkeypatch):
        stub = _RecordingStub(
            {
                "bars": {
                    "AAPL": [
                        {"t": "2026-08-01T00:00:00Z", "o": 100.0, "h": 102.0, "l": 99.0, "c": 101.0, "v": 1000},
                        {"t": "2026-08-02T00:00:00Z", "o": 101.0, "h": 103.0, "l": 100.0, "c": 102.5, "v": 1200},
                    ]
                },
                "next_page_token": None,
            }
        )
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.get_stock_bars("AAPL", lookback_days=30)

        assert stub.calls == [("get_stock_bars", {"symbols": "AAPL", "timeframe": "1Day", "days": 30})]
        assert result["symbol"] == "AAPL"
        assert result["closes"] == [101.0, 102.5]
        assert result["highs"] == [102.0, 103.0]
        assert result["lows"] == [99.0, 100.0]
        assert result["volumes"] == [1000, 1200]

    def test_missing_symbol_returns_empty_series(self, monkeypatch):
        stub = _RecordingStub({"bars": {}, "next_page_token": None})
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.get_stock_bars("ZZZZ")
        assert result["closes"] == []
        assert result["highs"] == []


class TestGetLatestQuote:
    def test_calls_correct_tool_and_computes_mid(self, monkeypatch):
        stub = _RecordingStub({"quotes": {"AAPL": {"bp": 199.98, "ap": 200.02}}})
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.get_latest_quote("AAPL")

        assert stub.calls == [("get_stock_latest_quote", {"symbols": "AAPL"})]
        assert result["bid"] == 199.98
        assert result["ask"] == 200.02
        assert result["price"] == pytest.approx(200.0)

    def test_falls_back_to_available_side_if_one_missing(self, monkeypatch):
        stub = _RecordingStub({"quotes": {"AAPL": {"bp": None, "ap": 200.0}}})
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.get_latest_quote("AAPL")
        assert result["price"] == 200.0


class TestGetNews:
    def test_calls_correct_tool_and_parses_headlines(self, monkeypatch):
        stub = _RecordingStub(
            {
                "news": [
                    {"headline": "AAPL beats earnings", "source": "benzinga", "created_at": "2026-08-25T12:00:00Z"},
                ]
            }
        )
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.get_news("AAPL", limit=5)

        assert stub.calls == [("get_news", {"symbols": "AAPL", "limit": 5})]
        assert result[0]["headline"] == "AAPL beats earnings"
        assert result[0]["source"] == "benzinga"


class TestGetAccount:
    def test_calls_correct_tool_and_casts_to_float_flat_response(self, monkeypatch):
        # Some alpaca-mcp-server versions return the account fields flat.
        stub = _RecordingStub({"cash": "12345.67", "portfolio_value": "50000.00", "buying_power": "24691.34"})
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.get_account()

        assert stub.calls == [("get_account_info", {})]
        assert result == {"cash": 12345.67, "portfolio_value": 50000.0, "buying_power": 24691.34}
        assert isinstance(result["cash"], float)

    def test_also_handles_wrapped_response(self, monkeypatch):
        # Other versions wrap the account fields under an "account" key --
        # this used to be a silent KeyError; get_account() must handle both.
        stub = _RecordingStub(
            {"account": {"cash": "999.00", "portfolio_value": "1000.00", "buying_power": "999.00"}}
        )
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.get_account()
        assert result == {"cash": 999.0, "portfolio_value": 1000.0, "buying_power": 999.0}


class TestGetPositions:
    def test_calls_correct_tool_and_casts_to_float_flat_response(self, monkeypatch):
        stub = _RecordingStub(
            [{"symbol": "AAPL260116C00150000", "qty": "10", "avg_entry_price": "1.50", "market_value": "1550.00"}]
        )
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.get_positions()

        assert stub.calls == [("get_all_positions", {})]
        assert result == [
            {"symbol": "AAPL260116C00150000", "qty": 10.0, "avg_entry_price": 1.5, "market_value": 1550.0}
        ]

    def test_also_handles_wrapped_response(self, monkeypatch):
        stub = _RecordingStub(
            {"positions": [{"symbol": "AAPL", "qty": "5", "avg_entry_price": "150.0", "market_value": "750.0"}]}
        )
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.get_positions()
        assert result == [{"symbol": "AAPL", "qty": 5.0, "avg_entry_price": 150.0, "market_value": 750.0}]

    def test_empty_positions_list(self, monkeypatch):
        stub = _RecordingStub([])
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)
        assert alpaca_tools.get_positions() == []


class TestGetClock:
    def test_open_market_flat_response(self, monkeypatch):
        stub = _RecordingStub({"is_open": True, "next_open": "2026-09-03T13:30:00Z", "next_close": "2026-09-02T20:00:00Z"})
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.get_clock()
        assert stub.calls == [("get_market_clock", {})]
        assert result["is_open"] is True

    def test_closed_market_wrapped_response(self, monkeypatch):
        stub = _RecordingStub({"clock": {"is_open": False, "next_open": "2026-09-03T13:30:00Z", "next_close": None}})
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.get_clock()
        assert result["is_open"] is False
        assert result["next_open"] == "2026-09-03T13:30:00Z"


class TestGetOptionChain:
    def test_calls_correct_tool_with_expiry_window(self, monkeypatch):
        stub = _RecordingStub({"option_contracts": [{"symbol": "AAPL260116C00150000"}]})
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.get_option_chain("AAPL", min_days_to_expiry=15, max_days_to_expiry=45)

        tool_name, args = stub.calls[0]
        assert tool_name == alpaca_tools.OPTION_CONTRACTS_TOOL
        assert args["underlying_symbols"] == "AAPL"
        assert "expiration_date_gte" in args and "expiration_date_lte" in args
        assert result == [{"symbol": "AAPL260116C00150000"}]

    def test_filters_by_option_type_when_given(self, monkeypatch):
        stub = _RecordingStub({"option_contracts": []})
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        alpaca_tools.get_option_chain("AAPL", option_type="call")
        _, args = stub.calls[0]
        assert args["type"] == "call"


class TestGetOptionLatestQuote:
    def test_calls_correct_tool_and_parses_quote(self, monkeypatch):
        stub = _RecordingStub({"quotes": {"AAPL260116C00150000": {"bp": 4.9, "ap": 5.1}}})
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.get_option_latest_quote("AAPL260116C00150000")

        assert stub.calls == [(alpaca_tools.OPTION_QUOTE_TOOL, {"symbols": "AAPL260116C00150000"})]
        assert result["bid"] == 4.9
        assert result["ask"] == 5.1


class TestPlaceMarketOrder:
    def test_buy_order_uses_correct_tool_and_exact_params(self, monkeypatch):
        """
        Confirms place_market_order calls the real 'place_stock_order' MCP
        tool with exactly the parameters Alpaca's API expects -- qty as a
        STRING, type/time_in_force pinned to market/day, and side passed
        through verbatim (never inverted).
        """
        stub = _RecordingStub({"id": "order-123", "status": "accepted"})
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.place_market_order("AAPL", qty=10.5, side="buy")

        assert stub.calls == [
            (
                "place_stock_order",
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": "10.5",
                    "type": "market",
                    "time_in_force": "day",
                },
            )
        ]
        assert result["order_id"] == "order-123"
        assert result["side"] == "buy"
        assert result["status"] == "accepted"

    def test_sell_order_side_is_never_flipped(self, monkeypatch):
        stub = _RecordingStub({"id": "order-456", "status": "accepted"})
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        alpaca_tools.place_market_order("AAPL", qty=5, side="sell")

        called_tool, called_args = stub.calls[0]
        assert called_tool == "place_stock_order"
        assert called_args["side"] == "sell"


class TestPlaceOptionOrder:
    def test_buy_call_uses_correct_tool_and_exact_params(self, monkeypatch):
        """
        The single most safety-critical test for v3: confirms
        place_option_order sends the OCC contract symbol, an INTEGER
        contract count as a STRING, and defaults to side="buy" (OrbiTrade
        never writes naked options).
        """
        stub = _RecordingStub({"id": "order-789", "status": "accepted"})
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        result = alpaca_tools.place_option_order("AAPL260116C00150000", qty=2)

        assert stub.calls == [
            (
                alpaca_tools.OPTION_ORDER_TOOL,
                {
                    "symbol": "AAPL260116C00150000",
                    "side": "buy",
                    "qty": "2",
                    "type": "market",
                    "time_in_force": "day",
                },
            )
        ]
        assert result["order_id"] == "order-789"
        assert result["contract_symbol"] == "AAPL260116C00150000"
        assert result["status"] == "accepted"

    def test_qty_is_always_sent_as_string(self, monkeypatch):
        stub = _RecordingStub({"id": "order-999", "status": "accepted"})
        monkeypatch.setattr(alpaca_tools.mcp_client, "call_tool", stub)

        alpaca_tools.place_option_order("AAPL260116P00140000", qty=1, side="buy")

        _, called_args = stub.calls[0]
        assert isinstance(called_args["qty"], str)
        assert called_args["qty"] == "1"
