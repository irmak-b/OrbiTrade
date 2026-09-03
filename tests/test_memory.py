"""
tests/test_memory.py
-----------------------
Tests for core/memory.py (Day-4 self-reflection lesson store). Every test
uses a tmp_path-scoped file so the real project-root trade_memory.json is
never touched by the test suite.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.memory import TradeMemory


class TestTradeMemory:
    def test_creates_empty_file_on_first_use(self, tmp_path):
        path = tmp_path / "trade_memory.json"
        assert not path.exists()
        TradeMemory(path)
        assert path.exists()
        assert path.read_text() == "[]"

    def test_save_and_read_back(self, tmp_path):
        store = TradeMemory(tmp_path / "trade_memory.json")
        store.save_trade_reflection(
            {"symbol": "AAPL", "strategy": "BUY_CALL", "contract_symbol": "AAPL260116C00150000", "pnl": None},
            "High-IV calls into earnings decayed faster than the thesis assumed.",
        )
        memories = store.get_all_memories()
        assert len(memories) == 1
        assert memories[0]["symbol"] == "AAPL"
        assert "decayed" in memories[0]["lesson"]

    def test_get_lessons_as_context_returns_none_when_empty(self, tmp_path):
        store = TradeMemory(tmp_path / "trade_memory.json")
        assert store.get_lessons_as_context("AAPL") is None

    def test_get_lessons_as_context_filters_by_symbol_and_limits(self, tmp_path):
        store = TradeMemory(tmp_path / "trade_memory.json")
        for i in range(5):
            store.save_trade_reflection({"symbol": "AAPL"}, f"Lesson {i}")
        store.save_trade_reflection({"symbol": "MSFT"}, "Unrelated MSFT lesson")

        context = store.get_lessons_as_context("AAPL", limit=3)
        assert context is not None
        assert "MSFT" not in context
        assert "Lesson 2" in context and "Lesson 3" in context and "Lesson 4" in context
        assert "Lesson 0" not in context  # only the most recent `limit` lessons

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "trade_memory.json"
        TradeMemory(path).save_trade_reflection({"symbol": "SPY"}, "First lesson.")
        # A brand-new TradeMemory instance pointed at the same file must see it.
        reopened = TradeMemory(path)
        assert reopened.get_lessons_as_context("SPY") == "- First lesson."
