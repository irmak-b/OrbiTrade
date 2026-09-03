"""
core/memory.py
-----------------
Day-4 "self-reflection / lesson memory" layer. This is deliberately the
simplest possible durable store (a single JSON file) rather than
SQLite/Chroma: for a hackathon demo, a database adds setup friction (an
extra binary/service, migration risk on the judges' machines) without
buying anything a flat JSON list of <100 trades won't -- it needs zero
config, is trivially inspectable (`cat trade_memory.json`), and will not
get in the way of wiring a dashboard on top of it later.

Storage: `trade_memory.json` at the project root (a plain JSON array of
records), kept intentionally distinct from `trade_log.jsonl` (JSON Lines,
written by agents/execution_agent.py) -- trade_log.jsonl is the raw,
append-only audit trail of every order attempt; trade_memory.json is the
curated, LLM-summarized *lessons* the Quant Agent re-reads before its next
decision (see agents/quant_agent.py::generate_thesis's `lessons_context`
argument, wired in by orchestration/graph.py::quant_node).

Thread-safety: OrbiTrade's LangGraph pipeline runs a single symbol at a
time on one thread, but the Streamlit dashboard may read memory from a
different thread while a CLI run is writing to it, so read-modify-write is
guarded by a module-level lock.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

MEMORY_FILE_PATH = Path(__file__).resolve().parents[1] / "trade_memory.json"

_lock = threading.Lock()


class TradeMemory:
    """Append-and-read store for post-trade lessons, keyed by symbol."""

    def __init__(self, file_path: Path | str = MEMORY_FILE_PATH):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            self.file_path.write_text("[]", encoding="utf-8")

    # -- writes ------------------------------------------------------------

    def save_trade_reflection(
        self,
        trade_data: dict[str, Any],
        reflection_lesson: str,
    ) -> None:
        """
        Appends one lesson record. `trade_data` is expected to carry at
        least `symbol`; `strategy` and `pnl` are optional (pnl is usually
        unknown at the moment a position is opened -- realized PnL is only
        known once the position is later closed, which is outside this
        pipeline's current scope -- so it is commonly logged as None and
        the lesson focuses on the *reasoning quality*, not the outcome).
        """
        with _lock:
            memories = self._read()
            memories.append(
                {
                    "symbol": trade_data.get("symbol"),
                    "strategy": trade_data.get("strategy"),
                    "contract_symbol": trade_data.get("contract_symbol"),
                    "pnl": trade_data.get("pnl"),
                    "lesson": reflection_lesson,
                }
            )
            self._write(memories)

    # -- reads ---------------------------------------------------------------

    def get_all_memories(self) -> list[dict[str, Any]]:
        with _lock:
            return self._read()

    def get_lessons_as_context(self, symbol: str, limit: int = 3) -> Optional[str]:
        """
        Returns the last `limit` lessons for `symbol` as a newline-bulleted
        string ready to drop into the Quant Agent's prompt, or None if
        there is no history yet (so quant_agent.py can omit the section
        entirely instead of printing a placeholder).
        """
        memories = self.get_all_memories()
        relevant = [m["lesson"] for m in memories if m.get("symbol") == symbol and m.get("lesson")]
        if not relevant:
            return None
        return "\n".join(f"- {lesson}" for lesson in relevant[-limit:])

    # -- internals -----------------------------------------------------------

    def _read(self) -> list[dict[str, Any]]:
        try:
            raw = self.file_path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else []
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, memories: list[dict[str, Any]]) -> None:
        self.file_path.write_text(json.dumps(memories, indent=2), encoding="utf-8")
