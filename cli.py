"""
cli.py
--------
Terminal CLI for OrbiTrade -- built for the hackathon demo and for quick
manual triggers, exactly like Alpaca's own CLI philosophy (structured JSON
output, no interactive prompts, safe to script/cron).

Usage:
    python cli.py run --symbol AAPL
    python cli.py run --symbol AAPL --max-debate-rounds 3 --verbose
    python cli.py account
    python cli.py positions
    python cli.py memory --symbol AAPL

Every command prints structured JSON to stdout and returns a non-zero exit
code on failure, so it's safe to pipe/script. `account` and `positions`
previously let raw exceptions (missing ALPACA_API_KEY, a closed MCP
session, a malformed Alpaca response) propagate as unhandled Python
tracebacks -- they are now caught here and reported as
`{"error": "..."}`  JSON instead, matching every other command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from core import alpaca_tools, memory
from core.state import initial_state as _initial_state
from orchestration.graph import run_pipeline, stream_pipeline


def _state_to_json(state: dict) -> dict:
    """Converts the final AgentState (which holds Pydantic model instances)
    into a plain JSON-serializable dict for CLI output."""
    return {
        "symbol": state.get("symbol"),
        "status": state.get("status"),
        "error": state.get("error"),
        "debate_round": state.get("debate_round"),
        "max_debate_rounds": state.get("max_debate_rounds"),
        "thesis": state["thesis"].model_dump() if state.get("thesis") else None,
        "verdict": state["verdict"].model_dump() if state.get("verdict") else None,
        "execution_result": (
            state["execution_result"].model_dump() if state.get("execution_result") else None
        ),
        "reflection": state.get("reflection"),
        "reflection_error": state.get("reflection_error"),
    }


def cmd_run(args: argparse.Namespace) -> int:
    symbol = args.symbol.upper()

    try:
        if args.verbose:
            print(f"Running OrbiTrade pipeline for {symbol} (max_debate_rounds={args.max_debate_rounds})...\n")
            final_state: dict = dict(_initial_state(symbol, max_debate_rounds=args.max_debate_rounds))
            for chunk in stream_pipeline(symbol, max_debate_rounds=args.max_debate_rounds):
                for node_name, node_update in chunk.items():
                    if node_update is None:
                        continue  # some LangGraph versions emit a sentinel chunk with no update here
                    print(f"[{node_name}] -> {list(node_update.keys())}")
                    final_state = {**final_state, **node_update}
        else:
            final_state = run_pipeline(symbol, max_debate_rounds=args.max_debate_rounds)
    except Exception as exc:
        # Fail-safe: a bug/crash anywhere the graph's own try/except didn't
        # catch (e.g. LangGraph/MCP session setup itself) still prints
        # structured JSON instead of a raw traceback.
        print(json.dumps({"symbol": symbol, "status": "error", "error": str(exc)}, indent=2))
        return 1

    print(json.dumps(_state_to_json(final_state), indent=2, default=str))

    return 0 if final_state.get("status") != "error" else 1


def cmd_account(args: argparse.Namespace) -> int:
    try:
        account = alpaca_tools.get_account()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    print(json.dumps(account, indent=2))
    return 0


def cmd_positions(args: argparse.Namespace) -> int:
    try:
        positions = alpaca_tools.get_positions()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    print(json.dumps(positions, indent=2))
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    """Day-4: inspect the self-reflection lesson store (trade_memory.json)."""
    try:
        store = memory.TradeMemory()
        if args.symbol:
            symbol = args.symbol.upper()
            lessons = store.get_lessons_as_context(symbol, limit=args.limit)
            print(json.dumps({"symbol": symbol, "lessons": lessons}, indent=2))
        else:
            print(json.dumps(store.get_all_memories(), indent=2))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orbitrade",
        description="OrbiTrade -- a math-first, multi-agent options trading system on Alpaca.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full pipeline for one symbol.")
    run_parser.add_argument("--symbol", required=True, help="Ticker to analyze, e.g. AAPL.")
    run_parser.add_argument(
        "--max-debate-rounds",
        type=int,
        default=2,
        help="Max number of Quant<->Risk debate rounds before giving up (default: 2).",
    )
    run_parser.add_argument(
        "--verbose", action="store_true", help="Print each pipeline stage as it completes."
    )
    run_parser.set_defaults(func=cmd_run)

    account_parser = subparsers.add_parser("account", help="Show the Alpaca paper account balance.")
    account_parser.set_defaults(func=cmd_account)

    positions_parser = subparsers.add_parser("positions", help="Show currently open positions.")
    positions_parser.set_defaults(func=cmd_positions)

    memory_parser = subparsers.add_parser(
        "memory", help="Show self-reflection lessons recorded from past trades (Day 4)."
    )
    memory_parser.add_argument(
        "--symbol", help="Only show lessons for this symbol. Omit to dump the full memory store."
    )
    memory_parser.add_argument(
        "--limit", type=int, default=3, help="Max lessons to show when --symbol is given (default: 3)."
    )
    memory_parser.set_defaults(func=cmd_memory)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())