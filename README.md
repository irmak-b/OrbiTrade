# OrbiTrade v.2

A multi-agent, math-first trading system built on the Alpaca API/MCP, combining
deterministic quantitative computation (Black-Scholes, Greeks, technical
indicators, risk sizing) with LLM-based multi-agent debate for qualitative
reasoning.

## Architecture

```
                    ┌─────────────────────────────┐
                    │      CLI / Streamlit         │
                    │   (cli.py, dashboard/app.py) │
                    └──────────────┬──────────────┘
                                   │  run_pipeline(symbol)
                                   ▼
                    ┌─────────────────────────────────────────┐
                    │        orchestration/graph.py            │
                    │            (LangGraph StateGraph)         │
                    │                                            │
                    │  data ──► quant ──► risk ──► execution     │
                    │             ▲         │                    │
                    │             └─ debate loop (veto, retry) ─┘ │
                    │  any node failure ──► status="error" [END] │
                    └───────┬─────────┬─────────┬────────────────┘
                            │         │         │
                            ▼         ▼         ▼
                    core/math_engine.py   core/llm_factory.py
                    (Black-Scholes,       (Featherless: Mistral-7B,
                     Greeks, RSI, ATR,     Qwen2.5-72B/32B, DeepSeek-V3)
                     Kelly, VaR)
                            │
                            ▼
                    core/alpaca_tools.py
                            │  call_tool(name, args)
                            ▼
                    core/mcp_client.py  ──(stdio, JSON-RPC)──►  alpaca-mcp-server
                                                                  (official Alpaca package)
                                                                          │
                                                                          ▼
                                                          Alpaca Paper Trading API
```

- **Quant Engine (math side)** eliminates the risk of LLM mathematical
  hallucination: option prices, Greek sensitivities, technical indicators,
  and risk sizing are computed directly by deterministic Python functions
  in `core/math_engine.py` -- never by an LLM.
- **Real Alpaca MCP integration**: `core/mcp_client.py` spawns the official,
  PyPI-published [`alpaca-mcp-server`](https://pypi.org/project/alpaca-mcp-server/)
  as a subprocess and talks to it over stdio using the standard `mcp`
  Python SDK -- the same protocol Claude Desktop, Cursor, or VS Code would
  use. `core/alpaca_tools.py` exposes plain Python functions
  (`get_stock_bars`, `get_account`, `place_market_order`, ...) that call
  the server's real tools (`get_stock_bars`, `get_account_info`,
  `place_stock_order`, ...) under the hood, so `agents/*.py` never touches
  Alpaca's REST API or the MCP protocol directly.
- **Debate & Risk Agent** challenges the Quant Agent's thesis
  (TradingAgents-style Devil's Advocate), runs a deterministic Greek/
  volatility threshold filter *before* ever calling an LLM, and estimates
  the win-probability / win-loss-ratio inputs Kelly sizing needs.
- **LangGraph orchestration** (`orchestration/graph.py`) wires the four
  agents into one autonomous pipeline with a bounded Quant<->Risk debate
  loop, and fails *closed*: any exception anywhere in the pipeline routes
  straight to `status="error"` and never reaches the execution node.
- **Execution Agent** applies the deterministic Kelly + position-cap safety
  ceiling and is the only module allowed to call `place_market_order`.

## Folder structure

```
OrbiTrade-v2/
├── agents/
│   ├── data_agent.py          # Alpaca MCP -> MarketState (build_market_state is pure/testable)
│   ├── quant_agent.py         # Thesis synthesis, revises itself on a Risk Agent objection
│   ├── risk_agent.py          # Deterministic Greek/vol filter + Devil's Advocate LLM debate
│   └── execution_agent.py     # Kelly sizing, position cap, places the order via alpaca_tools
├── core/
│   ├── math_engine.py         # Black-Scholes, Greeks, ATR, RSI, VaR calculations
│   ├── schemas.py             # Shared Pydantic contracts between agents (MarketState, QuantThesis, ...)
│   ├── state.py               # LangGraph AgentState TypedDict
│   ├── llm_factory.py         # Featherless model routing per agent role
│   ├── mcp_client.py          # Real MCP client: spawns & talks to alpaca-mcp-server over stdio
│   ├── alpaca_tools.py        # Alpaca-facing functions, implemented as real MCP tool calls
│   └── memory.py              # Self-improvement / reflection storage (Day 4)
├── orchestration/
│   └── graph.py               # LangGraph StateGraph: data -> quant -> risk -> execution + debate loop
├── dashboard/
│   └── app.py                 # Streamlit monitoring panel
├── cli.py                     # argparse CLI: `python cli.py run --symbol AAPL`
├── tests/
│   ├── test_math_engine.py    # 35 tests for the math engine
│   ├── test_agents_logic.py   # 16 tests for deterministic agent logic (Kelly sizing, Greek filter, ...)
│   ├── test_alpaca_tools.py   # 13 tests locking in exact MCP tool names/argument shapes
│   └── test_graph.py          # 8 tests for LangGraph routing, the debate loop, and fail-closed behavior
├── trade_log.jsonl            # Created at runtime by execution_agent.py (not shipped in the repo)
├── .env.example
├── config.py
├── requirements.txt
└── README.md
```

## Hackathon requirements checklist

| Requirement | Where it's satisfied |
|---|---|
| **Trading API** | `core/alpaca_tools.py` calls the real Alpaca Trading/Market Data API (bars, quotes, news, account, positions, orders) |
| **MCP server** | `core/mcp_client.py` spawns the official [`alpaca-mcp-server`](https://pypi.org/project/alpaca-mcp-server/) and talks JSON-RPC over stdio, exactly as Claude Desktop/Cursor/VS Code would |
| **Alpaca CLI** | `cli.py` -- `python cli.py run --symbol AAPL`, `python cli.py account`, `python cli.py positions` |
| **Paper trading environment** | `ALPACA_PAPER_TRADE` defaults to `true`; no real money is touched unless explicitly changed |
| **Options trading** | Every strategy trades options, not equity: `core/alpaca_tools.py::get_option_chain/place_option_order`, priced via `core/math_engine.py`'s Black-Scholes/Greeks in `agents/data_agent.py`, and `core/schemas.py::QuantThesis.recommended_action` is restricted to `BUY_CALL \| BUY_PUT \| HOLD` -- there is no bare-equity fallback |

## Agent -> model mapping (Featherless, license-unrestricted)

| Agent | Model | Why |
|---|---|---|
| `data_agent.py` | `mistralai/Mistral-7B-Instruct-v0.3` | fast news/JSON formatting, no gated license |
| `quant_agent.py` | `Qwen/Qwen2.5-72B-Instruct` | strong mathematical reasoning & thesis writing |
| `risk_agent.py` | `deepseek-ai/DeepSeek-V3` | adversarial "Devil's Advocate" review |
| `execution_agent.py` | `Qwen/Qwen2.5-32B-Instruct` | tight JSON-schema adherence, deterministic output |

All four share a single `FEATHERLESS_API_KEY` (see `.env.example`) via `core/llm_factory.py`.

## Getting started

```bash
pip install -r requirements.txt
pytest tests/ -v

# Run the full pipeline for one symbol (spawns the real alpaca-mcp-server
# under the hood; needs ALPACA_API_KEY/ALPACA_SECRET_KEY/FEATHERLESS_API_KEY
# set in .env)
python cli.py run --symbol AAPL --verbose

# Check the paper account / open positions directly
python cli.py account
python cli.py positions

# Visual dashboard
streamlit run dashboard/app.py
```

`run_pipeline()` / `stream_pipeline()` in `orchestration/graph.py` are the
single entry points into the whole system -- `cli.py`, `dashboard/app.py`,
or any future API endpoint all call the exact same function with one line:

```python
from orchestration.graph import run_pipeline
final_state = run_pipeline("AAPL", max_debate_rounds=2)
print(final_state["status"])  # "approved" | "no_trade" | "rejected" | "error"
```

## Roadmap (6-day build)

| Day | Focus | Deliverable |
|-----|-------|-------------|
| 1 | Core math engine | `core/math_engine.py`, fully tested (35/35 passing) |
| 2 | Data & tool integration | `agents/data_agent.py` wired to Alpaca MCP |
| 3 | Agent orchestration & debate | LangGraph state graph, Quant vs. Devil's Advocate |
| 4 | Feedback loop & order execution | Reflection memory + execution hardening |
| 5 | Dashboard | Streamlit panel with reasoning traces, Greeks, portfolio |
| 6 | Test, video & submission | End-to-end stress test, architecture diagram, demo video |

## Status

- [x] Day 1 — Math engine complete, 35/35 tests passing
- [x] Day 2 — All four agents scaffolded, Featherless model routing (`core/llm_factory.py`),
      shared Pydantic schemas (`core/schemas.py`). Deterministic logic (position sizing,
      Greek-risk filter, market-state assembly) is unit tested.
- [x] Day 3 — **Real Alpaca MCP integration** (`core/mcp_client.py` + `core/alpaca_tools.py`,
      talking to the official `alpaca-mcp-server` over stdio -- verified end-to-end with a live
      handshake and tool call), LangGraph orchestration (`core/state.py` + `orchestration/graph.py`)
      with a bounded Quant<->Risk debate loop and fail-closed error handling, and a `cli.py`
      terminal interface. 70/70 tests passing (35 math + 16 agent logic + 13 MCP tool-mapping +
      8 graph routing). LLM-calling and live-Alpaca branches require real `FEATHERLESS_API_KEY` /
      `ALPACA_API_KEY` credentials and are not exercised in CI/tests -- everything up to that
      boundary (tool names, argument shapes, routing, debate loop, error propagation) is.
- [x] Day 4 — **Options trading (v3) + reflection loop.** `agents/data_agent.py` now fetches
      and prices a live Alpaca option chain (Black-Scholes IV + Greeks per candidate contract);
      `agents/quant_agent.py`/`risk_agent.py`/`execution_agent.py` are options-first end to end
      (`BUY_CALL`/`BUY_PUT`/`HOLD`, whole-contract Kelly sizing, Greek-risk veto on the selected
      contract). `core/memory.py` (`trade_memory.json`) stores one-sentence lessons generated by
      `agents/quant_agent.py::generate_reflection` after every executed trade; `orchestration/graph.py`
      runs this as a `reflection_node` after `execution_node`, and every subsequent `quant_node`
      call re-reads that symbol's lessons before drafting its next thesis. `agents/execution_agent.py`
      adds fail-safe handling for a closed market or any Alpaca/MCP error (clean rejection, never a
      crash). `cli.py account`/`positions` no longer leak raw tracebacks, and a new `cli.py memory`
      command inspects the lesson store.
- [x] Day 5 — Dashboard first version (shipped early, alongside Day 1; still math-engine-only,
      not yet wired to live option-chain output -- see "Next up" below)
- [ ] Day 6 — Final testing & submission

### Next up
- Wire `dashboard/app.py` to `orchestration/graph.py::run_pipeline` so the panel shows live
  reasoning traces / selected contracts / reflections instead of only the standalone math engine.
- Verify the exact MCP tool names in `core/alpaca_tools.py` (`OPTION_CONTRACTS_TOOL`,
  `OPTION_QUOTE_TOOL`, `OPTION_ORDER_TOOL`, `CLOCK_TOOL`) against your installed
  `alpaca-mcp-server` version with `core/mcp_client.list_tools()` -- see that file's docstring.
