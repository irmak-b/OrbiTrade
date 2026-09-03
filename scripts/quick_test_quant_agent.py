"""
scripts/quick_test_quant_agent.py
------------------------------------
Standalone smoke test for quant_agent.py that does NOT touch Alpaca at all.
Use this before you've set up an Alpaca paper trading account, to confirm
your FEATHERLESS_API_KEY and the quant model are working end to end.

It builds a MarketState from synthetic bars (via agents.data_agent's pure,
network-free build_market_state function) and sends it straight to
quant_agent.generate_thesis().

Run with:
    python scripts/quick_test_quant_agent.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.data_agent import build_market_state
from agents.quant_agent import generate_thesis

# --- Fake price history: 60 synthetic daily bars with a mild uptrend ---
rng = np.random.default_rng(42)
closes = list(100 + np.cumsum(rng.normal(0.1, 1.0, 60)))
highs = [c + abs(rng.normal(0.5, 0.2)) for c in closes]
lows = [c - abs(rng.normal(0.5, 0.2)) for c in closes]

fake_news = [
    {"headline": "Company beats quarterly earnings estimates", "source": "Reuters", "published_at": "2026-08-25"},
]

market_state = build_market_state(
    symbol="TEST",
    closes=closes,
    highs=highs,
    lows=lows,
    latest_price=closes[-1],
    news=fake_news,
)

print("=== MarketState (synthetic, no Alpaca call) ===")
print(market_state.model_dump_json(indent=2))

print("\n=== Calling quant_agent (Featherless -> Qwen2.5-72B) ===")
thesis = generate_thesis(market_state)

print("\n=== QuantThesis ===")
print(thesis.model_dump_json(indent=2))
