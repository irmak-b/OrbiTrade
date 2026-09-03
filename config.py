"""
config.py
---------
OrbiTrade v.2 - Central configuration. Actual secrets are read from a .env
file (see .env.example); this file only defines defaults and risk limits.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Alpaca API ---
# Read here only for convenience/validation elsewhere in the app. The real
# alpaca-mcp-server subprocess (spawned by core/mcp_client.py) reads
# ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_PAPER_TRADE directly from the
# process environment itself -- there is no URL to configure since it's a
# local stdio subprocess, not an HTTP service.
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# --- LLM APIs (hybrid model architecture) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FEATHERLESS_API_KEY = os.getenv("FEATHERLESS_API_KEY", "")
FEATHERLESS_BASE_URL = os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")

# Agent -> model routing lives in core/llm_factory.py (the single source of
# truth) -- it is not duplicated here to avoid the two ever drifting apart.

# --- Risk Limits ---
MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "0.10"))  # max 10% of portfolio
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.5"))  # Half-Kelly default
VAR_CONFIDENCE = float(os.getenv("VAR_CONFIDENCE", "0.95"))
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))

# --- Watchlist ---
WATCHLIST = os.getenv("WATCHLIST", "AAPL,MSFT,SPY").split(",")

# --- Technical Indicator Parameters ---
RSI_PERIOD = 14
ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_NUM_STD = 2.0

# --- Options Trading (v3) ---
# Expiration window the Data Agent pulls candidate contracts from. 15-45
# days is a common sweet spot for directional swing strategies: enough
# time for a thesis to play out, without the Theta decay of very
# short-dated (<15d) contracts or the high capital cost of far-dated ones.
OPTION_MIN_DAYS_TO_EXPIRY = int(os.getenv("OPTION_MIN_DAYS_TO_EXPIRY", "15"))
OPTION_MAX_DAYS_TO_EXPIRY = int(os.getenv("OPTION_MAX_DAYS_TO_EXPIRY", "45"))
# Cap on how many priced contracts are handed to the Quant Agent per
# symbol -- keeps the LLM prompt small and decision-relevant (nearest the
# money first; see agents/data_agent.py::_price_option_chain).
OPTION_MAX_CANDIDATES = int(os.getenv("OPTION_MAX_CANDIDATES", "6"))
