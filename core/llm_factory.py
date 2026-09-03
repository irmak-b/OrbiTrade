"""
core/llm_factory.py
---------------------
Routes each agent role to its assigned Featherless-hosted open-source model,
per the OrbiTrade hybrid-model plan. Every agent shares a single
FEATHERLESS_API_KEY and talks to Featherless's OpenAI-compatible endpoint
via langchain_openai.ChatOpenAI, so swapping a model later is a one-line
change here rather than a change in every agent file.

Model assignment rationale (all license-unrestricted on Featherless):
    data      -> Mistral-7B-Instruct-v0.3     : fast news/JSON formatting, no gated license
    quant     -> Qwen2.5-72B-Instruct         : strong mathematical reasoning / thesis writing
    risk      -> DeepSeek-V3                  : deep, adversarial "Devil's Advocate" review
    execution -> Qwen2.5-32B-Instruct         : tight JSON-schema adherence, deterministic output

Low temperature (0.1) is used for roles that must stick tightly to a schema
(data, execution); slightly higher temperature (0.3) is used for roles that
need to *reason* over ambiguous evidence (quant, risk).
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
load_dotenv() 

from langchain_openai import ChatOpenAI

AgentRole = Literal["data", "quant", "risk", "execution"]

MODEL_MAP: dict[AgentRole, str] = {
    "data": "mistralai/Mistral-7B-Instruct-v0.3",
    "quant": "Qwen/Qwen2.5-72B-Instruct",
    "risk": "deepseek-ai/DeepSeek-V3-0324",
    "execution": "Qwen/Qwen2.5-32B-Instruct",
}

TEMPERATURE_MAP: dict[AgentRole, float] = {
    "data": 0.1,
    "quant": 0.3,
    "risk": 0.3,
    "execution": 0.1,
}


def get_agent_llm(role: AgentRole, **overrides) -> ChatOpenAI:
    """
    Returns a ChatOpenAI client pointed at Featherless, configured for the
    given agent role.

    Args:
        role: One of "data", "quant", "risk", "execution".
        **overrides: Any ChatOpenAI kwarg (e.g. temperature=0.0) to override
                     the role's default for a specific call.

    Raises:
        ValueError: If `role` isn't one of the known agent roles.
        EnvironmentError: If FEATHERLESS_API_KEY isn't set.
    """
    if role not in MODEL_MAP:
        raise ValueError(f"Unknown agent role '{role}'. Expected one of {list(MODEL_MAP)}.")

    api_key = os.getenv("FEATHERLESS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FEATHERLESS_API_KEY is not set. Add it to your .env file "
            "before instantiating an agent LLM."
        )

    base_url = os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")

    params = dict(
        model=MODEL_MAP[role],
        base_url=base_url,
        api_key=api_key,
        temperature=TEMPERATURE_MAP[role],
    )
    params.update(overrides)

    return ChatOpenAI(**params)
