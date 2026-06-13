"""Runtime configuration.

Two layers:
- `settings` for general framework config + subprocess infrastructure.
- `Tier` + `TIER` for the pipeline's three-tier model routing.

Tier modes
----------
- `agentic` (Tier 1 / EXPERT): Claude Agent SDK with the full tool loop.
- `completion` (Tier 2 / STANDARD, Tier 3 / QUICK): one-shot OpenAI-compatible
  completions via the Nebius Token Factory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


class Tier(StrEnum):
    """Pipeline stages pin to a tier; the tier resolves to model + endpoint + budgets."""

    QUICK = "quick"  # tier 3 — Cosmos3-Super-Reasoner
    STANDARD = "standard"  # tier 2 — Nemotron-3-Ultra-550b-a55b
    EXPERT = "expert"  # tier 1 — Opus 4.8 high


TierMode = Literal["agentic", "completion"]


@dataclass(frozen=True)
class TierConfig:
    model: str
    mode: TierMode
    api_base: str | None
    api_key_env: str
    max_turns: int  # only meaningful in `agentic` mode
    wall_clock_s: int  # cap on a single stage call — abort with TimeoutError


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw else default


TIER: dict[Tier, TierConfig] = {
    Tier.EXPERT: TierConfig(
        model=_env("AGENTIC_TIER1_MODEL", "claude-opus-4-8-high"),
        mode="agentic",
        api_base=os.getenv("AGENTIC_TIER1_BASE_URL"),
        api_key_env=_env("AGENTIC_TIER1_API_KEY_ENV", "ANTHROPIC_API_KEY"),
        max_turns=_env_int("AGENTIC_TIER1_MAX_TURNS", 20),
        wall_clock_s=_env_int("AGENTIC_TIER1_WALL_CLOCK_S", 300),
    ),
    Tier.STANDARD: TierConfig(
        model=_env("AGENTIC_TIER2_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b"),
        mode="completion",
        api_base=_env(
            "AGENTIC_TIER2_BASE_URL",
            "https://api.tokenfactory.us-central1.nebius.com/v1/",
        ),
        api_key_env=_env("AGENTIC_TIER2_API_KEY_ENV", "NEBIUS_API_KEY"),
        max_turns=1,
        wall_clock_s=_env_int("AGENTIC_TIER2_WALL_CLOCK_S", 180),
    ),
    Tier.QUICK: TierConfig(
        model=_env("AGENTIC_TIER3_MODEL", "nvidia/Cosmos3-Super-Reasoner"),
        mode="completion",
        api_base=_env(
            "AGENTIC_TIER3_BASE_URL",
            "https://api.tokenfactory.nebius.com/v1/",
        ),
        api_key_env=_env("AGENTIC_TIER3_API_KEY_ENV", "NEBIUS_API_KEY"),
        max_turns=1,
        wall_clock_s=_env_int("AGENTIC_TIER3_WALL_CLOCK_S", 300),
    ),
}


def _hf_home() -> str | None:
    """Honour AGENTIC_HF_HOME first, then HF_HOME, else leave unset (subprocess
    inherits the user's default cache)."""
    return os.getenv("AGENTIC_HF_HOME") or os.getenv("HF_HOME") or None


@dataclass(frozen=True)
class Settings:
    # Free-form orchestrator (used by `agentic run` / `agentic run-goal`).
    orchestrator_model: str = field(
        default_factory=lambda: os.getenv("AGENTIC_ORCHESTRATOR_MODEL", "claude-opus-4-7")
    )

    # External-LLM dispatch tool (separate from the tier system).
    external_model: str = field(
        default_factory=lambda: os.getenv("AGENTIC_EXTERNAL_MODEL", "openai/gpt-4.1")
    )
    external_api_base: str | None = field(
        default_factory=lambda: os.getenv("AGENTIC_EXTERNAL_API_BASE")
    )
    max_turns: int = field(default_factory=lambda: int(os.getenv("AGENTIC_MAX_TURNS", "30")))

    # Pipeline state + events.
    state_dir: str = field(default_factory=lambda: os.getenv("AGENTIC_STATE_DIR", "state"))
    blocks_file: str = field(default_factory=lambda: os.getenv("AGENTIC_BLOCKS_FILE", "BLOCKS.md"))
    event_webhook: str | None = field(default_factory=lambda: os.getenv("AGENTIC_EVENT_WEBHOOK"))

    # Subprocess execution.
    gpu_count: int = field(default_factory=lambda: int(os.getenv("AGENTIC_GPU_COUNT", "2")))
    hf_home: str | None = field(default_factory=_hf_home)


settings = Settings()
