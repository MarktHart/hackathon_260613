"""Smoke tests — no network, no SDK calls. Just verify wiring."""

from agentic.agents import DEFAULT_AGENTS
from agentic.config import TIER, Tier, settings
from agentic.tools import external_llm_mcp_server


def test_default_agents_present() -> None:
    assert "external-dispatcher" in DEFAULT_AGENTS
    assert "researcher" in DEFAULT_AGENTS
    assert "experimenter" in DEFAULT_AGENTS


def test_external_llm_mcp_server_built() -> None:
    assert external_llm_mcp_server is not None


def test_settings_have_defaults() -> None:
    assert settings.orchestrator_model
    assert settings.external_model
    assert settings.max_turns > 0


def test_tier_table_complete() -> None:
    assert set(TIER) == {Tier.QUICK, Tier.STANDARD, Tier.EXPERT}
    assert TIER[Tier.EXPERT].mode == "agentic"
    assert TIER[Tier.STANDARD].mode == "completion"
    assert TIER[Tier.QUICK].mode == "completion"
    # Nebius endpoints for tier 2/3 unless overridden.
    standard_base = TIER[Tier.STANDARD].api_base
    quick_base = TIER[Tier.QUICK].api_base
    assert standard_base is not None and "nebius" in standard_base
    assert quick_base is not None and "nebius" in quick_base
