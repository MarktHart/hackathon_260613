"""Smoke tests — no network, no SDK calls. Just verify wiring."""

from agentic.agents import DEFAULT_AGENTS
from agentic.config import settings
from agentic.tools import external_llm_mcp_server


def test_default_agents_present() -> None:
    assert "external-dispatcher" in DEFAULT_AGENTS
    assert "researcher" in DEFAULT_AGENTS


def test_external_llm_mcp_server_built() -> None:
    assert external_llm_mcp_server is not None


def test_settings_have_defaults() -> None:
    assert settings.orchestrator_model
    assert settings.external_model
    assert settings.max_turns > 0
