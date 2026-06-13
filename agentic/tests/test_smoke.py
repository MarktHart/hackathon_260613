"""Smoke tests — no network, no SDK calls. Just verify wiring."""

from agentic.config import TIER, Tier, settings


def test_settings_have_pipeline_defaults() -> None:
    assert settings.state_dir
    assert settings.blocks_file
    assert settings.gpu_count > 0


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
