import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    orchestrator_model: str = field(
        default_factory=lambda: os.getenv("AGENTIC_ORCHESTRATOR_MODEL", "claude-opus-4-7")
    )
    external_model: str = field(
        default_factory=lambda: os.getenv("AGENTIC_EXTERNAL_MODEL", "openai/gpt-4.1")
    )
    external_api_base: str | None = field(
        default_factory=lambda: os.getenv("AGENTIC_EXTERNAL_API_BASE")
    )
    max_turns: int = field(default_factory=lambda: int(os.getenv("AGENTIC_MAX_TURNS", "30")))


settings = Settings()
