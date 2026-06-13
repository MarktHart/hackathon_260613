"""Dispatch a task to an external (non-Claude) LLM via LiteLLM.

LiteLLM normalises the call shape across OpenAI, Gemini, Bedrock, Vertex, Ollama,
local OpenAI-compatible endpoints, etc. The model id follows LiteLLM's
`provider/model` convention (e.g. `openai/gpt-4.1`, `gemini/gemini-2.5-pro`,
`ollama/llama3.1`).
"""

from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool
from litellm import acompletion

from agentic.config import settings


@tool(
    name="dispatch_to_external_llm",
    description=(
        "Send a self-contained prompt to an external LLM and return its text response. "
        "Use when a task is better suited to a different model (cost, latency, "
        "specialised capability) or when you need a second opinion. The prompt must "
        "be fully self-contained — the external model does not see this conversation."
    ),
    input_schema={
        "prompt": str,
        "model": str,
        "system": str,
        "temperature": float,
    },
)
async def dispatch_to_external_llm(args: dict[str, Any]) -> dict[str, Any]:
    model = args.get("model") or settings.external_model
    messages: list[dict[str, str]] = []
    if system := args.get("system"):
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": args["prompt"]})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": args.get("temperature", 0.2),
    }
    if settings.external_api_base:
        kwargs["api_base"] = settings.external_api_base

    response = await acompletion(**kwargs)
    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    return {
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "_meta": {
            "model": model,
            "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else None,
        },
    }


external_llm_mcp_server = create_sdk_mcp_server(
    name="external-llm",
    version="0.0.1",
    tools=[dispatch_to_external_llm],
)
