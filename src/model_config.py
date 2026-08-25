"""OpenAI-compatible model provider configuration.

The project can run against TUD:AI/SCADS, OpenAI, or a local OpenAI-compatible
server such as Ollama. Environment variables intentionally stay simple:

- MODEL_PROVIDER=scads|openai|ollama
- MODEL_NAME=<model id>
- MODEL_BASE_URL=<OpenAI-compatible base URL override>
"""

from __future__ import annotations

import os
from dataclasses import dataclass

SCADS_BASE_URL = "https://llm.scads.ai/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    api_key: str
    base_url: str


def get_model_config(*, default_model: str) -> ModelConfig:
    provider = os.environ.get("MODEL_PROVIDER", "").strip().lower()
    if not provider:
        provider = "openai" if os.environ.get("OPENAI_API_KEY") else "scads"

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("MODEL_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL
        model = os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL") or default_model
    elif provider == "ollama":
        api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
        base_url = os.environ.get("MODEL_BASE_URL") or os.environ.get("OLLAMA_BASE_URL") or OLLAMA_BASE_URL
        model = os.environ.get("MODEL_NAME") or os.environ.get("OLLAMA_MODEL") or default_model
    elif provider == "scads":
        api_key = os.environ.get("SCADS_API_KEY")
        base_url = os.environ.get("MODEL_BASE_URL") or os.environ.get("SCADS_BASE_URL") or SCADS_BASE_URL
        model = os.environ.get("MODEL_NAME") or os.environ.get("SCADS_MODEL") or default_model
    else:
        raise ValueError("MODEL_PROVIDER must be one of: scads, openai, ollama.")

    if not api_key:
        key_name = {
            "openai": "OPENAI_API_KEY",
            "scads": "SCADS_API_KEY",
            "ollama": "OLLAMA_API_KEY",
        }[provider]
        raise EnvironmentError(f"Set {key_name} or choose another MODEL_PROVIDER.")

    return ModelConfig(provider=provider, model=model, api_key=api_key, base_url=base_url)
