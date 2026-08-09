from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.parse import urlparse


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "gpt-4.1"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    api_key_present: bool
    source: str


def _append_v1_if_needed(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        return DEFAULT_BASE_URL

    parsed = urlparse(value)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        return value

    # Keep Azure-native or explicitly deployment-scoped URLs unchanged.
    lowered = path.lower()
    if "/openai/" in lowered or "/deployments" in lowered:
        return value

    return value + "/v1"


def _load_model_map() -> dict[str, str]:
    raw = os.getenv("ONEGRID_MODEL_MAP", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def resolve_llm_config() -> LLMConfig:
    """Resolve OpenAI-compatible LLM settings without exposing secrets."""
    openai_base = os.getenv("OPENAI_BASE_URL", "").strip()
    azure_base = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()

    if openai_base:
        base_url = openai_base
        source = "OPENAI_BASE_URL"
    elif azure_base:
        base_url = azure_base
        source = "AZURE_OPENAI_ENDPOINT"
    else:
        base_url = DEFAULT_BASE_URL
        source = "default_localhost"

    model_map = _load_model_map()
    model = (
        model_map.get("o3-mini")
        or model_map.get("gpt-4o")
        or os.getenv("OPENAI_MODEL", "").strip()
        or DEFAULT_MODEL
    )
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    api_key_present = bool(api_key)
    if not api_key:
        api_key = "dummy"

    return LLMConfig(
        base_url=_append_v1_if_needed(base_url),
        api_key=api_key,
        model=model,
        api_key_present=api_key_present,
        source=source,
    )


def configure_openai_environment(config: LLMConfig | None = None) -> LLMConfig:
    """Set compatibility env vars for clients that discover OpenAI settings implicitly."""
    resolved = config or resolve_llm_config()
    os.environ["OPENAI_BASE_URL"] = resolved.base_url
    os.environ.setdefault("OPENAI_API_KEY", resolved.api_key)
    return resolved
