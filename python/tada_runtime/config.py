"""Explicit provider profiles; no discovery of unrelated agent configuration."""
from __future__ import annotations
import os
from pathlib import Path
import tomllib
from typing import Annotated, Literal
from urllib.parse import urlparse
from pydantic import Field, model_validator
from .contracts import StrictModel, LongRowsConfig
from .large_contracts import LargeScaleConfig

class ProviderConfig(StrictModel):
    base_url: str
    model: Annotated[str, Field(min_length=1)]
    protocol: Literal['anthropic-messages', 'openai-responses', 'openai-chat']
    api_key_env: str | None = None
    auth_mode: Literal['required', 'optional'] = 'required'
    auth_placeholder: str | None = None
    max_output_tokens: Annotated[int, Field(gt=0, le=16384)] = 4096
    timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 120.0
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    reasoning_effort: Literal['minimal', 'low', 'medium', 'high', 'xhigh'] | None = None

    @model_validator(mode='after')
    def endpoint(self):
        parsed = urlparse(self.base_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('base_url must be an HTTP API prefix without credentials/query/fragment')
        if self.auth_mode == 'optional' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('optional authentication is only allowed for local endpoints')
        if self.base_url.rstrip('/').endswith('/v1/v1'):
            raise ValueError('duplicate API version prefix')
        return self

    def api_key(self) -> str | None:
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        key = key or self.auth_placeholder
        if not key and self.auth_mode == 'required':
            raise ValueError(f'required credential unavailable: {self.api_key_env or "api_key_env not configured"}')
        return key

class ModelsConfig(StrictModel):
    agent: ProviderConfig
    operators: ProviderConfig

    @model_validator(mode='after')
    def operator_limits(self):
        if self.operators.protocol == 'anthropic-messages':
            raise ValueError('operators support explicit Responses or Chat adapters')
        if self.operators.max_output_tokens > 8192:
            raise ValueError('operator output limit exceeds 8192')
        return self

class RuntimeConfig(StrictModel):
    mode: Literal['trusted-local'] = 'trusted-local'
    large_scale: LargeScaleConfig = Field(default_factory=LargeScaleConfig)
    operator_concurrency: Annotated[int, Field(ge=1, le=2)] = 1
    reuse: Literal['exact', 'off'] = 'exact'
    cell_timeout_seconds: Annotated[float, Field(gt=0, le=1800)] = 180.0
    output_view_chars: Annotated[int, Field(ge=100, le=100000)] = 8000
    batch_rows: Annotated[int, Field(ge=1, le=1000)] = 32
    batch_chars: Annotated[int, Field(ge=100, le=1000000)] = 16000
    reduce_fan_in: Annotated[int, Field(ge=2, le=20)] = 4
    # Extra attempts for one tag batch whose response failed the output contract (rid coverage,
    # labels, JSON). Each retry is a new budget-counted, audited request linked by retry_of.
    # Default 0 keeps "one dispatch per batch"; transport/provider failures are never retried here.
    batch_retries: Annotated[int, Field(ge=0, le=2)] = 0
    # Long rows are segmented inside tag; capacity against batch_chars is checked when segmentation is used.
    long_rows: LongRowsConfig = Field(default_factory=LongRowsConfig)

class LoggingConfig(StrictModel):
    level: Literal['full'] = 'full'
    format: Literal['jsonl'] = 'jsonl'
    record_provider_exposed_reasoning: Literal[True] = True
    max_unflushed_events: Literal[0] = 0

class BudgetConfig(StrictModel):
    ledger_path: str
    max_requests: Annotated[int, Field(ge=1, le=2000)]
    deadline_seconds: Annotated[float, Field(gt=0, le=86400)]

class Config(StrictModel):
    schema_version: Literal['1.0'] = '1.0'
    models: ModelsConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    budget: BudgetConfig | None = None

    def require_budget(self) -> BudgetConfig:
        if self.budget is None:
            raise ValueError('a finite request budget and deadline are required for inference')
        return self.budget

def _merge(target: dict, incoming: dict) -> dict:
    # This merges syntactically decoded TOML only. Config validates the final domain.
    from collections.abc import Mapping
    for key, value in incoming.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            target[key] = _merge(dict(target[key]), value)
        else:
            target[key] = value
    return target

def load_config(path: str | Path | None = None, *, user_path: str | Path | None = None, overrides: dict | None = None) -> Config:
    merged: dict = {}
    user = Path(user_path) if user_path else Path.home() / '.config' / 'tada' / 'config.toml'
    project = Path(path) if path else Path('tada.toml')
    for candidate in (user, project):
        if candidate.exists():
            with candidate.open('rb') as handle:
                _merge(merged, tomllib.load(handle))
        elif path is not None and candidate == project:
            raise FileNotFoundError(candidate)
    if overrides:
        _merge(merged, overrides)
    return Config.model_validate(merged)


def parse_overrides(values: list[str]) -> dict:
    """Decode CLI syntax; the merged Config remains the domain authority."""
    from .contracts import JSON
    result = {}
    for entry in values:
        key, separator, raw = entry.partition('=')
        if not separator or not key or any(not part for part in key.split('.')):
            raise ValueError('--set requires dotted.path=JSON_VALUE')
        cursor = result
        parts = key.split('.')
        for part in parts[:-1]:
            current = cursor.setdefault(part, {})
            if not isinstance(current, dict):
                raise ValueError('conflicting override paths')
            cursor = current
        cursor[parts[-1]] = JSON.validate_json(raw)
    return result
