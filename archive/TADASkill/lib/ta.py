"""Text Analysis primitives for tabular data.

Public API
----------
tag(data, column_name, query) -> list
    Row-level labeling / transformation. Returns one value per row.

categorize(data, column_name, query) -> list[dict]
    Dataset-level summarization. Returns [{"name": ..., "description": ...}, ...].

materialize_tag(data, column_name, query, output_column, ...) -> SemanticState
    Runs tagging and returns a typed, row-aligned semantic-state object with
    validation/provenance rows and a manifest.

Each call may trigger multiple backend subcalls (intent detection, per-batch
tagging / categorization, cross-batch consolidation). Reported
runs use the OpenAI-compatible backend with `gpt-4.1` and local fallback
disabled; the Claude CLI backend remains a supported historical/local path.
The orchestrator sees only the public API; the underlying prompts live in
../prompts/ and are never surfaced back to the caller.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from shutil import which
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError:
    pd = None  # pandas is optional; list[str] input also supported


# -----------------------------------------------------------------------------
# Runtime config and defaults.
# -----------------------------------------------------------------------------


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value >= min_value else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


MAX_ROWS_PER_TAGGING_BATCH = _env_int("TA_TAG_BATCH_ROWS", 50)
MAX_ROWS_PER_CATEGORIZATION_BATCH = _env_int("TA_CATEGORIZE_BATCH_ROWS", 1000)
_MIN_SPLIT_BATCH_ROWS = _env_int("TA_MIN_SPLIT_BATCH_ROWS", 25)

_SKILL_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_DIR = _SKILL_ROOT / "prompts"

_PROMPT_TAGGING = _PROMPT_DIR / "tagging.md"
_PROMPT_CATEGORIZATION = _PROMPT_DIR / "categorization.md"
_PROMPT_INTENT = _PROMPT_DIR / "tagging_intent_detection.md"
_PROMPT_CONSOLIDATION = _PROMPT_DIR / "category_consolidation.md"

_DEFAULT_MODEL = os.environ.get("TA_MODEL", "haiku")
_FALLBACK_MODEL = os.environ.get("TA_FALLBACK_MODEL", "").strip() or None
_BACKEND = os.environ.get("TA_BACKEND", "auto").strip().lower()
_OPENAI_MODEL = os.environ.get("TA_OPENAI_MODEL", "").strip() or os.environ.get("OPENAI_MODEL", "").strip() or os.environ.get("TADA_LIVE_MODEL", "").strip() or "gpt-4.1"
_OPENAI_BASE_URL = os.environ.get("TA_OPENAI_BASE_URL", "").strip() or os.environ.get("OPENAI_BASE_URL", "").strip()
_OPENAI_API_KEY = os.environ.get("TA_OPENAI_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
_OPENAI_WIRE_API = (
    os.environ.get("TA_OPENAI_WIRE_API", "").strip()
    or os.environ.get("OPENAI_WIRE_API", "").strip()
    or "chat"
).lower()
_OPENAI_REQUIRES_AUTH = _env_bool("TA_OPENAI_REQUIRES_AUTH", False)
_MAX_CONCURRENCY = _env_int("TA_MAX_CONCURRENCY", 8)
_PER_CALL_TIMEOUT_S = _env_int("TA_CALL_TIMEOUT_S", 300)
_OPENAI_MAX_TOKENS = _env_int("TA_OPENAI_MAX_TOKENS", 4000)
_MAX_RETRIES = _env_int("TA_MAX_RETRIES", 2, min_value=0)
_RETRY_BACKOFF_S = float(os.environ.get("TA_RETRY_BACKOFF_S", "1.0"))
_ENABLE_CACHE = _env_bool("TA_ENABLE_CACHE", False)
_FORCE_INDEPENDENT_TAGGING = _env_bool("TA_FORCE_INDEPENDENT_TAGGING", False)
_ALLOW_RAW_SINGLE_TAG_CELL = _env_bool("TA_ALLOW_RAW_SINGLE_TAG_CELL", False)
_ENABLE_DEDUP = _env_bool("TA_ENABLE_DEDUP", True)
_DEDUP_MIN_ROWS = _env_int("TA_DEDUP_MIN_ROWS", 2000)
_CONSOLIDATE_SET_LIMIT = _env_int("TA_CONSOLIDATE_SET_LIMIT", 24)
_CACHE_MODE = os.environ.get("TA_CACHE_MODE", "single").lower()
_CACHE_PATH = Path(
    os.environ.get(
        "TA_CACHE_PATH",
        str(_SKILL_ROOT / "workspace" / "ta_cache.json"),
    )
)
_CACHE_DIR = Path(
    os.environ.get(
        "TA_CACHE_DIR",
        str(_SKILL_ROOT / "workspace" / "ta_cache"),
    )
)

_DEBUG_TRACE_ENABLED = _env_bool("TA_DEBUG_TRACE", False)
_DEBUG_TRACE_PATH = Path(
    os.environ.get(
        "TA_DEBUG_TRACE_PATH",
        str(_SKILL_ROOT / "workspace" / "ta_debug_trace.jsonl"),
    )
)
_LOCAL_FALLBACK_ENABLED = _env_bool("TA_ENABLE_LOCAL_FALLBACK", False)


class TextAnalysisError(RuntimeError):
    pass


class _RunStats:
    """Lightweight runtime statistics for a tag/categorize operation."""

    def __init__(self):
        self.start_time = time.time()
        self.rows_processed = 0
        self.batches_sent = 0
        self.dedup_ratio = 1.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.retries_used = 0
        self.subcall_errors = 0
        self.openai_calls = 0
        self.openai_prompt_tokens = 0
        self.openai_completion_tokens = 0
        self.openai_total_tokens = 0

    def elapsed_s(self) -> float:
        return time.time() - self.start_time

    def summary_line(self) -> str:
        dedup_pct = 100.0 * (1.0 - self.dedup_ratio)
        cache_rate = (
            100.0 * self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0.0
        )
        return (
            f"[ta.stats] rows={self.rows_processed} batches={self.batches_sent} "
            f"dedup={dedup_pct:.1f}% cache_hit={cache_rate:.0f}% "
            f"retries={self.retries_used} errors={self.subcall_errors} "
            f"openai_calls={self.openai_calls} tokens={self.openai_total_tokens} "
            f"elapsed={self.elapsed_s():.2f}s"
        )


def _trace_shorten(value: Any, limit: int = 800) -> Any:
    """Best-effort truncation for trace safety and readability."""
    try:
        if isinstance(value, str):
            return value if len(value) <= limit else (value[:limit] + "...<truncated>")
        s = json.dumps(value, ensure_ascii=False)
        if len(s) <= limit:
            return value
        return s[:limit] + "...<truncated>"
    except Exception:
        return str(value)[:limit]


def _trace_log(event: dict[str, Any]) -> None:
    if not _DEBUG_TRACE_ENABLED:
        return
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    try:
        _DEBUG_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        return


class _JsonCache:
    """Thread-safe JSON object store for optional sub-call caching.

    Supports two modes:
    - single: all cache in one JSON file (TA_CACHE_PATH)
    - sharded: split by hash prefix into subdirs to reduce contention (TA_CACHE_DIR)
    """

    def __init__(self, mode: str = "single", path: Path | None = None, dir_path: Path | None = None):
        self.mode = mode
        self.path = path or _CACHE_PATH  # for single mode
        self.dir_path = dir_path or _CACHE_DIR  # for sharded mode
        self._lock = threading.Lock()
        self._data: dict[str, Any] | None = None  # only used in single mode

    def _load(self) -> None:
        """Load cache from disk (single mode only)."""
        if self.mode != "single" or self._data is not None:
            return
        if self.path.is_file():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(self._data, dict):
                    self._data = {}
            except (OSError, json.JSONDecodeError):
                self._data = {}
        else:
            self._data = {}

    def _shard_path(self, key: str) -> Path:
        """Compute shard subdirectory from key hash."""
        prefix = key[:2]  # use first 2 chars of hash
        return self.dir_path / prefix / f"{key}.json"

    def get(self, key: str) -> Any | None:
        with self._lock:
            if self.mode == "single":
                self._load()
                assert self._data is not None
                return self._data.get(key)
            else:  # sharded
                shard_path = self._shard_path(key)
                if shard_path.is_file():
                    try:
                        data = json.loads(shard_path.read_text(encoding="utf-8"))
                        return data.get("value")
                    except (OSError, json.JSONDecodeError):
                        return None
                return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if self.mode == "single":
                self._load()
                assert self._data is not None
                self._data[key] = value
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
                except OSError:
                    pass
            else:  # sharded
                shard_path = self._shard_path(key)
                try:
                    shard_path.parent.mkdir(parents=True, exist_ok=True)
                    shard_path.write_text(
                        json.dumps({"key": key, "value": value}, ensure_ascii=False),
                        encoding="utf-8"
                    )
                except OSError:
                    pass


_CACHE = _JsonCache(mode=_CACHE_MODE, path=_CACHE_PATH, dir_path=_CACHE_DIR)


# -----------------------------------------------------------------------------
# LLM subprocess driver
# -----------------------------------------------------------------------------
def _llm_call(
    system_prompt_path: Path,
    user_payload: Any,
    model: str | None = None,
    stats: _RunStats | None = None,
    allow_raw_text: bool = False,
) -> Any:
    """Run one backend subcall and return parsed JSON from the model.

    `user_payload` is either a str or a dict (serialized to JSON). The
    OpenAI-compatible backend posts this payload as a chat message; the Claude
    CLI backend pipes it via stdin to avoid argv length limits.

    If `stats` is provided, updates cache hit/miss counters.
    """
    if isinstance(user_payload, (dict, list)):
        stdin_text = json.dumps(user_payload, ensure_ascii=False)
    else:
        stdin_text = str(user_payload)

    if _BACKEND in {"openai", "auto"} and _OPENAI_BASE_URL and (_OPENAI_API_KEY or not _OPENAI_REQUIRES_AUTH):
        try:
            if _OPENAI_WIRE_API in {"responses", "response"}:
                return _openai_responses_call(
                    system_prompt_path,
                    stdin_text,
                    model=model,
                    stats=stats,
                    allow_raw_text=allow_raw_text,
                )
            return _openai_chat_call(
                system_prompt_path,
                stdin_text,
                model=model,
                stats=stats,
                allow_raw_text=allow_raw_text,
            )
        except Exception as e:
            if _BACKEND == "openai":
                raise
            if stats:
                stats.subcall_errors += 1
            print(f"[ta.llm] OpenAI-compatible backend failed; falling back to claude -p: {str(e)[:200]}", file=sys.stderr)

    claude_bin = os.environ.get("TA_CLAUDE_BIN", "").strip()
    if not claude_bin:
        claude_bin = which("claude") or which("claude.cmd") or which("claude.exe") or "claude"

    cmd = [
        claude_bin,
        "-p",
        "--system-prompt-file", str(system_prompt_path),
        "--output-format", "json",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--tools", "",
        "--model", model or _DEFAULT_MODEL,
    ]

    # Signal to the project's SessionStart hook that this is a TA sub-call
    # invocation, so it does NOT wipe the workspace mid-run.
    sub_env = os.environ.copy()
    sub_env["TA_SKILL_SUBCALL"] = "1"

    model_candidates = [model or _DEFAULT_MODEL]
    if _FALLBACK_MODEL and _FALLBACK_MODEL not in model_candidates:
        model_candidates.append(_FALLBACK_MODEL)

    payload_hash = sha256(
        (str(system_prompt_path) + "\n" + stdin_text).encode("utf-8", errors="ignore")
    ).hexdigest()

    last_err: Exception | None = None
    for selected_model in model_candidates:
        cache_key = f"{selected_model}:{payload_hash}"
        if _ENABLE_CACHE:
            cached = _CACHE.get(cache_key)
            if cached is not None:
                if stats:
                    stats.cache_hits += 1
                return cached
            if stats:
                stats.cache_misses += 1

        for attempt in range(_MAX_RETRIES + 1):
            cmd[-1] = selected_model
            try:
                proc = subprocess.run(
                    cmd,
                    input=stdin_text,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=_PER_CALL_TIMEOUT_S,
                    check=False,
                    env=sub_env,
                )
            except subprocess.TimeoutExpired as e:
                last_err = TextAnalysisError(
                    f"claude -p timed out after {_PER_CALL_TIMEOUT_S}s (model={selected_model})"
                )
                if attempt < _MAX_RETRIES:
                    if stats:
                        stats.retries_used += 1
                    time.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
                    continue
                break

            if proc.returncode != 0:
                last_err = TextAnalysisError(
                    f"claude -p exited {proc.returncode} (model={selected_model})\n"
                    f"stderr: {proc.stderr[:2000]}"
                )
                if attempt < _MAX_RETRIES:
                    if stats:
                        stats.retries_used += 1
                    time.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
                    continue
                break

            try:
                envelope = json.loads(proc.stdout)
            except json.JSONDecodeError as e:
                last_err = TextAnalysisError(
                    f"claude -p returned non-JSON envelope (model={selected_model}): "
                    f"{proc.stdout[:500]}"
                )
                if attempt < _MAX_RETRIES:
                    if stats:
                        stats.retries_used += 1
                    time.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
                    continue
                break

            if envelope.get("is_error"):
                last_err = TextAnalysisError(
                    f"claude -p reported error (model={selected_model}): {envelope.get('result')}"
                )
                if attempt < _MAX_RETRIES:
                    if stats:
                        stats.retries_used += 1
                    time.sleep(_RETRY_BACKOFF_S * (2 ** attempt))
                    continue
                break

            raw_result = envelope.get("result", "")
            parsed = _parse_model_json(raw_result)
            _trace_log({
                "type": "llm_call",
                "prompt": system_prompt_path.name,
                "model": selected_model,
                "attempt": attempt,
                "input": _trace_shorten(user_payload),
                "output": _trace_shorten(parsed),
            })
            if _ENABLE_CACHE:
                _CACHE.set(cache_key, parsed)
            return parsed

    if last_err is not None:
        raise last_err
    raise TextAnalysisError("claude -p failed for unknown reason")


def _openai_chat_call(
    system_prompt_path: Path,
    stdin_text: str,
    model: str | None = None,
    stats: _RunStats | None = None,
    allow_raw_text: bool = False,
) -> Any:
    """Run one OpenAI-compatible chat completion and parse the model JSON.

    This gives TADASkill a non-CLI backend for benchmark runs.  The public API
    remains `ta.tag` / `ta.categorize`; only the subcall transport changes.
    """
    base_url = _OPENAI_BASE_URL.rstrip("/")
    url = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    selected_model = model or _OPENAI_MODEL
    payload: dict[str, Any] = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": stdin_text},
        ],
    }
    if selected_model.startswith("gpt-5"):
        payload["max_completion_tokens"] = _OPENAI_MAX_TOKENS
    else:
        payload["temperature"] = 0.0
        payload["max_tokens"] = _OPENAI_MAX_TOKENS
    headers = {"Content-Type": "application/json"}
    if _OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {_OPENAI_API_KEY}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_PER_CALL_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise TextAnalysisError(f"OpenAI-compatible backend HTTP {exc.code}: {detail}") from exc
    response = json.loads(raw)
    usage = response.get("usage") or {}
    if stats is not None:
        stats.openai_calls += 1
        try:
            stats.openai_prompt_tokens += int(usage.get("prompt_tokens") or 0)
            stats.openai_completion_tokens += int(usage.get("completion_tokens") or 0)
            stats.openai_total_tokens += int(usage.get("total_tokens") or 0)
        except (TypeError, ValueError):
            pass
    choices = response.get("choices") or []
    if not choices:
        raise TextAnalysisError("OpenAI-compatible backend returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        raw_result = "\n".join(str(part.get("text", part)) for part in content)
    else:
        raw_result = str(content)
    try:
        parsed = _parse_model_json(raw_result)
    except TextAnalysisError:
        if not allow_raw_text:
            raise
        parsed = raw_result.strip()
    _trace_log({
        "type": "llm_call",
        "backend": "openai-compatible",
        "prompt": system_prompt_path.name,
        "model": selected_model,
        "usage": usage,
        "input": _trace_shorten(stdin_text),
        "output": _trace_shorten(parsed),
    })
    return parsed


def _response_text_from_responses(response: dict[str, Any]) -> str:
    """Best-effort text extraction for OpenAI Responses-compatible payloads."""
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("text"), str):
            chunks.append(item["text"])
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            text = part.get("text") or part.get("output_text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def _responses_usage_as_chat_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _openai_responses_call(
    system_prompt_path: Path,
    stdin_text: str,
    model: str | None = None,
    stats: _RunStats | None = None,
    allow_raw_text: bool = False,
) -> Any:
    """Run one OpenAI Responses-compatible call and parse the model JSON."""
    base_url = _OPENAI_BASE_URL.rstrip("/")
    url = base_url if base_url.endswith("/responses") else base_url + "/responses"
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    selected_model = model or _OPENAI_MODEL
    payload: dict[str, Any] = {
        "model": selected_model,
        "instructions": system_prompt,
        "input": stdin_text,
        "max_output_tokens": _OPENAI_MAX_TOKENS,
    }
    headers = {"Content-Type": "application/json"}
    if _OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {_OPENAI_API_KEY}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_PER_CALL_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise TextAnalysisError(f"OpenAI Responses backend HTTP {exc.code}: {detail}") from exc
    response = json.loads(raw)
    usage = _responses_usage_as_chat_usage(response)
    if stats is not None:
        stats.openai_calls += 1
        stats.openai_prompt_tokens += usage["prompt_tokens"]
        stats.openai_completion_tokens += usage["completion_tokens"]
        stats.openai_total_tokens += usage["total_tokens"]
    raw_result = _response_text_from_responses(response)
    if not raw_result.strip():
        raise TextAnalysisError("OpenAI Responses backend returned no output text")
    try:
        parsed = _parse_model_json(raw_result)
    except TextAnalysisError:
        if not allow_raw_text:
            raise
        parsed = raw_result.strip()
    _trace_log({
        "type": "llm_call",
        "backend": "openai-responses",
        "prompt": system_prompt_path.name,
        "model": selected_model,
        "usage": usage,
        "input": _trace_shorten(stdin_text),
        "output": _trace_shorten(parsed),
    })
    return parsed


def _parse_model_json(text: str) -> Any:
    """Extract the JSON object/array from a model response, tolerating fences
    and trailing prose. Strategies tried in order:
      1. Parse as-is.
      2. Extract the content of the first ```json ... ``` (or ``` ... ```) fence.
      3. Find the first balanced { ... } or [ ... ] span and parse it.
    """
    s = text.strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
    if fence:
        inner = fence.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    span = _find_balanced_json_span(s)
    if span is not None:
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            pass

    raise TextAnalysisError(f"model returned unparseable JSON: {text[:500]}")


def _find_balanced_json_span(s: str) -> str | None:
    """Scan for the first balanced { ... } or [ ... ] substring, respecting strings."""
    start = None
    open_ch = None
    close_ch = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            open_ch = ch
            close_ch = "}" if ch == "{" else "]"
            break
    if start is None:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


# -----------------------------------------------------------------------------
# Input normalization
# -----------------------------------------------------------------------------
def _extract_column(data: Any, column_name: str) -> list[str]:
    """Accept DataFrame, Series, dict-of-lists, or list[str]; return list[str]."""
    if pd is not None and isinstance(data, pd.DataFrame):
        if column_name not in data.columns:
            raise TextAnalysisError(
                f"column {column_name!r} not in DataFrame; have {list(data.columns)}"
            )
        series = data[column_name]
    elif pd is not None and isinstance(data, pd.Series):
        series = data
    elif isinstance(data, dict):
        if column_name not in data:
            raise TextAnalysisError(f"column {column_name!r} not in dict keys {list(data.keys())}")
        return ["" if v is None else str(v) for v in data[column_name]]
    elif isinstance(data, list):
        return ["" if v is None else str(v) for v in data]
    else:
        raise TextAnalysisError(f"unsupported input type: {type(data).__name__}")

    return ["" if (v is None or (pd is not None and pd.isna(v))) else str(v) for v in series.tolist()]


def _format_items(texts: Iterable[str]) -> list[str]:
    """Mirror C# TextItemFormat: [0] first, [1] second, ..."""
    return [f"[{i}] {t}" for i, t in enumerate(texts)]


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _split_size_guard(n_rows: int) -> int:
    return min(_MIN_SPLIT_BATCH_ROWS, max(1, n_rows))


# -----------------------------------------------------------------------------
# Internal LLM-backed primitives
# -----------------------------------------------------------------------------
def _intent_detect(column_name: str, query: str, stats: _RunStats | None = None) -> tuple[str, str | None]:
    """Return (tagging_type, domain_identification_query).

    tagging_type ∈ {"Independent", "Joint", "Invalid"}.
    """
    payload = {"UserQuery": query, "ColumnName": column_name}
    result = _llm_call(_PROMPT_INTENT, payload, stats=stats)
    ttype = result.get("TaggingType")
    diq = result.get("DomainIdentificationQuery")
    if ttype not in ("Independent", "Joint", "Invalid"):
        raise TextAnalysisError(f"intent detection returned unexpected TaggingType: {result}")
    _trace_log({
        "type": "intent_detect",
        "column": column_name,
        "query": _trace_shorten(query, 300),
        "tagging_type": ttype,
        "domain_query": _trace_shorten(diq, 300),
    })
    return ttype, diq


def _has_explicit_tag_domain(query: str) -> bool:
    """Detect constrained row-wise tagging requests that do not need intent routing.

    Benchmark and skill-generated programs often already phrase the primitive as
    "assign each row to exactly one of ...".  Running a separate intent-detection
    LLM call for these requests adds latency and creates an extra failure point
    without changing the planned operation.
    """
    q = query.lower()
    domain_markers = [
        "exactly one of",
        "from this taxonomy",
        "from this closed taxonomy",
        "one of:",
        "one of the following",
        "return only the integer",
        "return only a number",
    ]
    rowwise_markers = ["assign each", "label each", "tag each", "classify each"]
    return any(marker in q for marker in domain_markers) and any(marker in q for marker in rowwise_markers)


def _extract_explicit_domain(query: str) -> list[str]:
    patterns = [
        r"exactly one of(?: the following)?\s*:\s*([^\.]+)",
        r"from this (?:closed )?taxonomy\s*:\s*([^\.]+)",
        r"one of(?: the following)?\s*:\s*([^\.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1)
        raw = re.split(r"\bwhere\b|\breturn\b", raw, flags=re.IGNORECASE)[0]
        labels = [part.strip(" `\t\n\r.;") for part in re.split(r",|;|/|\|", raw)]
        return [label for label in labels if label]
    if "return only the integer" in query.lower() or "return only a number" in query.lower():
        return ["1", "2", "3", "4", "5"]
    return []


def _normalize_to_domain(value: Any, domain: list[str]) -> Any:
    if not domain:
        return value
    text = str(value or "").strip()
    if not text:
        return domain[0]
    lookup = {label.lower(): label for label in domain}
    low = text.lower()
    if low in lookup:
        return lookup[low]
    for label in domain:
        label_low = label.lower()
        if re.search(rf"\b{re.escape(label_low)}\b", low):
            return label
    if domain == ["1", "2", "3", "4", "5"]:
        m = re.search(r"[1-5]", text)
        return m.group(0) if m else "3"
    return domain[0]


def _local_fallback_tag(texts: list[str], query: str) -> list[Any] | None:
    """Conservative local fallback for explicit-domain benchmark tags.

    This is not a replacement for open semantic discovery.  It only handles
    constrained label/score tasks, keeps row alignment, and records a trace event.
    """
    domain = _extract_explicit_domain(query)
    if not domain:
        return None
    q = query.lower()
    yes_no = {label.lower() for label in domain} == {"yes", "no"}

    def score_text(text: str) -> float:
        low = text.lower()
        neg_words = ["bad", "poor", "terrible", "awful", "disappointed", "return", "returned", "small", "large", "tight", "cheap", "waste", "not recommend", "wouldn't recommend", "uncomfortable", "hate", "broke", "broken", "defect", "defective", "scratchy", "itchy", "unflattering", "thin", "see through", "see-through", "wrinkle", "wrinkled", "problem", "issue", "sadly", "unfortunately", "too short", "too long", "too big", "too small", "runs small", "runs large", "didn't like", "doesn't fit", "not worth"]
        pos_words = ["love", "loved", "great", "excellent", "perfect", "beautiful", "recommend", "comfortable", "nice", "favorite", "good", "happy", "flattering", "soft", "well made", "quality", "cute", "gorgeous", "pleased", "compliment"]
        neg = sum(low.count(w) for w in neg_words)
        pos = sum(low.count(w) for w in pos_words)
        if re.search(r"\b(not|never|no)\s+(good|great|comfortable|flattering|recommend|worth)\b", low):
            neg += 2
        if re.search(r"\b(not|never|no)\s+(bad|terrible|awful|disappointed)\b", low):
            pos += 1
        fit = sum(low.count(w) for w in ["fit", "size", "small", "large", "tight", "loose", "petite", "waist", "length"])
        if "fit-or-sizing" in q or "fit or sizing" in q:
            return float(max(1, min(5, 1 + fit + neg)))
        if "risk" in q or "discourage recommendation" in q or "product-fix" in q:
            return float(max(1, min(5, 1 + 2 * neg - pos)))
        return float(max(1, min(5, 3 + pos - neg)))

    out: list[Any] = []
    for text in texts:
        low = text.lower()
        if domain == ["1", "2", "3", "4", "5"]:
            out.append(str(int(score_text(text))))
        elif yes_no:
            if "strong praise" in q or "praise or delight" in q:
                out.append("yes" if score_text(text) >= 5 else "no")
            elif "negative" in q or "complaint" in q or "would not recommend" in q:
                score = score_text(text)
                has_explicit_negative = any(w in low for w in ["not recommend", "wouldn't recommend", "return", "returned", "waste", "disappointed", "terrible", "awful", "poor", "defective", "broken", "doesn't fit", "didn't like", "not worth"])
                out.append("yes" if score <= 2 or has_explicit_negative else "no")
            else:
                out.append("no")
        elif {d.lower() for d in domain} == {"negative", "neutral", "positive"}:
            s = score_text(text)
            out.append("negative" if s <= 2 else ("positive" if s >= 4 else "neutral"))
        else:
            out.append(domain[0])
    _trace_log({"type": "local_fallback_tag", "domain": domain, "query": _trace_shorten(query, 300), "rows": len(texts)})
    print(f"[ta.tag] local constrained fallback used for {len(texts)} row(s)", file=sys.stderr)
    return out


def _tag_batch(texts: list[str], column_name: str, query: str, stats: _RunStats | None = None) -> list[Any]:
    payload = {
        "UserQuery": query,
        "ColumnName": column_name,
        "TextItems": _format_items(texts),
    }
    result = _llm_call(
        _PROMPT_TAGGING,
        payload,
        stats=stats,
        allow_raw_text=_ALLOW_RAW_SINGLE_TAG_CELL and len(texts) == 1,
    )
    # Normal contract: {"Results": {"Items": {"0": ..., "1": ...}}}.
    # Some models occasionally drop the envelope and return a bare array (or a
    # bare {index: value} map) for the batch — accept those by position too.
    if isinstance(result, str) and len(texts) == 1:
        items = {"0": result}
    elif isinstance(result, list):
        items = {str(i): v for i, v in enumerate(result)}
    elif isinstance(result, dict):
        inner = result.get("Results", result)
        if isinstance(inner, dict):
            items = inner.get("Items", inner)
        else:
            items = inner
        if isinstance(items, list):
            items = {str(i): v for i, v in enumerate(items)}
        elif not isinstance(items, dict):
            items = {}
    else:
        items = {}
    # 单行 raw-cell 模式用于结构化抽取；模型返回显式空 Items 时，语义是
    # “该行没有命中值”，不是行对齐丢失。多行批次仍保持严格缺索引报错。
    if _ALLOW_RAW_SINGLE_TAG_CELL and len(texts) == 1 and items == {}:
        items = {"0": ""}
        _trace_log({"type": "tag_batch_explicit_empty", "column": column_name})
    # items is keyed by stringified index; assemble in order.
    out: list[Any] = []
    for i in range(len(texts)):
        key = str(i)
        if key not in items:
            raise TextAnalysisError(
                f"tagging batch missing index {key}; got keys {list(items.keys())[:5]}..."
            )
        out.append(items[key])
    _trace_log({
        "type": "tag_batch",
        "column": column_name,
        "query": _trace_shorten(query, 300),
        "rows": len(texts),
        "sample_input": _trace_shorten(texts[:3]),
        "sample_output": _trace_shorten(out[:8]),
    })
    return out


def _tag_batch_resilient(texts: list[str], column_name: str, query: str, stats: _RunStats | None = None) -> list[Any]:
    """Retry a failed tagging batch by recursively splitting it."""
    try:
        return _tag_batch(texts, column_name, query, stats=stats)
    except Exception as e:
        if stats:
            stats.subcall_errors += 1
        min_size = _split_size_guard(len(texts))
        if len(texts) <= min_size:
            if _LOCAL_FALLBACK_ENABLED and _has_explicit_tag_domain(query):
                fallback = _local_fallback_tag(texts, query)
                if fallback is not None:
                    return fallback
            raise TextAnalysisError(
                f"tag batch failed at min split size={min_size}: {str(e)[:300]}"
            ) from e

        mid = len(texts) // 2
        print(
            f"[ta.tag] batch of {len(texts)} failed; splitting into {mid}+{len(texts)-mid}",
            file=sys.stderr,
        )
        left = _tag_batch_resilient(texts[:mid], column_name, query, stats=stats)
        right = _tag_batch_resilient(texts[mid:], column_name, query, stats=stats)
        return left + right


def _categorize_batch(texts: list[str], column_name: str, query: str, stats: _RunStats | None = None) -> dict[str, Any]:
    payload = {
        "UserQuery": query,
        "ColumnName": column_name,
        "TextItems": _format_items(texts),
    }
    result = _llm_call(_PROMPT_CATEGORIZATION, payload, stats=stats)
    cats = result.get("Categories", [])
    have_description = bool(result.get("HaveDescription")) or any(
        (c.get("Description") or "").strip() for c in cats
    )
    normalized = [
        {"Name": c.get("Name", ""), "Description": c.get("Description", "") or ""}
        for c in cats
    ]
    _trace_log({
        "type": "categorize_batch",
        "column": column_name,
        "query": _trace_shorten(query, 300),
        "rows": len(texts),
        "sample_input": _trace_shorten(texts[:3]),
        "categories": _trace_shorten(normalized),
    })
    return {"HaveDescription": have_description, "BatchRows": len(texts), "Categories": normalized}


def _categorize_batch_resilient(
    texts: list[str], column_name: str, query: str, stats: _RunStats | None = None
) -> list[dict[str, Any]]:
    """Retry a failed categorization batch by recursively splitting it."""
    try:
        return [_categorize_batch(texts, column_name, query, stats=stats)]
    except Exception as e:
        if stats:
            stats.subcall_errors += 1
        min_size = _split_size_guard(len(texts))
        if len(texts) <= min_size:
            raise TextAnalysisError(
                f"categorize batch failed at min split size={min_size}: {str(e)[:300]}"
            ) from e

        mid = len(texts) // 2
        print(
            f"[ta.categorize] batch of {len(texts)} failed; splitting into {mid}+{len(texts)-mid}",
            file=sys.stderr,
        )
        left = _categorize_batch_resilient(texts[:mid], column_name, query, stats=stats)
        right = _categorize_batch_resilient(texts[mid:], column_name, query, stats=stats)
        return left + right




def _normalize_category_key(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _batch_set_weight(batch_set: dict[str, Any]) -> float:
    raw = batch_set.get("BatchRows")
    if raw is None:
        raw = batch_set.get("SetWeight")
    try:
        v = float(raw)
        return v if v > 0 else 1.0
    except (TypeError, ValueError):
        return 1.0


def _decorate_batch_sets_with_weight_hints(batch_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach lightweight WeightHint to category descriptions for consolidation calls."""
    score_by_key: dict[str, float] = {}
    for s in batch_sets:
        cats = s.get("Categories") or []
        if not cats:
            continue
        per_cat = _batch_set_weight(s) / max(1, len(cats))
        for c in cats:
            key = _normalize_category_key(str(c.get("Name", "")))
            if not key:
                continue
            score_by_key[key] = score_by_key.get(key, 0.0) + per_cat

    decorated: list[dict[str, Any]] = []
    for s in batch_sets:
        cats_out: list[dict[str, Any]] = []
        for c in (s.get("Categories") or []):
            name = str(c.get("Name", ""))
            desc = str(c.get("Description", "") or "")
            key = _normalize_category_key(name)
            hint = score_by_key.get(key, 0.0)
            if hint > 0:
                suffix = f" [WeightHint={hint:.2f}]"
                if suffix not in desc:
                    desc = f"{desc}{suffix}" if desc else suffix.strip()
            cats_out.append({"Name": name, "Description": desc})

        decorated.append(
            {
                "HaveDescription": bool(s.get("HaveDescription")) or any((x.get("Description") or "").strip() for x in cats_out),
                "BatchRows": s.get("BatchRows"),
                "SetWeight": s.get("SetWeight"),
                "Categories": cats_out,
            }
        )

    return decorated


def _to_batch_category_set(categories: list[dict[str, str]], set_weight: float = 1.0) -> dict[str, Any]:
    """Convert normalized categories back to consolidation input schema."""
    have_description = any((c.get("description") or "").strip() for c in categories)
    return {
        "HaveDescription": have_description,
        "SetWeight": set_weight,
        "Categories": [
            {
                "Name": c.get("name", ""),
                "Description": c.get("description", "") or "",
            }
            for c in categories
        ],
    }


def _consolidate_categories_hierarchical(
    batch_sets: list[dict[str, Any]],
    column_name: str,
    query: str,
    stats: _RunStats | None = None,
) -> list[dict[str, str]]:
    """Consolidate large numbers of batch category sets in multiple rounds.

    This avoids oversized single prompts when the number of batch sets is high.
    """
    if len(batch_sets) <= _CONSOLIDATE_SET_LIMIT:
        return _consolidate_categories(batch_sets, column_name, query, stats=stats)

    current = batch_sets
    round_idx = 1
    while len(current) > _CONSOLIDATE_SET_LIMIT:
        groups = [
            current[i:i + _CONSOLIDATE_SET_LIMIT]
            for i in range(0, len(current), _CONSOLIDATE_SET_LIMIT)
        ]
        print(
            f"[ta.categorize] consolidation round {round_idx}: {len(current)} sets -> {len(groups)} grouped calls",
            file=sys.stderr,
        )

        next_round: list[dict[str, Any]] = []
        for group in groups:
            decorated_group = _decorate_batch_sets_with_weight_hints(group)
            merged = _consolidate_categories(decorated_group, column_name, query, stats=stats)
            group_weight = sum(_batch_set_weight(s) for s in group)
            next_round.append(_to_batch_category_set(merged, set_weight=group_weight))

        current = next_round
        round_idx += 1

    print(
        f"[ta.categorize] final consolidation on {len(current)} grouped set(s)",
        file=sys.stderr,
    )
    decorated_final = _decorate_batch_sets_with_weight_hints(current)
    return _consolidate_categories(decorated_final, column_name, query, stats=stats)

def _consolidate_categories(
    batch_sets: list[dict[str, Any]], column_name: str, query: str, stats: _RunStats | None = None
) -> list[dict[str, str]]:
    payload = {
        "UserQuery": query,
        "ColumnName": column_name,
        "BatchCategorySets": batch_sets,
    }
    result = _llm_call(_PROMPT_CONSOLIDATION, payload, stats=stats)
    merged = [
        {"name": c.get("Name", ""), "description": c.get("Description", "") or ""}
        for c in result.get("Categories", [])
    ]
    _trace_log({
        "type": "consolidate",
        "column": column_name,
        "query": _trace_shorten(query, 300),
        "input_sets": len(batch_sets),
        "output_categories": len(merged),
        "sample_output": _trace_shorten(merged[:10]),
    })
    return merged


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def tag(data: Any, column_name: str, query: str) -> list[Any]:
    """Row-level tagging. Returns one value per row, aligned with input order.

    Dispatches Independent vs Joint based on intent detection:
      * Independent: each item tagged in isolation.
      * Joint: first discover a tag vocabulary via categorize(), then assign.
    Batches rows with MAX_ROWS_PER_TAGGING_BATCH; batches run concurrently.
    """
    stats = _RunStats()
    texts = _extract_column(data, column_name)
    if not texts:
        return []

    stats.rows_processed = len(texts)
    explicit_domain = _extract_explicit_domain(query) if _has_explicit_tag_domain(query) else []
    if _FORCE_INDEPENDENT_TAGGING:
        ttype, diq = "Independent", None
        print("[ta.tag] Forced Independent path: skipping intent detection", file=sys.stderr)
        _trace_log({
            "type": "intent_forced_independent",
            "column": column_name,
            "query": _trace_shorten(query, 300),
            "tagging_type": ttype,
        })
    elif explicit_domain:
        ttype, diq = "Independent", None
        print("[ta.tag] Explicit domain fast path: skipping intent detection", file=sys.stderr)
        _trace_log({
            "type": "intent_fast_path",
            "column": column_name,
            "query": _trace_shorten(query, 300),
            "tagging_type": ttype,
        })
    else:
        ttype, diq = _intent_detect(column_name, query, stats=stats)

    if ttype == "Invalid":
        raise TextAnalysisError(
            f"Query is not a valid tagging task for column {column_name!r}: {query!r}"
        )

    if ttype == "Joint":
        if not diq:
            raise TextAnalysisError("Joint tagging requires DomainIdentificationQuery")
        print(f"[ta.tag] Joint path: discovering vocabulary via categorize()", file=sys.stderr)
        domain_cats = categorize(texts, column_name, diq)
        if not domain_cats:
            raise TextAnalysisError("Joint tagging: domain discovery returned empty vocabulary")
        effective_query = _compose_citation_query(domain_cats, query)
    else:
        print(f"[ta.tag] Independent path", file=sys.stderr)
        effective_query = query

    if _ENABLE_DEDUP and len(texts) >= _DEDUP_MIN_ROWS:
        uniques: list[str] = []
        index_by_text: dict[str, int] = {}
        inverse: list[int] = []
        for t in texts:
            idx = index_by_text.get(t)
            if idx is None:
                idx = len(uniques)
                index_by_text[t] = idx
                uniques.append(t)
            inverse.append(idx)
        if len(uniques) < len(texts):
            stats.dedup_ratio = len(uniques) / len(texts)
            print(
                f"[ta.tag] dedup enabled: {len(texts)} rows -> {len(uniques)} unique text items",
                file=sys.stderr,
            )
            unique_tags = _tag_texts_in_batches(uniques, column_name, effective_query, stats=stats)
            results = [unique_tags[i] for i in inverse]
            print(stats.summary_line(), file=sys.stderr)
            return results

    results = _tag_texts_in_batches(texts, column_name, effective_query, stats=stats)

    if len(results) != len(texts):
        raise TextAnalysisError(
            f"tag row-count mismatch: {len(results)} tags for {len(texts)} rows"
        )

    if explicit_domain:
        results = [_normalize_to_domain(result, explicit_domain) for result in results]

    # If Joint path: normalize results to ensure they match vocabulary exactly
    if ttype == "Joint":
        results = _normalize_citation_results(results, domain_cats)

    print(stats.summary_line(), file=sys.stderr)
    return results


def _tag_texts_in_batches(texts: list[str], column_name: str, query: str, stats: _RunStats | None = None) -> list[Any]:
    batches = _chunk(texts, MAX_ROWS_PER_TAGGING_BATCH)
    if stats:
        stats.batches_sent = len(batches)
    print(f"[ta.tag] {len(texts)} rows in {len(batches)} batch(es)", file=sys.stderr)

    # Preallocate result array and fill by start_idx to ensure deterministic alignment
    results: list[Any | None] = [None] * len(texts)
    start_indices = [i * MAX_ROWS_PER_TAGGING_BATCH for i in range(len(batches))]

    with ThreadPoolExecutor(max_workers=min(_MAX_CONCURRENCY, len(batches))) as ex:
        futures = [
            (start_idx, ex.submit(_tag_batch_resilient, b, column_name, query, stats=stats))
            for start_idx, b in zip(start_indices, batches)
        ]
        for start_idx, f in futures:
            batch_results = f.result()
            for i, result in enumerate(batch_results):
                results[start_idx + i] = result

    # Verify complete coverage
    if any(r is None for r in results):
        raise TextAnalysisError(
            f"tag batch alignment error: incomplete result coverage after {len(batches)} batch(es)"
        )
    return results


def categorize(data: Any, column_name: str, query: str) -> list[dict[str, str]]:
    """Dataset-level categorization. Returns [{"name", "description"}, ...].

    Batches rows with MAX_ROWS_PER_CATEGORIZATION_BATCH, runs per-batch calls
    concurrently, then consolidates batch-level category sets into a single
    globally coherent list (skipped when there is only one batch).
    """
    stats = _RunStats()
    texts = _extract_column(data, column_name)
    if not texts:
        return []

    stats.rows_processed = len(texts)
    batches = _chunk(texts, MAX_ROWS_PER_CATEGORIZATION_BATCH)
    stats.batches_sent = len(batches)
    print(
        f"[ta.categorize] {len(texts)} rows in {len(batches)} batch(es)", file=sys.stderr
    )

    with ThreadPoolExecutor(max_workers=min(_MAX_CONCURRENCY, len(batches))) as ex:
        futures = [ex.submit(_categorize_batch_resilient, b, column_name, query, stats=stats) for b in batches]
        nested_sets = [f.result() for f in futures]

    batch_sets: list[dict[str, Any]] = []
    for partial in nested_sets:
        batch_sets.extend(partial)

    if len(batch_sets) == 1:
        result = [
            {"name": c["Name"], "description": c["Description"]}
            for c in batch_sets[0]["Categories"]
        ]
        print(stats.summary_line(), file=sys.stderr)
        return result

    print(
        f"[ta.categorize] consolidating {len(batch_sets)} batch category sets",
        file=sys.stderr,
    )
    result = _consolidate_categories_hierarchical(batch_sets, column_name, query, stats=stats)
    print(stats.summary_line(), file=sys.stderr)
    return result


def _normalize_citation_results(
    results: list[Any], categories: list[dict[str, str]]
) -> list[Any]:
    """Normalize Joint tagging results to ensure they conform to discovered vocabulary.

    If a result doesn't exactly match a category name, attempt fuzzy matching or mark as invalid.
    This post-check guarantees cross-batch consistency in Joint workflows.
    """
    valid_names = {c.get("name") or c.get("Name", "") for c in categories}
    valid_names.discard("")  # remove empty string
    valid_names_list = sorted(valid_names)

    normalized: list[Any] = []
    for result in results:
        if isinstance(result, str):
            result_stripped = result.strip()
            # Exact match → keep
            if result_stripped in valid_names:
                normalized.append(result_stripped)
                continue
            # Fuzz match on case-insensitive prefix
            lower_result = result_stripped.lower()
            matched = False
            for valid_name in valid_names_list:
                if lower_result.startswith(valid_name.lower()) or lower_result == valid_name.lower():
                    normalized.append(valid_name)
                    matched = True
                    break
            if not matched:
                # Fallback: mark as ambiguous or pick first category
                print(
                    f"[ta.tag.normalize] result {result_stripped!r} not in vocabulary {valid_names_list}; "
                    f"keeping as-is for manual review",
                    file=sys.stderr,
                )
                normalized.append(result_stripped)
        else:
            # Non-string result (rare in tag) → keep as-is
            normalized.append(result)

    return normalized


def _compose_citation_query(categories: list[dict[str, str]], original_query: str) -> str:
    """Turn a discovered vocabulary into a citation-style tagging query.

    Mirrors LargeScaleLLMTextAnalysis.ComposeTaggingWithPreSpecifiedDomainQuery.
    """
    lines = ["The categories list is ["]
    for c in categories:
        name = c.get("name") or c.get("Name") or ""
        desc = c.get("description") or c.get("Description") or ""
        if desc.strip():
            lines.append(f"CategoryName : {name}, Description : {desc}")
        else:
            lines.append(f"CategoryName : {name}")
    lines.append("]")
    lines.append("Map each item to the most relevant one or a few category name(s) from this list.")
    lines.append("Strictly preserve all output constraints from the original user query below.")
    lines.append(f"Original user query: {original_query}")
    return "\n".join(lines)


@dataclass
class SemanticState:
    """Materialized semantic column plus audit metadata.

    This is a thin state contract over the existing list-returning primitives.
    It keeps the old `tag` API stable while giving experiments and agents a
    first-class object for typed, row-aligned semantic table state.
    """

    frame: Any
    output_column: str
    manifest: dict[str, Any]
    audit_rows: list[dict[str, Any]]

    def to_dir(self, path: str | Path) -> None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        if pd is None:
            raise TextAnalysisError("SemanticState.to_dir requires pandas")
        self.frame.to_csv(out_dir / "semantic_state.csv", index=False)
        with (out_dir / "audit_rows.jsonl").open("w", encoding="utf-8") as f:
            for row in self.audit_rows:
                f.write(json.dumps(row, ensure_ascii=False, default=_json_scalar) + "\n")
        (out_dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False, default=_json_scalar) + "\n",
            encoding="utf-8",
        )


def _json_scalar(value: Any) -> Any:
    """把 NumPy/pandas 标量降为标准 JSON 可序列化的 Python 值。"""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    if pd is not None:
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return str(value)


def _coerce_frame_for_state(data: Any, column_name: str, row_id: str | None = None) -> Any:
    if pd is None:
        raise TextAnalysisError("materialize_tag requires pandas")
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, dict):
        frame = pd.DataFrame(data)
    elif isinstance(data, list):
        frame = pd.DataFrame({column_name: data})
    elif isinstance(data, pd.Series):
        frame = pd.DataFrame({column_name: data.tolist()})
    else:
        raise TextAnalysisError(f"unsupported input type for materialize_tag: {type(data).__name__}")
    if column_name not in frame.columns:
        raise TextAnalysisError(f"column {column_name!r} not in materialized frame")
    if row_id and row_id not in frame.columns:
        raise TextAnalysisError(f"row_id column {row_id!r} not in materialized frame")
    return frame


def _valid_for_domain(value: Any, domain: list[str] | None) -> bool:
    if not domain:
        return True
    if isinstance(value, list):
        return all(str(v) in domain for v in value)
    text = str(value)
    parts = [part.strip() for part in re.split(r"[;,\n]", text) if part.strip()]
    if len(parts) > 1:
        return all(part in domain for part in parts)
    return text.strip() in domain


def materialize_tag(
    data: Any,
    column_name: str,
    query: str,
    output_column: str,
    *,
    domain: list[str] | None = None,
    row_id: str | None = None,
) -> SemanticState:
    """Run `tag` and return a typed, row-aligned semantic-state object.

    Parameters mirror `tag`, with explicit metadata needed by analytical
    pipelines. `domain` is optional but recommended for closed-taxonomy tasks;
    it is used to mark domain/schema violations in the audit artifact.
    """
    frame = _coerce_frame_for_state(data, column_name, row_id=row_id)
    texts = _extract_column(frame, column_name)
    values = tag(frame, column_name, query)
    if len(values) != len(frame):
        raise TextAnalysisError(f"materialize_tag row-count mismatch: {len(values)} values for {len(frame)} rows")

    frame[output_column] = values
    valid_column = f"{output_column}__valid"
    frame[valid_column] = [_valid_for_domain(value, domain) for value in values]

    audit_rows: list[dict[str, Any]] = []
    for idx, (text, value, valid) in enumerate(zip(texts, values, frame[valid_column].tolist())):
        record_id = frame.iloc[idx][row_id] if row_id else idx
        audit_rows.append(
            {
                "row_index": idx,
                "row_id": record_id,
                "input_hash": sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
                "output_column": output_column,
                "parsed_value": value,
                "valid": bool(valid),
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": "materialize_tag",
        "input_column": column_name,
        "output_column": output_column,
        "row_id": row_id,
        "rows": len(frame),
        "domain": domain or [],
        "query": query,
        "backend": _BACKEND,
        "model": _OPENAI_MODEL if _OPENAI_BASE_URL else _DEFAULT_MODEL,
        "openai_wire_api": _OPENAI_WIRE_API if _OPENAI_BASE_URL else "",
        "openai_base_url_present": bool(_OPENAI_BASE_URL),
        "schema_valid_rate": sum(bool(x) for x in frame[valid_column].tolist()) / len(frame) if len(frame) else 1.0,
    }
    return SemanticState(frame=frame, output_column=output_column, manifest=manifest, audit_rows=audit_rows)
