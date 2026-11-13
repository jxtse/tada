"""Blind extraction: the extractor never receives expected fact values."""
from __future__ import annotations

import json
from .clients import ModelClient


def canonical(value) -> str:
    # JSON equality must distinguish bool from int and preserve value types.
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def extract_facts(text: str, fact_names: list[str], client: ModelClient) -> dict:
    response = client.complete("extract_facts", {
        "instruction": "Extract the named facts from this document only. Return JSON facts; do not infer missing values.",
        "text": text, "fact_names": fact_names,
    })
    facts = response.get("facts")
    if not isinstance(facts, dict):
        raise ValueError("Extractor must return an object under facts")
    return facts


def verify_document(text: str, expected: dict, client: ModelClient) -> dict:
    try:
        extracted = extract_facts(text, sorted(expected), client)
        matched = canonical(extracted) == canonical(expected)
        return {"pass": matched, "extracted": extracted, "error": None}
    except (ValueError, TypeError, KeyError) as exc:
        return {"pass": False, "extracted": None, "error": str(exc)}
