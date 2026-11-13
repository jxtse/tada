"""Generate fact carriers with bounded, independently verified regeneration."""
from __future__ import annotations

from .clients import FakeClient, ModelClient, InvalidModelOutputError
from .roundtrip import verify_document


def generate_document(document: dict, writer: ModelClient | None = None,
                      extractor: ModelClient | None = None, max_regenerations: int = 3) -> dict:
    writer = writer or FakeClient("fake-writer")
    extractor = extractor or FakeClient("fake-extractor")
    if writer.model_id == extractor.model_id:
        raise ValueError("Round-trip extraction must use a different model")
    if not 0 <= max_regenerations <= 3:
        raise ValueError("At most three regenerations are permitted")
    facts = document.get("facts", {})
    if not isinstance(facts, dict):
        raise ValueError("Document facts must map fact identifiers to JSON values")
    attempts = []
    result = {"id": document["id"], "path": document.get("path", document["id"] + ".txt"),
              "kind": document.get("kind", "notice"), "writer_model": writer.model_id,
              "extractor_model": extractor.model_id,
              "fake": bool(getattr(writer, "fake", False) and getattr(extractor, "fake", False))}
    # One initial generation plus at most three regenerations.
    for number in range(max_regenerations + 1):
        try:
            response = writer.complete("write_document", {
                "instruction": "Write a realistic professional email, notice or minutes conveying every fact exactly in natural, varied prose. Do not list facts as a table, JSON or a key-value dump.",
                "id": document["id"], "kind": document.get("kind", "notice"),
                "facts": facts, "attempt": number + 1,
            })
        except InvalidModelOutputError:
            response = {}
        text = response.get("text", "")
        check = verify_document(text, facts, extractor) if isinstance(text, str) and text else {
            "pass": False, "extracted": None, "error": "Empty/non-text generation"}
        attempts.append({"attempt": number + 1, **check})
        if check["pass"]:
            return {**result, "status": "verified", "pass": True, "text": text,
                    "facts": facts, "attempts": attempts, "dropped_facts": []}
    return {**result, "status": "dropped", "pass": False, "text": "", "facts": {},
            "attempts": attempts, "dropped_facts": sorted(facts)}


def generate_documents(documents: list[dict], writer: ModelClient | None = None,
                       extractor: ModelClient | None = None) -> dict:
    results = [generate_document(doc, writer, extractor) for doc in documents]
    return {"documents": results, "pass": all(d["pass"] for d in results),
            "fake": all(d["fake"] for d in results),
            "dropped_facts": [f for d in results for f in d["dropped_facts"]]}
