#!/usr/bin/env python3
"""Run TADA (Tag primitive) on the BioDEX adverse-drug-reaction benchmark.

Design (option A, decided with the author):
  * ta.tag extracts free-text reaction phrases from each paper (the semantic
    judgement the LLM is good at). It never sees the 24k official term list.
  * Deterministic code embeds each free-text phrase and each official term with
    text-embedding-3-small, then nearest-neighbour aligns the phrases to the
    official vocabulary and ranks them. This is the exact-match/retrieval work
    that should stay out of the prompt.

Scoring reuses the official DocETL/Abacus evaluator verbatim: rank_precision@k
and term_recall (see baseline_paper/docetl/.../evaluation/biodex.py).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline_paper"
DEFAULT_DATA = BASELINE / "docetl/experiments/reasoning/data/test/biodex.json"
DEFAULT_OUT_DIR = ROOT / "bench/results/biodex_tada"
SKILL_LIB = ROOT / "TADA_skill/TADASkill/lib"
EMBED_CACHE = ROOT / "bench/results/biodex_tada/label_embeddings.npz"

EMBED_MODEL = "text-embedding-3-small"

QUERY = """For each medical paper, extract every adverse drug reaction or
patient-reported medical condition the paper describes. Report each reaction as
a short clinical phrase in the paper's own clinical language (for example
"acute kidney injury", "nausea", "rash"), ordered from most to least clearly
reported. Return the reactions as a semicolon-separated list. Do not include
drug names, dosages, or commentary."""


# --------------------------------------------------------------------------- #
# Official evaluator (copied verbatim from
# baseline_paper/docetl/experiments/reasoning/evaluation/biodex.py)
# --------------------------------------------------------------------------- #
def rank_precision_at_k(preds: list[str], targets: list[str], k: int) -> float:
    if not preds:
        return 0.0
    preds = [p.strip().lower().replace("'", "").replace("^", "") for p in preds]
    targets = set(t.strip().lower().replace("'", "").replace("^", "") for t in targets)
    rn = len(targets)
    denom = min(k, rn)
    if denom == 0:
        return 0.0
    total = 0.0
    for i in range(min(k, len(preds))):
        if preds[i] in targets:
            total += 1.0
    return total / denom


def term_recall(preds: list[str], targets: list[str]) -> float:
    if not preds:
        return 0.0
    pred_terms = set(
        term.strip()
        for pred in preds
        for term in pred.lower().replace("'", "").replace("^", "").split(" ")
    )
    target_terms = set(
        term.strip()
        for target in targets
        for term in target.lower().replace("'", "").replace("^", "").split(" ")
    )
    if len(target_terms) == 0:
        return 0.0
    return len(pred_terms.intersection(target_terms)) / len(target_terms)


# --------------------------------------------------------------------------- #
# Embedding-based alignment of free-text phrases -> official 24k vocabulary
# --------------------------------------------------------------------------- #
def _openai_client():
    """Embedding client.

    The tag step (ta.tag) may run on an endpoint that has no embeddings API
    (e.g. OpenRouter/Llama), so the embedding endpoint is configured
    separately via TA_EMBED_BASE_URL / TA_EMBED_API_KEY. Falls back to the
    TA_OPENAI_* vars when they are not set (single-endpoint setups).
    """
    from openai import OpenAI

    base_url = (
        os.environ.get("TA_EMBED_BASE_URL", "").strip()
        or os.environ.get("TA_OPENAI_BASE_URL", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "").strip()
    )
    api_key = (
        os.environ.get("TA_EMBED_API_KEY", "").strip()
        or os.environ.get("TA_OPENAI_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or "sk-none"
    )
    if not base_url:
        raise SystemExit(
            "Set TA_EMBED_BASE_URL (or TA_OPENAI_BASE_URL) to the embeddings endpoint."
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def embed_texts(client, texts: list[str], batch: int = 256) -> "np.ndarray":
    import numpy as np

    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        out.extend(d.embedding for d in resp.data)
    arr = np.asarray(out, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def get_label_embeddings(client, labels: list[str], cache: Path):
    import numpy as np

    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        if list(data["labels"]) == labels:
            return data["emb"]
    emb = embed_texts(client, labels)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, labels=np.array(labels, dtype=object), emb=emb)
    return emb


def align_and_rank(
    phrase_lists: list[list[str]],
    labels: list[str],
    label_emb,
    client,
) -> list[list[str]]:
    """Map each doc's free-text phrases to nearest official terms, dedup, keep order."""
    import numpy as np

    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for phrases in phrase_lists:
        start = len(flat)
        flat.extend(phrases)
        spans.append((start, len(flat)))

    phrase_emb = embed_texts(client, flat) if flat else np.zeros((0, label_emb.shape[1]), dtype="float32")

    ranked_all: list[list[str]] = []
    for (start, end) in spans:
        ranked: list[str] = []
        seen: set[str] = set()
        if end > start:
            sims = phrase_emb[start:end] @ label_emb.T  # (n_phrases, n_labels)
            best_idx = sims.argmax(axis=1)
            for j in best_idx:
                term = labels[int(j)]
                if term not in seen:
                    seen.add(term)
                    ranked.append(term)
        ranked_all.append(ranked)
    return ranked_all


# --------------------------------------------------------------------------- #
def parse_phrases(raw: Any) -> list[str]:
    if isinstance(raw, list):
        parts = raw
    elif isinstance(raw, dict):
        parts = raw.get("reactions") or raw.get("labels") or list(raw.values())
    else:
        text = str(raw or "").strip()
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                parts = json.loads(m.group(0))
            except Exception:
                parts = re.split(r"[,;\n]", m.group(0).strip("[]"))
        else:
            parts = re.split(r"[,;\n]", text)
    out = []
    for p in parts:
        s = str(p).strip().strip('"\'').strip()
        if s:
            out.append(s)
    return out


def run_tag(input_texts: list[str], query: str) -> tuple[list[Any], float]:
    """Tag each paper independently.

    BioDEX is a per-document extraction task (each paper -> its own reaction
    list), so we call ta.tag once per paper. This also isolates any single
    malformed document instead of failing the whole batch.
    """
    sys.path.insert(0, str(SKILL_LIB))
    import ta  # type: ignore

    started = time.time()
    raw: list[Any] = []
    gap = float(os.getenv("TA_TAG_GAP_S", "3"))
    for i, text in enumerate(input_texts):
        out_val: Any = "[]"
        for attempt in range(6):
            try:
                out = ta.tag({"paper": [text]}, "paper", query)
                out_val = out[0]
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                is_rate = "429" in msg or "502" in msg or "too many" in msg.lower()
                if attempt == 5:
                    print(
                        f"[biodex] doc {i} tag failed after retries: {msg[:160]}",
                        file=sys.stderr,
                    )
                else:
                    back = min(60.0, (2 ** attempt) * (5.0 if is_rate else 1.0))
                    time.sleep(back)
        raw.append(out_val)
        time.sleep(gap)
    return raw, time.time() - started


def shorten(text: str, limit: int = 8000) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n...<truncated>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--limit", type=int, default=None, help="only first N docs")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--call-llm", action="store_true")
    ap.add_argument("--allow-local-fallback", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = json.loads(Path(args.data).read_text())
    if args.limit:
        docs = docs[: args.limit]
    labels = docs[0]["possible_labels"].split("\n")
    labels = [l.strip() for l in labels if l.strip()]

    input_texts = [
        f"PMID: {d['id']}\n{shorten(d['fulltext_processed'])}" for d in docs
    ]

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "docs": len(docs),
        "n_official_labels": len(labels),
        "embed_model": EMBED_MODEL,
        "call_llm": bool(args.call_llm),
        "ta_openai_model": os.getenv("TA_OPENAI_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4.1",
        "ta_openai_wire_api": os.getenv("TA_OPENAI_WIRE_API", ""),
        "ta_openai_base_url_present": bool(os.getenv("TA_OPENAI_BASE_URL")),
        "design": "ta.tag extracts free-text reactions; code embeds+aligns to 24k official vocab",
        "evaluator": "official DocETL biodex rank_precision@k + term_recall",
    }

    if not args.call_llm:
        (out_dir / "input_preview.json").write_text(
            json.dumps(input_texts[:3], indent=2, ensure_ascii=False)
        )
        manifest["note"] = "dry run: no LLM called"
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Prepared {len(docs)} BioDEX docs (dry run). {out_dir/'manifest.json'}")
        return

    os.environ["TA_BACKEND"] = "openai"
    if not args.allow_local_fallback:
        os.environ["TA_ENABLE_LOCAL_FALLBACK"] = "0"

    raw_tags, elapsed = run_tag(input_texts, QUERY)
    phrase_lists = [parse_phrases(r) for r in raw_tags]

    client = _openai_client()
    label_emb = get_label_embeddings(client, labels, EMBED_CACHE)
    ranked = align_and_rank(phrase_lists, labels, label_emb, client)

    results = []
    rp5, rp10, tr = [], [], []
    for d, phrases, ranked_terms in zip(docs, phrase_lists, ranked):
        gt = d["ground_truth_reactions"]
        r5 = rank_precision_at_k(ranked_terms, gt, 5)
        r10 = rank_precision_at_k(ranked_terms, gt, 10)
        t = term_recall(ranked_terms, gt)
        rp5.append(r5)
        rp10.append(r10)
        tr.append(t)
        results.append(
            {
                "id": d["id"],
                "free_text_phrases": phrases,
                "ranked_reaction_labels": ranked_terms,
                "ground_truth_reactions": gt,
                "rp_at_5": r5,
                "rp_at_10": r10,
                "term_recall": t,
            }
        )

    n = len(results) or 1
    summary = {
        "method": "tada_tag_embed_align",
        "total_documents": len(results),
        "avg_rp_at_5": sum(rp5) / n,
        "avg_rp_at_10": sum(rp10) / n,
        "avg_term_recall": sum(tr) / n,
        "elapsed_s": round(elapsed, 2),
    }
    manifest["metrics"] = summary

    (out_dir / "predictions.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))
    print(out_dir / "summary.json")


if __name__ == "__main__":
    main()
