#!/usr/bin/env python3
"""Fair baselines for BioDEX: DocETL-style and LOTUS-style, on the SAME 100 docs.

Both official systems' native BioDEX recipe puts the entire 24313-term MedDRA
vocabulary into the prompt so the LLM emits official terms directly. That single
prompt is ~150k tokens and does not fit gpt-4.1/gpt-5.5. To make the comparison
BOTH fair (same information as TADA) and actually runnable, every system here
first retrieves the top-K candidate terms for each paper via the SAME
text-embedding-3-small nearest-neighbour step TADA uses, then the baseline LLM
ranks/selects official terms from that candidate shortlist.

  * docetl style: one map prompt per paper (mirrors biodex.yaml's
    extract_and_match_reactions), asked to return ranked_reaction_labels chosen
    from the shortlisted official terms.
  * lotus  style: a sem_extract-then-rank framing — extract reactions, then map
    each to the single best official term from the shortlist, ranked.

Scoring reuses the official DocETL evaluator verbatim (rank_precision@k,
term_recall), identical to run_biodex_tada.py.
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
EMBED_CACHE = ROOT / "bench/results/biodex_tada/label_embeddings.npz"
EMBED_MODEL = "text-embedding-3-small"
TOP_K_CANDIDATES = 200


# ---- official evaluator (verbatim) ---------------------------------------- #
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
    if not target_terms:
        return 0.0
    return len(pred_terms & target_terms) / len(target_terms)


# ---- embeddings / retrieval (same as TADA path) --------------------------- #
def _client():
    """Embedding client (see run_biodex_tada._openai_client)."""
    from openai import OpenAI

    base_url = (
        os.environ.get("TA_EMBED_BASE_URL", "").strip()
        or os.environ.get("TA_OPENAI_BASE_URL", "").strip()
    )
    api_key = (
        os.environ.get("TA_EMBED_API_KEY", "").strip()
        or os.environ.get("TA_OPENAI_API_KEY", "").strip()
        or "sk-none"
    )
    if not base_url:
        raise SystemExit("Set TA_EMBED_BASE_URL (or TA_OPENAI_BASE_URL)")
    return OpenAI(base_url=base_url, api_key=api_key)


def _llm_client():
    """Generation client for baseline reasoning; may differ from embeddings."""
    from openai import OpenAI

    base_url = (
        os.environ.get("TA_OPENAI_BASE_URL", "").strip()
        or os.environ.get("TA_EMBED_BASE_URL", "").strip()
    )
    api_key = (
        os.environ.get("TA_OPENAI_API_KEY", "").strip()
        or os.environ.get("TA_EMBED_API_KEY", "").strip()
        or "sk-none"
    )
    if not base_url:
        raise SystemExit("Set TA_OPENAI_BASE_URL")
    return OpenAI(base_url=base_url, api_key=api_key)


def embed_texts(client, texts: list[str], batch: int = 256):
    import numpy as np

    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + batch])
        out.extend(d.embedding for d in resp.data)
    arr = np.asarray(out, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def load_label_embeddings(labels: list[str]):
    import numpy as np

    if not EMBED_CACHE.exists():
        raise SystemExit(f"Missing label cache {EMBED_CACHE}; run run_biodex_tada.py first.")
    data = np.load(EMBED_CACHE, allow_pickle=True)
    if list(data["labels"]) != labels:
        raise SystemExit("Label cache does not match this dataset's possible_labels.")
    return data["emb"]


def retrieve_candidates(client, doc_text: str, labels, label_emb, k: int) -> list[str]:
    import numpy as np

    q = embed_texts(client, [doc_text[:8000]])
    sims = (q @ label_emb.T)[0]
    idx = np.argsort(-sims)[:k]
    return [labels[int(i)] for i in idx]


# ---- baseline LLM call ---------------------------------------------------- #
def chat(client, model: str, system: str, user: str) -> str:
    # Endpoints differ: the gpt-5.5 endpoint serves the Responses API, while
    # OpenRouter/Llama serves chat/completions. Pick via TA_OPENAI_WIRE_API.
    wire = os.environ.get("TA_OPENAI_WIRE_API", "responses").strip().lower()
    if wire in {"chat", "completions", "chat/completions"}:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
    resp = client.responses.create(
        model=model,
        instructions=system,
        input=user,
    )
    text = getattr(resp, "output_text", None)
    if text:
        return text
    chunks: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                chunks.append(t)
    return "".join(chunks)


DOCETL_SYS = (
    "You are an expert medical reviewer specializing in adverse drug reactions (ADR)."
)


def docetl_prompt(fulltext: str, candidates: list[str]) -> str:
    cand = "\n".join(candidates)
    return f"""Your task:
1. Read the medical paper and identify every ADR or medical condition reported by patients.
2. Match each identified reaction to the single best-fitting official medical reaction term from the provided candidate list.
3. Produce a ranked JSON array named ranked_reaction_labels ordered from most to least relevant.

<<MEDICAL_PAPER>>
{fulltext[:12000]}
</MEDICAL_PAPER>>

<<OFFICIAL_REACTION_TERMS>> (choose ONLY from these)
{cand}
</OFFICIAL_REACTION_TERMS>>

Guidelines:
- Only return official terms that appear in the candidate list above.
- If multiple official terms could match, choose the most specific one.
- Return ONLY a valid JSON array, e.g. ["term-A", "term-B"]. No commentary, no markdown."""


LOTUS_SYS = "You apply semantic operators (sem_extract then sem_map) over a medical paper."


def lotus_prompt(fulltext: str, candidates: list[str]) -> str:
    cand = "\n".join(candidates)
    return f"""Step 1 (sem_extract): extract every adverse drug reaction / patient-reported
condition from the paper below as short clinical phrases.
Step 2 (sem_map + rank): map each extracted reaction to the single best-matching
official term drawn ONLY from the candidate vocabulary, then rank the mapped
official terms from most to least clearly reported.

PAPER:
{fulltext[:12000]}

CANDIDATE OFFICIAL TERMS (choose ONLY from these):
{cand}

Return ONLY a JSON array of the ranked official terms, e.g. ["term-A", "term-B"].
No commentary, no markdown."""


def parse_labels(raw: str, candidates: list[str]) -> list[str]:
    text = (raw or "").strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    parts: list[str] = []
    if m:
        try:
            parts = json.loads(m.group(0))
        except Exception:
            parts = re.split(r"[,;\n]", m.group(0).strip("[]"))
    else:
        parts = re.split(r"[,;\n]", text)
    cand_lower = {c.lower(): c for c in candidates}
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        s = str(p).strip().strip('"\'').strip()
        if not s:
            continue
        key = s.lower()
        term = cand_lower.get(key, s)  # snap to candidate casing when possible
        if term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["docetl", "lotus"], required=True)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default=os.environ.get("TA_OPENAI_MODEL", "gpt-5.5"))
    ap.add_argument("--gap", type=float, default=float(os.environ.get("TA_TAG_GAP_S", "8")))
    ap.add_argument("--topk", type=int, default=TOP_K_CANDIDATES)
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = json.loads(Path(args.data).read_text())
    if args.limit:
        docs = docs[: args.limit]
    labels = [l.strip() for l in docs[0]["possible_labels"].split("\n") if l.strip()]

    client = _client()
    llm = _llm_client()
    label_emb = load_label_embeddings(labels)
    build = docetl_prompt if args.style == "docetl" else lotus_prompt
    system = DOCETL_SYS if args.style == "docetl" else LOTUS_SYS

    predictions: list[dict[str, Any]] = []
    started = time.time()
    for i, d in enumerate(docs):
        gt = d.get("ground_truth_reactions", [])
        full = d["fulltext_processed"]
        cands = retrieve_candidates(client, full, labels, label_emb, args.topk)
        preds: list[str] = []
        for attempt in range(6):
            try:
                raw = chat(llm, args.model, system, build(full, cands))
                preds = parse_labels(raw, cands)
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if attempt == 5:
                    print(f"[{args.style}] doc {i} failed: {msg[:160]}", file=sys.stderr)
                else:
                    is_rate = "429" in msg or "too many" in msg.lower()
                    time.sleep(min(60.0, (2 ** attempt) * (5.0 if is_rate else 1.0)))
        predictions.append(
            {
                "id": d.get("id"),
                "ranked_reaction_labels": preds,
                "ground_truth_reactions": gt,
                "rp_at_5": rank_precision_at_k(preds, gt, 5),
                "rp_at_10": rank_precision_at_k(preds, gt, 10),
                "term_recall": term_recall(preds, gt),
            }
        )
        print(f"[{args.style}] doc {i+1}/{len(docs)} preds={len(preds)}", file=sys.stderr)
        time.sleep(args.gap)

    n = len(predictions) or 1
    summary = {
        "method": f"baseline_{args.style}_retrieval_rank",
        "model": args.model,
        "topk_candidates": args.topk,
        "total_documents": len(predictions),
        "avg_rp_at_5": sum(p["rp_at_5"] for p in predictions) / n,
        "avg_rp_at_10": sum(p["rp_at_10"] for p in predictions) / n,
        "avg_term_recall": sum(p["term_recall"] for p in predictions) / n,
        "elapsed_s": time.time() - started,
    }
    (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "style": args.style,
                "model": args.model,
                "topk": args.topk,
                "data": str(args.data),
                "note": "candidates retrieved via text-embedding-3-small top-K, same as TADA",
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
