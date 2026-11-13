#!/usr/bin/env python3
"""Run a TADA full-clause extraction baseline on the DocETL CUAD split."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    REPO_ROOT,
    base_manifest,
    finish_manifest,
    import_tada,
    make_run_dir,
    summarize_trace,
    temporary_tada_env,
    write_csv,
    write_json,
    write_jsonl,
    write_readme,
)


DOCETL_REASONING = REPO_ROOT / "baseline_paper" / "docetl" / "experiments" / "reasoning"
TEST_JSON = DOCETL_REASONING / "data" / "test" / "cuad.json"
GROUND_TRUTH_CSV = DOCETL_REASONING / "data" / "CUAD-master_clauses.csv"
OFFICIAL_EVAL = DOCETL_REASONING / "evaluation" / "cuad.py"
DOCETL_PIPELINE = DOCETL_REASONING / "pipelines" / "cuad.yaml"
DOCETL_VENV_PYTHON = Path("/tmp/tada-docetl-venv/bin/python")

CLAUSE_TYPES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice to Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit of Customers",
    "Competitive Restriction Exception",
    "No-Solicit of Employees",
    "Non-Disparagement",
    "Termination for Convenience",
    "Right of First Refusal",
    "Change of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restriction",
    "Minimum Commitment",
    "Volume Restriction",
    "IP Ownership Assignment",
    "Joint IP Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate IP License-Licensor",
    "Affiliate IP License-Licensee",
    "Unlimited License",
    "Irrevocable or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap on Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not to Sue",
    "Third Party Beneficiary",
]

TUNED_VARIANTS = {"untuned", "tuned_clause_family_v1"}

TUNED_CLAUSE_NOTES = {
    "Covenant Not to Sue": "a promise, covenant, release, or agreement not to sue or bring claims; do not confuse it with ordinary indemnity language",
    "Irrevocable or Perpetual License": "license rights described as irrevocable, perpetual, permanent, continuing, or surviving termination; do not output a generic license grant unless those durability terms are present",
    "IP Ownership Assignment": "assignment or ownership transfer of intellectual property, work product, inventions, data, deliverables, or improvements; distinguish ownership transfer from a mere license",
    "Volume Restriction": "minimum, maximum, quota, cap, volume, quantity, unit, territory-volume, or purchase/sales amount restrictions; distinguish from monetary payment minimums",
    "Source Code Escrow": "deposit, escrow, release, or third-party holding of source code or software materials",
}

CLAUSE_NAME_BY_NORMALIZED = {
    "".join(ch for ch in clause.lower() if ch.isalnum()): clause for clause in CLAUSE_TYPES
}
CLAUSE_ALIASES = {
    "ipassignment": "IP Ownership Assignment",
    "ipownership": "IP Ownership Assignment",
    "intellectualpropertyassignment": "IP Ownership Assignment",
    "covenantnottosue": "Covenant Not to Sue",
    "nottosue": "Covenant Not to Sue",
    "perpetuallicense": "Irrevocable or Perpetual License",
    "irrevocablelicense": "Irrevocable or Perpetual License",
    "irrevocableperpetuallicense": "Irrevocable or Perpetual License",
    "volumerestrictions": "Volume Restriction",
    "volumecap": "Volume Restriction",
    "sourcecodeescrow": "Source Code Escrow",
    "codeescrow": "Source Code Escrow",
}


def load_rows(offset: int, limit: int | None, context_limit_chars: int | None) -> list[dict[str, Any]]:
    rows = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    if limit is not None:
        rows = rows[offset : offset + limit]
    else:
        rows = rows[offset:]
    out = []
    for row in rows:
        doc = str(row["document"])
        truncated = False
        if context_limit_chars and len(doc) > context_limit_chars:
            doc = doc[:context_limit_chars] + "\n...[truncated for benchmark prompt budget]"
            truncated = True
        out.append(
            {
                "document": doc,
                "name": row["name"],
                "id": row["id"],
                "original_document_chars": len(str(row["document"])),
                "truncated": truncated,
            }
        )
    return out


def make_query(variant: str = "untuned") -> str:
    clause_list = "\n".join(f"{idx + 1}. {name}" for idx, name in enumerate(CLAUSE_TYPES))
    base = (
        "For each contract independently, extract text spans for the CUAD clause types below. "
        "Return the value for each item as a JSON array, not prose. Each array element must be an object "
        "with exactly two keys: clause_type and text_span. Include only clause types that are present and supported by an exact or near-exact span in the contract. "
        "Use the clause_type names exactly as listed. Keep each text_span as the shortest contract span that supports the clause; do not summarize, do not invent clauses, and omit absent or uncertain clause types.\n\n"
        f"CUAD clause types:\n{clause_list}"
    )
    if variant == "untuned":
        return base
    if variant != "tuned_clause_family_v1":
        raise ValueError(f"unknown CUAD variant: {variant}")
    tuned_notes = "\n".join(f"- {name}: {note}." for name, note in TUNED_CLAUSE_NOTES.items())
    return (
        "For each contract independently, extract CUAD clause evidence spans as structured data. "
        "Return a JSON array, not prose. Each array element must contain exactly two keys: clause_type and text_span. "
        "Use clause_type names exactly as listed below; do not create abbreviations, aliases, merged clause names, explanations, or confidence fields. "
        "Output only clauses that are explicitly supported by a contract span. If a clause is absent or only weakly implied, omit it. "
        "Keep text_span short: prefer the sentence, paragraph, bullet, or subsection containing the operative language, not the entire agreement. "
        "When related clauses occur near each other, output separate entries only when each clause has its own supporting language. "
        "Scan all clause types, but pay special attention to the ambiguous CUAD families below because they are often confused.\n\n"
        f"Ambiguous clause guidance:\n{tuned_notes}\n\n"
        f"Allowed CUAD clause_type names:\n{clause_list}"
    )


def normalize_clause_type(value: str) -> str:
    normalized = "".join(ch for ch in value.lower() if ch.isalnum())
    return CLAUSE_NAME_BY_NORMALIZED.get(normalized) or CLAUSE_ALIASES.get(normalized) or value.strip()


def parse_clause_output(raw: Any) -> tuple[list[dict[str, str]], str]:
    value = raw
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"null", "none", "[]"}:
            return [], "empty_string"
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            clauses = []
            for part in text.split(";"):
                left, sep, right = part.partition(":")
                if sep and left.strip() and right.strip():
                    clauses.append({"clause_type": left.strip(), "text_span": right.strip()})
            return clauses, "semicolon_fallback" if clauses else "unparsed_string"
    if isinstance(value, dict):
        if isinstance(value.get("clauses"), list):
            value = value["clauses"]
        elif isinstance(value.get("Clauses"), list):
            value = value["Clauses"]
        else:
            return [], "dict_without_clauses"
    if not isinstance(value, list):
        return [], f"unsupported_{type(value).__name__}"
    clauses = []
    for item in value:
        if not isinstance(item, dict):
            continue
        clause_type = str(item.get("clause_type") or item.get("Clause Type") or item.get("type") or "").strip()
        text_span = str(item.get("text_span") or item.get("Text Span") or item.get("span") or "").strip()
        if clause_type and text_span:
            clauses.append({"clause_type": normalize_clause_type(clause_type), "text_span": text_span})
    return clauses, "ok"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            return str(value)
    return value


def evaluate_official(run_dir: Path, raw_outputs: Path, input_rows: Path) -> dict[str, Any]:
    eval_script = run_dir / "evaluate_with_docetl_official.py"
    eval_metrics = run_dir / "official_eval_metrics.json"
    eval_script.write_text(
        """
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
results_file = sys.argv[2]
ground_truth_file = sys.argv[3]
original_json_file = sys.argv[4]
output_file = sys.argv[5]
sys.path.insert(0, str(repo / 'baseline_paper' / 'docetl'))
from experiments.reasoning.evaluation.cuad import evaluate_results

metrics = evaluate_results('tada_cuad_full_extraction', results_file, ground_truth_file, original_json_file)

def clean(value):
    try:
        import math
        import numpy as np
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, np.generic):
            return clean(value.item())
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value

Path(output_file).write_text(json.dumps(clean(metrics), indent=2) + '\\n', encoding='utf-8')
""".lstrip(),
        encoding="utf-8",
    )
    python = DOCETL_VENV_PYTHON if DOCETL_VENV_PYTHON.exists() else Path(sys.executable)
    proc = subprocess.run(
        [
            str(python),
            str(eval_script),
            str(REPO_ROOT),
            str(raw_outputs),
            str(GROUND_TRUTH_CSV),
            str(input_rows),
            str(eval_metrics),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    write_json(
        run_dir / "official_eval_command.json",
        {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"official CUAD evaluator failed: {proc.stderr[-1000:]}")
    return json.loads(eval_metrics.read_text(encoding="utf-8"))


def run(limit: int | None, offset: int, context_limit_chars: int | None, run_id: str | None, variant: str = "untuned") -> Path:
    if variant not in TUNED_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(TUNED_VARIANTS)}, got {variant!r}")
    workload = "CUAD full clause extraction"
    system = "TADA gpt-4.1 tuned" if variant != "untuned" else "TADA gpt-4.1"
    system_dir = "tada_gpt41_tuned" if variant != "untuned" else "tada_gpt41"
    run_dir = make_run_dir("RQ1", "cuad_full_clause_extraction", system_dir, rid=run_id)
    trace_path = run_dir / "ta_debug_trace.jsonl"


    # Keep one contract per TADASkill call: CUAD documents are long and outputs can be large.
    os.environ["TA_TAG_BATCH_ROWS"] = "1"
    os.environ["TA_MIN_SPLIT_BATCH_ROWS"] = "1"
    os.environ["TA_MAX_CONCURRENCY"] = "1"
    os.environ.setdefault("TA_CALL_TIMEOUT_S", "900")
    os.environ.setdefault("TA_MAX_RETRIES", "1")
    os.environ.setdefault("TA_OPENAI_MAX_TOKENS", "8000")
    os.environ.setdefault("TA_OPENAI_API_KEY", "dummy")

    manifest = base_manifest(
        rq="RQ1",
        workload=workload,
        system=system,
        source_status="rerun",
        model=DEFAULT_MODEL,
        base_url=DEFAULT_BASE_URL,
        runner="run_tada_cuad_full_extraction.py",
        data_paths=[TEST_JSON, GROUND_TRUTH_CSV, OFFICIAL_EVAL, DOCETL_PIPELINE],
        notes=(
            "TADA full-clause extraction on the official DocETL CUAD test split, scored by the official DocETL CUAD evaluator. "
            f"Prompt/schema variant: {variant}."
        ),
    )
    manifest["variant"] = variant
    rows = load_rows(offset=offset, limit=limit, context_limit_chars=context_limit_chars)
    input_path = run_dir / "input_rows.json"
    write_json(input_path, rows)
    query = make_query(variant)
    prompt_dir = run_dir / "prompts_or_queries"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    write_json(prompt_dir / "queries.json", {"cuad_full_extraction": query, "clause_types": CLAUSE_TYPES, "variant": variant})
    if variant != "untuned":
        write_json(
            prompt_dir / "tuning_diff.json",
            {
                "variant": variant,
                "base_variant": "untuned",
                "target_clause_families": list(TUNED_CLAUSE_NOTES),
                "changes": [
                    "adds explicit ambiguous-clause guidance",
                    "requires exact allowed clause_type names",
                    "asks for short operative evidence spans",
                    "normalizes common clause_type aliases before official evaluation",
                ],
            },
        )

    started = time.time()
    raw_values: list[Any] = []
    parsed_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    row_errors: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "checkpoint_outputs.jsonl"

    with temporary_tada_env(DEFAULT_BASE_URL, DEFAULT_MODEL, trace_path):
        ta = import_tada()
        for idx, row in enumerate(rows):
            row_started = time.time()
            try:
                value = ta.tag([row["document"]], "document", query)[0]
                clauses, parse_status = parse_clause_output(value)
            except Exception as exc:  # keep partial artifacts instead of losing a long run
                value = {"error": repr(exc)}
                clauses = []
                parse_status = "error"
                row_errors.append({"row_index": idx, "name": row["name"], "error": repr(exc)})
            elapsed = time.time() - row_started
            raw_values.append(value)
            output_row = {"document": row["document"], "name": row["name"], "id": row["id"], "clauses": clauses}
            output_rows.append(output_row)
            parsed = {
                "row_index": idx,
                "id": row["id"],
                "name": row["name"],
                "clauses_extracted": len(clauses),
                "parse_status": parse_status,
                "row_elapsed_s": elapsed,
                "truncated": row["truncated"],
                "original_document_chars": row["original_document_chars"],
            }
            parsed_rows.append(parsed)
            with checkpoint_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"parsed": parsed, "raw": value, "clauses": clauses}, ensure_ascii=False) + "\n")
            print(f"[cuad-tada] row {idx + 1}/{len(rows)} clauses={len(clauses)} status={parse_status} elapsed={elapsed:.1f}s", file=sys.stderr)

    elapsed_s = time.time() - started
    raw_path = run_dir / "raw_outputs.json"
    parsed_path = run_dir / "parsed_predictions.jsonl"
    write_json(raw_path, output_rows)
    write_json(run_dir / "raw_tag_values.json", raw_values)
    write_jsonl(parsed_path, parsed_rows)
    trace_summary = summarize_trace(trace_path)
    eval_metrics = evaluate_official(run_dir, raw_path, input_path)
    metrics = {
        **json_safe(eval_metrics),
        "rows": len(rows),
        "offset": offset,
        "limit": limit if limit is not None else "all",
        "context_limit_chars": context_limit_chars or "full",
        "truncated_input_rows": sum(1 for row in rows if row["truncated"]),
        "row_errors": len(row_errors),
        "parsed_rows": sum(1 for row in parsed_rows if row["parse_status"] in {"ok", "semicolon_fallback", "empty_string"}),
        "elapsed_s": elapsed_s,
        "variant": variant,
        **trace_summary,
    }
    metrics["native_metric"] = "avg_precision/avg_recall/avg_f1"
    metrics["precision"] = metrics.get("avg_precision")
    metrics["recall"] = metrics.get("avg_recall")
    metrics["f1"] = metrics.get("avg_f1")
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "cost_latency.json", {"elapsed_s": elapsed_s, **trace_summary})
    write_json(run_dir / "row_errors.json", row_errors)
    write_csv(run_dir / "parsed_predictions.csv", parsed_rows)
    write_readme(
        run_dir,
        "TADA CUAD Full Clause Extraction",
        f"TADA full-clause extraction on the official DocETL CUAD test split. Outputs are scored with the official DocETL CUAD evaluator against CUAD-master_clauses.csv. Variant: {variant}.",
    )
    finish_manifest(
        run_dir,
        manifest,
        metrics=metrics,
        artifacts={
            "input_rows": "input_rows.json",
            "raw_outputs": "raw_outputs.json",
            "raw_tag_values": "raw_tag_values.json",
            "parsed_predictions": "parsed_predictions.jsonl",
            "metrics": "metrics.json",
            "cost_latency": "cost_latency.json",
            "prompts_or_queries": "prompts_or_queries/queries.json",
            "checkpoint": "checkpoint_outputs.jsonl",
            "official_eval_command": "official_eval_command.json",
        },
    )
    print(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Optional number of CUAD test rows to run.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--context-limit-chars", type=int, default=0, help="Optional truncation limit; 0 means full document.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--variant", default="untuned", choices=sorted(TUNED_VARIANTS))
    args = parser.parse_args()
    run(
        limit=args.limit,
        offset=args.offset,
        context_limit_chars=args.context_limit_chars or None,
        run_id=args.run_id,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
