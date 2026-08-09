#!/usr/bin/env python3
"""Unified matrix runner v2 — new model configuration (2026-07-26).

Config changes vs run_matrix.py (v1, opus-4.8 era):
  host agent:          claude-opus-5      (was claude-opus-4.8)
  semantic sub-calls:  claude-sonnet-5    (was claude-opus-4.8; restores the
                       strong-host / weaker-atoms deployment assumption)
  judge:               unchanged gpt-5.5 (human calibration stays valid)

Task set: 23 cases. Q11 (doc-to-table), Q12 (semantic imputation), Q18
(semantic join) are EXCLUDED from the benchmark runs by decision 2026-07-26:
text-to-table and semantic join are out-of-scope for TADA and move to the
Discussion section.

Methods:
  skill_on     textanalysis skill visible (semanticoperators hidden)
  operator_on  semanticoperators skill visible (textanalysis hidden)
  skill_off    both skills hidden
  code_only    both skills hidden, deterministic-code-only constraint
  direct       single /v1/messages call, no tools

Output: matrix_v2_runs/<method>_r<k>/<qid>/
Resumable: existing claude_output.json / raw_response.md dirs are skipped.
"""
import argparse, json, os, shutil, subprocess, time, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
INPUTS = HERE / "full26_inputs"
RUNS = HERE / "matrix_v2_runs"
PROJECT = Path(os.environ.get("TADA_PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))
SKILLS = PROJECT / ".claude/skills"
TA, TA_BAK = SKILLS / "textanalysis", SKILLS / "_textanalysis_disabled_full26"
OPS, OPS_BAK = SKILLS / "semanticoperators", SKILLS / "_semanticoperators_disabled"
QUERIES = json.loads((HERE / "queries_snapshot_v2.json").read_text())
EXP_INPUTS = json.loads((HERE / "expansion_inputs_map.json").read_text())

EXCLUDE = {"Q11", "Q12", "Q18"}          # text-to-table & semantic join -> Discussion
# Q41/Q43 are already absent from expansion_queries (same boundary decision).
# Old cases run first; expansion cases (Q27+) queue after them.
ORDERED_QS = ([q for q in QUERIES if q not in EXCLUDE and int(q[1:]) <= 26] +
              [q for q in QUERIES if q not in EXCLUDE and int(q[1:]) >= 27])
HOST_MODEL = "claude-opus-5"
ATOM_MODEL = "claude-sonnet-5"
BASE_URL = os.environ.get("TADA_ANTHROPIC_BASE_URL", "http://localhost:8000")


def set_visible(ta_on: bool, ops_on: bool):
    if ta_on and TA_BAK.exists() and not TA.exists(): TA_BAK.rename(TA)
    if not ta_on and TA.exists() and not TA_BAK.exists(): TA.rename(TA_BAK)
    if ops_on and OPS_BAK.exists() and not OPS.exists(): OPS_BAK.rename(OPS)
    if not ops_on and OPS.exists() and not OPS_BAK.exists(): OPS.rename(OPS_BAK)


def input_src(q):
    if q in EXP_INPUTS:
        return Path(EXP_INPUTS[q])
    return INPUTS / f"{q}.csv"


def claude_run(q, method, td):
    src = input_src(q)
    dst = td / src.name
    shutil.copy2(src, dst)
    input_desc = f"Input CSV: {dst}."
    if method == "skill_on":
        note = ("Use the textanalysis skill and its ta.tag/ta.categorize primitives whenever applicable; "
                "compose with pandas for symbolic/group/join steps. ")
    elif method == "operator_on":
        note = ("Use the semanticoperators skill (sem_filter/sem_topk/sem_group_by/sem_agg/sem_join/sem_map) "
                "whenever applicable; compose with pandas for symbolic/group/join steps. ")
    elif method == "skill_off":
        note = ("Solve directly without any textanalysis/TADA skill. You may use ordinary Python/pandas "
                "and your own semantic reasoning. ")
    else:  # code_only
        note = ("STRICT CONSTRAINT: you may ONLY use deterministic code (pandas, regex, keyword lists, "
                "numeric rules). You must NOT semantically judge individual rows yourself and must NOT "
                "call any LLM. Encode your best heuristic rules in code, run them, and save the outputs. ")
    prompt = (f"{input_desc} {QUERIES[q]} {note}Write every requested output inside {td}. "
              "Do not ask for permissions or clarification; execute the task now. Never overwrite the input. "
              "Run every step to completion in this session: do NOT launch background jobs, do NOT exit "
              "while any step is still running; write all output files before finishing. "
              "Include a concise method.md describing actual operations.")
    env = os.environ.copy()
    env.update({"ANTHROPIC_BASE_URL": BASE_URL, "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "dummy"),
                "TA_MODEL": ATOM_MODEL, "TA_FALLBACK_MODEL": ""})
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", HOST_MODEL,
           "--dangerously-skip-permissions"]
    try:
        r = subprocess.run(cmd, cwd=PROJECT, env=env, capture_output=True, text=True, timeout=1500)
        (td / "claude_output.json").write_text(r.stdout)
        if r.stderr: (td / "stderr.txt").write_text(r.stderr)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        (td / "claude_output.json").write_text(json.dumps({"error": "timeout"}))
        return False


def direct_run(q, td):
    src = input_src(q)
    shutil.copy2(src, td / src.name)
    data = f"=== FILE {src.name} ===\n" + src.read_text(errors="replace")
    if len(data) > 350_000:
        data = data[:350_000] + "\n...[truncated]"
    prompt = (f"{QUERIES[q]}\n\nYou cannot execute code. Read the data below and produce the requested "
              f"output DIRECTLY as CSV/JSON/markdown text in your reply. For CSV output, start it with a "
              f"line 'FILE: result.csv' followed by the complete CSV content (all rows, no elision). "
              f"For markdown/JSON use 'FILE: result.md' / 'FILE: result.json' likewise.\n\n{data}")
    body = json.dumps({"model": HOST_MODEL, "max_tokens": 32000,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{BASE_URL}/v1/messages", data=body,
                                 headers={"content-type": "application/json",
                                          "x-api-key": os.environ.get("ANTHROPIC_API_KEY", "dummy"),
                                          "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=880) as r:
            resp = json.load(r)
    except Exception as e:
        (td / "error.txt").write_text(str(e))
        return False
    text = "".join(c.get("text", "") for c in resp.get("content", []))
    (td / "raw_response.md").write_text(text)
    cur, buf = None, []
    for line in text.splitlines():
        if line.strip().startswith("FILE:"):
            if cur: (td / cur).write_text("\n".join(buf).strip() + "\n")
            cur, buf = line.split("FILE:", 1)[1].strip().strip("`"), []
        elif cur:
            if line.strip().startswith("```"): continue
            buf.append(line)
    if cur: (td / cur).write_text("\n".join(buf).strip() + "\n")
    return True


VISIBILITY = {"skill_on": (True, False), "operator_on": (False, True),
              "skill_off": (False, False), "code_only": (False, False),
              "direct": (False, False)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True)
    ap.add_argument("--repeats", default="1")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    methods = a.methods.split(",")
    repeats = [int(x) for x in a.repeats.split(",")]
    qs = [x.strip().upper() for x in a.only.split(",") if x.strip()] or ORDERED_QS
    try:
        for m in methods:
            set_visible(*VISIBILITY[m])
            for r in repeats:
                tag = f"{m}_r{r}"
                base = RUNS / tag
                for q in qs:
                    if q in EXCLUDE:
                        print(f"excluded {q}, skipping", flush=True)
                        continue
                    td = base / q
                    if (td / "claude_output.json").exists() or (td / "raw_response.md").exists():
                        print(f"skip {tag}/{q}", flush=True)
                        continue
                    td.mkdir(parents=True, exist_ok=True)
                    t0 = time.time()
                    ok = direct_run(q, td) if m == "direct" else claude_run(q, m, td)
                    print(f"{tag}/{q} ok={ok} {time.time()-t0:.0f}s", flush=True)
                (base / "run_info.json").write_text(json.dumps(
                    {"method": m, "repeat": r, "host_model": HOST_MODEL,
                     "atom_model": ATOM_MODEL, "excluded": sorted(EXCLUDE),
                     "date": time.strftime("%F %T")}, indent=1))
    finally:
        set_visible(True, True)
    print("MATRIX V2 DONE")

if __name__ == "__main__":
    main()
