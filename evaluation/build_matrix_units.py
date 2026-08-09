#!/usr/bin/env python3
"""Build judge units for the full method matrix (anchor = skill_on_r1).

For every pair (method_rK vs skill_on_r1) on row-aligned tasks:
  - equal values (normalized) -> deterministic tie, not judged
  - missing rows in candidate -> coverage failure, recorded, not judged
  - differing values -> judge unit
Writes matrix_units.jsonl + matrix_coverage.json.
"""
import csv, json, re
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "full26_runs"
INPUTS = HERE / "full26_inputs"
ANCHOR = RUNS / "full26_20260722" / "skill_on"
QUERIES = json.loads((HERE / "queries_snapshot.json").read_text())

KEYS = {"Q01":"CustomerID","Q02":"TicketID","Q03":"ContractID","Q09":"ContractID","Q10":"IncidentID",
        "Q12":"SubmissionID","Q19":"ReviewID","Q20":"NoteID","Q23":"RecordID","Q24":"ProductID",
        "Q25":"RecordID","Q26":"TaskID"}
SEM = {"Q01":["sentiment"],"Q02":["priority"],"Q03":["risk"],"Q09":["party a|party_a","party b|party_b"],
       "Q10":["time","location","scope","root"],"Q12":["revised"],"Q19":["sentiment|score"],
       "Q20":["relevance|score"],"Q23":["normalized"],"Q24":["summary"],"Q25":["notes_en|english|translat"],
       "Q26":["priority"]}
TEXTCOL = {"Q01":"customer_feedback","Q02":"customer_notes","Q03":"contract_clause","Q09":"contract_notes",
           "Q10":"incident_report","Q12":"original_document","Q19":"review","Q20":"notes","Q23":"country",
           "Q24":"description","Q25":"notes","Q26":"description"}

def load(p):
    for enc in ("utf-8-sig","latin-1"):
        try: return list(csv.DictReader(open(p, newline="", encoding=enc)))
        except UnicodeDecodeError: pass
    return []

def best_csv(d, q):
    cands=[p for p in Path(d,q).glob("*.csv") if p.name not in (f"{q}.csv","Q18_1.csv","Q18_2.csv","Q11_structured_output.csv")]
    return max(cands, key=lambda x:x.stat().st_size) if cands else None

def find_col(rows, pat):
    opts = pat.lower().split("|")
    for h in rows[0].keys():
        if any(o in (h or "").lower() for o in opts): return h
    return None

def norm(v): return re.sub(r"\s+"," ",(v or "").strip().lower())

def method_dir(m):
    if m == "skill_off_r1": return RUNS/"full26_20260722"/"skill_off"
    return RUNS/m

methods = (["skill_off_r1","skill_off_r2","skill_off_r3","direct_r1","direct_r2","direct_r3",
            "code_only_r1","code_only_r2","code_only_r3","skill_on_r2","skill_on_r3"])

inputs_text = {}
for q in KEYS:
    inputs_text[q] = {r[KEYS[q]]: r.get(TEXTCOL[q],"") for r in load(INPUTS/f"{q}.csv")}

units, coverage = [], {}
for m in methods:
    md = method_dir(m)
    for q, key in KEYS.items():
        a_p, b_p = best_csv(ANCHOR,q), best_csv(md,q)
        a = load(a_p) if a_p else []
        b = load(b_p) if b_p else []
        cov = {"anchor_rows": len(a), "cand_rows": len(b), "missing": 0, "equal": 0, "diff": 0}
        if not a or not b or key not in (a[0] if a else {}) or key not in (b[0] if b else {}):
            cov["missing"] = len(a) or 200
            coverage[f"{m}|{q}"] = cov
            continue
        am = {r[key]: r for r in a}
        bm = {r[key]: r for r in b}
        for pat in SEM[q]:
            ca, cb = find_col(a, pat), find_col(b, pat)
            for k, ar in am.items():
                br = bm.get(k)
                if br is None:
                    cov["missing"] += 1
                    continue
                va, vb = ar.get(ca,"") if ca else "", br.get(cb,"") if cb else ""
                if norm(va) == norm(vb):
                    cov["equal"] += 1
                else:
                    cov["diff"] += 1
                    units.append({"pair": m, "qid": q, "row_id": k, "field": pat if len(SEM[q])>1 else "",
                                  "query": QUERIES[q], "text": inputs_text[q].get(k,""),
                                  "anchor": va, "cand": vb})
        coverage[f"{m}|{q}"] = cov

with open(HERE/"matrix_units.jsonl","w") as f:
    for u in units:
        f.write(json.dumps(u, ensure_ascii=False)+"\n")
json.dump(coverage, open(HERE/"matrix_coverage.json","w"), indent=1)
print(f"units={len(units)} -> projected calls={len(units)*6}")
per = {}
for u in units:
    per[u["pair"]] = per.get(u["pair"],0)+1
for m in methods:
    print(f"  {m:15s} {per.get(m,0):5d}")
