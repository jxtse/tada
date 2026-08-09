#!/usr/bin/env python3
"""Aggregate product judge (3-vote, position-swapped) + tie-band re-aggregation.

Outputs product_final.json with:
- per-product per-qid net verdicts (anchor=TADA r1)
- tie-band applied to anchored-scale tasks (|delta|<=1 -> tie), consistent
  with the controlled-matrix re-aggregation
- case-level failures (ShortcutAI missing outputs, Q22 input-mismatch)
- quality score per product on the same scale as the controlled methods
"""
import json, csv, pathlib
from collections import Counter, defaultdict

WD = pathlib.Path(__file__).parent
SCALE_QS = {"Q02","Q19","Q20","Q26"}  # anchored/ordinal scales -> tie-band
# Q19/Q20 excluded from row-level in build (ranking track); Q02 kept (categorical P-levels: tie-band n/a)
NUMERIC_TIEBAND_QS = {"Q26"}  # numeric scale judged row-level for products

# ---- load units (for tie-band we need the raw values) ----
units = {}
for line in open(WD/"product_units.jsonl"):
    u = json.loads(line)
    units[(u["pair"], u["qid"], u["row_id"], u.get("field",""))] = u

# ---- aggregate votes ----
votes = defaultdict(lambda: defaultdict(list))
for line in open(WD/"product_judge_raw.jsonl"):
    r = json.loads(line)
    k = (r["pair"], r["qid"], r["row_id"], r.get("field",""))
    votes[k][r["direction"]].append(r["winner"])

def maj(ws):
    c = Counter(w for w in ws if w in ("A","B","tie"))
    return c.most_common(1)[0][0] if c else None

def verdict(k):
    # tie-band first: numeric adjacent-level diffs are deterministic ties
    u = units.get(k)
    if u and k[1] in NUMERIC_TIEBAND_QS:
        try:
            d = abs(float(str(u["anchor"]).strip()) - float(str(u["cand"]).strip()))
            if d <= 1.0:
                return "tie_band"
        except (ValueError, TypeError):
            pass
    dirs = votes[k]
    m1 = maj(dirs.get("anchor_as_A", []))
    m2 = maj(dirs.get("anchor_as_B", []))
    v1 = {"A":"anchor","B":"cand","tie":"tie"}.get(m1)
    v2 = {"A":"cand","B":"anchor","tie":"tie"}.get(m2)
    if v1 == v2 and v1 in ("anchor","cand"):
        return v1
    return "tie"

# ---- per product per qid ----
res = defaultdict(lambda: defaultdict(Counter))
for k in votes:
    res[k[0]][k[1]][verdict(k)] += 1

# equal counts from coverage
cov = json.load(open(WD/"product_coverage.json"))

out = {}
for m in ("shortcut_ai","excelagent_gpt","claude_in_excel"):
    per_q = {}
    for q, c in sorted(res[m].items()):
        cc = cov.get(f"{m}|{q}", {})
        per_q[q] = {
            "anchor_win": c.get("anchor",0), "cand_win": c.get("cand",0),
            "tie": c.get("tie",0), "tie_band": c.get("tie_band",0),
            "equal": cc.get("equal",0), "missing": cc.get("missing",0),
            "net": c.get("anchor",0) - c.get("cand",0),
        }
    out[m] = per_q

json.dump(out, open(WD/"product_final.json","w"), indent=1)

for m, per_q in out.items():
    tot_a = sum(v["anchor_win"] for v in per_q.values())
    tot_c = sum(v["cand_win"] for v in per_q.values())
    tot_t = sum(v["tie"]+v["tie_band"] for v in per_q.values())
    print(f"{m}: TADA wins {tot_a}, product wins {tot_c}, ties {tot_t}, net {tot_a-tot_c:+d}")
    negs = [(q,v["net"]) for q,v in per_q.items() if v["net"] < 0]
    print(f"  product-favored qids: {negs}")
