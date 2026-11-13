#!/usr/bin/env python3
"""Re-aggregate controlled matrix with tie-band (adjacent scale level -> deterministic tie).

Applies to Q19 (1-10), Q20 (1-10), Q26 (1-5): |delta| <= 1 scored as tie_band.
Rewrites matrix_result_tieband.csv and recomputes v3-style quality scores.
"""
import json, csv, statistics, pathlib
from collections import Counter, defaultdict

WD = pathlib.Path(__file__).parent
TIEBAND_QS = {"Q19","Q20","Q26"}

units = {}
for line in open(WD/"matrix_units.jsonl"):
    u = json.loads(line)
    units[(u["pair"], u["qid"], u["row_id"], u.get("field",""))] = u

votes = defaultdict(lambda: defaultdict(list))
for line in open(WD/"matrix_judge_raw.jsonl"):
    r = json.loads(line)
    votes[(r["pair"], r["qid"], r["row_id"], r.get("field",""))][r["direction"]].append(r["winner"])

def maj(ws):
    c = Counter(w for w in ws if w in ("A","B","tie"))
    return c.most_common(1)[0][0] if c else None

def verdict(k):
    u = units.get(k)
    if u and k[1] in TIEBAND_QS:
        try:
            if abs(float(str(u["anchor"]).strip()) - float(str(u["cand"]).strip())) <= 1.0:
                return "tie_band"
        except (ValueError, TypeError):
            pass
    dirs = votes[k]
    m1 = maj(dirs.get("anchor_as_A", []))
    m2 = maj(dirs.get("anchor_as_B", []))
    v1 = {"A":"anchor","B":"cand","tie":"tie"}.get(m1)
    v2 = {"A":"cand","B":"anchor","tie":"tie"}.get(m2)
    return v1 if v1 == v2 and v1 in ("anchor","cand") else "tie"

per_pair = defaultdict(Counter)
for k in votes:
    per_pair[k[0]][verdict(k)] += 1

# merge with equal_auto / missing from original matrix_result.csv
orig = {r["pair"]: r for r in csv.DictReader(open(WD/"matrix_result.csv"))}
rows = []
for pair, c in sorted(per_pair.items()):
    o = orig[pair]
    eq, miss = int(o["equal_auto"]), int(o["missing_rows"])
    anchor, cand = c.get("anchor",0), c.get("cand",0)
    tie = c.get("tie",0) + c.get("tie_band",0)
    rows.append({"pair":pair,"anchor_win":anchor,"cand_win":cand,"tie_judged":tie,
                 "tie_band":c.get("tie_band",0),"equal_auto":eq,"missing_rows":miss,
                 "net_anchor":anchor-cand})

with open(WD/"matrix_result_tieband.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# quality score (v3 definition: good = equal + tie + cand_win over full denominator)
scores = defaultdict(list)
for r in rows:
    m = r["pair"].rsplit("_r",1)[0]
    total = r["equal_auto"]+r["tie_judged"]+r["cand_win"]+r["anchor_win"]+r["missing_rows"]
    good = r["equal_auto"]+r["tie_judged"]+r["cand_win"]
    scores[m].append(100.0*good/total)

print("=== tie-band quality scores (mean ± sd over runs) ===")
for m in ("skill_on","skill_off","code_only","direct"):
    v = scores[m]
    sd = statistics.stdev(v) if len(v)>1 else 0.0
    print(f"{m}: {statistics.mean(v):.1f} ± {sd:.1f}  (runs: {[f'{x:.1f}' for x in sorted(v)]})")

# product quality on the same scale (single run each)
pf = json.load(open(WD/"product_final.json"))
cov = json.load(open(WD/"product_coverage.json"))
print("\n=== product quality (same formula, single snapshot) ===")
prod_scores = {}
for m, per_q in pf.items():
    eq = sum(v["equal"] for v in per_q.values())
    tie = sum(v["tie"]+v["tie_band"] for v in per_q.values())
    cand = sum(v["cand_win"] for v in per_q.values())
    anchor = sum(v["anchor_win"] for v in per_q.values())
    miss = sum(v["missing"] for v in per_q.values())
    # add coverage rows for tasks with no diff units at all (fully equal or missing)
    for key_, cc in cov.items():
        mm, qq = key_.split("|")
        if mm == m and qq not in per_q:
            eq += cc.get("equal",0); miss += cc.get("missing",0)
    total = eq+tie+cand+anchor+miss
    good = eq+tie+cand
    prod_scores[m] = 100.0*good/total
    print(f"{m}: {prod_scores[m]:.1f}  (eq={eq} tie={tie} prod_win={cand} tada_win={anchor} miss={miss} total={total})")

json.dump({"controlled":{m:{"mean":statistics.mean(v),"sd":(statistics.stdev(v) if len(v)>1 else 0.0),"runs":sorted(v)} for m,v in scores.items()},
           "products":prod_scores}, open(WD/"final_scores.json","w"), indent=1)
print("\nsaved final_scores.json")
