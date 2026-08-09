#!/usr/bin/env python3
"""Reproduce the human-calibration analysis for the 39-unit stratified sample.

Inputs (same directory layout as this repository):
  calibration39_key.json   - 39 judgment units with internal answer key and strata
  raters/rater*.csv        - blind A/B/tie verdicts from six independent annotators
  judge_verdicts.csv       - per-unit LLM-judge verdict (aggregated from raw judge
                             logs; see paper appendix for the aggregation protocol:
                             anonymized A/B presentation, position swapping, three
                             votes per direction, direction disagreement -> tie)

Outputs (printed):
  - inter-annotator agreement (pairwise raw agreement, Fleiss' kappa)
  - human-majority vs judge agreement, Cohen's kappa, direction reversals
  - per-stratum and per-task breakdown (paper appendix table)

Usage:
  python3 analyze_calibration.py
"""
import csv
import glob
import itertools
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CATS = ("A", "B", "tie")


def load_key():
    with open(os.path.join(HERE, "calibration39_key.json"), encoding="utf-8") as f:
        return json.load(f)


def load_raters():
    raters = {}
    for path in sorted(glob.glob(os.path.join(HERE, "raters", "rater*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        rows = {}
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                v = r["verdict"].strip()
                assert v in CATS, (path, r)
                rows[int(r["idx"])] = v
        assert sorted(rows) == list(range(1, 40)), f"{path}: incomplete"
        raters[name] = rows
    return raters


def load_judge():
    with open(os.path.join(HERE, "judge_verdicts.csv"), newline="", encoding="utf-8") as f:
        return {int(r["idx"]): r["judge_verdict"] for r in csv.DictReader(f)}


def cohen_kappa(l1, l2):
    n = len(l1)
    po = sum(a == b for a, b in zip(l1, l2)) / n
    c1, c2 = Counter(l1), Counter(l2)
    pe = sum(c1[c] * c2[c] for c in CATS) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def fleiss_kappa(item_counts, n_raters):
    N = len(item_counts)
    pj = Counter()
    for row in item_counts:
        pj.update(row)
    pj = {c: pj[c] / (N * n_raters) for c in CATS}
    p_bar = sum(
        (sum(v * v for v in row.values()) - n_raters) / (n_raters * (n_raters - 1))
        for row in item_counts
    ) / N
    p_e = sum(p * p for p in pj.values())
    return p_bar, (p_bar - p_e) / (1 - p_e)


def majority(votes):
    c = Counter(votes)
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "split"
    return top[0][0]


def is_flip(h, j):
    return {h, j} == {"A", "B"}


def main():
    key = load_key()
    raters = load_raters()
    judge = load_judge()
    names = list(raters)
    n = len(names)
    idxs = list(range(1, 40))
    strat = {it["idx"]: it["_stratum"] for it in key}
    qid = {it["idx"]: it["qid"] for it in key}

    print(f"annotators: {n}, units: {len(idxs)}")

    # 1. inter-annotator agreement
    pas = []
    print("\n-- pairwise raw agreement --")
    for a, b in itertools.combinations(names, 2):
        po, k = cohen_kappa([raters[a][i] for i in idxs], [raters[b][i] for i in idxs])
        pas.append(po)
        print(f"  {a} vs {b}: agree={po:.3f} kappa={k:.3f}")
    print(f"mean pairwise agreement: {sum(pas)/len(pas):.3f}")
    p_bar, fk = fleiss_kappa([Counter(raters[m][i] for m in names) for i in idxs], n)
    print(f"Fleiss kappa: P-bar={p_bar:.3f} kappa={fk:.3f}")

    # 2. human majority vs judge
    hm = {i: majority([raters[m][i] for m in names]) for i in idxs}
    maj_items = [i for i in idxs if hm[i] != "split"]
    agree = sum(hm[i] == judge[i] for i in maj_items)
    flips = sum(is_flip(hm[i], judge[i]) for i in maj_items)
    po, ck = cohen_kappa([hm[i] for i in maj_items], [judge[i] for i in maj_items])
    print(f"\n-- human majority vs judge --")
    print(f"majority reached on {len(maj_items)}/39 units")
    print(f"agreement: {agree}/{len(maj_items)} = {po:.3f}, Cohen kappa={ck:.3f}")
    print(f"direction reversals (A vs B): {flips}")

    # 3. judge vs individual annotators
    print("\n-- judge vs individual annotators --")
    vals = []
    for m in names:
        po_i, k_i = cohen_kappa([raters[m][i] for i in idxs], [judge[i] for i in idxs])
        vals.append(po_i)
        print(f"  {m}: agree={po_i:.3f} kappa={k_i:.3f}")
    print(f"mean judge-annotator agreement: {sum(vals)/len(vals):.3f}")

    # 4. breakdown (paper appendix table)
    def block(label, subset):
        m = [i for i in subset if hm[i] != "split"]
        a = sum(hm[i] == judge[i] for i in m)
        f = sum(is_flip(hm[i], judge[i]) for i in m)
        print(f"  {label:28s} units={len(subset):2d} maj={len(m):2d} agree={a:2d} flip={f}")

    print("\n-- by stratum --")
    for s in ("contested", "low_conf", "high_conf"):
        block(s, [i for i in idxs if strat[i] == s])
    print("-- by task --")
    for q in sorted(set(qid.values())):
        block(q, [i for i in idxs if qid[i] == q])

    print("\n-- majority-free (split) units --")
    for i in idxs:
        if hm[i] == "split":
            dist = dict(Counter(raters[m][i] for m in names))
            print(f"  idx {i} ({qid[i]}, {strat[i]}): {dist} judge={judge[i]}")


if __name__ == "__main__":
    main()
