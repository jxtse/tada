#!/usr/bin/env python3
"""Build row-level product-baseline judge units vs TADA skill_on r1.

Methods: shortcut_ai, excelagent_gpt, claude_in_excel. No ExcelAgent+TA.
Same semantics as matrix pipeline:
- exact equal -> deterministic tie
- product missing row/task/output -> coverage failure (not judged)
- differing answer -> product_units.jsonl for anonymized 3-vote judge
Covers row-aligned Q01,Q02,Q03,Q09,Q10,Q12,Q18,Q19,Q20,Q23,Q24,Q25,Q26.
Filter/holistic cases are recorded for separate handling.
"""
import csv, json, re, unicodedata
from pathlib import Path

HERE=Path(__file__).parent
P=HERE/"product_csv"
RUN=HERE/"full26_runs/full26_20260722/skill_on"
INP=HERE/"full26_inputs"
QUERIES=json.load(open(HERE/"queries_snapshot.json"))

KEYS={"Q01":"CustomerID","Q02":"TicketID","Q03":"ContractID","Q09":"ContractID","Q10":"IncidentID",
      "Q12":"SubmissionID","Q18":"EmailID","Q23":"RecordID","Q24":"ProductID",
      "Q25":"RecordID","Q26":"TaskID"}
TEXT={"Q01":"customer_feedback","Q02":"customer_notes","Q03":"contract_clause","Q09":"contract_notes","Q10":"incident_report",
      "Q12":"original_document","Q18":"email_excerpt","Q19":"review","Q20":"notes","Q23":"country","Q24":"description",
      "Q25":"notes","Q26":"description"}
PATS={"Q01":["sentiment"],"Q02":["priority"],"Q03":["risk"],"Q09":["party a|party_a","party b|party_b"],
      "Q10":["time","location","scope","root"],"Q12":["revised"],"Q18":["contradicted|statementid"],
      "Q19":["sentiment|score"],"Q20":["relevance|score"],"Q23":["country_normalized|normalized"],"Q24":["summary"],
      "Q25":["notes_en|english|translat"],"Q26":["priority"]}

METHODS=["shortcut_ai","excelagent_gpt","claude_in_excel"]

def load(p, skip_leading=False):
    if not p or not Path(p).exists(): return []
    for enc in ("utf-8-sig","latin-1"):
        try:
            lines=Path(p).read_text(encoding=enc).splitlines()
            if skip_leading and lines and lines[0].startswith("[Analysis Theme]"): lines=lines[1:]
            return list(csv.DictReader(lines))
        except UnicodeDecodeError: pass
    return []

def find(rows,pat):
    opts=pat.lower().split("|")
    # exact header match wins (e.g. Claude Q26 has both 'PriorityScore' and final 'priority')
    for h in rows[0]:
        if (h or "").lower().strip() in opts: return h
    for h in rows[0]:
        if any(o in (h or "").lower() for o in opts): return h
    return None

def find_q23_product(rows):
    # Products overwrote `country` in place (no country_normalized column).
    # Their normalized output IS the country column; record the
    # preserve-original violation separately, but judge output-vs-output.
    h = find(rows, "country_normalized|normalized")
    if h: return h
    return find(rows, "country")

def norm(s):
    s=str(s or "").strip().lower()
    s=unicodedata.normalize("NFKC",s)
    s=re.sub(r"\s+"," ",s)
    return s

def anchor_file(q):
    td=RUN/q
    p=td/"result.csv"
    if p.exists(): return p
    # Q18 result exists; no issue. fallback largest generated CSV
    c=[x for x in td.glob("*.csv") if x.name not in (f"{q}.csv","Q18_1.csv","Q18_2.csv")]
    return max(c,key=lambda x:x.stat().st_size) if c else None

def product_file(m,q):
    d=P/m
    candidates=[]
    # explicitly prefer output-bearing multi-sheet files
    prefs={
      "Q18":["Q18__Emails.csv"],
    }.get(q,[])
    for n in prefs:
        if (d/n).exists(): return d/n
    # copy filenames are official artifacts for Claude Q04/Q11/Q24/Q26
    for n in [f"{q}_copy.csv",f"{q}.csv"]:
        if (d/n).exists(): return d/n
    return None

def align_by_position(a, b, key, textcol):
    """ShortcutAI Q02: no TicketID; align by row position (both 200 rows)."""
    if len(a)!=len(b): return None
    bm={}
    for ar,br in zip(a,b):
        bm[str(ar[key]).strip()]=br
    return bm

inputs={}
for q,k in KEYS.items():
    f=INP/("Q18_2.csv" if q=="Q18" else f"{q}.csv")
    inputs[q]={r[k]:r.get(TEXT[q],"") for r in load(f)}

units=[]; cov={}
for m in METHODS:
  for q,key in KEYS.items():
    af=anchor_file(q); pf=product_file(m,q)
    a=load(af); b=load(pf,skip_leading=(q=="Q20"))
    c={"anchor_rows":len(a),"product_rows":len(b),"equal":0,"diff":0,"missing":0,"status":"ok","file":str(pf or "")}
    allow_positional = (m=="shortcut_ai" and q=="Q02")
    if not a or not b or key not in (a[0] if a else {}) or (key not in (b[0] if b else {}) and not allow_positional):
        c["status"]="missing_or_unparseable"; c["missing"]=len(a) or 200; cov[f"{m}|{q}"]=c; continue
    am={str(r[key]).strip():r for r in a}
    if m=="shortcut_ai" and q=="Q02" and key not in b[0]:
        # positional alignment: ShortcutAI dropped TicketID but kept row order & count
        input_rows=load(INP/"Q02.csv")
        if len(b)==len(input_rows):
            bm={str(ir[key]).strip():br for ir,br in zip(input_rows,b)}
            c["status"]="positional_alignment"
        else:
            c["status"]="missing_or_unparseable"; c["missing"]=len(a); cov[f"{m}|{q}"]=c; continue
    else:
        bm={str(r[key]).strip():r for r in b}
    for pat in PATS[q]:
      ca=find(a,pat)
      cb=find_q23_product(b) if q=="Q23" else find(b,pat)
      if q=="Q23" and cb=="country": c["note"]="in_place_overwrite"
      if not ca or not cb:
        c["status"]="missing_output_col"; c["missing"]+=len(am); continue
      for rid,ar in am.items():
        br=bm.get(rid)
        if br is None: c["missing"]+=1; continue
        va,vb=ar.get(ca,""),br.get(cb,"")
        if norm(va)==norm(vb): c["equal"]+=1
        else:
          c["diff"]+=1
          units.append({"pair":m,"qid":q,"row_id":rid,"field":pat if len(PATS[q])>1 else "",
                        "query":QUERIES[q],"text":inputs[q].get(rid,""),"anchor":va,"cand":vb})
    cov[f"{m}|{q}"]=c

with open(HERE/"product_units.jsonl","w") as f:
    for u in units: f.write(json.dumps(u,ensure_ascii=False)+"\n")
json.dump(cov,open(HERE/"product_coverage.json","w"),indent=1)
print("units",len(units),"calls",len(units)*6)
for m in METHODS:
    us=sum(1 for u in units if u["pair"]==m)
    miss=sum(v["missing"] for k,v in cov.items() if k.startswith(m+"|"))
    bad=[k.split('|')[1]+":"+v['status'] for k,v in cov.items() if k.startswith(m+'|') and v['status']!='ok']
    print(m,"units",us,"missing",miss,"bad",bad)
