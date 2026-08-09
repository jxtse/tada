#!/usr/bin/env python3
"""Judge the full method matrix: anchor (skill_on_r1) vs candidate, anonymized
A/B x position swap x 3 votes. Resumable via v2_judge_raw.jsonl."""
import json, re, time, urllib.request, threading, concurrent.futures as cf
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "v2_judge_raw.jsonl"
BASE_URL = os.environ.get("TADA_JUDGE_BASE_URL", "http://localhost:8000/v1/responses")
MODEL = "gpt-5.5"

def call_judge(prompt, retries=3):
    body = json.dumps({"model": MODEL, "input": prompt, "max_output_tokens": 1500}).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(BASE_URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                resp = json.load(r)
            for o in resp.get("output", []):
                if o.get("type") == "message":
                    for c in o.get("content", []):
                        if c.get("type") == "output_text":
                            return c["text"]
        except Exception:
            if i == retries - 1:
                return None
            time.sleep(2 ** (i + 1))
    return None

def parse_json(text):
    if not text:
        return {}
    m = re.search(r"\{.*?\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}

PAIR = """You are judging two candidate answers for a text-analysis task on one spreadsheet row.

Task: {query}
{fn}
Row text: {text}

Answer A: {a}
Answer B: {b}

Which answer is semantically more correct for this task on this row?
Do not prefer longer or more detailed answers. If both are equally acceptable, answer tie.
Output exactly one JSON object: {{"winner":"A"|"B"|"tie","reason":"<15 words"}}"""

done = set()
if RAW.exists():
    for line in open(RAW):
        try:
            r = json.loads(line)
            done.add((r["pair"], r["qid"], r["row_id"], r["field"], r["direction"], r["vote"]))
        except Exception:
            pass

units = [json.loads(l) for l in open(HERE / "v2_units.jsonl")]
jobs = []
for u in units:
    for d in ("anchor_as_A", "anchor_as_B"):
        for v in (1, 2, 3):
            if (u["pair"], u["qid"], u["row_id"], u["field"], d, v) not in done:
                jobs.append((u, d, v))
print(f"units={len(units)} pending_calls={len(jobs)} (done={len(done)})", flush=True)

raw_f = open(RAW, "a")
lock = threading.Lock()

def run(job):
    u, d, v = job
    a, b = (u["anchor"], u["cand"]) if d == "anchor_as_A" else (u["cand"], u["anchor"])
    fn = f"(Judging the field: {u['field']})" if u.get("field") else ""
    res = parse_json(call_judge(PAIR.format(query=u["query"], fn=fn, text=str(u["text"])[:1200],
                                            a=str(a)[:800], b=str(b)[:800])))
    rec = {"pair": u["pair"], "qid": u["qid"], "row_id": u["row_id"], "field": u["field"],
           "direction": d, "vote": v, "winner": res.get("winner"), "reason": res.get("reason")}
    with lock:
        raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        raw_f.flush()

t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    n = 0
    for _ in ex.map(run, jobs):
        n += 1
        if n % 500 == 0:
            el = time.time() - t0
            eta = el / n * (len(jobs) - n)
            print(f"  {n}/{len(jobs)} ({el:.0f}s, ETA {eta/3600:.1f}h)", flush=True)
raw_f.close()
print(f"V2 JUDGE DONE {time.time()-t0:.0f}s")
