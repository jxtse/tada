#!/usr/bin/env python3
"""Controller check (MAUD families, round 3+): cost of one misread upstream label.

For every expert label the gold actually uses, replace it with each other codebook option, recompute the full
deliverable with the package's own compute_gold(), write it as a submission and score it against the real gold.
Reward loss = 1 - score. Gate (declared 2026-11-11 before any r3 measurement): the worst single-label loss <= 0.20
and the median <= 0.10. Labels that select which agreement is used (descriptor predicates) are reported separately
because changing the selected source is the intended dependency, but they are still gated.

Needs: build/build.py exposing compute_gold(sp, label_fn) -> (sheets, info) with info['labels'] {"cid:dp": answer},
SHEET_SPECS, and tests/scenario_params.json. Runs inside the controller sandbox with maud_sources on PYTHONPATH.
Usage: label_flip_check.py --task T --out O
"""
import argparse, importlib.util, json, statistics, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('--task', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--scorer-py', default=sys.executable); ap.add_argument('--runtime', required=True)
a = ap.parse_args()
task, out = Path(a.task), Path(a.out); out.mkdir(parents=True, exist_ok=True)
if json.loads((task / 'tests/gold.json').read_text()).get('schema_version') == 5:
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'v5'))
    from tada_taskgen.label_cost import label_flip
    print(json.dumps(label_flip(task, out), ensure_ascii=False))
    raise SystemExit(0)
sys.path.insert(0, str(task / 'build'))
spec = importlib.util.spec_from_file_location('pkgbuild', task / 'build/build.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
import maud_sources as ms
from openpyxl import Workbook

sp = json.loads((task / 'tests/scenario_params.json').read_text())
sheets, info = b.compute_gold(sp, ms.label)
used = sorted(info['labels'])
opts = {k: v['answer_options'] for k, v in ms.deal_points().items()}
norm = getattr(b, 'norm_answer', lambda x: x)


def write(sheets_, path):
    wb = Workbook(); wb.remove(wb.active)
    for name, _keys, cols in b.SHEET_SPECS:
        ws = wb.create_sheet(name); ws.append(cols)
        for r in sheets_[name]:
            ws.append([r.get(c) for c in cols])
    wb.save(path)


def score(path, od):
    p = subprocess.run([a.scorer_py, str(task / 'tests/scoring.py'), str(path), str(task / 'tests/gold.json'), str(od)],
                       env={'PYTHONPATH': a.runtime, 'PYTHONDONTWRITEBYTECODE': '1', 'PATH': '/usr/bin:/bin'},
                       capture_output=True, text=True, timeout=600)
    return float((od / 'reward.txt').read_text())


write(sheets, out / 'self.xlsx')
self_score = score(out / 'self.xlsx', out / 'self')
rows = []
for k in used:
    cid, dp = k.split(':')
    orig = info['labels'][k]
    for alt in opts.get(dp, []):
        alt_n = norm(alt)
        if alt_n == orig:
            continue

        def lf(c, d, _cid=cid, _dp=dp, _alt=alt):
            r = ms.label(c, d)
            return ((_alt,) + tuple(r[1:])) if (c == _cid and d == _dp and r is not None) else r
        try:
            s2, _ = b.compute_gold(sp, lf)
        except Exception as exc:  # a flip the build cannot represent is reported, not gated
            rows.append({'label': k, 'alt': alt, 'error': repr(exc)[:200]}); continue
        f = out / 'flip.xlsx'; write(s2, f)
        r = score(f, out / 'flip')
        rows.append({'label': k, 'orig': orig, 'alt': alt, 'loss': round(1 - r, 6)})
losses = [r['loss'] for r in rows if 'loss' in r]
worst = sorted((r for r in rows if 'loss' in r), key=lambda r: -r['loss'])[:10]
res = {'self_score': self_score, 'labels_used': len(used), 'flips': len(losses),
       'max_loss': max(losses) if losses else None, 'median_loss': statistics.median(losses) if losses else None,
       'gate': {'max_loss': 0.20, 'median_loss': 0.10}, 'worst': worst,
       'errors': [r for r in rows if 'error' in r][:10]}
res['pass'] = bool(losses) and self_score == 1.0 and res['max_loss'] <= 0.20 and res['median_loss'] <= 0.10
(out / 'label_flip.json').write_text(json.dumps({**res, 'all': rows}, indent=1, ensure_ascii=False))
print(json.dumps(res, ensure_ascii=False))
