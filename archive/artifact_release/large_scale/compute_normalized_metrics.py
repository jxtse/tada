#!/usr/bin/env python3
"""Compute the full RQ3 metric set from saved predictions + pre-voting labeler rounds.

Per task:
  reference pairwise accuracy / macro-F1 among the 3 strong-labeler rounds
  unanimous fraction (all 3 rounds agree)
  per candidate run: raw acc, macro-F1, normalized acc/F1 (raw / reference pairwise),
  unanimous-subset and uncertain-subset accuracy.
Pure post-processing; no LLM calls.
"""
import json, csv, glob, os, re
from collections import defaultdict

BASE = '<LOCAL_ARTIFACT_PATH>'

def norm(s):
    return re.sub(r'[^a-z/&\- ]', '', str(s).strip().lower().replace('_', ' ')).strip()

def load_json_labels(p):
    return {str(k): norm(v) for k, v in json.load(open(p)).items()}

def load_csv_labels(p):
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            out[str(row['rid']).strip()] = norm(row.get('label', ''))
    return out

def macro_f1(y_true, y_pred):
    labs = set(y_true) | set(y_pred)
    f1s = []
    for c in labs:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        f1s.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0)
    return sum(f1s)/len(f1s) if f1s else 0.0

def pairwise(rounds):
    keys = sorted(set.intersection(*[set(r) for r in rounds]))
    accs, f1s = [], []
    for i in range(len(rounds)):
        for j in range(i+1, len(rounds)):
            a = [rounds[i][k] for k in keys]
            b = [rounds[j][k] for k in keys]
            accs.append(sum(x == y for x, y in zip(a, b))/len(keys))
            f1s.append(macro_f1(a, b))
    unan = [k for k in keys if len({r[k] for r in rounds}) == 1]
    return (sum(accs)/len(accs), sum(f1s)/len(f1s), len(unan)/len(keys),
            set(unan), set(keys))

TASKS = {
  'sentiment_ps': ('gold_sentiment.json', ['ckpt_sent_s1','ckpt_sent_s2','ckpt_sent_s3'], 'labels_fixed'),
  'joint_ps_fixed': ('gold_joint_ps.json', ['ckpt_fixed_ps_s1','ckpt_fixed_ps_s2','ckpt_fixed_ps_s3'], None),
  'cat_tm_fixed': ('gold_joint_tm.json', ['ckpt_fixed_tm_s1','ckpt_fixed_tm_s2','ckpt_fixed_tm_s3'], None),
}

def load_round(stem):
    js = f'{BASE}/gold/{stem}.jsonl'.replace('ckpt_', 'labels_').replace('.jsonl', '.json')
    # prefer labels_*.json when present, else parse ckpt jsonl
    cands = glob.glob(js)
    if cands:
        return load_json_labels(cands[0])
    out = {}
    with open(f'{BASE}/gold/{stem}.jsonl') as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            for k, v in d.get('labels', d if isinstance(d, dict) else {}).items():
                if isinstance(v, str):
                    out[str(k)] = norm(v)
    return out

results = {}
for task, (goldfile, round_stems, _) in TASKS.items():
    gold = load_json_labels(f'{BASE}/gold/{goldfile}')
    rounds = [load_round(s) for s in round_stems]
    rounds = [r for r in rounds if len(r) > 100]
    if len(rounds) < 2:
        results[task] = {'error': f'only {len(rounds)} usable rounds'}
        continue
    ref_acc, ref_f1, unan_frac, unan, keys = pairwise(rounds)
    entry = {'reference': {'pairwise_acc': round(ref_acc, 4), 'pairwise_macro_f1': round(ref_f1, 4),
                           'unanimous_frac': round(unan_frac, 4), 'n_ref_rows': len(keys)},
             'runs': {}}
    # explicit stem prefixes per task (TADA assignment runs carry no _fixed suffix)
    PREFIXES = {
      'sentiment_ps': ('tada_sentiment_ps_rep', 'claudecode_sentiment_ps_rep'),
      'joint_ps_fixed': ('tada_joint_ps_rep', 'claudecode_joint_ps_fixed_rep'),
      'cat_tm_fixed': ('tada_cat_tm_rep', 'claudecode_cat_tm_fixed_rep'),
    }
    for f in sorted(glob.glob(f'{BASE}/e2e/artifacts/*_labels.csv') +
                    glob.glob(f'{BASE}/e2e/artifacts/*_labels.json')):
        stem = os.path.basename(f).rsplit('_labels', 1)[0]
        if not stem.startswith(PREFIXES[task]):
            continue
        pred = load_json_labels(f) if f.endswith('.json') else load_csv_labels(f)
        common = [k for k in gold if k in pred]
        if not common:
            continue
        yt = [gold[k] for k in common]; yp = [pred[k] for k in common]
        acc = sum(a == b for a, b in zip(yt, yp))/len(common)
        f1 = macro_f1(yt, yp)
        u = [k for k in common if k in unan]
        nu = [k for k in common if k in keys and k not in unan]
        entry['runs'][stem] = {
            'raw_acc': round(acc, 4), 'raw_macro_f1': round(f1, 4),
            'norm_acc': round(acc/ref_acc, 4), 'norm_macro_f1': round(f1/ref_f1, 4),
            'unanimous_subset_acc': round(sum(gold[k] == pred[k] for k in u)/len(u), 4) if u else None,
            'uncertain_subset_acc': round(sum(gold[k] == pred[k] for k in nu)/len(nu), 4) if nu else None,
            'n_scored': len(common)}
    results[task] = entry

out = f'{BASE}/e2e/artifacts/normalized_metrics.json'
json.dump(results, open(out, 'w'), indent=2)
print(json.dumps(results, indent=2))
