#!/usr/bin/env python3
"""Score discovery-task outputs whose label vocabulary differs from gold.

Two-step: (1) build a label->gold-label mapping via one LLM call over the
label vocabularies (semantic alignment, not string match); (2) row accuracy
under that mapping. A discovered label with no gold equivalent maps to
'__none__' (rows counted wrong). This mirrors how Phase-0 taxonomies were
aligned and is arm-agnostic: same scorer for TADA / operators / Claude Code.
"""
import os, sys, json, csv, argparse, re
for v in ('ALL_PROXY','all_proxy','HTTP_PROXY','http_proxy','HTTPS_PROXY','https_proxy','NO_PROXY','no_proxy'):
    os.environ.pop(v, None)
from openai import OpenAI

BASE = '<LOCAL_ARTIFACT_PATH>'

def norm(s):
    return re.sub(r'[^a-z/&\- ]', '', str(s).strip().lower().replace('_', ' ')).strip()

def semantic_map(src_labels, gold_labels, judge_model='gpt-5.5'):
    cl = OpenAI(base_url='http://127.0.0.1:3000/v1', api_key='dummy', timeout=300, max_retries=2)
    prompt = {
        'source_labels': sorted(src_labels),
        'target_labels': sorted(gold_labels),
        'instruction': ('Map each source label to the single semantically equivalent target label, '
                        "or to '__none__' if no target label covers the same concept. "
                        'Return strict JSON: {"mapping": {source: target}}. '
                        'Equivalence means the same review would get both labels (synonyms, '
                        'rewordings, granularity-equivalent). Do NOT force a mapping.')}
    r = cl.chat.completions.create(model=judge_model, temperature=0.0,
        messages=[{'role': 'system', 'content': 'Return strict JSON only.'},
                  {'role': 'user', 'content': json.dumps(prompt)}])
    txt = re.sub(r'^```(?:json)?\s*|\s*```$', '', (r.choices[0].message.content or '').strip())
    return json.loads(txt).get('mapping', {})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True, help='labels file: .csv (rid,label) or .json {rid: label}')
    ap.add_argument('--gold', required=True, help='gold json filename under gold/')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    gold = json.load(open(f'{BASE}/gold/{args.gold}'))
    if args.labels.endswith('.csv'):
        got = {str(r['rid']).strip(): r['label'] for r in csv.DictReader(open(args.labels))}
    else:
        got = {str(k): v for k, v in json.load(open(args.labels)).items()}

    gold_vocab = sorted({norm(v) for v in gold.values()})
    src_vocab = sorted({norm(v) for v in got.values()})
    exact = [s for s in src_vocab if s in gold_vocab]
    need_map = [s for s in src_vocab if s not in gold_vocab]
    mapping = {s: s for s in exact}
    if need_map:
        m = semantic_map(need_map, gold_vocab)
        for s in need_map:
            mapping[s] = norm(m.get(s, '__none__')) if norm(m.get(s, '')) in gold_vocab else '__none__'

    hit = tot = 0
    for rid, g in gold.items():
        if rid in got:
            tot += 1
            hit += (mapping.get(norm(got[rid]), '__none__') == norm(g))
    res = {'acc_mapped': hit / tot if tot else 0.0, 'gold_rows_scored': tot,
           'coverage': len(got), 'label_mapping': mapping,
           'unmapped_labels': [s for s, t in mapping.items() if t == '__none__']}
    json.dump(res, open(args.out, 'w'), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != 'label_mapping'}, indent=2))

if __name__ == '__main__':
    main()
