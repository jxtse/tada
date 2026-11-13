"""Private build-time source API for TADA family B (Evidence Inference + ClinicalTrials.gov).

The task designer writes a build script against this API but never runs it and never sees
its outputs. Only the controller runs the build script, on the host, with read-only access
to these caches. Labels returned here must flow only into tests/ (gold) and never into
environment/ (solver-visible files) except where the design explicitly declares a
document that a real professional would have (and then only as the source text itself).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from local_config import path as local_path, require

CACHE = local_path('legacy_cache')
QUARANTINE = local_path('quarantine')
DIRECTION = {-1: 'significantly decreased', 0: 'no significant difference', 1: 'significantly increased'}


@lru_cache(maxsize=1)
def _ei():
    require('legacy_cache')
    return json.loads((CACHE / 'ei.json').read_text())


@lru_cache(maxsize=1)
def _ctg():
    require('legacy_cache')
    return json.loads((CACHE / 'ctg_relevant.json').read_text())['records']


@lru_cache(maxsize=1)
def quarantine() -> dict:
    """{'papers': set(pmc), 'prompts': set(prompt_id)} provisionally excluded from eligibility."""
    q = json.loads(QUARANTINE.read_text())
    papers = {str(x['pmc']) for x in q.get('B_papers', [])}
    prompts = {str(x.get('prompt_id') or x.get('PromptID')) for x in q.get('B_prompts', [])}
    return {'papers': papers, 'prompts': prompts}


def _eligible(p: dict) -> bool:
    anns = p.get('annotations') or []
    if not p.get('nct') or p['nct'] not in _ctg():
        return False
    if not anns or len({a.get('UserID') for a in anns}) < 2:
        return False
    if not all(a.get('Valid Label') in (True, 'True') and a.get('Valid Reasoning') in (True, 'True') for a in anns):
        return False
    codes = {int(a['Label Code']) for a in anns}
    if len(codes) != 1 or p['label'] not in (-1, 0, 1):
        return False
    q = quarantine()
    return str(p['pmc']) not in q['papers'] and str(p['PromptID']) not in q['prompts']


@lru_cache(maxsize=1)
def _permutation():
    """Controller-only leak test: when TADA_LABEL_PERMUTE=<seed> is set, expert labels and evidence
    spans are shuffled across eligible comparisons (emptiness of span lists preserved). A correct build
    must render byte-identical solver-visible files with and without this permutation."""
    seed = os.environ.get('TADA_LABEL_PERMUTE')
    if not seed:
        return None
    import random
    rows = [p for p in _ei()['prompts'] if _eligible(p)]
    rng = random.Random(int(seed))
    dirs = [int(p['label']) for p in rows]
    rng.shuffle(dirs)
    with_span = [i for i, p in enumerate(rows) if any(a.get('Annotations') for a in p['annotations'])]
    spans = [[a.get('Annotations') for a in rows[i]['annotations'] if a.get('Annotations')] for i in with_span]
    rng.shuffle(spans)
    span_map = {str(rows[i]['PromptID']): s for i, s in zip(with_span, spans)}
    return {str(p['PromptID']): d for p, d in zip(rows, dirs)}, span_map


def iter_comparisons(eligible_only: bool = True):
    """Yield one dict per expert-labelled comparison (Evidence Inference 'prompt').

    Keys: prompt_id, pmc, nct, intervention, comparator, outcome,
          direction (int -1/0/1, expert label), direction_text,
          evidence_spans (list of annotator evidence strings; private).
    """
    perm = _permutation()
    for p in _ei()['prompts']:
        eligible = _eligible(p)
        if eligible_only and not eligible:
            continue
        pid = str(p['PromptID'])
        direction = int(p['label'])
        spans = [a.get('Annotations') for a in p['annotations'] if a.get('Annotations')]
        if perm and eligible:
            direction = perm[0][pid]
            spans = perm[1].get(pid, spans)
        yield {
            'prompt_id': pid, 'pmc': str(p['pmc']), 'nct': p['nct'],
            'intervention': p['Intervention'], 'comparator': p['Comparator'], 'outcome': p['Outcome'],
            'direction': direction, 'direction_text': DIRECTION[direction],
            'evidence_spans': spans,
        }


def paper_text(pmc: str) -> str:
    """Full text of one paper (PMC open-access text with TITLE/ABSTRACT/BODY section markers)."""
    return (CACHE / 'ei_text' / f'{pmc}.txt').read_text(encoding='utf-8', errors='strict')


def registry(nct: str) -> dict:
    """ClinicalTrials.gov record: {'protocol': {...modules}, 'derived': {'conditionBrowseModule': ...}}.
    Posted results are excluded from this cache and must not be used."""
    r = _ctg()[nct]
    return {'protocol': r['protocol'], 'derived': r['derived']}


def condition_mesh(nct: str) -> list[dict]:
    """Registry-derived MeSH condition terms and ancestors: [{'id','term','kind':'mesh'|'ancestor'}]."""
    cb = _ctg()[nct]['derived'].get('conditionBrowseModule', {})
    out = [{'id': m['id'], 'term': m['term'], 'kind': 'mesh'} for m in cb.get('meshes', [])]
    out += [{'id': m['id'], 'term': m['term'], 'kind': 'ancestor'} for m in cb.get('ancestors', [])]
    return out
