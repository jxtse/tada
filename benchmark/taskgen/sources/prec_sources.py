"""Private PREC source API (controller-side only). Designer sees only the label-free pack.

Text: Caselaw Access Project (CAP) U.S. Reports volumes 329-572 (1946-2014), public domain case law,
opinions separated by CAP into majority / concurrence / dissent / etc.
Labels (private): Library of Congress, Constitution Annotated, "Table of Supreme Court Decisions Overruled
by Subsequent Decisions" (fetched 2026-11-07; pairs overruling -> overruled, '(in part)' flags), and SCDB
precedentAlteration (CC BY-NC 3.0 US; used only for gold/diagnostics, never rendered).
Neither table is a complete statement of current law: the LoC table lists explicit overrulings recognised
by its editors; SCDB codes whether the majority opinion says it formally alters precedent.
"""
from __future__ import annotations
import json, os, random, re, zipfile
from functools import lru_cache
from pathlib import Path
from local_config import path as local_path, require

CACHE = local_path('private_cache') / 'precedent_meta.json'
CAP = local_path('datasets_root') / 'raw/legal/cap_us_reports_scdb'
QUARANTINE = local_path('quarantine')


@lru_cache(maxsize=1)
def _meta():
    require('private_cache')
    return json.loads(CACHE.read_text())


@lru_cache(maxsize=1)
def _quarantine() -> set:
    q = json.loads(QUARANTINE.read_text()) if QUARANTINE.exists() else {}
    return {(x['overruling'], x['overruled']) for x in q.get('PREC_pairs', [])}


def iter_cases(min_volume: int = 329, max_volume: int = 572):
    """Label-free case metadata. Yields dict: cite ('V U.S. P', '#n' suffix when several decisions share a page),
    name, name_abbr, date (YYYY-MM-DD), docket, opinion_types [majority|concurrence|dissent|...],
    opinion_chars, cites_us (U.S. Reports cites listed by CAP as cited in this decision)."""
    for cite, c in _meta()['cases'].items():
        v = int(cite.split()[0])
        if min_volume <= v <= max_volume:
            yield dict(c, cite=cite, cites_us=list(c['cites_us']), opinion_types=list(c['opinion_types']),
                       opinion_chars=list(c['opinion_chars']))


def opinions(cite: str) -> list[dict]:
    """Full text of every opinion in one decision: [{type, author, text}], in CAP order; plus head matter as
    a pseudo-opinion of type 'head_matter' first (syllabus, counsel)."""
    require('datasets_root')
    c = _meta()['cases'][cite]
    with zipfile.ZipFile(CAP / c['zip']) as z:
        d = json.loads(z.read(c['member']))
    body = d['casebody']
    out = [{'type': 'head_matter', 'author': None, 'text': body.get('head_matter') or ''}]
    out += [{'type': o.get('type'), 'author': o.get('author'), 'text': o.get('text') or ''}
            for o in body.get('opinions', [])]
    return out


@lru_cache(maxsize=1)
def _pairs():
    rows = []
    for r in _meta()['loc']:
        for i in r['overruled']:
            if not (r['overruling_cite'] and i['cite']):
                continue
            if (r['overruling_cite'], i['cite']) in _quarantine():
                continue
            rows.append({'overruling': r['overruling_cite'], 'overruled': i['cite'],
                         'in_part': bool(i['in_part'] or r['overruling_in_part']),
                         'loc_text': i['text'], 'overruling_year': r['overruling_year']})
    seed = os.environ.get('TADA_LABEL_PERMUTE')
    if seed:  # controller-only leak test: shuffle overruled targets and in-part flags across pairs
        rng = random.Random(int(seed))
        targets = [(p['overruled'], p['in_part'], p['loc_text']) for p in rows]
        rng.shuffle(targets)
        for p, (t, ip, lt) in zip(rows, targets):
            p['overruled'], p['in_part'], p['loc_text'] = t, ip, lt
    return rows


def overruling_pairs() -> list[dict]:
    """PRIVATE labels: [{overruling, overruled, in_part, loc_text, overruling_year}] where both sides have a
    U.S. Reports cite (either side may fall outside the CAP volume range). Provenance id: 'overruling>overruled'."""
    return [dict(p) for p in _pairs()]


@lru_cache(maxsize=1)
def _scdb():
    s = {k: v['precedentAlteration'] for k, v in _meta()['scdb'].items()}
    seed = os.environ.get('TADA_LABEL_PERMUTE')
    if seed:
        rng = random.Random(int(seed) + 1)
        ks, vs = list(s), list(s.values())
        rng.shuffle(vs)
        s = dict(zip(ks, vs))
    return s


def scdb_precedent_alteration(cite: str) -> str | None:
    """PRIVATE: SCDB precedentAlteration ('1' = majority says it formally alters precedent, '0' = no) or None."""
    return _scdb().get(cite.split('#')[0])


def cite_pattern() -> str:
    """Regex matching U.S. Reports citations as printed in CAP text ('410 U. S. 113')."""
    return r'(\d+)\s*U\.\s?S\.,?\s*(\d+)'


def find_quote(cite: str, text: str) -> bool:
    norm = lambda s: re.sub(r'\s+', ' ', s).strip()
    return norm(text) in norm(' '.join(o['text'] for o in opinions(cite)))
