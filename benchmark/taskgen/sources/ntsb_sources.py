"""Private NTSB source API (controller-side only). Designer sees only the label-free pack.

Source: NTSB aviation accident database avall.mdb (US Government work, public domain), events 2008-2019 with
Cause_Factor-coded findings (NTSB stopped populating Cause_Factor in 2020), a factual report narrative
(narr_accp) >= 1500 chars and a probable-cause narrative. Events whose factual report contains the phrase
'probable cause', 'NTSB determined/determines' or 'contributing to the accident' are excluded (the factual text
would hand over the conclusion).

PUBLIC-able: event metadata, aircraft rows, the factual report (narr_accp).
PRIVATE: findings (coded, with C = cause / F = factor), probable cause text (narr_cause), the analysis narrative
(narr_accf, which contains evaluative language in ~91% of events).
"""
from __future__ import annotations
import json, os, random, re
from functools import lru_cache
from pathlib import Path
from local_config import path as local_path, require

CACHE = local_path('private_cache') / 'ntsb.json'
QUARANTINE = local_path('quarantine')
TELLS = re.compile(r'probable cause|ntsb (?:determines|determined)|contributing to the accident', re.I)
TOP = {'01': 'Aircraft', '02': 'Personnel issues', '03': 'Environmental issues', '04': 'Organizational issues',
       '05': 'Not determined'}


@lru_cache(maxsize=1)
def _quarantine() -> set:
    q = json.loads(QUARANTINE.read_text()) if QUARANTINE.exists() else {}
    return {x['ev_id'] for x in q.get('NTSB_events', [])}


@lru_cache(maxsize=1)
def _events():
    require('private_cache')
    d = json.loads(CACHE.read_text())['events']
    keep = {e: v for e, v in d.items() if not TELLS.search(v['factual']) and e not in _quarantine()}
    seed = os.environ.get('TADA_LABEL_PERMUTE')
    if seed:  # controller-only leak test: shuffle findings / cause / analysis across events
        rng = random.Random(int(seed))
        ids = sorted(keep)
        priv = [(keep[e]['findings'], keep[e]['cause'], keep[e]['analysis']) for e in ids]
        rng.shuffle(priv)
        keep = {e: dict(keep[e], findings=f, cause=c, analysis=a) for e, (f, c, a) in zip(ids, priv)}
    return keep


def iter_events():
    """Yield label-free dicts: ev_id, ntsb_no, date, state, city, ev_type, highest_injury, light, wx,
    aircraft [{key, make, model, far_part, category, damage, regis_no}], factual (full factual report text)."""
    for e, v in sorted(_events().items()):
        yield {'ev_id': e, **{k: v[k] for k in ('ntsb_no', 'date', 'state', 'city', 'ev_type', 'highest_injury',
                                                  'light', 'wx', 'factual')},
               'aircraft': [dict(a) for a in v['aircraft']]}


def findings(ev_id: str) -> list[dict]:
    """PRIVATE: [{no, aircraft_key, code, description ('Top-level-...-...'), category ('01'..'05'), cause_factor 'C'|'F'}].
    Provenance id: '<ev_id>:<aircraft_key>:<no>'."""
    return [dict(f) for f in _events()[ev_id]['findings']]


def probable_cause(ev_id: str) -> str:
    """PRIVATE: NTSB probable-cause statement."""
    return _events()[ev_id]['cause']


def analysis_text(ev_id: str) -> str:
    """PRIVATE: NTSB analysis narrative (evaluative). Never render into solver-visible files."""
    return _events()[ev_id]['analysis']


def top_categories() -> dict:
    return dict(TOP)
