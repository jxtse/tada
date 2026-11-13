"""Private MAUD source API (controller-side only). Designer sees only the label-free pack.

MAUD v1 (Wang et al. 2023), merger agreements annotated by lawyers on 144 deal points; CC BY 4.0.
Only the 100 agreements with full text are used, and of those only the agreements whose principal parties
(target "Company" and acquirer "Parent"/"Purchaser"/...) can be parsed from the opening paragraph (77).
Party names are pseudonymised deterministically from the contract id (label-independent), in full text and
evidence alike. Scorable (contract, deal point) pairs are those whose every evidence excerpt segment is
locatable in the full text after normalisation (rule uses evidence location only, never the answer).
"""
from __future__ import annotations
import hashlib, json, os, random, re
from functools import lru_cache
from pathlib import Path
from local_config import path as local_path, require

ROOT = local_path('datasets_root') / 'raw/legal/maud/MAUD_v1'
QUARANTINE = local_path('quarantine')
ROLE_RE = re.compile(r'([A-Z][A-Za-z0-9.,&\'’ \-]{1,90}?),? (?:an?|the) [A-Za-z ]{0,60}?(?:corporation|limited liability company|'
                     r'limited partnership|company|société|N\.V\.|plc|public limited company|exempted company|trust|partnership|'
                     r'S\.A\.|AG|GmbH|Ltd\.?)[^“"(]{0,140}?\(?(?:the |hereinafter )?[“"](Parent|Company|Merger Sub(?: \w+)?|Purchaser|'
                     r'Buyer|Acquiror|Acquirer|Merger Subsidiary|Sub|Seller|Target|Offeror)[”"]')
SUFFIX = re.compile(r',?\s+(?:Inc\.?|Incorporated|Corp\.?|Corporation|Company|Co\.|LLC|L\.L\.C\.|Ltd\.?|Limited|plc|N\.V\.|S\.A\.|AG|'
                    r'GmbH|L\.P\.|LP|Holdings?|Group|Trust)\s*$', re.I)
ACQ = ('Parent', 'Purchaser', 'Buyer', 'Acquiror', 'Acquirer', 'Offeror')
FIRST = ['Alder', 'Birchwood', 'Cobalt', 'Dunmore', 'Easton', 'Fairlight', 'Granite', 'Harbor', 'Ironvale', 'Juniper', 'Kestrel',
         'Larkspur', 'Meridian', 'Northgate', 'Orchard', 'Pinecrest', 'Quarry', 'Redwater', 'Silverline', 'Tamarack', 'Upland',
         'Vantage', 'Westbrook', 'Yarrow', 'Zephyr']
SECOND = ['Analytics', 'Biosystems', 'Capital', 'Dynamics', 'Energy', 'Foods', 'Genomics', 'Health', 'Industries', 'Logistics',
          'Materials', 'Networks', 'Pharma', 'Resources', 'Semiconductor', 'Technologies', 'Therapeutics', 'Systems']


def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


def _segments(excerpt: str):
    return [p for p in re.split(r'\(Page \d+\)|<omitted>|\.\.\.', excerpt) if len(_norm(p)) > 40]


@lru_cache(maxsize=None)
def _parties(cid: str):
    require('datasets_root')
    t = re.sub(r'\s+', ' ', (ROOT / 'contracts' / f'{cid}.txt').read_text(errors='ignore')[:30000])
    found = {}
    for m in ROLE_RE.finditer(t):
        name = m.group(1).strip(' ,')
        name = re.sub(r'.*\b(?:among|between)\s+', '', name, flags=re.I)
        name = re.sub(r'^(?:and|,)\s+', '', name, flags=re.I).strip()
        if 3 <= len(name) <= 80:
            found.setdefault(m.group(2), name)
    tgt = found.get('Company')
    acq = next((found[k] for k in ACQ if k in found), None)
    if not tgt or not acq:
        return None
    subs = [v for k, v in found.items() if k.startswith(('Merger Sub', 'Sub', 'Merger Subsidiary'))]
    return {'target': tgt, 'acquirer': acq, 'subs': subs}


@lru_cache(maxsize=None)
def _mapping(cid: str):
    """Deterministic, label-independent pseudonym map for one agreement: [(regex, replacement), ...]."""
    p = _parties(cid)
    rng = random.Random(int(hashlib.sha256(cid.encode()).hexdigest()[:12], 16))
    a, b = rng.sample(FIRST, 2)
    names = {'target': f'{a} {rng.choice(SECOND)}, Inc.', 'acquirer': f'{b} {rng.choice(SECOND)} Corporation'}
    pairs = []
    for role in ('target', 'acquirer'):
        full = p[role]
        core = SUFFIX.sub('', full).strip(' ,')
        new_full, new_core = names[role], SUFFIX.sub('', names[role]).strip(' ,')
        pairs.append((full, new_full))
        if len(core) >= 4 and core.lower() not in ('the company', 'parent'):
            pairs.append((core, new_core))
        # straightforward variant: core in upper case (signature blocks)
        if core.upper() != core:
            pairs.append((core.upper(), new_core.upper()))
    for i, s in enumerate(p['subs']):
        pairs.append((s, f'{b} Merger Sub {i + 1}, Inc.'))
    pairs.sort(key=lambda x: -len(x[0]))
    return [(re.compile(r'(?<![A-Za-z])' + re.escape(o) + r'(?![A-Za-z])'), n) for o, n in pairs]


def _pseudo(cid: str, text: str) -> str:
    for rx, new in _mapping(cid):
        text = rx.sub(new, text)
    return text


@lru_cache(maxsize=1)
def _labels_raw():
    require('datasets_root')
    import pandas as pd
    df = pd.concat([pd.read_csv(ROOT / f'MAUD_{s}.csv') for s in ('train', 'dev', 'test')])
    df = df[df.data_type == 'main'].copy()
    df['answer'] = df['answer'].astype(str)
    df['subquestion'] = df['subquestion'].fillna('<NONE>').astype(str)
    df['question'] = df['question'].astype(str)
    return df


@lru_cache(maxsize=1)
def deal_points() -> dict:
    """Public schema: {dp_id: {category, question, subquestion, answer_options}}. answer_options is the dataset-wide
    option list for that deal point (MAUD codebook), not any record's answer."""
    m = _labels_raw()
    out = {}
    for i, ((q, s), g) in enumerate(sorted(m.groupby(['question', 'subquestion']), key=lambda x: x[0])):
        out[f'DP{i:03}'] = {'category': g.category.iloc[0], 'question': q, 'subquestion': s,
                            'answer_options': sorted(map(str, g.answer.astype(str).unique()))}
    return out


@lru_cache(maxsize=1)
def eligible_contracts() -> list[str]:
    require('datasets_root')
    files = sorted(f[:-4] for f in os.listdir(ROOT / 'contracts'))
    return [c for c in files if _parties(c)]


@lru_cache(maxsize=1)
def _quarantine() -> set:
    """(contract_id, dp_id) pairs excluded after independent review (exclusion only; labels are never edited)."""
    q = json.loads(QUARANTINE.read_text()) if QUARANTINE.exists() else {}
    return {(x['contract_id'], x['dp']) for x in q.get('MAUD_labels', [])}


@lru_cache(maxsize=1)
def _labels():
    m = _labels_raw()
    key = {(v['question'], v['subquestion']): k for k, v in deal_points().items()}
    out = {}
    for c in eligible_contracts():
        full = _norm(_pseudo(c, (ROOT / 'contracts' / f'{c}.txt').read_text(errors='ignore')))
        for _, r in m[m.contract_name == c].iterrows():
            ex = _pseudo(c, str(r.text))
            segs = _segments(ex)
            k = (c, key[(r.question, r.subquestion)])
            if k in _quarantine():
                continue
            if segs and all(_norm(s) in full for s in segs):
                out[k] = (str(r.answer), segs)
    seed = os.environ.get('TADA_LABEL_PERMUTE')
    if seed:  # controller-only leak test: shuffle (answer, segments) across agreements within each deal point
        rng = random.Random(int(seed))
        by = {}
        for (c, dp), v in out.items():
            by.setdefault(dp, []).append((c, v))
        out = {}
        for dp, items in sorted(by.items()):
            vals = [v for _, v in items]
            rng.shuffle(vals)
            for (c, _), v in zip(items, vals):
                out[(c, dp)] = v
    return out


def iter_contracts():
    """Label-free, pseudonymised: contract_id, target_name, acquirer_name (pseudonyms), text."""
    for c in eligible_contracts():
        t = _pseudo(c, (ROOT / 'contracts' / f'{c}.txt').read_text(errors='ignore'))
        names = {o: n for o, n in [(rx.pattern, n) for rx, n in _mapping(c)]}
        p = _parties(c)
        yield {'contract_id': c, 'target_name': _pseudo(c, p['target']), 'acquirer_name': _pseudo(c, p['acquirer']), 'text': t}


def scorable(contract_id: str, dp_id: str) -> bool:
    """Label-independent eligibility of a (contract, deal point) pair: evidence locatable in the full text."""
    return (contract_id, dp_id) in _labels()


def label(contract_id: str, dp_id: str):
    """PRIVATE: (answer, [evidence segment text, pseudonymised]) or None. Provenance id '<contract_id>:<dp_id>'."""
    return _labels().get((contract_id, dp_id))


def residual_original_names() -> dict:
    """PRIVATE audit: count of original principal full/core names still present after pseudonymisation."""
    out = {}
    for c in eligible_contracts():
        t = _pseudo(c, (ROOT / 'contracts' / f'{c}.txt').read_text(errors='ignore'))
        p = _parties(c)
        n = sum(t.count(SUFFIX.sub('', p[r]).strip(' ,')) for r in ('target', 'acquirer') if len(SUFFIX.sub('', p[r]).strip(' ,')) >= 4)
        if n:
            out[c] = n
    return out
