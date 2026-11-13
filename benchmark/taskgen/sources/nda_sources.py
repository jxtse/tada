"""Private NDA source API (controller-side only). Designer sees only the label-free pack.

Source: ContractNLI (Koreeda & Manning 2021), 607 non-disclosure agreements with 17 fixed hypotheses each,
expert labels Entailment / Contradiction / NotMentioned and evidence spans. CC BY 4.0.
Party names are not renamed here (NDAs are mostly templates; principal parties are rarely famous). The
designer may rename parties in rendering; renaming must be label-independent.
"""
from __future__ import annotations
import json, os, random, zipfile
from functools import lru_cache
from pathlib import Path
from local_config import path as local_path, require

ZIP = local_path('datasets_root') / 'raw/legal/contract_nli/contract-nli.zip'
QUARANTINE = local_path('quarantine')


@lru_cache(maxsize=1)
def _raw():
    require('datasets_root')
    docs, labels = [], None
    with zipfile.ZipFile(ZIP) as z:
        for split in ('train', 'dev', 'test'):
            d = json.loads(z.read(f'contract-nli/{split}.json'))
            labels = d['labels']
            for doc in d['documents']:
                doc = dict(doc, split=split)
                docs.append(doc)
    return docs, labels


@lru_cache(maxsize=1)
def _quarantine() -> set:
    q = json.loads(QUARANTINE.read_text()) if QUARANTINE.exists() else {}
    return {(str(x['doc_id']), x['hypothesis']) for x in q.get('NDA_labels', [])}


@lru_cache(maxsize=1)
def _labels():
    docs, _ = _raw()
    out = {}
    for d in docs:
        for h, a in d['annotation_sets'][0]['annotations'].items():
            if (str(d['id']), h) in _quarantine():
                continue
            spans = [d['text'][d['spans'][i][0]:d['spans'][i][1]] for i in a['spans']]
            out[(str(d['id']), h)] = (a['choice'], spans)
    seed = os.environ.get('TADA_LABEL_PERMUTE')
    if seed:  # controller-only leak test: shuffle each hypothesis's (choice, spans) across documents
        rng = random.Random(int(seed))
        by_h = {}
        for (doc, h), v in out.items():
            by_h.setdefault(h, []).append((doc, v))
        out = {}
        for h, items in sorted(by_h.items()):
            vals = [v for _, v in items]
            rng.shuffle(vals)
            for (doc, _), v in zip(items, vals):
                out[(doc, h)] = v
    return out


def hypotheses() -> dict:
    """{hypothesis_id: {'short': short description, 'text': hypothesis sentence}} (public dataset definitions)."""
    _, labels = _raw()
    return {k: {'short': v['short_description'], 'text': v['hypothesis']} for k, v in labels.items()}


def iter_documents():
    """Label-free: doc_id, file_name, document_type (search-pdf / sec-text / sec-html), url, split, text."""
    docs, _ = _raw()
    for d in docs:
        yield {'doc_id': str(d['id']), 'file_name': d['file_name'], 'document_type': d['document_type'],
               'url': d['url'], 'split': d['split'], 'text': d['text']}


def label(doc_id: str, hypothesis: str):
    """PRIVATE: ('Entailment'|'Contradiction'|'NotMentioned', [evidence span text, ...]) or None if quarantined.
    Provenance id: '<doc_id>:<hypothesis>'."""
    return _labels().get((str(doc_id), hypothesis))
