"""Private CUADX source API (controller-side only). Designer sees only the label-free pack.

Records come from private_cache/cuadx.json: the same 198 v4p-audited, renamed CUAD contracts as C2 (197 admitted
after a per-category span/answer and renamed-span grounding check), with 14 CUAD clause categories that C2 does
not use. Labels are expert clause-presence judgements (broad attorney-review categories).
"""
from __future__ import annotations
import json, os, random
from functools import lru_cache
from pathlib import Path
from local_config import path as local_path, require

CACHE = local_path('private_cache') / 'cuadx.json'
CATEGORY_QUESTIONS = {
    'cap_on_liability': 'Does the contract include a cap on liability upon the breach of a party\'s obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.',
    'uncapped_liability': 'Is a party\'s liability uncapped upon the breach of its obligation in the contract? This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.',
    'insurance': 'Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?',
    'audit_rights': 'Does a party have the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?',
    'post_termination_services': 'Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?',
    'minimum_commitment': 'Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?',
    'revenue_profit_sharing': 'Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?',
    'ip_ownership_assignment': 'Does intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?',
    'license_grant': 'Does the contract contain a license granted by one party to its counterparty?',
    'covenant_not_to_sue': 'Is a party restricted from contesting the validity of the counterparty\'s ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?',
    'liquidated_damages': 'Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?',
    'warranty_duration': 'What is the duration of any warranty against defects or errors in technology, products, or services provided under the contract?',
    'most_favored_nation': 'Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?',
    'volume_restriction': 'Is there a fee increase or consent requirement, etc. if one party\'s use of the product/services exceeds certain threshold?',
}


@lru_cache(maxsize=1)
def _data():
    require('private_cache')
    d = json.loads(CACHE.read_text())
    seed = os.environ.get('TADA_LABEL_PERMUTE')
    if seed:  # controller-only leak test: shuffle each category's labels/spans across contracts
        rng = random.Random(int(seed))
        cs = d['contracts']
        for k in d['categories']:
            pairs = [(c['labels'][k], c['evidence_spans'][k]) for c in cs]
            rng.shuffle(pairs)
            for c, (l, s) in zip(cs, pairs):
                c['labels'][k], c['evidence_spans'][k] = l, s
    return d


def categories() -> list[str]:
    return list(CATEGORY_QUESTIONS)


def iter_contracts():
    """Yield dicts: contract_id, document_type, organizations (renamed party names, >=2), text (renamed full
    contract text), labels {category: 'Yes'|'No'} (expert, private), evidence_spans {category: [str]} (private)."""
    for c in _data()['contracts']:
        yield {'contract_id': c['contract_id'], 'document_type': c['document_type'],
               'organizations': list(c['organizations']), 'text': c['text'], 'labels': dict(c['labels']),
               'evidence_spans': {k: list(v) for k, v in c['evidence_spans'].items()}}
