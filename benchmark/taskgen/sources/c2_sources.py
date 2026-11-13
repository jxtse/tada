"""Private C2 source API (controller-side only). Designer sees only the label-free pack.

Records come from private_cache/c2.json: 198 CUAD contracts admitted by the v4p source audit
(span/answer consistency, dedup, length <= 60k chars, identity renaming of principal parties) plus a
per-category span/answer and renamed-span grounding check for the eight C2 categories.
"""
from __future__ import annotations
import json, os, random
from functools import lru_cache
from pathlib import Path
from local_config import path as local_path, require

CACHE = local_path('private_cache') / 'c2.json'
CATEGORY_QUESTIONS = {
    'change_of_control': 'Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?',
    'anti_assignment': 'Is consent or notice required of a party if the contract is assigned to a third party?',
    'non_compete': 'Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?',
    'exclusivity': 'Is there an exclusive dealing commitment with the counterparty?',
    'no_solicit_customers': 'Is a party restricted from contracting or soliciting customers or partners of the counterparty?',
    'non_transferable_license': 'Does the contract limit the ability of a party to transfer the license being granted to a third party?',
    'termination_for_convenience': 'Can a party terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire)?',
    'rofr_rofo_rofn': 'Is there a clause granting one party a right of first refusal, offer or negotiation?',
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
        yield dict(c, labels=dict(c['labels']), evidence_spans={k: list(v) for k, v in c['evidence_spans'].items()})
