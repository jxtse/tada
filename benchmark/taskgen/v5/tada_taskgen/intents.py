"""Local O*NET 31.0 task/DWA citations, clean-source question types, and quotas."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .adapters import get_adapter

from local_config import path as local_path, require
ONET = local_path('onet_root')
DELIVERABLES = {
    'MAUD': {'name': 'Merger diligence issue register', 'purpose': 'Compare agreement terms for a transaction team',
             'patterns': [('scope-review-summary', ['Establish the transaction scope', 'Review agreement terms', 'Summarize issues']),
                          ('revision-review', ['Resolve current instructions', 'Review the requested agreement terms'])]},
    'CUAD': {'name': 'Contract transfer review schedule', 'purpose': 'Identify contractual restrictions for a corporate transaction',
             'patterns': [('scope-review-summary', ['Establish the transaction scope', 'Review contract restrictions', 'Summarize issues']),
                          ('revision-review', ['Resolve current instructions', 'Review the requested contract restrictions'])]},
}


def read_tsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def load_onet(root=ONET):
    if root == ONET:
        require('onet_root')
    root = Path(root)
    occupations = {r['O*NET-SOC Code']: r for r in read_tsv(root / 'Occupation Data.txt')}
    tasks = read_tsv(root / 'Task Statements.txt')
    dwa_names = {r['DWA Element ID']: r['DWA Element Name'] for r in read_tsv(root / 'GWAs to IWAs to DWAs.txt')}
    linked = {}
    for row in read_tsv(root / 'Tasks to DWAs.txt'):
        linked.setdefault((row['O*NET-SOC Code'], row['Task ID']), []).append(
            {'id': row['DWA Element ID'], 'text': dwa_names[row['DWA Element ID']]})
    return occupations, tasks, linked


def review_intent(intent, client):
    """Cheap-model review hook, with exact citations and declared clean-pool availability."""
    payload = {'operation': 'intent_review', 'intent': intent,
               'criteria': ['occupational realism', 'O*NET alignment', 'clean label availability', 'work-chain compatibility']}
    result = client(payload) if callable(client) else client.complete('review_intent', payload)
    return bool(result.get('approved', result.get('accepted', False))), result


def generate_intents(adapters, occupations=('23-2011.00',), limit=20, review=None, onet_root=ONET,
                     screening_reports=None):
    """Build intents from scorable labels; historical screening is report-only."""
    from .screening import DEFAULT_OUTPUT, load_screening, ScreeningError
    if limit < 0:
        raise ValueError('limit must be nonnegative')
    occupation_data, statements, dwa = load_onet(onet_root)
    selected = [row for row in statements if row['O*NET-SOC Code'] in occupations
                and any(term in row['Task'].lower() for term in ('document', 'contract', 'legal', 'research', 'review', 'analy'))]
    out = []
    for adapter in sorted(adapters, key=lambda a: a.source):
        try:
            report = (screening_reports or {}).get(adapter.source)
            if report is None:
                report = DEFAULT_OUTPUT / (adapter.source + '.json')
            # This report is optional. Do not hash the full corpus merely to
            # discover that its historical screening file does not exist.
            if isinstance(report, (str, Path)) and not Path(report).is_file():
                raise ScreeningError(f'Missing screening report {report}')
            screening = load_screening(adapter, report=report)
            screening_status = {'status': 'reported_only', 'used_for_admission': False,
                                'eligible_questions': screening['eligible_questions']}
        except ScreeningError as exc:
            screening = {}
            screening_status = {'status': 'unavailable', 'used_for_admission': False, 'reason': str(exc)}
        pool = adapter.label_pool()
        available = {q for _, q in pool}
        from .adapters.base import disabled_questions
        disabled = disabled_questions(adapter.quarantine, adapter.source)
        question_ids = sorted(q for q in available if q in adapter.questions() and q not in disabled)
        if not question_ids:
            continue
        deliverable = DELIVERABLES.get(adapter.source, {
            'name': 'Evidence review register', 'purpose': 'Review source evidence for a professional decision',
            'patterns': [('scope-review-summary', ['Establish the review scope', 'Review source evidence', 'Summarize findings']),
                         ('revision-review', ['Resolve current instructions', 'Review the requested source evidence'])]})
        for row in sorted(selected, key=lambda r: (r['O*NET-SOC Code'], r['Task ID'])):
            occupation = row['O*NET-SOC Code']
            for pattern, items in deliverable['patterns']:
                # O*NET and work-chain patterns are context, never task-space quotas.
                key = adapter.source
                identifier = hashlib.sha256(f'{occupation}|{row["Task ID"]}|{adapter.source}|{pattern}'.encode()).hexdigest()[:16]
                intent = {'id': 'intent-' + identifier, 'occupation': occupation,
                          'occupation_title': occupation_data[occupation]['Title'], 'occupation_group': occupation[:2],
                          'onet_version': '31.0', 'onet_tasks': [{'id': row['Task ID'], 'text': row['Task'],
                              'source': str(Path(onet_root) / 'Task Statements.txt')}],
                          'dwa': dwa.get((occupation, row['Task ID']), []), 'source': adapter.source,
                          'screening_version': screening.get('version'),
                          'screening_policy': screening.get('policy'),
                          'screening_report': screening_status,
                          'questions': question_ids, 'question_types': {q: adapter.questions()[q]['question'] for q in question_ids},
                          'clean_label_counts': {q: sum(k[1] == q for k in pool) for q in question_ids},
                          'deliverable': deliverable['name'], 'back_translation': deliverable['purpose'],
                          'work_chain_pattern': pattern,
                          'work_chain': [{'id': f'w{i + 1}', 'title': title, 'input_type': 'documents' if i == 0 else f'w{i}.output',
                                          'output_type': 'scope' if i == 0 else 'reviewed_labels' if i == 1 else 'summary'}
                                         for i, title in enumerate(items)], 'quota_key': key}
                if review is not None:
                    approved, result = review_intent(intent, review)
                    intent['review'] = result
                    if not approved:
                        continue
                else:
                    intent['review'] = {'approved': False, 'status': 'pending_injected_reviewer'}
                out.append(intent)
    # Interleave quota cells before truncating so a prolific occupation/source cannot monopolize the batch.
    groups = {}
    for intent in out:
        groups.setdefault(intent['quota_key'], []).append(intent)
    ordered = []
    while any(groups.values()) and len(ordered) < limit:
        for key in sorted(groups):
            if groups[key] and len(ordered) < limit:
                ordered.append(groups[key].pop(0))
    return ordered


def generate_combination_intents(adapters, enumeration, *, review=None, onet_root=ONET):
    """Choose occupational context inside each fixed dataset combination.

    Source catalogues are supplied separately by the designer entry point; no
    private records or labels are copied into the intent.
    """
    contexts = {}
    for adapter in adapters:
        from .batch1 import GROUPS
        if adapter.source in ('B', 'biodex'):
            occupation = ('19-1042.00',)
        elif adapter.source in GROUPS['B'] + GROUPS['C'] or adapter.source == 'sembench':
            occupation = ('15-2051.00',)
        elif adapter.source == 'asap-chinese-restaurant-absa':
            occupation = ('13-1161.00',)
        elif adapter.source == 'ntsb_aviation_avall':
            occupation = ('17-2111.00',)
        else:
            occupation = ('23-2011.00',)
        contexts[adapter.source] = generate_intents([adapter], occupations=occupation, limit=2,
                                                    onet_root=onet_root)
    out = []
    for combination in enumeration['combinations']:
        sources = combination['sources']
        if any(not contexts.get(source) for source in sources):
            raise ValueError('No local occupational context for combination ' + combination['key'])
        for index in range(2):
            parts = [contexts[source][index % len(contexts[source])] for source in sources]
            base = dict(parts[0])
            identifier = hashlib.sha256((combination['key'] + '|' + str(index)).encode()).hexdigest()[:16]
            base.update(id='intent-' + identifier, sources=sources, combination=combination['key'],
                        quota_key=combination['key'], source=sources[0] if len(sources) == 1 else None,
                        questions={p['source']: p['questions'] for p in parts},
                        question_types={p['source']: p['question_types'] for p in parts},
                        clean_label_counts={p['source']: p['clean_label_counts'] for p in parts},
                        source_contexts=[{k: p[k] for k in ('source', 'occupation', 'occupation_title', 'onet_tasks', 'dwa')}
                                         for p in parts])
            if review is not None:
                approved, result = review_intent(base, review)
                base['review'] = result
                if not approved:
                    continue
            out.append(base)
    return out


def select_undercovered(intents, counts, quotas=None):
    quotas = quotas or {}
    eligible = [i for i in intents if counts.get(i['quota_key'], 0) < quotas.get(i['quota_key'], 1)]
    if not eligible:
        return None
    return min(eligible, key=lambda i: (counts.get(i['quota_key'], 0) / quotas.get(i['quota_key'], 1), i['quota_key'], i['id']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['MAUD', 'CUAD'], action='append', required=True)
    parser.add_argument('--occupation', action='append', default=None)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    intents = generate_intents([get_adapter(s) for s in args.source], args.occupation or ['23-2011.00'], args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(intents, indent=2) + '\n')
    print(json.dumps({'intents': len(intents), 'path': str(args.out), 'review': 'pending'}))


if __name__ == '__main__':
    main()
