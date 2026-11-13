"""Blind double readings produce exclusion proposals; applying is a separate command.

No model transport is built in. A caller supplies a callable client. Never relabel.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

from .adapters import AVAILABLE_SOURCES, available_sources, get_adapter
from .adapters.base import QUARANTINE, exclusions, load_source, normalize


def agreement_definitions(text):
    """Attach the agreement's own defined-term clauses, preserving their exact text."""
    pattern = r'[“"]([^”"\n]{2,100})[”"]\s*(?:shall\s+)?(?:means?\b|shall have the meaning|has the meaning)'
    result = []
    for match in re.finditer(pattern, text, re.I):
        end = re.search(r'\n\s*\n|(?=\n[^\n]{0,60}[“"])', text[match.end():])
        stop = match.end() + end.start() if end else min(len(text), match.end() + 3000)
        result.append({'term': match.group(1), 'start': match.start(), 'text': text[match.start():stop]})
    return result


def structure_signature(segments):
    """Conservative pool grouping: identical normalized evidence, not mere keywords."""
    return hashlib.sha256('\n'.join(normalize(s) for s in segments).encode()).hexdigest()


def find_suspects(adapter, questions=None, records=None):
    selected = set(questions or adapter.questions())
    pool = {k: v for k, v in adapter.label_pool().items() if k[1] in selected}
    groups = defaultdict(list)
    for key, (answer, segments) in pool.items():
        if segments:
            groups[key[1], structure_signature(segments)].append((key, answer))
    structural = {}
    for (question, signature), items in groups.items():
        counts = Counter(answer for _, answer in items)
        majority, count = counts.most_common(1)[0]
        if len(items) >= 5 and count / len(items) >= .8:
            for key, answer in items:
                if answer != majority:
                    structural[key] = {'kind': 'pool_majority', 'signature': signature,
                                       'answer': majority, 'support': count, 'total': len(items),
                                       'members': [list(k) for k, _ in items]}
    suspects = []
    for (record, question), (answer, segments) in sorted(pool.items()):
        if records is not None and record not in records:
            continue
        prediction = adapter.keyword_predict(question, adapter.records()[record]['text'])
        if prediction != answer or (record, question) in structural:
            suspects.append({'record': record, 'question': question,
                             'reasons': ['heuristic_disagreement'] + (['pool_inconsistency'] if (record, question) in structural else []),
                             'structural_evidence': structural.get((record, question))})
    return suspects


def _call(client, payload):
    if callable(client):
        return client(payload)
    return client.complete('preaudit_reading', payload)


def typed_equal(a, b):
    return type(a) is type(b) and a == b


def preaudit(adapter, client, questions=None, suspects=None, out=None):
    pool, rows, proposals = adapter.label_pool(), [], []
    for suspect in (find_suspects(adapter, questions) if suspects is None else suspects):
        record, question = suspect['record'], suspect['question']
        answer, segments = pool[record, question]
        text = adapter.records()[record]['text']
        metadata = adapter.questions()[question]
        payload = {'question': metadata['question'], 'options': metadata['options'],
                   'coding_conventions': metadata['coding_conventions'], 'agreement_text': text,
                   'definitions': agreement_definitions(text), 'evidence_segments': segments}
        # No original label, previous reading, heuristic answer, or model trials enter either prompt.
        readings = [_call(client, dict(payload, read_index=i)) for i in range(2)]
        agreed = (len(readings) == 2 and any(typed_equal(readings[0].get('answer'), option) for option in metadata['options'])
                  and typed_equal(readings[0].get('answer'), readings[1].get('answer')))
        high = all(r.get('confidence') == 'high' for r in readings)
        structural = suspect.get('structural_evidence')
        if structural and structural.get('kind') == 'pool_majority':
            valid = (structural.get('total', 0) >= 5 and structural.get('support', 0) / structural['total'] >= .8
                     and typed_equal(structural.get('answer'), readings[0].get('answer')))
            structural = structural if valid else None
        else:
            structural = None
        # A clause citation must occur verbatim in this same agreement. Both readers must cite it.
        if not structural:
            quotes = [r.get('clause_quote', '') for r in readings]
            if all(len(q.strip()) >= 25 and q in text for q in quotes):
                structural = {'kind': 'citable_clause', 'quotes': quotes,
                              'offsets': [text.index(q) for q in quotes]}
        eligible = agreed and high and not typed_equal(readings[0]['answer'], answer) and bool(structural)
        row = {'source': adapter.source, 'record': record, 'question': question,
               'original_answer': answer, 'answer_options': metadata['options'], 'readings': readings, 'structural_evidence': structural,
               'status': 'propose_exclusion' if eligible else 'no_exclusion',
               'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
               'definitions_count': len(payload['definitions'])}
        rows.append(row)
        if eligible:
            proposals.append(dict(row, action='exclude', reason='Independent high-confidence readings and structural evidence'))
    report = {'source': adapter.source, 'policy': 'exclusion_only_never_relabel', 'rows': rows, 'proposals': proposals,
              'question_pool_sizes': dict(Counter(q for _, q in pool)), 'calls': 2 * len(rows)}
    if out is not None:
        output = Path(out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return report


def apply_proposals(report, quarantine, pool_sizes, notify_path=None):
    """Explicit apply operation. Validates the whole batch before backup and atomic replacement.

    pool_sizes must count the original eligible pool including already excluded labels.
    The default global quarantine is protected during this implementation run: the CLI
    requires a separate explicit --apply and --quarantine path; tests only use scratch.
    """
    path = Path(quarantine)
    data = json.loads(path.read_text()) if path.exists() else {}
    proposals = report.get('proposals', [])
    candidates = defaultdict(set)
    existing = defaultdict(set)
    excluded_records = {}
    # Historical tasks retain their supported source identity even when an
    # adapter (CUADX) is not a separate member of the current batch task space.
    allowed = set(available_sources()) | set(AVAILABLE_SOURCES)
    if any(item.get('source') not in allowed for item in proposals):
        raise ValueError('Unsupported source for automatic exclusion')
    for source in {item['source'] for item in proposals}:
        pairs, record_exclusions = exclusions(path, source)
        if source == 'B':
            # Historical biomedical quarantine can exclude whole papers or prompts.
            # Count every affected prompt conservatively against the same 5% cap.
            record_exclusions |= {str(row.get('prompt_id') or row.get('PromptID')) for row in data.get('B_prompts', [])}
            papers = {str(row['pmc']) for row in data.get('B_papers', [])}
            if papers:
                record_exclusions |= {str(row['PromptID']) for row in load_source('b_sources', '')._ei()['prompts']
                                      if str(row['pmc']) in papers}
        excluded_records[source] = record_exclusions
        for record, question in pairs:
            existing[source, question].add(record)
    for item in proposals:
        reads = item.get('readings', [])
        structural = item.get('structural_evidence') or {}
        valid_reads = (len(reads) == 2 and all(r.get('confidence') == 'high' for r in reads)
                       and reads[0].get('answer') is not None and typed_equal(reads[0].get('answer'), reads[1].get('answer'))
                       and not typed_equal(reads[0]['answer'], item.get('original_answer')))
        if item.get('answer_options') is not None:
            valid_reads = valid_reads and all(any(typed_equal(r.get('answer'), option) for option in item['answer_options']) for r in reads)
        elif item['source'] == 'B':
            valid_reads = valid_reads and all(type(r.get('answer')) is int and r['answer'] in (-1, 0, 1) for r in reads)
        if item['source'] == 'B':
            valid_reads = (valid_reads and type(item.get('original_answer')) is int and item['original_answer'] in (-1, 0, 1)
                           and all(type(r.get('answer')) is int and r['answer'] in (-1, 0, 1) for r in reads))
        valid_structure = (structural.get('kind') == 'citable_clause' and len(structural.get('quotes', [])) == 2
                           or structural.get('kind') == 'pool_majority' and structural.get('total', 0) >= 5
                           and structural.get('support', 0) / structural['total'] >= .8)
        if item.get('action') != 'exclude' or not valid_reads or not valid_structure:
            raise ValueError('Unqualified exclusion proposal')
        candidates[item['source'], item['question']].add(item['record'])
    violations = []
    for (source, question), records in candidates.items():
        size = pool_sizes.get(source + ':' + question, pool_sizes.get(question, 0))
        count = len(records | existing[source, question] | excluded_records[source])
        if size <= 0 or count / size > .05:
            violations.append({'source': source, 'question': question, 'excluded': count, 'pool_size': size, 'cap': .05})
    if violations:
        notice = {'status': 'stopped_notify_richard', 'reason': '5_percent_per_question_cap', 'violations': violations}
        if notify_path:
            Path(notify_path).write_text(json.dumps(notice, indent=2) + '\n')
        return notice
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = path.with_name(path.name + '.backup-' + stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, backup)
    else:
        backup.write_text('{}\n')
    for item in proposals:
        key = item['source'] + '_labels'
        known = existing[item['source'], item['question']]
        if item['record'] in known:
            continue
        row = dict(item, contract_id=item['record'])
        row['dp' if item['source'] == 'MAUD' else 'question'] = item['question']
        if item['source'] == 'NDA':
            row.update(doc_id=item['record'], hypothesis=item['question'])
        elif item['source'] == 'B':
            row['prompt_id'] = item['record']
        elif item['source'] == 'PREC':
            row['cite'] = item['record']
        data.setdefault(key, []).append(row)
        known.add(item['record'])
    temp = path.with_name(path.name + '.tmp-' + stamp)
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)
    return {'status': 'applied', 'backup': str(backup), 'proposals': len(proposals)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', type=Path, help='Apply an already reviewed proposal report; never calls a model')
    parser.add_argument('--quarantine', type=Path, required=True)
    parser.add_argument('--pool-sizes', type=Path, required=True)
    parser.add_argument('--notify', type=Path, required=True)
    args = parser.parse_args()
    if not args.apply:
        parser.error('This CLI only applies proposals; inject a client into preaudit() to create them')
    print(json.dumps(apply_proposals(json.loads(args.apply.read_text()), args.quarantine,
                                     json.loads(args.pool_sizes.read_text()), args.notify), indent=2))


if __name__ == '__main__':
    main()
