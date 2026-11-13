"""Public-only rule baseline and generic adversarial fixtures."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from .compiler import write_workbook, compute_gold, sheet_specs, dump


def read_facts(text):
    """A deliberately transparent baseline for offline generated fixtures."""
    for line in text.splitlines():
        if line.startswith('Recorded facts: '):
            return json.loads(line.removeprefix('Recorded facts: '))
    return {}


def keyword_baseline(public, out):
    public = Path(public)
    contract = json.loads((public / 'coding_conventions.json').read_text())
    specs = contract['sheets']
    scope, coding, summary = [x[0] for x in specs[:3]]
    rows = {name: [] for name, _, _ in specs}
    facts = read_facts((public / contract['scope_documents']['current']).read_text())
    questions = {q: dict(v, options=[x[0] for x in v['encoding']], answer_options=[x[0] for x in v['encoding']])
                 for q, v in contract['questions'].items()}
    def predictor(q, text):
        if questions[q].get('kind') == 'benchmark_item':
            return None
        source = questions[q].get('source', contract['source'])
        original = questions[q].get('source_question', q)
        if source == 'MAUD':
            from .adapters.maud_keywords import predict
            return predict(original, text, {original: questions[q]})
        if source in ('CUAD', 'C2'):
            from .adapters.cuad import CUADAdapter
            return CUADAdapter.keyword_predict(None, original, text)
        if source == 'B':
            return 0
        if source == 'PREC':
            import re
            return '1' if re.search(r'\b(?:we (?:now )?overrule|is hereby overruled)\b', text, re.I) else '0'
        from .adapters.base import generic_term_predict
        return generic_term_predict(questions[q]['options'], text)
    for record in contract['records']:
        vid = record['visible_id']
        if vid in contract.get('scope_records', [r['visible_id'] for r in contract['records']]):
            rows[scope].append({'Record_ID': vid, 'Included': facts.get(vid, 'Yes')})
        text = (public / record['visible_file']).read_text()
        for q in record['questions']:
            answer = predictor(q, text)
            encoded = dict(contract['questions'][q]['encoding']).get(answer, 'INVALID')
            rows[coding].append({'Record_ID': vid, 'Question': q, 'Answer': encoded})
    for q, value in contract['questions'].items():
        if value.get('kind') == 'benchmark_item':
            rows[summary].append({'Question': q, 'Positive_Count': 0})
            continue
        option = dict(value['encoding'])[value['summary_option']]
        rows[summary].append({'Question': q, 'Positive_Count': sum(r['Answer'] == option for r in rows[coding] if r['Question'] == q)})
    if len(specs) > 3:
        for node in contract.get('extra_facts', []):
            values = read_facts((public / node['visible_file']).read_text())
            rows[specs[3][0]].append({'Fact_ID': node['id'], 'Value': values.get(node['fact_key'])})
    write_workbook(rows, specs, out)
    return out


def wrong_answers(task, out):
    task, out = Path(task), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sp = json.loads((task / 'tests/scenario_params.json').read_text())
    specs = sheet_specs(sp['spec'])
    scope, coding, summary = [x[0] for x in specs[:3]]
    correct, _ = compute_gold(sp)
    cases = {}
    included = copy.deepcopy(correct)
    for row in included[scope]:
        row['Included'] = 'Yes'
    cases['include_all'] = included
    ignored = copy.deepcopy(correct)
    for row in ignored[scope]:
        row['Included'] = sp['prior_scope_facts'][row['Record_ID']]
    cases['ignore_revisions'] = ignored
    independent = copy.deepcopy(correct)
    # True per-document labels + public rules; no cross-document scope facts.
    for row in independent[scope]:
        row['Included'] = 'Yes'
    cases['independent_then_filter'] = independent
    draft = copy.deepcopy(ignored)
    for row in draft[coding]:
        row['Answer'] = 'INVALID'
    cases['copy_old_draft'] = draft
    from collections import Counter
    majority = copy.deepcopy(correct)
    for q in sp['spec']['questions']:
        top = Counter(r['Answer'] for r in correct[coding] if r['Question'] == q).most_common(1)[0][0]
        for row in majority[coding]:
            if row['Question'] == q:
                row['Answer'] = top
    cases['all_majority'] = majority
    expected = {}
    for name, rows in cases.items():
        write_workbook(rows, specs, out / f'{name}.xlsx')
        expected[name] = [0, sp['spec']['thresholds'].get(name, 1.0)]
    # For these two fixtures the declared interval applies to scope F1, not reward.
    dump(out / 'expected_metrics.json', {'include_all': 'scope_f1', 'ignore_revisions': 'scope_f1'})
    dump(out / 'expected.json', expected)
    return expected


def sensitivity(task):
    task = Path(task)
    sp = json.loads((task / 'tests/scenario_params.json').read_text())
    sheets, info = compute_gold(sp)
    gold = json.loads((task / 'tests/gold.json').read_text())
    assert info['cells'] == gold['cells'], 'saved scenario does not reproduce original labels'
    from .scoring import score
    dest = task / 'tests/sensitivity_reference.xlsx'
    write_workbook(sheets, sheet_specs(sp['spec']), dest)
    assert score(dest, gold)[0] == 1.0
    changed = copy.deepcopy(sp)
    if not changed['scope_facts']:
        dest.unlink()
        return {'pass': True, 'exact_instance': True, 'scope_edit_changes_prerequisite_only': None}
    key = next(iter(changed['scope_facts']))
    changed['scope_facts'][key] = 'No' if changed['scope_facts'][key] == 'Yes' else 'Yes'
    new, new_info = compute_gold(changed)
    assert new != sheets and new_info['cells'] == info['cells'], 'scope edit must be local to prerequisites'
    dest.unlink()
    return {'pass': True, 'exact_instance': True, 'scope_edit_changes_prerequisite_only': True}


def keyword_main():
    p = argparse.ArgumentParser(); p.add_argument('--public', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); keyword_baseline(a.public, a.out)


def wrong_main():
    p = argparse.ArgumentParser(); p.add_argument('--task', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); wrong_answers(a.task, a.out)


def sensitivity_main():
    p = argparse.ArgumentParser(); p.add_argument('--task', required=True)
    a = p.parse_args(); print(json.dumps(sensitivity(a.task)))
