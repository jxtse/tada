"""Owner policy 2026-11-12: minor (severity-2) review findings left after the revision budget do not block;
scope status is scored once; a component without items is dropped and the others renormalised."""
import json
from pathlib import Path

import pytest

from tada_taskgen import design_loop
from tada_taskgen.scheduler import review_failures
from tada_taskgen.scoring import score_records

REPO = Path(__file__).resolve().parents[2]


def review(severity, key='scoring_fairness'):
    questions = [{'id': k, 'severity': severity if k == key else 0,
                  'evidence': 'Public evidence.' if k == key and severity else 'Clear.',
                  'required_change': 'Change it.' if k == key and severity else ''}
                 for k in ('instruction_sufficiency', 'scoring_fairness', 'realism_factual_correctness',
                           'text_analysis_difficulty')]
    return {'questions': questions}


def run(tmp_path, severity):
    spec = json.loads((REPO / 'v5/specs/maud_screened.json').read_text())
    gates = lambda s, out: {'task': str(tmp_path / 'task'),
                            'hard_gates': {k: {'pass': True} for k in design_loop.HARD_GATES}}
    return design_loop.prepare_candidate({}, tmp_path / 'c', designer=lambda *a: spec, compile_and_gate=gates,
                                         reviewer=lambda *a: review(severity), max_revisions=1)


def test_severity_two_left_after_budget_is_admitted_with_notes(tmp_path):
    result = run(tmp_path, 2)
    assert result['verdict'] == 'ready' and result['revisions'] == 1
    assert [n['id'] for n in result['admitted_with_review_notes']] == ['scoring_fairness']
    assert review_failures(result['review']) == []


def test_severity_three_still_blocks(tmp_path):
    result = run(tmp_path, 3)
    assert result['verdict'] == 'discarded'
    assert review_failures(result['review'])


def test_hard_gate_failure_still_blocks(tmp_path):
    spec = json.loads((REPO / 'v5/specs/maud_screened.json').read_text())
    gates = lambda s, out: {'task': str(tmp_path / 'task'), 'hard_gates': {
        k: {'pass': k != 'keyword_baseline'} for k in design_loop.HARD_GATES}}
    result = design_loop.prepare_candidate({}, tmp_path / 'c', designer=lambda *a: spec, compile_and_gate=gates,
                                           reviewer=lambda *a: review(2), max_revisions=0)
    assert result['verdict'] == 'discarded' and 'keyword_baseline' in result['failures']


def _gold(with_doc_fact):
    sheets = {'S': {'key_columns': ['Record_ID'], 'columns': ['Record_ID', 'Included'],
                    'rows': {'A': {'Record_ID': 'A', 'Included': 'Yes'}, 'B': {'Record_ID': 'B', 'Included': 'No'}}},
              'C': {'key_columns': ['Record_ID', 'Question'], 'columns': ['Record_ID', 'Question', 'Answer'],
                    'rows': {'A|Q': {'Record_ID': 'A', 'Question': 'Q', 'Answer': 'Yes'},
                             'B|Q': {'Record_ID': 'B', 'Question': 'Q', 'Answer': 'No'}}}}
    facts = [{'id': f'scope:{k}', 'sheet': 'S', 'key': [k], 'column': 'Included', 'value': v, 'work_item': 'scope',
              'counted_in': 'scope'} for k, v in (('A', 'Yes'), ('B', 'No'))]
    if with_doc_fact:
        sheets['F'] = {'key_columns': ['Fact_ID'], 'columns': ['Fact_ID', 'Value'],
                       'rows': {'d': {'Fact_ID': 'd', 'Value': '2024-10-02'}}}
        facts.append({'id': 'd', 'sheet': 'F', 'key': ['d'], 'column': 'Value', 'value': '2024-10-02',
                      'work_item': 'summary'})
    enc = [['No', 'No'], ['Yes', 'Yes']]
    cells = [{'id': f'label:{r}:Q', 'sheet': 'C', 'key': [r, 'Q'], 'column': 'Answer', 'record': r, 'question': 'Q',
              'answer': a, 'encoding': enc, 'work_item': 'coding'} for r, a in (('A', 'Yes'), ('B', 'No'))]
    return {'label_aggregation': 'per_question_chance_corrected_balanced_accuracy', 'sheets': sheets, 'cells': cells,
            'facts': facts, 'consistency': [],
            'scopes': [{'work_item': 'coding', 'sheet': 'S', 'column': 'Included', 'expected_keys': [['A']]}],
            'weights': {'labels': .70, 'scope': .15, 'facts': .10, 'consistency': .05}}


def _parsed(scope_b, fact=None):
    sheet = lambda rows: {'status': 'ok', 'records': rows, 'extra': 0}
    parsed = {'S': sheet([{'Record_ID': 'A', 'Included': 'Yes'}, {'Record_ID': 'B', 'Included': scope_b}]),
              'C': sheet([{'Record_ID': 'A', 'Question': 'Q', 'Answer': 'Yes'},
                          {'Record_ID': 'B', 'Question': 'Q', 'Answer': 'No'}])}
    if fact is not None:
        parsed['F'] = sheet([{'Fact_ID': 'd', 'Value': fact}])
    return parsed


def test_scope_status_is_not_double_counted_and_components_renormalise():
    perfect, _ = score_records(_parsed('No'), _gold(False))
    assert perfect == pytest.approx(1.0)
    wrong_scope, detail = score_records(_parsed('Yes'), _gold(False))
    assert detail['components']['facts'] == 0  # no document facts: dropped, not zero-weighted
    # Only scope F1 drops (2/3): reward = (.70*1 + .15*2/3) / .90 (no consistency/facts items).
    assert wrong_scope == pytest.approx((.70 + .15 * 2 / 3) / .85)
    with_fact, detail = score_records(_parsed('No', '2024-10-02'), _gold(True))
    assert with_fact == pytest.approx(1.0) and detail['components']['facts'] == 1
