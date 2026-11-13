"""Sparse admission must reconstruct availability, draw, drops and public scope."""
import copy
import json

import pytest

from tada_taskgen import adapters, compiler, gates, schema


class SparseSource:
    """Exactly 16 candidates, one rare-class question and one ineligible record."""
    def reference(self):
        return self

    def questions(self):
        return {f'Q{i}': {'options': ['A', 'B']} for i in range(1, 7)}

    def records(self):
        return {f'r{i}': {'text': 'Full source text'} for i in range(17)}

    def visible_id(self, record):
        return record.upper()

    def label_pool(self):
        return {(f'r{i}', q): ('A' if i < 8 or q == 'Q6' else 'B', [])
                for i in range(17) for q in self.questions()
                if (i < 16 or q in ('Q1', 'Q2')) and not (q == 'Q5' and i in (0, 8))}

    def scorable(self, record, question):
        return (record, question) in self.label_pool()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def sparse_task(tmp_path, monkeypatch):
    source = SparseSource()
    monkeypatch.setattr(adapters, 'get_adapter', lambda name: source)
    spec = {'source': 'MAUD', 'questions': list(source.questions()), 'sample_size': 16,
            'work_items': [{'id': 'coding'}]}
    sp = compiler.resolve(spec, 7)
    cells = [{'record': record['record_id'], 'question': q,
              'answer': source.label_pool()[(record['record_id'], q)][0]}
             for record in sp['records'] for q in record['questions']]
    gold = {'cells': cells, 'label_class_counts': compiler.class_counts(cells),
            'label_aggregation': compiler.SCORING_POLICY['label_aggregation'],
            'weights': compiler.WEIGHTS}
    task = tmp_path/'task'
    save(task/'tests/scenario_spec.json', spec)
    save(task/'tests/scenario_params.json', sp)
    save(task/'tests/instance_guard.json', {**compiler.guard_instance(sp), 'sampling': sp['sampling']})
    save(task/'tests/gold.json', gold)
    save(task/'environment/public/coding_conventions.json', {
        'records': [{key: row[key] for key in ('visible_id', 'visible_file', 'questions')} for row in sp['records']],
        'questions': {q: {} for q in sp['spec']['questions']}, 'applicability_memo': compiler.REVIEW_MEMO})
    (task/'environment/public'/compiler.REVIEW_MEMO).write_text(compiler.review_memo(sp))
    return task


def mutate(task, relative, fn):
    path = task/relative
    value = json.loads(path.read_text())
    fn(value)
    save(path, value)


def test_sparse_audit_replays_original_candidates_and_records_dropped_question(sparse_task):
    result = gates.sparse_instance_gate(sparse_task)
    assert result['pass']
    assert result['sampling']['candidate_count'] == 16
    assert result['sampling']['retained_questions'] == ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    assert result['sampling']['dropped_questions']['Q6']['class_counts'] == {'A': 16}


def test_dropped_question_cannot_shrink_denominator_with_matching_metadata(sparse_task):
    def remove(gold):
        gold['cells'] = [c for c in gold['cells'] if c['question'] != 'Q5']
        gold['label_class_counts'] = compiler.class_counts(gold['cells'])
    mutate(sparse_task, 'tests/gold.json', remove)
    result = gates.sparse_instance_gate(sparse_task)
    assert not result['pass']
    assert 'exact sampled record/question pairs' in result['error']


def test_internally_consistent_question_removal_cannot_change_fixed_drop_rule(sparse_task):
    # Removing all mention from effective spec, records and audit is still invalid.
    def remove(sp):
        sp['spec']['questions'].remove('Q5')
        sp['sampling']['retained_questions'].remove('Q5')
        sp['sampling']['dropped_questions']['Q5'] = {'reason': 'hard question', 'class_counts': {'A': 7, 'B': 7}}
        for r in sp['records']:
            r['questions'] = [q for q in r['questions'] if q != 'Q5']
    mutate(sparse_task, 'tests/scenario_params.json', remove)
    assert 'fixed sparse draw' in gates.sparse_instance_gate(sparse_task)['error']


def test_source_unavailable_pair_cannot_be_added(sparse_task):
    mutate(sparse_task, 'tests/scenario_params.json',
           lambda sp: next(r for r in sp['records'] if r['record_id'] == 'r0')['questions'].append('Q5'))
    assert not gates.sparse_instance_gate(sparse_task)['pass']


def test_source_available_pair_cannot_be_hidden(sparse_task):
    mutate(sparse_task, 'tests/scenario_params.json', lambda sp: sp['records'][0]['questions'].remove('Q1'))
    assert not gates.sparse_instance_gate(sparse_task)['pass']


@pytest.mark.parametrize('target', ['memo', 'contract', 'guard', 'gold_duplicate', 'metadata'])
def test_sparse_audit_rejects_inconsistent_materials(sparse_task, target):
    if target == 'memo':
        (sparse_task/'environment/public'/compiler.REVIEW_MEMO).write_text('No checklist supplied')
    elif target == 'contract':
        mutate(sparse_task, 'environment/public/coding_conventions.json', lambda obj: obj['records'][0].update(questions=[]))
    elif target == 'guard':
        mutate(sparse_task, 'tests/instance_guard.json', lambda obj: obj['sampling'].update(dropped_questions={}))
    elif target == 'gold_duplicate':
        mutate(sparse_task, 'tests/gold.json', lambda obj: obj['cells'].append(copy.deepcopy(obj['cells'][0])))
    else:
        mutate(sparse_task, 'tests/gold.json', lambda obj: obj.update(label_class_counts={}))
    assert not gates.sparse_instance_gate(sparse_task)['pass']


def test_requested_questions_and_sparse_guard_remain_reported(sparse_task, monkeypatch):
    seen = []
    monkeypatch.setattr(schema, 'validate_spec', lambda spec: seen.append(spec['questions']))
    monkeypatch.setattr(gates, 'invertibility', lambda task: {'pass': True})
    monkeypatch.setattr(gates, 'roundtrip_gate', lambda task: {'pass': True})
    result = gates.extensions(sparse_task, {'verdict': {}, 'steps': {
        'keyword_baseline': {'details': {'per_work_item': {'coding': 0.1}}}}})
    assert seen == [['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']]
    assert result['verdict']['reported_only']['v5_instance_class_guard']
    assert 'v5_instance_class_guard' not in result['hard_gates']
