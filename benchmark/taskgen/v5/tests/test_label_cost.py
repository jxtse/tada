"""R3 cost regressions use actual compute_gold and scorer, with a tiny fake source."""
import json

import pytest
from tada_taskgen import adapters, label_cost
from tada_taskgen.compiler import compute_gold, sheet_specs, write_workbook
from tada_taskgen.scoring import LABEL_AGGREGATION


class SourceFixture:
    def __init__(self, majority, minority):
        self.pool = {(f'r{i}', 'Q1'): (answer, ['Evidence'])
                     for i, answer in enumerate(['A'] * majority + ['B'] * minority)}

    def label_pool(self):
        return self.pool

    def questions(self):
        return {'Q1': {'options': ['A', 'B']}}

    def encode(self, question, answer):
        if answer not in self.questions()[question]['options']:
            raise ValueError(answer)
        return answer


def task_fixture(tmp_path, monkeypatch, majority, minority):
    source = SourceFixture(majority, minority)
    monkeypatch.setattr(adapters, 'get_adapter', lambda source_name: source)
    monkeypatch.setattr(label_cost, 'get_adapter', lambda source_name: source)
    records = [{'record_id': rid, 'visible_id': rid, 'questions': ['Q1']} for rid, _ in source.pool]
    spec = {'source': 'MAUD', 'questions': ['Q1'], 'dependency_graph': {'nodes': [
        {'template': 'record_question', 'work_item': 'coding'}]}}
    sp = {'spec': spec, 'records': records, 'scope_facts': {r['visible_id']: 'Yes' for r in records}}
    sheets, info = compute_gold(sp)
    specs = sheet_specs(spec)
    gold = {
        'schema_version': 5, 'label_aggregation': LABEL_AGGREGATION,
        'weights': {'labels': .70, 'scope': .15, 'facts': .10, 'consistency': .05},
        'sheets': {name: {'columns': cols, 'key_columns': keys,
                         'rows': {'|'.join(row[k] for k in keys): dict(row) for row in sheets[name]}}
                   for name, keys, cols in specs},
        'cells': info['cells'],
        'facts': [{'id': f'scope:{r["visible_id"]}', 'sheet': 'Scope', 'key': [r['visible_id']],
                   'column': 'Included', 'value': 'Yes', 'work_item': 'scope'} for r in records],
        'scopes': [{'work_item': 'coding', 'sheet': 'Scope', 'column': 'Included',
                    'expected_keys': [[r['visible_id']] for r in records]}],
        'consistency': [{'id': 'summary:Q1', 'sheet': 'Summary', 'key': ['Q1'], 'column': 'Positive_Count',
                         'operation': 'count_option', 'option': 'A', 'work_item': 'summary',
                         'dependencies': [c['id'] for c in info['cells']]}],
    }
    gold['sheets']['Summary']['rows']['Q1'].pop('Positive_Count')
    task = tmp_path / 'task'
    (task / 'tests').mkdir(parents=True)
    (task / 'solution').mkdir()
    (task / 'tests/gold.json').write_text(json.dumps(gold))
    (task / 'tests/scenario_params.json').write_text(json.dumps(sp))
    write_workbook(sheets, specs, task / 'solution/reference.xlsx')
    return task


@pytest.mark.parametrize('majority,minority,expected_max,expected_median,passed', [
    (4, 2, .40, .225, False),
    (8, 4, .225, .1375, False),
    (16, 8, .1375, .09375, True),
])
def test_label_cost_uses_balanced_denominators_and_fixed_gates(
        tmp_path, monkeypatch, majority, minority, expected_max, expected_median, passed):
    task = task_fixture(tmp_path, monkeypatch, majority, minority)
    result = label_cost.label_flip(task, tmp_path / 'cost')
    assert result['self_score'] == 1
    assert result['label_aggregation'] == LABEL_AGGREGATION
    assert result['gold_class_counts'] == {'Q1': {'A': majority, 'B': minority}}
    assert result['gold_codebook_option_counts'] == {'Q1': 2}
    assert json.loads((tmp_path / 'cost/label_flip.json').read_text()) == result
    assert (tmp_path / 'cost/label_flip.json').read_bytes() == (tmp_path / 'cost/report.json').read_bytes()
    assert result['flips'] == majority + minority
    assert result['errors'] == []
    assert result['max_loss'] == pytest.approx(expected_max)
    assert result['median_loss'] == pytest.approx(expected_median)
    assert result['gate'] == {'max': .20, 'median': .10}
    assert result['pass'] is passed
    # In the 4:2 case a minority flip yields a submitted 5:1 distribution.
    # It must still be scored against 4:2 gold, not rejected as a new instance.
    flips = json.loads((tmp_path / 'cost/flips.json').read_text())
    assert len(flips) == majority + minority


def test_failed_flip_never_disappears_from_gate(tmp_path, monkeypatch):
    task = task_fixture(tmp_path, monkeypatch, 8, 4)
    original = label_cost.compute_gold

    def fail_only_counterfactual(sp, label_fn=None, *, adapter=None):
        if label_fn is not None:
            raise ValueError('synthetic failure')
        return original(sp, adapter=adapter)

    monkeypatch.setattr(label_cost, 'compute_gold', fail_only_counterfactual)
    result = label_cost.label_flip(task, tmp_path / 'cost')
    assert not result['pass']
    assert len(result['errors']) == 12
    assert result['flips'] == 0
    assert result['max_loss'] is None
