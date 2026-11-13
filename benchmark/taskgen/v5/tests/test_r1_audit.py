"""Independent R1 admission audit: original labels govern guards, failures retain seeds."""
import json

import pytest
from tada_taskgen import adapters, compiler, gates, schema


class GuardSource:
    def __init__(self, original, synthetic=None):
        self.original = original
        self.synthetic = synthetic if synthetic is not None else original

    def reference(self):
        return GuardSource(self.original)

    def label_pool(self):
        return {(f'r{i}', q): (answer, []) for i, answer in enumerate(self.synthetic)
                for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5')}


def guard_params():
    return {'seed': 7, 'spec': {'source': 'MAUD', 'questions': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']},
            'records': [{'record_id': f'r{i}', 'questions': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']} for i in range(4)]}


def test_guard_uses_authoritative_classes_when_permutation_has_singleton(monkeypatch):
    source = GuardSource(['A', 'A', 'B', 'B'], ['A', 'A', 'A', 'B'])
    monkeypatch.setattr(adapters, 'get_adapter', lambda source_name: source)
    result = compiler.guard_instance(guard_params())
    assert result['pass']
    assert result['class_counts'] == {q: {'A': 2, 'B': 2} for q in guard_params()['spec']['questions']}


def test_balanced_permutation_cannot_rescue_invalid_original_instance(monkeypatch):
    source = GuardSource(['A', 'A', 'A', 'B'], ['A', 'A', 'B', 'B'])
    monkeypatch.setattr(adapters, 'get_adapter', lambda source_name: source)
    with pytest.raises(compiler.InstanceCompilationError) as error:
        compiler.guard_instance(guard_params())
    assert error.value.details['invalid_questions'] == {q: {'A': 3, 'B': 1} for q in guard_params()['spec']['questions']}


def test_all_failed_compilations_keep_five_seeds_and_null_metrics(tmp_path, monkeypatch):
    spec = {'source': 'MAUD', 'work_items': [{'id': 'scope'}, {'id': 'coding'}],
            'thresholds': {'include_all': .90, 'ignore_revisions': .90}}
    monkeypatch.setattr(schema, 'validate_spec', lambda value: value)
    attempted = []

    def fail(spec, seed, out, **kwargs):
        attempted.append(seed)
        raise compiler.InstanceCompilationError('instance class guard failed',
                                                {'class_counts': {'Q': {'A': 3, 'B': 1}}, 'seed': seed})

    monkeypatch.setattr(compiler, 'compile_task', fail)
    monkeypatch.setattr(gates, 'run_controller', lambda *_args: pytest.fail('controller must not run after failed compile'))
    result = gates.run_multiseed(spec, tmp_path / 'multiseed')
    assert attempted == [7, 11, 19, 23, 31]
    assert [row['seed'] for row in result['instances']] == attempted
    assert result['compilation_failures'] == 5
    assert result['keyword_gate']['passes'] == 0
    assert result['keyword_gate']['worst_value'] is None
    assert not result['keyword_gate']['pass']
    assert not result['pass'] and not result['engineering_pass']
    for row in result['instances']:
        assert row['task'] is None and row['keyword'] is None
        assert all(metric['value'] is None and not metric['pass'] for metric in row['metrics'].values())
        assert row['failure']['stage'] == 'compile'
        report = json.loads((tmp_path / 'multiseed' / f'seed-{row["seed"]}' / 'failure.json').read_text())
        assert report['failure']['details']['seed'] == row['seed']
