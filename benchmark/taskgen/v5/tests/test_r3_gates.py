"""Scope-test gating is component-specific and retains the original reward."""
import copy
import json
from pathlib import Path

import pytest

from tada_taskgen.gates import classify, evaluate_wrong_answer, run_multiseed, POLICY
from tada_taskgen.schema import validate_spec_structure


def scenario():
    return {'version': 5, 'id': 'x', 'source': 'MAUD', 'category': 'parallel_coding',
            'questions': ['q'], 'sample_size': 16, 'documents': [], 'instruction': 'Review.', 'design': 'Work.',
            'work_items': [{'id': 'scope', 'title': 'Scope', 'order': 1, 'depends_on': []},
                           {'id': 'coding', 'title': 'Code', 'order': 2, 'depends_on': ['scope']}],
            'dependency_graph': {'nodes': [], 'edges': []},
            'thresholds': {'include_all': .90, 'ignore_revisions': .90}}


@pytest.mark.parametrize('key', ['include_all', 'ignore_revisions'])
@pytest.mark.parametrize('bound', [1.1, True, -1, float('nan'), float('inf')])
def test_scope_threshold_validator_rejects_invalid_report_bound(key, bound):
    spec = scenario()
    spec['thresholds'][key] = bound
    with pytest.raises(ValueError, match='scope F1 bound'):
        validate_spec_structure(spec)


def test_scope_threshold_boundary_and_machine_schema_agree():
    assert validate_spec_structure(scenario())
    schema = json.loads((Path(__file__).parents[1]/'schemas/scenario.schema.json').read_text())
    for field in ('include_all', 'ignore_revisions'):
        assert schema['properties']['thresholds']['properties'][field]['maximum'] == 1
        reported = scenario()
        reported['thresholds'][field] = 1
        assert validate_spec_structure(reported)


@pytest.mark.parametrize('name', ['include_all', 'ignore_revisions', 'ignore_current_notice'])
def test_scope_fixture_uses_f1_even_when_total_reward_exceeds_bound(name):
    result = evaluate_wrong_answer(name, .954, {'components': {'scope': 6/7}}, [0, .90], scenario()['thresholds'])
    assert result['in_range'] and result['scope_f1'] == 6/7
    assert result['reward'] == .954 and result['gate_metric'] == 'scope_f1'
    assert result['expected'] == [0, .90]


@pytest.mark.parametrize('scope', [None, True, '0.1', float('nan'), float('inf'), -1, 1.1, .91])
def test_scope_fixture_never_substitutes_low_total_reward_for_invalid_scope(scope):
    result = evaluate_wrong_answer('include_all', .1, {'components': {'scope': scope}}, [0, .9], scenario()['thresholds'])
    assert not result['in_range']


@pytest.mark.parametrize('details', [{}, None, [], {'components': []}, {'components': 'invalid'}])
def test_scope_fixture_rejects_missing_or_malformed_component_container(details):
    assert not evaluate_wrong_answer('include_all', .1, details, [0, .9], scenario()['thresholds'])['in_range']


def test_other_fixtures_continue_to_use_total_reward():
    result = evaluate_wrong_answer('all_majority', .4, {'components': {'scope': 1}}, [0, .5], scenario()['thresholds'])
    assert result['in_range'] and result['gate_metric'] == 'reward'
    assert 'scope_f1' not in result


def test_classify_reads_scope_only_and_reports_total_rewards():
    spec = scenario()
    report = {'verdict': {}, 'steps': {'wrong_answers': {'results': {
        'include_all': {'scope_f1': 6/7, 'reward': .954},
        'ignore_revisions': {'scope_f1': 6/7, 'reward': .954}}}}}
    result = classify(report, spec)
    for key in spec['thresholds']:
        metric = result['metrics'][key]
        assert metric['value'] == 6/7 and metric['pass']
        assert metric['component'] == 'scope_f1' and metric['total_rewards'][key] == .954
    missing = copy.deepcopy(report)
    del missing['steps']['wrong_answers']['results']['include_all']['scope_f1']
    result = classify(missing, spec)
    assert result['metrics']['include_all']['value'] is None
    assert not result['metrics']['include_all']['pass']


def test_controller_reads_exact_score_directory_and_preserves_legacy(tmp_path):
    from controller import wrong_answer_result
    task = tmp_path/'task'
    (task/'tests').mkdir(parents=True)
    (task/'tests/gold.json').write_text(json.dumps({'schema_version': 5}))
    (task/'tests/scenario_spec.json').write_text(json.dumps(scenario()))
    score_dir = tmp_path/'score_include_all'
    score_dir.mkdir()
    (score_dir/'details.json').write_text(json.dumps({'components': {'scope': 6/7}}))
    scored = {'exit': 0, 'reward': .954}
    assert wrong_answer_result(task, 'include_all', scored, [0, .90], score_dir)['in_range']
    assert not wrong_answer_result(task, 'include_all', scored, [0, .90], tmp_path/'missing')['in_range']
    assert not wrong_answer_result(task, 'include_all', {**scored, 'exit': 1}, [0, .90], score_dir)['in_range']
    (task/'tests/gold.json').write_text('{}')
    legacy = wrong_answer_result(task, 'include_all', scored, [0, .90], score_dir)
    assert legacy == {'reward': .954, 'expected': [0, .90], 'in_range': False}


@pytest.mark.parametrize('mode', ['fake', 'real'])
def test_multiseed_forwards_client_mode_without_calling_model(mode, monkeypatch, tmp_path):
    from tada_taskgen import compiler, gates, schema
    seen = []
    monkeypatch.setattr(schema, 'validate_spec', lambda spec: spec)
    def stopped(spec, seed, out, *, client):
        seen.append((seed, client))
        raise compiler.InstanceCompilationError('Injected pre-model stop')
    monkeypatch.setattr(compiler, 'compile_task', stopped)
    monkeypatch.setattr(gates, 'run_controller', lambda *a, **k: pytest.fail('No controller after compile failure'))
    result = run_multiseed(scenario(), tmp_path/'multiseed', client=mode)
    assert seen == [(seed, mode) for seed in POLICY['seeds']]
    assert result['client'] == mode
    assert result['compilation_failures'] == 5
    frozen = json.loads((tmp_path/'multiseed/frozen_policy.json').read_text())
    assert frozen['client'] == mode
