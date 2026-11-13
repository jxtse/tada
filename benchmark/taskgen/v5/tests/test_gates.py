"""Mutation checks for the v5 correctness gates and frozen multi-seed policy."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from tada_taskgen.compiler import compile_task
from screened_support import screened_spec, integration_adapters
from tada_taskgen.gates import aggregate, invertibility, roundtrip_gate, POLICY


@pytest.fixture(scope='module', autouse=True)
def synthetic_pools():
    with integration_adapters():
        yield


@pytest.fixture(scope='module')
def task_fixture(tmp_path_factory):
    spec, report = screened_spec('CUAD')
    target = tmp_path_factory.mktemp('gates') / 'task'
    compile_task(spec, 7, target, screening_report=report)
    return target


@pytest.fixture
def task(task_fixture, tmp_path):
    target = tmp_path / 'task'
    shutil.copytree(task_fixture, target)
    return target


def mutate(task, relative, change):
    path = task / relative
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_compiler_passes_invertibility_and_roundtrip(task_fixture):
    assert invertibility(task_fixture)['pass']
    assert roundtrip_gate(task_fixture)['pass']


def test_duplicate_label_cell_is_not_bijection(task):
    mutate(task, 'tests/gold.json', lambda g: g['cells'].append(copy.deepcopy(g['cells'][0])))
    result = invertibility(task)
    assert not result['pass']
    assert any('bijection' in failure for failure in result['failures'])


def test_changed_original_label_rejected(task):
    mutate(task, 'tests/gold.json', lambda g: g['cells'][0].update(answer='not-original'))
    assert not invertibility(task)['pass']


def test_unpublished_encoding_rejected(task):
    mutate(task, 'environment/public/coding_conventions.json',
           lambda d: next(iter(d['questions'].values()))['encoding'].append(['Other', 'Other']))
    assert not invertibility(task)['pass']


def test_derived_independent_gold_rejected(task):
    mutate(task, 'tests/gold.json', lambda g: g['consistency'][0].update(value=100))
    assert not invertibility(task)['pass']


def test_solver_derived_value_leak_rejected(task):
    token = 'PRIVATE-DERIVED-RESULT-7926'
    mutate(task, 'tests/dependency_graph.json',
           lambda g: next(n for n in g['nodes'] if n['type'] == 'derived_quantity').update(value=token))
    (task / 'environment/public/leak.txt').write_text(token)
    assert not invertibility(task)['pass']


def test_roundtrip_modified_carrier_rejected(task):
    report = json.loads((task / 'tests/roundtrip_report.json').read_text())
    doc = next(d for d in report['documents'] if d['status'] == 'verified')
    (task / 'environment/public' / doc['path']).write_text('Different facts')
    assert not roundtrip_gate(task)['pass']


def test_roundtrip_same_model_rejected(task):
    mutate(task, 'tests/roundtrip_report.json',
           lambda r: r['documents'][0].update(extractor_model=r['documents'][0]['writer_model']))
    assert not roundtrip_gate(task)['pass']


def test_roundtrip_wrong_extraction_rejected(task):
    mutate(task, 'tests/roundtrip_report.json',
           lambda r: r['documents'][0]['attempts'][-1].update(extracted={'wrong': 'value'}))
    assert not roundtrip_gate(task)['pass']


def instances(failing=0, over_cap=False):
    result = []
    for index, seed in enumerate(POLICY['seeds']):
        passes = index >= failing
        value = .29 if passes else .36 if over_cap else .34
        result.append({'seed': seed, 'engineering_pass': True, 'numeric_pass': passes,
                       'all_pass': passes, 'metrics': {'keyword': {'value': value, 'pass': passes,
                       'within_worst_cap': value <= .35}}})
    return result


def test_multiseed_four_of_five_under_worst_cap():
    result = aggregate(instances(failing=1), 'frozen-config-digest')
    assert result['pass'] and result['numeric_passes'] == 4
    assert len(result['eligible_instances']) == 4
    assert POLICY['seeds'][0] not in result['eligible_instances']


def test_multiseed_three_of_five_rejected():
    assert not aggregate(instances(failing=2), 'frozen')['pass']


def test_multiseed_four_of_five_over_worst_cap_rejected():
    assert not aggregate(instances(failing=1, over_cap=True), 'frozen')['pass']


def test_multiseed_one_engineering_failure_rejects_design():
    cases = instances()
    cases[0]['engineering_pass'] = False
    cases[0]['all_pass'] = False
    assert not aggregate(cases, 'frozen')['pass']


def test_multiseed_duplicate_or_wrong_count_rejected():
    cases = instances()
    cases[0]['seed'] = cases[1]['seed']
    with pytest.raises(ValueError):
        aggregate(cases, 'frozen')
    with pytest.raises(ValueError):
        aggregate(instances()[:4], 'frozen')


def test_unregistered_gold_value_rejected(task):
    def change(g):
        sheet = next(iter(g['sheets'].values()))
        next(iter(sheet['rows'].values()))['Unauthorized_Gold'] = 'Yes'
    mutate(task, 'tests/gold.json', change)
    assert not invertibility(task)['pass']


def test_missing_deliverable_column_rejected(task):
    mutate(task, 'tests/dependency_graph.json',
           lambda g: g.update(nodes=[n for n in g['nodes'] if n['type'] != 'deliverable_column']))
    assert not invertibility(task)['pass']


def test_missing_dependency_trace_rejected(task):
    mutate(task, 'tests/dependency_graph.json', lambda g: g.update(edges=[]))
    assert not invertibility(task)['pass']


def test_rewritten_original_source_rejected(task):
    graph = json.loads((task / 'tests/dependency_graph.json').read_text())
    label = next(n for n in graph['nodes'] if n['type'] == 'original_label')
    source = task / 'environment/public' / label['carrier']
    source.write_text(source.read_text() + '\nRewritten source text')
    assert not invertibility(task)['pass']


def test_malformed_gate_inputs_fail_closed(task):
    mutate(task, 'tests/gold.json', lambda g: g.update(cells=[{}]))
    assert not invertibility(task)['pass']
    mutate(task, 'tests/roundtrip_report.json', lambda g: g.update(documents=[{}]))
    assert not roundtrip_gate(task)['pass']


def test_unscored_input_fact_needs_no_cell(task):
    report = json.loads((task / 'tests/roundtrip_report.json').read_text())
    carrier = next(d for d in report['documents'] if d['status'] == 'verified')
    key = next(iter(carrier['facts']))
    graph_path = task / 'tests/dependency_graph.json'
    graph = json.loads(graph_path.read_text())
    graph['nodes'].append({'id': 'input-fact', 'type': 'construction_fact', 'provision': 'in-document',
                           'scoring': 'input', 'carrier': carrier['path'], 'fact_key': key,
                           'value': carrier['facts'][key], 'work_item': 'scope'})
    graph_path.write_text(json.dumps(graph))
    assert roundtrip_gate(task)['pass']


def test_roundtrip_boolean_integer_substitution_rejected(task):
    report_path = task / 'tests/roundtrip_report.json'
    report = json.loads(report_path.read_text())
    document = report['documents'][0]
    document['facts']['typed_fact'] = True
    document['attempts'][-1]['extracted']['typed_fact'] = 1
    report_path.write_text(json.dumps(report))
    assert not roundtrip_gate(task)['pass']


@pytest.mark.parametrize('alias', ['lower', 'path', 'extension', 'spaces'])
def test_normalized_label_key_collision_rejected(task, alias):
    gold_path = task / 'tests/gold.json'
    graph_path = task / 'tests/dependency_graph.json'
    gold = json.loads(gold_path.read_text())
    graph = json.loads(graph_path.read_text())
    first, second = next((a, b) for i, a in enumerate(gold['cells']) for b in gold['cells'][i + 1:]
                         if a['question'] == b['question'])
    old_key = second['key']
    name = first['key'][0]
    visible = {'lower': name.lower(), 'path': 'agreements/' + name,
               'extension': name + '.TXT', 'spaces': ' ' + name + ' '}[alias]
    second['key'] = [visible, first['question']]
    row = gold['sheets'][second['sheet']]['rows'].pop('|'.join(old_key))
    row['Record_ID'] = visible
    gold['sheets'][second['sheet']]['rows']['|'.join(second['key'])] = row
    next(n for n in graph['nodes'] if n['id'] == second['id'])['cell']['key'] = second['key']
    gold_path.write_text(json.dumps(gold))
    graph_path.write_text(json.dumps(graph))
    result = invertibility(task)
    assert not result['pass']
    assert any('bijection' in failure for failure in result['failures'])
    assert any('adapter record/question' in failure for failure in result['failures'])


def test_normalized_fact_key_collision_rejected(task):
    gold_path = task / 'tests/gold.json'
    graph_path = task / 'tests/dependency_graph.json'
    gold = json.loads(gold_path.read_text())
    graph = json.loads(graph_path.read_text())
    first, second = gold['facts'][:2]
    old_key = second['key']
    second['key'] = [first['key'][0].lower()]
    row = gold['sheets'][second['sheet']]['rows'].pop('|'.join(old_key))
    row['Record_ID'] = second['key'][0]
    gold['sheets'][second['sheet']]['rows']['|'.join(second['key'])] = row
    next(n for n in graph['nodes'] if n['id'] == second['id'])['cell']['key'] = second['key']
    gold_path.write_text(json.dumps(gold))
    graph_path.write_text(json.dumps(graph))
    result = invertibility(task)
    assert not result['pass']
    assert any('bijection' in failure for failure in result['failures'])
