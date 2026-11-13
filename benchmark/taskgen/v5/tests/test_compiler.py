import copy
import hashlib
import json
from pathlib import Path

import pytest
from tada_taskgen.compiler import compile_task, compute_gold, resolve
from tada_taskgen.baselines import sensitivity, keyword_baseline, wrong_answers
from tada_taskgen.schema import validate_graph
from tada_taskgen.scoring import score
from screened_support import screened_spec, integration_adapters

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module', autouse=True)
def synthetic_pools():
    with integration_adapters():
        yield


@pytest.fixture(scope='module', params=['MAUD', 'CUAD'])
def compiled(request, tmp_path_factory):
    spec, report = screened_spec(request.param)
    target = tmp_path_factory.mktemp(request.param) / 'task'
    compile_task(spec, 7, target, screening_report=report)
    return target, spec


def test_reference_and_private_structure(compiled):
    task, spec = compiled
    gold = json.loads((task / 'tests/gold.json').read_text())
    assert score(task / 'solution/reference.xlsx', gold)[0] == pytest.approx(1)
    assert len(gold['cells']) == len(set((c['record'], c['question']) for c in gold['cells']))
    for check in gold['consistency']:
        assert check['column'] not in gold['sheets'][check['sheet']]['rows']['|'.join(check['key'])]
    graph = json.loads((task / 'tests/dependency_graph.json').read_text())
    validate_graph(graph, spec['work_items'], resolved=True)
    assert sensitivity(task)['pass']


def test_seed_and_rebuild_determinism(compiled, tmp_path):
    task, spec = compiled
    from tada_taskgen.adapters import get_adapter
    from tada_taskgen.screening import screen_adapter
    other = compile_task(spec, 7, tmp_path / 'task', screening_report=screen_adapter(get_adapter(spec['source'])))
    digest = lambda root: {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in root.rglob('*') if p.is_file()}
    assert digest(task / 'environment/public') == digest(other / 'environment/public')
    first = json.loads((task / 'tests/scenario_params.json').read_text())
    assert resolve(spec, 8)['records'] != first['records']


def test_baseline_and_scoring_source_identical(compiled, tmp_path):
    task, spec = compiled
    gold = json.loads((task / 'tests/gold.json').read_text())
    output = tmp_path / 'keyword.xlsx'
    keyword_baseline(task / 'environment/public', output)
    reward, details = score(output, gold)
    assert 0 <= reward <= 1
    assert details['per_work_item']
    assert (task / 'tests/scoring.py').read_bytes() == (REPO / 'v5/tada_taskgen/scoring.py').read_bytes()
    expected = wrong_answers(task, tmp_path / 'wrong')
    assert 'ignore_revisions' in expected
    assert score(tmp_path / 'wrong/ignore_revisions.xlsx', gold)[0] < 1


def test_unsupported_graph_fails_closed(tmp_path):
    spec, report = screened_spec('CUAD')
    spec['dependency_graph']['edges'] = []
    with pytest.raises(ValueError, match='edges must include'):
        compile_task(spec, 7, tmp_path, screening_report=report)


@pytest.mark.parametrize('count', [2, 4])
def test_two_and_four_work_item_graphs(count, tmp_path):
    spec, report = screened_spec('CUAD')
    if count == 2:
        spec['work_items'].pop()
        spec['dependency_graph']['nodes'][2]['work_item'] = 'coding'
    else:
        spec['work_items'].append({'id': 'verify', 'order': 4, 'depends_on': ['summary'], 'title': 'Confirm transaction cutoff'})
        spec['documents'].append({'id': 'cutoff', 'path': 'correspondence/cutoff.txt', 'kind': 'email',
                                  'facts': {'cutoff': '2024-09-30'}})
        spec['dependency_graph']['nodes'].append({'id': 'cutoff', 'type': 'construction_fact', 'template': 'document_fact',
              'provision': 'in-document', 'carrier': 'cutoff', 'scoring': 'prerequisite', 'work_item': 'verify', 'fact_key': 'cutoff'})
        spec['dependency_graph']['edges'].append({'from': 'summary_counts', 'to': 'cutoff'})
    task = compile_task(spec, 7, tmp_path / 'task', screening_report=report)
    gold = json.loads((task / 'tests/gold.json').read_text())
    assert score(task / 'solution/reference.xlsx', gold)[0] == pytest.approx(1)
    graph = json.loads((task / 'tests/dependency_graph.json').read_text())
    validate_graph(graph, spec['work_items'], resolved=True)
    if count == 4:
        assert any(f['id'] == 'cutoff' for f in gold['facts'])
        assert any(e['to'] == 'cutoff' for e in graph['edges'])
    keyword_baseline(task / 'environment/public', tmp_path / 'keyword.xlsx')
