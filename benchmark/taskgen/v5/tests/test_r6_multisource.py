"""Source identity, optional scopes and cross-source carriers use the fixed gates."""
import copy
import json

import pytest

from tada_taskgen import adapters
from tada_taskgen.adapters.base import SourceAdapter
from tada_taskgen.compiler import compile_task
from tada_taskgen.gates import invertibility, roundtrip_gate, sparse_instance_gate, extensions
from tada_taskgen.hard_gates import NAMES
from tada_taskgen.multisource import example_spec, label_permutation, adapter_for
from tada_taskgen.schema import validate_compilable, validate_spec
from tada_taskgen.scoring import score


class CollisionSource(SourceAdapter):
    """Two datasets intentionally share all original record and question IDs."""
    def __init__(self, source, permutation_seed=None):
        super().__init__(permutation_seed=permutation_seed)
        self.source = source

    def reference(self):
        return type(self)(self.source, '')

    def questions(self):
        return {'shared': {'question': 'Apply the published coding convention.',
                           'options': [-1, 0, 1] if self.source == 'B' else ['Entailment', 'Contradiction', 'NotMentioned'],
                           'coding_conventions': 'Use the original categorical answer.'}}

    def records(self):
        return {f'record-{i}': {'text': 'The complete source record describes the underlying observations.'}
                for i in range(12)}

    def label_pool(self):
        options = self.questions()['shared']['options']
        offset = bool(self.permutation_seed)
        return {(record, 'shared'): (options[(i + offset) % 3], [])
                for i, record in enumerate(self.records())}

    def encode(self, question, answer):
        if not any(type(answer) is type(value) and answer == value for value in self.questions()[question]['options']):
            raise ValueError('invalid typed answer')
        return str(answer)

    def decode(self, question, answer):
        return {self.encode(question, value): value for value in self.questions()[question]['options']}[answer]


@pytest.fixture
def source_spec(monkeypatch):
    monkeypatch.setattr(adapters, 'get_adapter', lambda source, **kw: CollisionSource(source, **kw))
    return example_spec(['B', 'NDA'])


def read(task, name):
    return json.loads((task / 'tests' / (name + '.json')).read_text())


def test_two_source_compile_through_all_six_real_gates(source_spec, tmp_path):
    task = compile_task(source_spec, 7, tmp_path / 'task')
    gold = read(task, 'gold')
    identities = {(c['source'], c['origin']['record'], c['origin']['question']) for c in gold['cells']}
    assert len(identities) == len(gold['cells']) == 24
    assert {c['question'] for c in gold['cells']} == {'B::shared', 'NDA::shared'}
    assert sparse_instance_gate(task)['pass']
    result = extensions(task, {'steps': {}, 'verdict': {}})
    assert set(result['hard_gates']) == set(NAMES)
    assert result['pass'], {k: v for k, v in result['hard_gates'].items() if not v['pass']}
    assert result['hard_gates']['leakage']['label_permutation']['gold_changed_under_permutation']
    assert result['steps']['question_screening']['pass'] is None


@pytest.mark.parametrize('scoped', [[], ['NDA']])
def test_optional_source_scope_preserves_original_label_scores(source_spec, tmp_path, scoped):
    graph = source_spec['dependency_graph']
    removed = {n['id'] for n in graph['nodes'] if n['template'] == 'record_scope' and n['source'] not in scoped}
    graph['nodes'] = [n for n in graph['nodes'] if n['id'] not in removed]
    graph['edges'] = [e for e in graph['edges'] if e['from'] not in removed and e['to'] not in removed]
    validate_spec(source_spec); validate_compilable(source_spec)
    task = compile_task(source_spec, 7, tmp_path / 'task')
    gold = read(task, 'gold')
    scope_facts = [f for f in gold['facts'] if f.get('counted_in') == 'scope']
    assert {f['source'] for f in scope_facts} == set(scoped)
    assert bool(gold['scopes']) == bool(scoped)
    assert len(gold['cells']) == 24
    assert score(task / 'solution/reference.xlsx', gold)[0] == 1
    assert invertibility(task, slim=True)['pass']


def test_cross_source_fact_requires_blind_verified_carrier(source_spec, tmp_path):
    source_spec['documents'] = [{'id': 'link', 'kind': 'notice', 'path': 'correspondence/link.txt',
                                 'facts': {'review_group': 'Combined review'}}]
    source_spec['dependency_graph']['nodes'].append({'id': 'link', 'template': 'document_fact',
        'type': 'construction_fact', 'provision': 'in-document', 'scoring': 'prerequisite',
        'carrier': 'link', 'fact_key': 'review_group', 'work_item': 'scope'})
    source_spec['dependency_graph']['edges'] += [
        {'from': 'link', 'to': source + '_record_question'} for source in ('B', 'NDA')]
    task = compile_task(source_spec, 7, tmp_path / 'task')
    graph = read(task, 'dependency_graph'); gold = read(task, 'gold')
    linked = {edge['to'] for edge in graph['edges'] if edge['from'] == 'link'}
    assert linked == {c['id'] for c in gold['cells']}
    assert roundtrip_gate(task)['pass']
    (task / 'environment/public/correspondence/link.txt').write_text('Unverified substituted linking fact.')
    assert not roundtrip_gate(task)['pass']


@pytest.mark.parametrize('tamper', ['source', 'bool_answer', 'bool_encoding'])
def test_original_triple_and_numeric_type_fail_closed(source_spec, tmp_path, tamper):
    task = compile_task(source_spec, 7, tmp_path / 'task')
    gold = read(task, 'gold')
    cell = next(c for c in gold['cells'] if c['source'] == 'B' and c['answer'] == 0)
    if tamper == 'source':
        cell['origin']['source'] = 'NDA'
    elif tamper == 'bool_answer':
        cell['answer'] = False
    else:
        next(pair for pair in cell['encoding'] if pair[0] == 0)[0] = False
    (task / 'tests/gold.json').write_text(json.dumps(gold))
    assert not invertibility(task, slim=True)['pass']
    reward, details = score(task / 'solution/reference.xlsx', gold)
    assert reward == 0
    assert details['error']


def test_permutation_context_is_local_to_each_worker(source_spec):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(2)
    def shuffled():
        with label_permutation(991):
            barrier.wait()
            return adapter_for(source_spec).label_pool()[('B::record-0', 'B::shared')][0]
    def original():
        barrier.wait()
        return adapter_for(source_spec).label_pool()[('B::record-0', 'B::shared')][0]
    with ThreadPoolExecutor(2) as pool:
        a, b = pool.submit(shuffled), pool.submit(original)
        assert (a.result(), b.result()) == (0, -1)


def test_source_qualified_label_flip_controller_check(source_spec, tmp_path, monkeypatch):
    from tada_taskgen import compiler
    from tada_taskgen.label_cost import label_flip
    task = compile_task(source_spec, 7, tmp_path / 'task')
    monkeypatch.setattr(compiler, 'adapter_for', lambda *args: pytest.fail('label flip rebuilt its source adapter'))
    report = label_flip(task, tmp_path / 'flips')
    assert report['self_score'] == 1
    assert report['labels_used'] == 24
    assert report['flips'] == 48
    assert report['errors'] == []
    assert report['pass']


@pytest.mark.parametrize('change', ['missing_source', 'duplicate_source', 'missing_template', 'cross_pair'])
def test_invalid_source_contract_rejected(source_spec, change):
    spec = copy.deepcopy(source_spec)
    if change == 'missing_source':
        spec['questions'].pop('B')
    elif change == 'duplicate_source':
        spec['sources'] = ['B', 'B']
    elif change == 'missing_template':
        spec['dependency_graph']['nodes'] = [n for n in spec['dependency_graph']['nodes'] if n['id'] != 'B_record_question']
    else:
        spec['dependency_graph']['nodes'][0]['source'] = 'MAUD'
    with pytest.raises(ValueError):
        validate_spec(spec); validate_compilable(spec)
