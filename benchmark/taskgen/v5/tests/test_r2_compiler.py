"""Sparse compilation tests use synthetic labels, never tune the real regression."""
import copy
import json
from pathlib import Path

import pytest

from tada_taskgen.adapters.base import SourceAdapter
from tada_taskgen.compiler import (
    InstanceCompilationError, REVIEW_MEMO, compile_task, guard_instance,
    resolve, review_memo,
)
from tada_taskgen.scoring import score


class SparseAdapter(SourceAdapter):
    source = 'MAUD'

    def __init__(self):
        self.pool = {(f'r{i:02}', f'DP{900 + q}'): ('Yes' if i % 2 else 'No', [])
                     for i in range(16) for q in range(6) if q < 5 or i < 8}
        self.pool.update({(f'r{i:02}', 'DP906'): ('Yes', []) for i in range(16)})
        # A record with two scorable questions must never enter the sample.
        self.pool.update({('r16', q): ('Yes', []) for q in ('DP900', 'DP901')})
        self.original = self

    def questions(self):
        return {f'DP{900 + q}': {'question': f'Clause {q}', 'options': ['Yes', 'No', 'Other'],
                                'coding_conventions': 'Use the original answer.'}
                for q in range(7)}

    def records(self):
        return {f'r{i:02}': {'text': f'Agreement {i}. The full source text remains unchanged.'}
                for i in range(17)}

    def label_pool(self):
        return self.pool

    def reference(self):
        return self.original


@pytest.fixture
def sparse(monkeypatch):
    from tada_taskgen import adapters, screening
    adapter = SparseAdapter()
    monkeypatch.setattr(adapters, 'get_adapter', lambda *args, **kwargs: adapter)
    # Screening is tested separately; isolate compiler behavior with a synthetic pool.
    monkeypatch.setattr(screening, 'require_eligible', lambda *args, **kwargs: {})
    spec = json.loads((Path(__file__).resolve().parents[1] / 'specs/maud_r7.json').read_text())
    spec['questions'] = list(adapter.questions())
    spec['sample_size'] = 16
    return spec, adapter


def test_sparse_draw_keeps_available_pairs_and_records_drops(sparse):
    spec, adapter = sparse
    sp = resolve(spec, 7)
    assert len(sp['records']) == 16
    assert 'r16' not in {r['record_id'] for r in sp['records']}
    assert sp['requested_spec'] == spec
    assert spec['questions'][-1] == 'DP906'  # No caller mutation.
    assert sp['spec']['questions'] == list(adapter.questions())[:-1]
    assert sp['sampling']['candidate_count'] == 16
    assert sp['sampling']['dropped_questions'] == {
        'DP906': {'class_counts': {'Yes': 16},
                  'reason': 'requires >= 2 present classes, each with >= 2 cells'}}
    pairs = {(r['record_id'], q) for r in sp['records'] for q in r['questions']}
    expected = {pair for pair in adapter.pool if pair[0] != 'r16' and pair[1] != 'DP906'}
    assert pairs == expected
    assert guard_instance(sp)['pass']
    assert resolve(spec, 7) == sp
    assert resolve(spec, 11)['records'] != sp['records']


def test_guard_drops_a_question_with_one_rare_third_class(sparse):
    spec, adapter = sparse
    adapter.pool[('r00', 'DP905')] = ('Other', [])
    sp = resolve(spec, 7)
    assert len(sp['spec']['questions']) == 5
    assert sp['sampling']['dropped_questions']['DP905']['class_counts'] == {'No': 3, 'Other': 1, 'Yes': 4}


def test_too_few_retained_questions_fails_one_draw_without_retries(sparse, monkeypatch):
    from tada_taskgen import compiler
    spec, adapter = sparse
    for pair in adapter.pool:
        if pair[1] in ('DP904', 'DP905'):
            adapter.pool[pair] = ('Yes', [])
    calls = []
    original = compiler.random.Random.sample

    def recorded(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(compiler.random.Random, 'sample', recorded)
    with pytest.raises(InstanceCompilationError, match='retains 4 questions') as error:
        resolve(spec, 7)
    assert len(calls) == 1
    assert error.value.details['sampling']['retained_questions'] == spec['questions'][:4]
    assert len(error.value.details['sampled_records']) == 16


def test_sparse_population_shortage_fails_before_draw(sparse, monkeypatch):
    from tada_taskgen import compiler
    spec, adapter = sparse
    spec['sample_size'] = 17
    monkeypatch.setattr(compiler.random.Random, 'sample', lambda *args, **kwargs: pytest.fail('must not draw'))
    with pytest.raises(InstanceCompilationError, match='at least 3 questions; only 16 eligible'):
        resolve(spec, 7)


def test_permuted_labels_do_not_change_drop_decision(sparse):
    spec, adapter = sparse
    reference = copy.deepcopy(adapter)
    expected = resolve(spec, 7)
    adapter.original = reference
    adapter.pool = {pair: ('Yes', evidence) for pair, (_answer, evidence) in adapter.pool.items()}
    assert resolve(spec, 7) == expected
    assert guard_instance(expected)['pass']


def test_compile_publishes_exact_checklist_and_retains_original_build_spec(sparse, tmp_path):
    spec, adapter = sparse
    task = compile_task(spec, 7, tmp_path / 'task')
    sp = json.loads((task / 'tests/scenario_params.json').read_text())
    original = json.loads((task / 'tests/scenario_spec.json').read_text())
    public = json.loads((task / 'environment/public/coding_conventions.json').read_text())
    gold = json.loads((task / 'tests/gold.json').read_text())
    assert original == spec == sp['requested_spec']
    assert 'DP906' not in public['questions']
    assert set(public['questions']) == set(sp['spec']['questions'])
    assert public['applicability_memo'] == REVIEW_MEMO
    memo = (task / 'environment/public' / REVIEW_MEMO).read_text()
    from tada_taskgen.canary import text_header
    assert memo == text_header(review_memo(sp))
    assert 'DP906' not in memo
    assert 'class_counts' not in memo
    for record in sp['records']:
        assert f'| {record["visible_id"]} | {", ".join(record["questions"])} |' in memo
    assert {(c['record'], c['question']) for c in gold['cells']} == {
        (r['record_id'], q) for r in sp['records'] for q in r['questions']}
    assert {c['key'][0] for c in gold['consistency']} == set(sp['spec']['questions'])
    assert score(task / 'solution/reference.xlsx', gold)[0] == pytest.approx(1)
    assert 'DP906' in (task / 'build/build.py').read_text()
    guard = json.loads((task / 'tests/instance_guard.json').read_text())
    assert guard['sampling'] == sp['sampling']


def test_failed_sparse_compile_writes_no_partial_package(sparse, tmp_path):
    spec, adapter = sparse
    for pair in adapter.pool:
        adapter.pool[pair] = ('Yes', [])
    destination = tmp_path / 'task'
    with pytest.raises(InstanceCompilationError, match='retains 0 questions'):
        compile_task(spec, 7, destination)
    assert not destination.exists()


def test_permuted_rebuild_keeps_visible_checklist_and_archive_identical(sparse, tmp_path):
    spec, adapter = sparse
    first = compile_task(spec, 7, tmp_path / 'original')
    archive = json.loads((first / 'tests/roundtrip_report.json').read_text())
    reference = copy.deepcopy(adapter)
    adapter.original = reference
    adapter.pool = {pair: ('Yes', evidence) for pair, (_answer, evidence) in adapter.pool.items()}
    second = compile_task(spec, 7, tmp_path / 'permuted', document_archive=archive)
    visible = lambda task: {str(p.relative_to(task / 'environment/public')): p.read_bytes()
                            for p in (task / 'environment/public').rglob('*') if p.is_file()}
    assert visible(first) == visible(second)
    assert (first / 'tests/gold.json').read_bytes() != (second / 'tests/gold.json').read_bytes()
    assert (first / 'tests/instance_guard.json').read_bytes() == (second / 'tests/instance_guard.json').read_bytes()


def test_nonapplicable_pair_keeps_inherited_precision_penalty(sparse, tmp_path):
    from openpyxl import load_workbook
    spec, adapter = sparse
    task = compile_task(spec, 7, tmp_path / 'task')
    gold = json.loads((task / 'tests/gold.json').read_text())
    book = load_workbook(task / 'solution/reference.xlsx')
    coding = spec['sheets']['coding']
    assert ('r08', 'DP905') not in adapter.pool
    book[coding].append([adapter.visible_id('r08'), 'DP905', 'Yes'])
    output = tmp_path / 'extra.xlsx'
    book.save(output)
    reward, detail = score(output, gold)
    cells = len(gold['cells'])
    assert reward < 1
    assert detail['sheets'][coding]['extra_rows'] == 1
    # The public codebook includes Other even though only Yes/No are present.
    assert detail['components']['labels'] == pytest.approx(1.5 * cells / (cells + 1) - .5)
    assert detail['components']['scope'] == 1
    # No document facts in this spec: facts has no items (component dropped, weights renormalised).
    assert detail['components']['facts'] == 0 and not [f for f in gold['facts'] if f.get('counted_in') != 'scope']
