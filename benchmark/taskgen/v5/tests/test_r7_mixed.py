"""R7 mixed-item mechanics measured by all six unchanged hard groups."""
import json
from pathlib import Path

import pytest

from tada_taskgen import adapters
from tada_taskgen.adapters.base import SourceAdapter
from tada_taskgen.compiler import compile_task, InstanceCompilationError
from tada_taskgen.multisource import example_spec
from tada_taskgen.gates import extensions, invertibility
from tada_taskgen.scoring import score


class FixtureSource(SourceAdapter):
    def __init__(self, source, permutation_seed=None, asset=None):
        super().__init__(permutation_seed=permutation_seed)
        self.source, self.asset = source, asset
        self.kind = 'record_label' if source in ('B', 'NDA') else 'benchmark_item'

    def reference(self):
        return FixtureSource(self.source, '', self.asset)

    def questions(self):
        return {'answer': {'question': 'Read the assigned inputs and answer the item.',
                           'kind': self.kind, 'options': ['Yes', 'No'] if self.kind == 'record_label' else [],
                           'coding_conventions': ['Use the original categorical code or JSON item answer as declared.']}}

    def records(self):
        return {str(i): {'text': 'The source provides the complete observations for this item.',
                        **({'assets': [{'source_path': str(self.asset), 'path': 'observations.bin'}]} if self.asset else {})}
                for i in range(20)}

    def label_pool(self):
        offset = bool(self.permutation_seed)
        return {(record, 'answer'): (['Yes', 'No'][(int(record) + offset) % 2] if self.kind == 'record_label'
                                     else {'result': (int(record) + offset) % 20, 'variable': 'observed'}, [])
                for record in self.records()}

    def encode(self, question, answer):
        return answer if self.kind == 'record_label' else json.dumps(answer, sort_keys=True)

    def decode(self, question, answer):
        return answer if self.kind == 'record_label' else json.loads(answer)

    def benchmark_checker(self, record, question):
        return {'kind': 'exact'}


def fixture_spec(monkeypatch, sources, asset=None):
    monkeypatch.setattr(adapters, 'get_adapter', lambda source, **kwargs: FixtureSource(source, asset=asset, **kwargs))
    spec = example_spec(sources)
    removed = {n['id'] for n in spec['dependency_graph']['nodes'] if n['template'] == 'record_scope'}
    spec['dependency_graph']['nodes'] = [n for n in spec['dependency_graph']['nodes'] if n['id'] not in removed]
    spec['dependency_graph']['edges'] = [e for e in spec['dependency_graph']['edges'] if e['from'] not in removed]
    return spec


@pytest.mark.parametrize('sources', [['B', 'NDA'], ['oolong-synth'], ['dabstep'], ['NDA', 'dabstep']])
def test_group_a_b_c_and_mixed_through_six_groups(monkeypatch, tmp_path, sources):
    spec = fixture_spec(monkeypatch, sources)
    task = compile_task(spec, 7, tmp_path / 'task')
    result = extensions(task, {'steps': {}, 'verdict': {}})
    assert result['pass'], {name: row for name, row in result['hard_gates'].items() if not row['pass']}
    assert len(result['hard_gates']) == 6
    public = json.loads((task / 'environment/public/coding_conventions.json').read_text())
    for row in public['questions'].values():
        if row['kind'] == 'benchmark_item':
            assert row['encoding'] == []
    gold = json.loads((task / 'tests/gold.json').read_text())
    assert score(task / 'solution/reference.xlsx', gold)[0] == 1


def test_original_item_checker_and_answer_identity_tampering(monkeypatch, tmp_path):
    spec = fixture_spec(monkeypatch, ['dabstep'])
    task = compile_task(spec, 7, tmp_path / 'task')
    gold_path = task / 'tests/gold.json'
    gold = json.loads(gold_path.read_text())
    gold['cells'][0]['checker'] = {'kind': 'invented'}
    gold_path.write_text(json.dumps(gold))
    assert not invertibility(task, slim=True)['pass']
    assert score(task / 'solution/reference.xlsx', gold)[0] < 1


def test_original_asset_bytes_are_bound_to_public_canary_manifest(monkeypatch, tmp_path):
    from tada_taskgen.hard_gates import canary_check
    source = tmp_path / 'input.bin'
    source.write_bytes(b'\x00original database bytes\xff')
    spec = fixture_spec(monkeypatch, ['dabstep'], source)
    task = compile_task(spec, 7, tmp_path / 'task')
    manifest = json.loads((task / 'environment/public/input_assets.json').read_text())
    asset = task / 'environment/public' / manifest['assets'][0]['path']
    assert asset.read_bytes() == source.read_bytes()
    assert canary_check(task)['pass']
    asset.write_bytes(b'tampered')
    assert not canary_check(task)['pass']


def test_unsafe_asset_destination_is_rejected(monkeypatch, tmp_path):
    source = tmp_path / 'input.bin'; source.write_bytes(b'data')
    spec = fixture_spec(monkeypatch, ['dabstep'], source)
    original = FixtureSource.records
    def bad(self):
        result = original(self)
        for row in result.values():
            row['assets'][0]['path'] = '/escape.bin'
        return result
    monkeypatch.setattr(FixtureSource, 'records', bad)
    with pytest.raises(ValueError, match='unsafe'):
        compile_task(spec, 7, tmp_path / 'task')


def test_crlf_original_text_remains_exactly_bound(monkeypatch, tmp_path):
    spec = fixture_spec(monkeypatch, ['dabstep'])
    original = FixtureSource.records
    def windows_text(self):
        result = original(self)
        for row in result.values():
            row['text'] = 'Original first line\r\nOriginal second line\r\n'
        return result
    monkeypatch.setattr(FixtureSource, 'records', windows_text)
    task = compile_task(spec, 7, tmp_path / 'task')
    assert invertibility(task, slim=True)['pass']


def test_nested_archive_copies_only_declared_inner_asset(monkeypatch, tmp_path):
    import io
    import zipfile
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, 'w') as archive:
        archive.writestr('db/input.sqlite', b'original database')
        archive.writestr('secret/answers.txt', b'never copy this gold')
    source = tmp_path / 'outer.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('nested/data.zip', inner.getvalue())
    spec = fixture_spec(monkeypatch, ['dabstep'], source)
    original = FixtureSource.records
    def nested(self):
        result = original(self)
        for row in result.values():
            row['assets'][0].update(nested_member='nested/data.zip', member='db/input.sqlite')
        return result
    monkeypatch.setattr(FixtureSource, 'records', nested)
    task = compile_task(spec, 7, tmp_path / 'task')
    manifest = json.loads((task / 'environment/public/input_assets.json').read_text())
    assert all((task/'environment/public'/a['path']).read_bytes()==b'original database' for a in manifest['assets'])
    assert not list((task/'environment/public').rglob('answers.txt'))


def test_sql_verifier_assets_are_packaged_and_bound(monkeypatch, tmp_path):
    import sqlite3
    from tada_taskgen.hard_gates import score_checks
    source=tmp_path/'db.sqlite'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE items (id INTEGER, result INTEGER)')
        db.executemany('INSERT INTO items VALUES (?,?)', [(i,i*3) for i in range(20)])
    spec=fixture_spec(monkeypatch,['dabstep'],source)
    monkeypatch.setattr(FixtureSource,'label_pool',lambda self: {(r,'answer'):(f'SELECT result FROM items WHERE id = {int(r)}',[]) for r in self.records()})
    monkeypatch.setattr(FixtureSource,'benchmark_checker',lambda *args: {'kind':'sqlite_result_set','database':'observations.bin'})
    task=compile_task(spec,7,tmp_path/'task')
    gold=json.loads((task/'tests/gold.json').read_text());gold['_task_root']=str(task)
    assert score(task/'solution/reference.xlsx',gold)[0]==1
    assert invertibility(task,slim=True)['pass']
    packaged=list((task/'tests/input_assets').rglob('observations.bin'))
    assert len(packaged)==20 and all(p.read_bytes()==source.read_bytes() for p in packaged)
    assert score_checks(task,tmp_path/'score_checks')['pass']
    packaged[0].write_bytes(b'tampered verifier database')
    assert not score_checks(task,tmp_path/'tampered_checks')['pass']


def test_workspace_cap_excludes_oversized_items(monkeypatch, tmp_path):
    from tada_taskgen import compiler
    source=tmp_path/'input.bin';source.write_bytes(b'small fixture for declared large input')
    spec=fixture_spec(monkeypatch,['dabstep'],source)
    monkeypatch.setattr(compiler,'asset_bytes',lambda _:2*1024**3+1)
    with pytest.raises(InstanceCompilationError) as exc:
        compile_task(spec,7,tmp_path/'task')
    assert exc.value.details['oversized_items_dropped']==20


def test_streamed_json_leak_detection_crosses_chunks_and_validates(monkeypatch, tmp_path):
    from tada_taskgen.hard_gates import _json_has_labels
    path=tmp_path/'large.json'
    path.write_text(json.dumps({'input':'x'*200000, 'rows':[{'question':'q','record':'r','answer':'a'}]}))
    assert _json_has_labels(path)
    path.write_text(json.dumps({'input':'x'*200000, 'rows':[{'question':'q'},{'record':'r'},{'answer':'a'}]}))
    assert not _json_has_labels(path)
    path.write_text('{"input":'+'"x"'*20000+'}')
    with pytest.raises(ValueError):_json_has_labels(path)
