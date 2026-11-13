"""Blind rereading and exclusion caps operate on original source triples."""
import copy
import json
from types import SimpleNamespace

import pytest

from tada_taskgen.common_error import adjudicate_common_errors, quarantine_common_errors
from tada_taskgen.preaudit import preaudit, apply_proposals
from tada_taskgen.scheduler import qualified_exclusions


QUOTE = 'The supplier must obtain prior written consent before any assignment.'


class Adapter:
    def __init__(self, source, quarantine, size=20):
        self.source, self.quarantine, self.size = source, quarantine, size
        self.options = [-1, 0, 1] if source == 'B' else ['No', 'Yes']
        self.original, self.alternative = (0, 1) if source == 'B' else ('No', 'Yes')

    def questions(self):
        return {'q': {'question': 'Is consent required?', 'options': self.options, 'coding_conventions': []}}

    def label_pool(self):
        return {(f'r{i}', 'q'): (self.original if i == 0 else self.alternative, [QUOTE]) for i in range(self.size)}

    def records(self):
        # Common-error reads must not open all other texts merely to find the
        # full-pool identical-evidence grouping.
        return {'r0': {'text': '"Assignment" means transfer of obligations.\n\n' + QUOTE}}

    def keyword_predict(self, *args):
        return self.alternative


def setup(tmp_path, sources, size=20):
    adapters = {source: Adapter(source, tmp_path / 'quarantine.json', size) for source in sources}
    cells = [{'id': 'label:' + source, 'record': source + '::r0', 'question': source + '::q', 'source': source,
              'origin': {'source': source, 'record': 'r0', 'question': 'q'}} for source in sources]
    tests = tmp_path / 'tests'; tests.mkdir()
    gold = {'source': '+'.join(sources), 'sources': sources, 'qualified_labels': True, 'cells': cells}
    (tests / 'gold.json').write_text(json.dumps(gold))
    (tests / 'scenario_spec.json').write_text(json.dumps({'sources': sources}))
    payloads = []
    def client(payload):
        payloads.append(copy.deepcopy(payload))
        return {'answer': 1 if payload['options'] == [-1, 0, 1] else 'Yes',
                'confidence': 'high', 'clause_quote': QUOTE}
    return adapters, cells, gold, payloads, client


def test_overlapping_record_question_ids_are_grouped_by_original_source(tmp_path):
    adapters, cells, _, payloads, client = setup(tmp_path, ['CUAD', 'CUADX'])
    report = adjudicate_common_errors(tmp_path, [cell['id'] for cell in cells], client=client,
                                      adapter=SimpleNamespace(adapters=adapters))
    assert report['sources'] == ['CUAD', 'CUADX'] and report['calls'] == 4
    assert report['confirmed_exclusions'] == ['CUAD:r0:q', 'CUADX:r0:q']
    assert report['confirmed_cell_ids'] == ['label:CUAD', 'label:CUADX']
    assert qualified_exclusions(report)
    assert report['question_pool_sizes'] == {'CUAD:q': 20, 'CUADX:q': 20}
    assert {(row['source'], row['record'], row['question']) for row in report['proposals']} == {
        ('CUAD', 'r0', 'q'), ('CUADX', 'r0', 'q')}
    assert [payload['read_index'] for payload in payloads] == [0, 1, 0, 1]
    assert all(payload['question'] == 'Is consent required?' for payload in payloads)
    assert all(not set(payload) & {'source', 'record', 'original_answer', 'trial', 'readings', 'origin'} for payload in payloads)
    path = tmp_path / 'scratch.json'; path.write_text('{}')
    result = quarantine_common_errors(report, tmp_path / 'notice.json', quarantine=path)
    assert result['status'] == 'applied'
    data = json.loads(path.read_text())
    assert set(data) == {'CUAD_labels', 'CUADX_labels'}
    assert all(rows[0]['contract_id'] == 'r0' and rows[0]['question'] == 'q' for rows in data.values())
    quarantine_common_errors(report, tmp_path / 'notice.json', quarantine=path)
    assert all(len(rows) == 1 for rows in json.loads(path.read_text()).values())


def test_singleton_sources_form_keeps_integer_labels_and_qualified_cells(tmp_path):
    adapters, cells, _, _, client = setup(tmp_path, ['B'])
    report = adjudicate_common_errors(tmp_path, [cells[0]['id']], client=client, adapter=SimpleNamespace(adapters=adapters))
    assert report['sources'] == ['B'] and report['confirmed_exclusions'] == ['B:r0:q']
    assert qualified_exclusions(report)
    assert report['confirmed_cell_ids'] == ['label:B']
    assert type(report['proposals'][0]['original_answer']) is int
    assert report['proposals'][0]['record'] == 'r0'


def test_original_pool_denominators_include_prior_exclusions_per_source(tmp_path):
    adapters, cells, _, _, client = setup(tmp_path, ['CUAD', 'NDA'])
    (tmp_path / 'quarantine.json').write_text(json.dumps({
        'CUAD_labels': [{'contract_id': 'r20', 'question': 'q'}],
        'NDA_labels': [{'doc_id': 'r20', 'hypothesis': 'q'}]}))
    report = adjudicate_common_errors(tmp_path, [cell['id'] for cell in cells], client=client,
                                      adapter=SimpleNamespace(adapters=adapters))
    assert report['question_pool_sizes'] == {'CUAD:q': 21, 'NDA:q': 21}


@pytest.mark.parametrize('mutation', ['wrong_origin', 'unknown_cell'])
def test_common_error_input_is_bound_to_gold_source_provenance(tmp_path, mutation):
    adapters, cells, gold, _, _ = setup(tmp_path, ['CUAD', 'CUADX'])
    if mutation == 'wrong_origin':
        gold['cells'][0]['origin']['record'] = 'different-record'
        (tmp_path / 'tests/gold.json').write_text(json.dumps(gold))
    ids = ['unknown'] if mutation == 'unknown_cell' else [cells[0]['id']]
    with pytest.raises(ValueError):
        adjudicate_common_errors(tmp_path, ids, client=lambda _: pytest.fail('must reject before blind calls'),
                                 adapter=SimpleNamespace(adapters=adapters))


def test_one_source_over_cap_prevents_the_entire_multi_source_apply(tmp_path):
    adapters, cells, _, _, client = setup(tmp_path, ['CUAD', 'CUADX'])
    report = adjudicate_common_errors(tmp_path, [c['id'] for c in cells], client=client, adapter=SimpleNamespace(adapters=adapters))
    report['question_pool_sizes']['CUADX:q'] = 19
    path = tmp_path / 'scratch.json'; path.write_text('{}')
    result = quarantine_common_errors(report, tmp_path / 'notify.json', quarantine=path)
    assert result['status'] == 'stopped_notify_richard'
    assert result['violations'] == [{'source': 'CUADX', 'question': 'q', 'excluded': 1, 'pool_size': 19, 'cap': .05}]
    assert path.read_text() == '{}' and not list(tmp_path.glob('scratch.json.backup-*'))


@pytest.mark.parametrize('source,field', [('NDA', 'doc_id'), ('B', 'prompt_id'), ('PREC', 'cite')])
def test_new_source_quarantine_serializes_original_record_fields(tmp_path, source, field):
    adapter = Adapter(source, tmp_path / 'none')
    report = preaudit(adapter, lambda _: {'answer': adapter.alternative, 'confidence': 'high', 'clause_quote': QUOTE},
                      suspects=[{'record': 'r0', 'question': 'q'}])
    path = tmp_path / 'scratch.json'
    assert apply_proposals(report, path, {source + ':q': 20})['status'] == 'applied'
    row = json.loads(path.read_text())[source + '_labels'][0]
    assert row[field] == 'r0' and row['question'] == 'q'
    if source == 'NDA':
        assert row['hypothesis'] == 'q'


@pytest.mark.parametrize('answer', [True, 1.0])
def test_b_blind_reading_requires_typed_original_options(tmp_path, answer):
    adapter = Adapter('B', tmp_path / 'none')
    report = preaudit(adapter, lambda _: {'answer': answer, 'confidence': 'high', 'clause_quote': QUOTE},
                      suspects=[{'record': 'r0', 'question': 'q'}])
    assert not report['proposals']
    valid = preaudit(adapter, lambda _: {'answer': 1, 'confidence': 'high', 'clause_quote': QUOTE},
                     suspects=[{'record': 'r0', 'question': 'q'}])
    for reading in valid['proposals'][0]['readings']:
        reading['answer'] = answer
    with pytest.raises(ValueError, match='Unqualified'):
        apply_proposals(valid, tmp_path / 'scratch.json', {'B:q': 20})
    assert not (tmp_path / 'scratch.json').exists()


def test_b_historical_prompt_and_paper_exclusions_count_against_same_cap(tmp_path, monkeypatch):
    from tada_taskgen import preaudit as module
    adapter = Adapter('B', tmp_path / 'none')
    report = preaudit(adapter, lambda _: {'answer': 1, 'confidence': 'high', 'clause_quote': QUOTE},
                      suspects=[{'record': 'r0', 'question': 'q'}])
    monkeypatch.setattr(module, 'load_source', lambda *args: SimpleNamespace(_ei=lambda: {'prompts': [
        {'PromptID': 'r2', 'pmc': 'paper-1'}, {'PromptID': 'r3', 'pmc': 'unaffected'}]}))
    path = tmp_path / 'scratch.json'
    path.write_text(json.dumps({'B_prompts': [{'prompt_id': 'r1'}], 'B_papers': [{'pmc': 'paper-1'}]}))
    before = path.read_bytes()
    result = apply_proposals(report, path, {'B:q': 40})
    assert result['status'] == 'stopped_notify_richard'
    assert result['violations'][0]['excluded'] == 3 and path.read_bytes() == before
