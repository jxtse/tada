"""Dataset combinations are fixed designer inputs, independent of O*NET."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from tada_taskgen.designer_sources import catalogues, validate_assignment

ROOT = Path(__file__).resolve().parents[2]


def designer():
    spec = importlib.util.spec_from_file_location('r6_designer', ROOT / 'run_designer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalogues_do_not_open_records_or_labels(monkeypatch):
    from tada_taskgen.adapters.maud import MAUDAdapter
    from tada_taskgen.adapters.cuad import CUADAdapter
    for cls in (MAUDAdapter, CUADAdapter):
        monkeypatch.setattr(cls, 'records', lambda *a: pytest.fail('designer requested records'))
        monkeypatch.setattr(cls, 'label_pool', lambda *a: pytest.fail('designer requested labels'))
    result = catalogues(['MAUD', 'C2'])
    assert set(result) == {'MAUD', 'CUAD'}
    assert result['MAUD']['questions']['DP068']['provision'] == 'Intervening Event Definition'
    assert all('records' not in row for row in result.values())


def test_exact_combination_and_aliases():
    value = {'sources': ['CUAD', 'MAUD']}
    validate_assignment(value, {'sources': ['MAUD', 'C2'], 'occupation': 'irrelevant'})
    with pytest.raises(ValueError, match='every assigned dataset'):
        validate_assignment({'source': 'MAUD'}, {'sources': ['MAUD', 'CUAD']})
    with pytest.raises(ValueError, match='every assigned dataset'):
        validate_assignment(value, {'sources': ['MAUD']})


def test_prompt_contains_only_assigned_catalogues(tmp_path):
    intent = tmp_path / 'intent.json'
    intent.write_text(json.dumps({'sources': ['MAUD', 'CUAD'], 'combination': 'CUAD+MAUD'}))
    text = designer().assemble('B', True, str(intent))
    inputs = json.loads(text.split('## Inputs\n\n', 1)[1].split('\n\n## Schemas', 1)[0])
    assert inputs['combination'] == 'CUAD+MAUD'
    assert set(inputs['source_catalogues']) == {'CUAD', 'MAUD'}
    assert '{{' not in text.replace('agreements/{record}.txt', '')


def test_two_source_example_compiles_with_fake_clients(tmp_path):
    from tada_taskgen.compiler import compile_task
    from tada_taskgen.designer_check import check
    from tada_taskgen.schema import load_designer_output
    example = copy.deepcopy(designer().two_source_example())
    directory = tmp_path / 'designer'; directory.mkdir()
    (directory / 'scenario_spec.json').write_text(json.dumps(example))
    (directory / 'dependency_graph.json').write_text(json.dumps(example['dependency_graph']))
    (directory / 'instruction.md').write_text('Prepare the two-source review workbook.')
    (directory / 'design.md').write_text('Combine the two assigned dataset contributions.')
    assert check(directory) == {'ok': True, 'error': None}
    task = compile_task(load_designer_output(directory), 7, tmp_path / 'task', client='fake')
    gold = json.loads((task / 'tests/gold.json').read_text())
    assert {cell['source'] for cell in gold['cells']} == {'CUAD', 'MAUD'}
    from controller import evidence_index, family_of, leak_scan
    from weekend_checks import strict_provenance
    (tmp_path / 'receipt.json').write_text(json.dumps({'family': 'CUAD+MAUD'}))
    assert family_of(tmp_path) == 'CUAD+MAUD'
    evidence = evidence_index('CUAD+MAUD', task)
    assert all(f'{cell["record"]}:{cell["question"]}' in evidence for cell in gold['cells'])
    provenance = strict_provenance(task, 'CUAD+MAUD', evidence)
    assert provenance['pass'], provenance
    assert leak_scan(task, 'CUAD+MAUD')['pass']
    gold['cells'][0]['origin']['source'] = 'NDA'
    (task / 'tests/gold.json').write_text(json.dumps(gold))
    assert not strict_provenance(task, 'CUAD+MAUD', evidence)['pass']
