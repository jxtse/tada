"""Section 0A gate authority, output contract and total-reward regressions."""
import json
from pathlib import Path

import pytest
from openpyxl import Workbook

from tada_taskgen import gates, hard_gates


def contract_task(tmp_path):
    task = tmp_path/'task'
    (task/'tests').mkdir(parents=True)
    (task/'environment/public').mkdir(parents=True)
    gold = {'sheets': {'Review': {'columns': ['Record_ID', 'Answer'], 'key_columns': ['Record_ID']}},
            'cells': [{'encoding': [['Original-A', 'A'], ['Original-B', 'B']]}]}
    (task/'tests/gold.json').write_text(json.dumps(gold))
    (task/'tests/validation_contract.json').write_text(json.dumps({'sheets': {
        'Review': {'categorical_values': {'Answer': ['A', 'B']}}}}))
    (task/'instruction.md').write_text('Choose A or B. Save the Review sheet.')
    wb = Workbook(); wb.active.title = 'Review'; wb.active.append(['Record_ID', 'Answer'])
    wb.save(task/'environment/public/review_template.xlsx')
    return task


def test_output_contract_accepts_instruction_options_and_template_headers(tmp_path):
    assert hard_gates.output_contract(contract_task(tmp_path))['pass']


@pytest.mark.parametrize('missing', ['sheet', 'column', 'option'])
def test_output_contract_detects_undeclared_scorer_input(tmp_path, missing):
    task = contract_task(tmp_path)
    path = task/'tests/gold.json'; gold = json.loads(path.read_text())
    if missing == 'sheet':
        gold['sheets']['Undeclared_Table'] = gold['sheets'].pop('Review')
    elif missing == 'column':
        gold['sheets']['Review']['columns'].append('Undeclared_Column')
    else:
        gold['cells'][0]['encoding'].append(['Original-C', 'Undeclared_Option'])
    path.write_text(json.dumps(gold))
    result = hard_gates.output_contract(task)
    assert not result['pass']
    assert any(row['kind'] == missing for row in result['missing'])


def test_codebook_alone_does_not_satisfy_output_contract(tmp_path):
    task = contract_task(tmp_path)
    (task/'instruction.md').write_text('Complete the workbook.')
    (task/'environment/public/coding_conventions.json').write_text('{"options": ["A", "B"]}')
    assert not hard_gates.output_contract(task)['pass']


def test_canary_requires_every_public_file_and_metadata(tmp_path, monkeypatch):
    task = contract_task(tmp_path)
    config = Path(hard_gates.__file__).parents[1]/'config.toml'
    import tomllib
    guid = tomllib.loads(config.read_text())['canary']['guid']
    (task/'task.toml').write_text(f'[metadata]\ncanary = "{guid}"\n')
    result = hard_gates.canary_check(task)
    assert not result['pass'] and 'instruction.md' in result['missing']
    (task/'instruction.md').write_text(guid)
    from openpyxl import load_workbook
    path = task/'environment/public/review_template.xlsx'
    wb = load_workbook(path); wb.properties.description = guid; wb.save(path); wb.close()
    assert hard_gates.canary_check(task)['pass']
    (task/'environment/public/extra.txt').write_text('Missing marker')
    assert not hard_gates.canary_check(task)['pass']


def test_six_groups_alone_control_verdict(tmp_path, monkeypatch):
    checks = {'reference': {'reward': 1.0}, 'identical': True, 'offline': True,
              'trivial': {'pass': True, 'metric': 'total_reward', 'results': {}},
              'include_all': {'status': 'reported_only', 'used_for_admission': False,
                              'reward': .95, 'scope_f1': 6/7}}
    monkeypatch.setattr(hard_gates, 'score_checks', lambda *args: checks)
    monkeypatch.setattr(gates, 'invertibility', lambda *args, **kwargs: {'pass': True})
    monkeypatch.setattr(hard_gates, 'output_contract', lambda *args: {'pass': True})
    monkeypatch.setattr(hard_gates, 'canary_check', lambda *args: {'pass': True})
    monkeypatch.setattr(hard_gates, 'visible_gold_check', lambda *args: {'pass': True})
    report = {'verdict': {'one_label_error_bounded': False, 'sensitivity': False, 'wrong_answers': False},
              'steps': {'keyword_baseline': {'reward': .2}, 'leak': {'pass': True},
                  'leak_permutation': {'pass': True}, 'extra_rows_nonperfect': {'pass': True},
                  'extra_rows_tail_gap': {'runs': [{'adjacent': .8, 'after_gap': .8}]}}}
    result = gates.evaluate_task(tmp_path/'task', report, out=tmp_path/'checks')
    assert result['pass']
    assert set(result['hard_gates']) == set(hard_gates.NAMES)
    assert set(result['verdict']) == set(hard_gates.NAMES) | {'reported_only'}
    assert result['verdict']['reported_only']['sensitivity'] is False
    assert result['verdict']['reported_only']['include_all']['reward'] == .95
    assert result['reported_only']['measurements']['include_all_reported_only']['scope_f1'] == 6/7
    classified = gates.classify(result, {'work_items': [], 'thresholds': {'include_all': .9, 'ignore_revisions': .9}})
    assert classified['all_pass'] and not classified['numeric_pass']
    checks['trivial'] = {'pass': False, 'metric': 'total_reward', 'results': {
        'constant': {'reward': .95, 'limit': .3, 'pass': False}}}
    result = gates.evaluate_task(tmp_path/'task', report, out=tmp_path/'checks')
    assert not result['pass']
    assert not result['hard_gates']['trivial_answers']['pass']


def test_multiseed_report_is_explicitly_nonbinding():
    rows = [{'seed': seed, 'engineering_pass': True, 'numeric_pass': True, 'all_pass': True,
             'metrics': {'keyword': {'value': .1, 'pass': True, 'within_worst_cap': True}}}
            for seed in gates.POLICY['seeds']]
    result = gates.aggregate(rows, 'digest')
    assert result['status'] == 'reported_only' and result['used_for_admission'] is False


@pytest.mark.parametrize('unavailable', [False, True])
def test_screening_measurements_are_distinct_from_structure(tmp_path, monkeypatch, unavailable):
    from tada_taskgen import adapters, screening, schema
    # The report is a local fixture; public exports intentionally ship no cache.
    monkeypatch.setattr(screening, 'DEFAULT_OUTPUT', tmp_path / 'screening')
    screening.DEFAULT_OUTPUT.mkdir()
    (screening.DEFAULT_OUTPUT / 'MAUD.json').write_text('{}')
    task = tmp_path/'task'; (task/'tests').mkdir(parents=True)
    spec = {'source': 'MAUD', 'questions': ['q1', 'q2'], 'work_items': []}
    (task/'tests/scenario_spec.json').write_text(json.dumps(spec))
    monkeypatch.setattr(adapters, 'get_adapter', lambda source: object())
    def load(adapter):
        if unavailable:
            raise screening.ScreeningError('Stale screening report')
        return {'version': {'digest': 'frozen'}, 'policy': {'minimum': 50}, 'questions': {
            'q1': {'eligible': True, 'reasons': [], 'kw_bal_q': .1},
            'q2': {'eligible': False, 'reasons': ['historical shortcut threshold'], 'kw_bal_q': .8}}}
    monkeypatch.setattr(screening, 'load_screening', load)
    monkeypatch.setattr(schema, 'validate_spec', lambda spec: spec)
    for name in ('invertibility', 'roundtrip_gate', 'sparse_instance_gate'):
        monkeypatch.setattr(gates, name, lambda task: {'pass': True})
    monkeypatch.setattr(gates, 'evaluate_task', lambda task, report: report)
    report = gates.extensions(task, {'steps': {}, 'verdict': {}})
    assert report['steps']['spec_structure']['pass']
    measured = report['steps']['question_screening']
    assert measured['status'] == 'reported_only' and measured['used_for_admission'] is False
    assert measured['available'] is not unavailable
    if unavailable:
        assert measured['pass'] is None and 'Stale' in measured['error']
    else:
        assert measured['pass'] is False
        assert measured['eligible_questions'] == ['q1']
        assert measured['questions']['q2']['kw_bal_q'] == .8
        assert measured['version']['digest'] == 'frozen'


def test_actual_trivial_checks_use_total_and_repeat_details(tmp_path):
    from screened_support import integration_adapters, screened_spec
    from tada_taskgen.compiler import compile_task
    with integration_adapters():
        spec, screening = screened_spec('CUAD')
        task = compile_task(spec, 7, tmp_path/'task', screening_report=screening)
        result = hard_gates.score_checks(task, tmp_path/'checks')
    assert result['reference']['reward'] == 1.0
    assert result['identical'] and result['offline']
    trivial = result['trivial']
    assert trivial['metric'] == 'total_reward'
    assert set(trivial['results']) == {'empty', 'all_majority', 'constant'}
    assert all(row['limit'] == .3 for row in trivial['results'].values())
    assert result['include_all']['reward'] > .3
    assert result['include_all']['scope_f1'] == pytest.approx(6/7)
    assert result['include_all']['used_for_admission'] is False
    assert trivial['pass']
    assert hard_gates.visible_gold_check(task)['pass']
    import shutil
    shutil.copyfile(task/'solution/reference.xlsx', task/'environment/public/innocent.xlsx')
    assert not hard_gates.visible_gold_check(task)['pass']
    (task/'environment/public/innocent.xlsx').unlink()
    shutil.copyfile(task/'tests/gold.json', task/'environment/public/innocent.json')
    assert not hard_gates.visible_gold_check(task)['pass']
    (task/'tests/test.sh').write_text('#!/bin/sh\npython untrusted_judge.py\n')
    with integration_adapters():
        assert not hard_gates.score_checks(task, tmp_path/'untrusted')['offline']
