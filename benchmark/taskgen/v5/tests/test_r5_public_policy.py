"""Public policy describes the actual partial-credit and population semantics."""
import json

import pytest
from openpyxl import load_workbook

from tada_taskgen.compiler import WEIGHTS, compile_task, public_scoring_policy
from tada_taskgen.scoring import score
from screened_support import integration_adapters, screened_spec


@pytest.fixture(scope='module', params=['MAUD', 'CUAD'])
def compiled(request, tmp_path_factory):
    with integration_adapters():
        spec, screening = screened_spec(request.param)
        spec['work_items'].append({'id': 'verify', 'order': 4,
                                  'depends_on': ['summary'], 'title': 'Confirm the cutoff'})
        spec['documents'].append({'id': 'cutoff', 'path': 'correspondence/cutoff.txt', 'kind': 'email',
                                  'facts': {'cutoff': 'PRIVATE_CONSTRUCTION_VALUE'}})
        spec['dependency_graph']['nodes'].append({
            'id': 'cutoff', 'type': 'construction_fact', 'template': 'document_fact',
            'provision': 'in-document', 'carrier': 'cutoff', 'scoring': 'prerequisite',
            'work_item': 'verify', 'fact_key': 'cutoff'})
        spec['dependency_graph']['edges'].append({'from': 'summary_counts', 'to': 'cutoff'})
        task = compile_task(spec, 7, tmp_path_factory.mktemp('public-policy-' + request.param) / 'task',
                            screening_report=screening)
    gold = json.loads((task / 'tests/gold.json').read_text())
    public = json.loads((task / 'environment/public/coding_conventions.json').read_text())
    return task, spec, gold, public


def test_policy_inputs_match_every_scored_component_without_private_values(compiled):
    task, spec, gold, public = compiled
    policy = public['scoring_policy']
    assert policy == public_scoring_policy(spec)
    assert policy['weights'] == gold['weights'] == WEIGHTS
    assert set(policy['components']) == {'labels', 'scope', 'facts', 'consistency'}
    actual = {'labels': gold['cells'], 'scope': gold['scopes'],
              'facts': [f for f in gold['facts'] if f.get('counted_in') != 'scope'],
              'consistency': gold['consistency']}
    for component, fields in actual.items():
        disclosed = policy['components'][component]
        assert disclosed['weight'] == gold['weights'][component]
        assert {(item['sheet'], item['column']) for item in disclosed['inputs']} == {
            (item['sheet'], item['column']) for item in fields}
        assert disclosed['rule']
    assert policy['extra_rows'] and policy['missing_answers'] and policy['summary_population']
    serialized = json.dumps(policy)
    assert 'PRIVATE_CONSTRUCTION_VALUE' not in serialized
    assert all(cell['id'] not in serialized and cell['record'] not in serialized for cell in gold['cells'])
    memo = (task / 'environment/public/review_memo.md').read_text()
    assert 'regardless' in memo.lower() and 'scope' in memo.lower()
    if spec['source'] == 'MAUD':
        for row in public['questions'].values():
            assert row['provision'] and row['question'].startswith(row['provision'] + ': ')


def test_scope_mistakes_do_not_remove_label_or_summary_credit(compiled, tmp_path):
    task, _, gold, _ = compiled
    reference_reward, reference = score(task / 'solution/reference.xlsx', gold)
    assert reference_reward == 1
    scope = gold['scopes'][0]
    wb = load_workbook(task / 'solution/reference.xlsx')
    ws = wb[scope['sheet']]
    column = gold['sheets'][scope['sheet']]['columns'].index(scope['column']) + 1
    for row in range(2, ws.max_row + 1):
        ws.cell(row, column).value = 'No' if ws.cell(row, column).value == 'Yes' else 'Yes'
    output = tmp_path / 'wrong-scope.xlsx'
    wb.save(output); wb.close()
    reward, details = score(output, gold)
    assert reward < reference_reward
    assert details['components']['labels'] == reference['components']['labels'] == 1
    assert details['components']['consistency'] == reference['components']['consistency'] == 1
    assert details['components']['scope'] < reference['components']['scope']
    # Scope status is scored once (scope F1), never again as a fact.
    assert details['components']['facts'] == reference['components']['facts']


def test_summary_counts_excluded_records_too(compiled, tmp_path):
    task, _, gold, _ = compiled
    in_scope = {key[0] for key in gold['scopes'][0]['expected_keys']}
    by_id = {cell['id']: cell for cell in gold['cells']}
    candidate = None
    for check in gold['consistency']:
        dependencies = [by_id[key] for key in check['dependencies']]
        all_count = sum(cell['answer'] == check['option'] for cell in dependencies)
        scope_count = sum(cell['answer'] == check['option'] and cell['key'][0] in in_scope
                          for cell in dependencies)
        if all_count != scope_count:
            candidate = check, all_count, scope_count
            break
    assert candidate is not None, 'Fixture must contain a counted out-of-scope answer'
    check, all_count, scope_count = candidate
    wb = load_workbook(task / 'solution/reference.xlsx')
    ws = wb[check['sheet']]
    headers = gold['sheets'][check['sheet']]['columns']
    key_col = headers.index('Question') + 1
    value_col = headers.index(check['column']) + 1
    row = next(row for row in range(2, ws.max_row + 1) if ws.cell(row, key_col).value == check['key'][0])
    assert ws.cell(row, value_col).value == all_count
    ws.cell(row, value_col).value = scope_count
    output = tmp_path / 'in-scope-only-summary.xlsx'
    wb.save(output); wb.close()
    _, details = score(output, gold)
    assert details['components']['labels'] == details['components']['scope'] == 1
    assert details['components']['consistency'] < 1


def test_duplicate_coding_row_reduces_precision_and_reward(compiled, tmp_path):
    task, _, gold, public = compiled
    wb = load_workbook(task / 'solution/reference.xlsx')
    sheet = gold['cells'][0]['sheet']
    ws = wb[sheet]
    ws.append([cell.value for cell in ws[2]])
    output = tmp_path / 'duplicate-row.xlsx'
    wb.save(output); wb.close()
    reward, details = score(output, gold)
    assert reward < 1
    assert details['sheets'][sheet]['extra_rows'] == 1
    assert details['sheets'][sheet]['precision'] < 1
    assert public['scoring_policy']['extra_rows']
