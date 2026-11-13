import json
from pathlib import Path
import zipfile

import openpyxl
import pytest
from tada_taskgen.compiler import write_workbook
from tada_taskgen.scoring import score, decoded, BoundedText, INVALID, LIMITS


@pytest.fixture
def example(tmp_path):
    specs = [('Scope', ['Record_ID'], ['Record_ID', 'Included']),
             ('Coding', ['Record_ID', 'Question'], ['Record_ID', 'Question', 'Answer']),
             ('Summary', ['Question'], ['Question', 'Positive_Count'])]
    sheets = {'Scope': [{'Record_ID': 'REC-001', 'Included': 'Yes'}, {'Record_ID': 'REC-002', 'Included': 'No'}],
              'Coding': [{'Record_ID': 'REC-001', 'Question': 'Q1', 'Answer': 'Yes'},
                         {'Record_ID': 'REC-002', 'Question': 'Q1', 'Answer': 'No'}],
              'Summary': [{'Question': 'Q1', 'Positive_Count': 1}]}
    gold = {'sheets': {name: {'columns': cols, 'key_columns': keys,
            'rows': {'|'.join(r[k] for k in keys): dict(r) for r in sheets[name]}} for name, keys, cols in specs},
            'cells': [{'id': f'c{i}', 'sheet': 'Coding', 'key': [r['Record_ID'], 'Q1'], 'column': 'Answer',
                       'question': 'Q1', 'answer': r['Answer'], 'encoding': [['Yes', 'Yes'], ['No', 'No']], 'work_item': 'coding'}
                      for i, r in enumerate(sheets['Coding'])],
            'facts': [{'id': f'f{i}', 'sheet': 'Scope', 'key': [r['Record_ID']], 'column': 'Included',
                       'value': r['Included'], 'work_item': 'scope'} for i, r in enumerate(sheets['Scope'])],
            'consistency': [{'id': 'summary', 'sheet': 'Summary', 'key': ['Q1'], 'column': 'Positive_Count',
                              'operation': 'count_option', 'option': 'Yes', 'dependencies': ['c0', 'c1']}],
            'scopes': [{'work_item': 'coding', 'sheet': 'Scope', 'column': 'Included', 'expected_keys': [['REC-001']]}],
            'label_aggregation': 'per_question_chance_corrected_balanced_accuracy',
            'weights': {'labels': .70, 'facts': .10, 'scope': .15, 'consistency': .05}}
    path = tmp_path / 'reference.xlsx'; write_workbook(sheets, specs, path)
    return path, gold


def mutate(path, tmp_path, fn):
    wb = openpyxl.load_workbook(path); fn(wb)
    out = tmp_path / 'mutated.xlsx'; wb.save(out); return out


def test_reference_missing_empty(example, tmp_path):
    path, gold = example
    assert score(path, gold)[0] == pytest.approx(1)
    assert score(tmp_path / 'missing.xlsx', gold)[0] == 0
    out = mutate(path, tmp_path, lambda wb: [ws.delete_rows(2, ws.max_row) for ws in wb])
    assert score(out, gold)[0] == 0


def test_scope_does_not_cascade(example, tmp_path):
    path, gold = example
    out = mutate(path, tmp_path, lambda wb: setattr(wb['Scope']['B2'], 'value', 'No'))
    reward, details = score(out, gold)
    assert details['components']['labels'] == 1
    assert details['components']['consistency'] == 1
    assert details['scope']['coding']['recall'] == 0


def test_self_consistent_wrong_labels_do_not_earn_derived(example, tmp_path):
    path, gold = example
    def change(wb):
        wb['Coding']['C2'] = 'No'; wb['Summary']['B2'] = 0
    out = mutate(path, tmp_path, change)
    assert score(out, gold)[1]['components']['consistency'] == 0


@pytest.mark.parametrize('row', [['UNKNOWN', 'Q1', 'Yes'], ['', 'Q1', 'No'], ['REC-001', 'Q1', 'Yes']])
def test_extra_rows_never_increase(example, tmp_path, row):
    path, gold = example
    base = mutate(path, tmp_path, lambda wb: setattr(wb['Coding']['C2'], 'value', 'No'))
    before = score(base, gold)[0]
    out = mutate(base, tmp_path, lambda wb: wb['Coding'].append(row))
    assert score(out, gold)[0] <= before


def test_blank_gap_same_penalty(example, tmp_path):
    path, gold = example
    near = mutate(path, tmp_path, lambda wb: wb['Coding'].append(['UNKNOWN', 'Q1', 'Yes']))
    expected = score(near, gold)[0]
    def gap(wb):
        for j, v in enumerate(['UNKNOWN', 'Q1', 'Yes'], 1): wb['Coding'].cell(25001, j).value = v
    far = mutate(path, tmp_path, gap)
    assert score(far, gold)[0] == expected


@pytest.mark.parametrize('value', ['not Yes', '≠Yes', '不是 Yes', 'YES', '= "Yes"', 'x' * 2100])
def test_bad_text_is_local(example, tmp_path, value):
    path, gold = example
    out = mutate(path, tmp_path, lambda wb: setattr(wb['Coding']['C2'], 'value', value))
    reward, details = score(out, gold)
    assert details['components']['labels'] == 0
    assert reward > 0


def test_reader_limits_and_bijection():
    text = BoundedText(4); text.add('123'); text.add('45'); text.add('x' * 1000)
    assert text.value() == INVALID and not text.parts
    assert decoded('yes', [['Yes', 'Yes']]) is None
    assert decoded('Yes', [['Yes', 'Yes']]) == 'Yes'


def test_canonical_sheet_survives_bad_extra_sheet(example, tmp_path):
    path, gold = example
    def extra(wb):
        ws = wb.create_sheet('unrelated'); ws['A1'] = 'x' * 10000
    out = mutate(path, tmp_path, extra)
    assert score(out, gold)[0] == pytest.approx(1)


def test_malformed_archive_fails_closed(tmp_path):
    path = tmp_path / 'bad.xlsx'; path.write_bytes(b'not a ZIP')
    assert score(path, {'sheets': {}})[0] == 0


def rewrite_part(source, target, part, transform):
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, 'w') as dst:
        for name in src.namelist():
            data = src.read(name)
            dst.writestr(name, transform(data) if name == part else data)
    return target


def test_unresolved_shared_string_is_per_cell(example, tmp_path):
    import re
    path, gold = example
    out = rewrite_part(path, tmp_path / 'unresolved.xlsx', 'xl/worksheets/sheet2.xml',
                       lambda data: re.sub(rb'<c r="C2"[^>]*>.*?</c>', b'<c r="C2" t="s"><v>999</v></c>', data))
    reward, details = score(out, gold)
    assert details['components']['labels'] == 0
    assert details['components']['facts'] == 1
    assert details['shared_strings']['per_sheet']['Coding']['unresolved'] == 1
    assert details['shared_strings']['per_sheet']['Scope']['unresolved'] == 0


def test_dimension_does_not_allocate_rows(example, tmp_path):
    import re
    path, gold = example
    out = rewrite_part(path, tmp_path / 'dimension.xlsx', 'xl/worksheets/sheet2.xml',
                       lambda data: re.sub(rb'<dimension ref="[^"]+"', b'<dimension ref="A1:XFD1048576"', data))
    assert score(out, gold)[0] == pytest.approx(1)


def test_xml_entity_is_rejected_per_sheet(example, tmp_path):
    path, gold = example
    out = rewrite_part(path, tmp_path / 'entity.xlsx', 'xl/worksheets/sheet2.xml',
                       lambda data: b'<!DOCTYPE worksheet [<!ENTITY bad "boom">]>' + data)
    reward, details = score(out, gold)
    assert details['components']['labels'] == 0
    assert details['components']['facts'] == 1
    assert details['sheets']['Coding']['status'] == 'abnormal'


def balanced_case(answer_sets, predictions, *, sheet_for=None, item_for=None, extra_rows=None):
    """Small trusted fixtures isolate aggregation from the XLSX reader."""
    from tada_taskgen.scoring import LABEL_AGGREGATION
    cells, schemas, parsed = [], {}, {}
    sheet_for, item_for, extra_rows = sheet_for or {}, item_for or {}, extra_rows or {}
    for question, answers in answer_sets.items():
        for index, answer in enumerate(answers):
            identifier = f'{question}-{index}'
            sheet = sheet_for.get(identifier, 'Coding')
            schemas.setdefault(sheet, {'columns': ['ID', 'Answer'], 'key_columns': ['ID'], 'rows': {}})
            parsed.setdefault(sheet, {'status': 'ok', 'records': [], 'extra': 0})
            schemas[sheet]['rows'][identifier] = {'ID': identifier, 'Answer': answer}
            prediction = predictions[question][index]
            if prediction is not None:
                parsed[sheet]['records'].append({'ID': identifier, 'Answer': prediction})
            cells.append({'id': identifier, 'record': identifier, 'question': question, 'sheet': sheet,
                          'key': [identifier], 'column': 'Answer', 'answer': answer,
                          'encoding': [[v, v] for v in sorted(set(answers))],
                          'work_item': item_for.get(question, 'coding')})
    for sheet, count in extra_rows.items():
        parsed[sheet]['records'].extend({'ID': f'UNKNOWN-{i}', 'Answer': 'A'} for i in range(count))
    return parsed, {'sheets': schemas, 'cells': cells, 'label_aggregation': LABEL_AGGREGATION,
                    'weights': {'labels': .70, 'scope': .15, 'facts': .10, 'consistency': .05}}


def test_balanced_majority_accuracy_does_not_dominate():
    from tada_taskgen.scoring import score_records
    parsed, gold = balanced_case({'Q1': ['A'] * 8 + ['B'] * 2}, {'Q1': ['A'] * 10})
    reward, details = score_records(parsed, gold)
    assert details['components']['labels'] == 0
    assert details['per_work_item']['coding'] == 0
    assert details['per_question']['Q1']['class_counts'] == {'A': 8, 'B': 2}
    assert details['per_question']['Q1']['class_recalls'] == {'A': 1, 'B': 0}
    assert reward == 0


def test_questions_have_equal_weight_despite_different_cell_counts():
    from tada_taskgen.scoring import score_records
    parsed, gold = balanced_case({'Q1': ['A'] * 8 + ['B'] * 2, 'Q2': ['A', 'B']},
                                {'Q1': ['A'] * 10, 'Q2': ['A', 'B']})
    _, details = score_records(parsed, gold)
    assert details['components']['labels'] == .5
    assert details['per_work_item']['coding'] == .5


def test_three_present_classes_and_single_class_instances():
    from tada_taskgen.scoring import score_records
    parsed, gold = balanced_case({'Q1': ['A'] * 8 + ['B'] * 2 + ['C']}, {'Q1': ['A'] * 11})
    assert score_records(parsed, gold)[1]['components']['labels'] == 0
    parsed, gold = balanced_case({'Q1': ['A'] * 4}, {'Q1': ['A', 'A', None, 'A']})
    for cell in gold['cells']:
        cell['encoding'].append(['B', 'B'])
    _, details = score_records(parsed, gold)
    assert details['components']['labels'] == .5
    assert details['per_question']['Q1']['balanced_accuracy'] == .75
    assert details['per_question']['Q1']['present_classes'] == ['A']
    # Absent options affect the chance floor, not the class-recall denominator.
    for cell in gold['cells']:
        cell['encoding'].append(['C', 'C'])
    assert score_records(parsed, gold)[1]['components']['labels'] == pytest.approx(.625)


def test_sheet_precision_applies_to_each_class_contribution():
    from tada_taskgen.scoring import score_records
    parsed, gold = balanced_case({'Q1': ['A', 'A', 'B', 'B']}, {'Q1': ['A', 'A', 'B', 'B']},
                                sheet_for={'Q1-0': 'Other', 'Q1-2': 'Other'}, extra_rows={'Other': 2})
    _, details = score_records(parsed, gold)
    assert details['sheets']['Other']['precision'] == .5
    assert details['per_question']['Q1']['class_recalls'] == {'A': .75, 'B': .75}
    assert details['components']['labels'] == .5
    assert details['per_work_item']['coding'] == .5


def test_work_item_label_scores_use_balanced_question_means():
    from tada_taskgen.scoring import score_records
    parsed, gold = balanced_case({'Q1': ['A'] * 8 + ['B'] * 2, 'Q2': ['A', 'B']},
                                {'Q1': ['A'] * 10, 'Q2': ['A', 'B']},
                                item_for={'Q1': 'first', 'Q2': 'second'})
    _, details = score_records(parsed, gold)
    assert details['per_work_item'] == {'first': 0, 'second': 1}
    assert details['per_work_item_details']['first']['components'] == {'labels': 0}


def test_mixed_work_item_uses_component_weights(example):
    path, gold = example
    from tada_taskgen.scoring import load_book, parse_sheet, score_records
    gold['facts'][0]['work_item'] = 'coding'
    gold['facts'][1]['work_item'] = 'coding'
    book, status, _ = load_book(path, gold['sheets'])
    parsed = {}
    for name, spec in gold['sheets'].items():
        rows, extras, _ = parse_sheet(book[name], spec['columns'], spec['key_columns'])
        parsed[name] = {'records': rows, 'extra': extras, 'status': status[name]}
    parsed['Coding']['records'][0]['Answer'] = 'No'
    _, details = score_records(parsed, gold)
    assert details['per_work_item']['coding'] == pytest.approx(.10 / .80)


def test_repaired_weights_and_consistency_gate(example, tmp_path):
    path, gold = example
    out = mutate(path, tmp_path, lambda wb: setattr(wb['Coding']['C2'], 'value', 'No'))
    reward, details = score(out, gold)
    assert reward == pytest.approx(.10 + .15)
    assert details['components']['consistency'] == 0


def test_unsupported_aggregation_fails_closed(example):
    path, gold = example
    gold['label_aggregation'] = 'majority_correction'
    reward, details = score(path, gold)
    assert reward == 0
    assert 'unsupported label aggregation' in details['error']
