"""Known compiler failures produce actionable aggregate feedback, never answers."""
import copy
import json
from pathlib import Path

import pytest

from tada_taskgen import compiler, design_loop, gates
from tada_taskgen.clients import FakeClient
from tada_taskgen.compile_errors import CompilationFailure, from_exception, safe_diagnostic
from screened_support import integration_adapters, screened_spec


def candidate():
    spec = json.loads((Path(__file__).parents[1] / 'specs/maud_screened.json').read_text())
    spec['documents'].append({'id': 'library_sop', 'path': 'correspondence/library_sop.txt',
                             'kind': 'SOP', 'facts': {'version': 'HIDDEN_FACT_VALUE', 'cutoff': '2024-09-30'}})
    return spec


def failure():
    return CompilationFailure('HIDDEN_RAW_EXCEPTION /private/gold.json',
        {'record': 'PRIVATE_RECORD', 'answer': 'HIDDEN_GOLD'}, code='carrier_roundtrip_failed',
        context={'document_id': 'library_sop', 'fact_keys': ['version'],
                 'extracted': {'version': 'HIDDEN_EXTRACTED'}, 'answer': 'HIDDEN_GOLD'})


def assert_no_private(message):
    for value in ('HIDDEN_', 'PRIVATE_RECORD', '/private', 'class_counts', 'extracted', 'original_answer'):
        assert value not in message


def test_carrier_diagnostic_only_names_public_identifiers():
    spec = candidate()
    diagnostic = from_exception(failure(), spec)
    assert diagnostic['document_id'] == 'library_sop'
    assert diagnostic['fact_keys'] == ['version']
    assert_no_private(json.dumps(diagnostic))
    message = design_loop.feedback_message({'compile_diagnostic': diagnostic}, {}, spec=spec)
    assert 'library_sop' in message and 'version' in message
    assert 'blind round-trip' in message
    assert_no_private(message)


@pytest.mark.parametrize('code', ['class_guard_failed', 'too_few_retained_questions'])
def test_class_guard_diagnostics_omit_distributions_and_records(code):
    spec = candidate()
    question = spec['questions'][0]
    value = {'code': code, 'context': {'question_ids': [question, 'PRIVATE_RECORD'],
             'retained_count': 3, 'minimum_required': 5,
             'class_counts': {question: {'HIDDEN_GOLD': 1}}, 'sampled_records': ['PRIVATE_RECORD']}}
    diagnostic = safe_diagnostic(value, spec)
    assert diagnostic['question_ids'] == [question]
    assert diagnostic['retained_count'] == 3 and diagnostic['minimum_required'] == 5
    message = design_loop.feedback_message({'compile_diagnostic': diagnostic}, {}, spec=spec)
    assert question in message and 'class guard' in message
    assert_no_private(message)


def test_unknown_errors_and_forged_identifiers_are_not_forwarded():
    spec = candidate()
    assert from_exception(ValueError('required fact carrier dropped: HIDDEN_GOLD'), spec) is None
    assert safe_diagnostic({'code': {'answer': 'HIDDEN_GOLD'}}, spec) is None
    value = safe_diagnostic({'code': 'carrier_roundtrip_failed', 'document_id': '/private/gold.json',
                             'fact_keys': ['version'], 'reason': 'HIDDEN_RAW_EXCEPTION'}, spec)
    assert set(value) == {'code', 'reason'}
    value = safe_diagnostic({'code': 'carrier_roundtrip_failed', 'document_id': 'library_sop',
                             'fact_keys': ['PRIVATE_RECORD', 'version', 'HIDDEN_GOLD']}, spec)
    assert value['fact_keys'] == ['version']
    assert_no_private(json.dumps(value))


def test_scope_carrier_does_not_expose_record_level_fact_keys():
    value = safe_diagnostic({'code': 'carrier_roundtrip_failed', 'document_id': 'scope_revision',
                             'fact_keys': ['PRIVATE_RECORD']}, candidate())
    assert value['document_id'] == 'scope_revision' and value['fact_keys'] == []


@pytest.mark.parametrize('through_report', [False, True])
def test_compile_failure_reaches_next_designer_round(tmp_path, monkeypatch, through_report):
    spec = candidate()
    feedback = []
    def designer(intent, out, previous, message):
        feedback.append(message)
        return copy.deepcopy(spec)
    def fail(*args, **kwargs):
        raise failure()
    monkeypatch.setattr(compiler, 'compile_task', fail)
    compile_and_gate = (lambda spec, out: gates.run_single(spec, out)) if through_report else fail
    result = design_loop.prepare_candidate({}, tmp_path / 'loop', designer=designer,
        compile_and_gate=compile_and_gate, reviewer=lambda *args: pytest.fail('No compiled task to review'),
        max_revisions=1)
    assert result['verdict'] == 'discarded' and result['revisions'] == 1
    assert len(feedback) == 2 and 'library_sop' in feedback[1] and 'version' in feedback[1]
    assert_no_private(feedback[1])


def test_actual_failed_carrier_names_only_fields_that_failed_final_read(tmp_path):
    with integration_adapters():
        spec, screening = screened_spec('CUAD')
        spec['work_items'].append({'id': 'verify', 'order': 4, 'depends_on': ['summary'], 'title': 'Confirm cutoff'})
        spec['documents'].append({'id': 'library_sop', 'path': 'correspondence/library_sop.txt', 'kind': 'SOP',
                                 'facts': {'version': 'HIDDEN_FACT_VALUE', 'cutoff': '2024-09-30'}})
        spec['dependency_graph']['nodes'].append({'id': 'version_check', 'type': 'construction_fact',
            'template': 'document_fact', 'provision': 'in-document', 'carrier': 'library_sop',
            'scoring': 'prerequisite', 'work_item': 'verify', 'fact_key': 'version'})
        spec['dependency_graph']['edges'].append({'from': 'summary_counts', 'to': 'version_check'})
        baseline = FakeClient('fake-extractor-base')
        def extract(payload):
            response = baseline.complete('extract_facts', payload)
            if 'library_sop' in payload['text']:
                response['facts']['version'] = 'HIDDEN_EXTRACTED'
            return response
        extractor = FakeClient('fake-extractor', {'extract_facts': extract})
        with pytest.raises(CompilationFailure) as error:
            compiler.compile_task(spec, 7, tmp_path / 'task', screening_report=screening, extractor=extractor)
    diagnostic = from_exception(error.value, spec)
    assert diagnostic['code'] == 'carrier_roundtrip_failed'
    assert diagnostic['document_id'] == 'library_sop'
    assert diagnostic['fact_keys'] == ['version']  # cutoff did round-trip correctly.
    assert_no_private(json.dumps(diagnostic))


def test_actual_class_guard_exposes_question_ids_not_answers(monkeypatch):
    from tada_taskgen import adapters
    from test_r2_compiler import SparseAdapter
    adapter = SparseAdapter()
    adapter.pool = {key: ('HIDDEN_GOLD', []) for key in adapter.pool}
    monkeypatch.setattr(adapters, 'get_adapter', lambda *args, **kwargs: adapter)
    spec = candidate()
    spec['questions'] = list(adapter.questions())
    with pytest.raises(CompilationFailure) as error:
        compiler.resolve(spec, 7)
    diagnostic = from_exception(error.value, spec)
    assert diagnostic['code'] == 'too_few_retained_questions'
    assert diagnostic['question_ids'] == sorted(spec['questions'])
    assert diagnostic['retained_count'] == 0 and diagnostic['minimum_required'] == 5
    assert_no_private(json.dumps(diagnostic))
