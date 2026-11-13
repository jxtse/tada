import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tada_taskgen import design_loop

REPO = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, REPO / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def review(severity=0):
    return {'questions': [{'id': key, 'severity': severity, 'evidence': 'Public instructions omit an audience.'
                          if severity else 'The public requirement is clear.',
                          'required_change': 'State the intended audience.' if severity else ''}
                         for key in module('review_package').V5_QUESTIONS]}


def gate_report(task, passes=True):
    return {'pass': passes, 'task': str(task), 'hard_gates': {
        key: {'pass': passes, 'reward': .45 if not passes else .1, 'limit': .3,
              'private': {'gold': 'HIDDEN_GOLD_SENTINEL', 'wrong_cells': ['cell-secret']},
              'error': 'RuntimeError: /private/gold.json HIDDEN_GOLD_SENTINEL'}
        for key in design_loop.HARD_GATES}}


def spec():
    return json.loads((REPO / 'v5/specs/maud_screened.json').read_text())


def write_four(output, candidate):
    output.mkdir(parents=True)
    for name, value in [('scenario_spec.json', candidate), ('dependency_graph.json', candidate['dependency_graph'])]:
        (output / name).write_text(json.dumps(value))
    for name in ('instruction', 'design'):
        (output / (name + '.md')).write_text(candidate[name])


@pytest.mark.parametrize('severity,required', [(0, False), (1, False), (2, True), (3, True)])
def test_four_question_decision(severity, required):
    result = module('review_package').validate_v5_review(review(severity))
    assert result['revision_required'] is required
    assert result['max_severity'] == severity
    assert len(result['questions']) == 4
    assert len(result['findings']) == (4 if required else 0)


@pytest.mark.parametrize('invalid', ['missing', 'extra', 'duplicate', 'boolean', 'high', 'low', 'empty_change', 'old'])
def test_four_question_schema_rejects_other_forms(invalid):
    value = review(2)
    if invalid == 'missing':
        value['questions'].pop()
    elif invalid == 'extra':
        value['unrequested_dimension'] = 'gold derivability'
    elif invalid == 'duplicate':
        value['questions'][1]['id'] = value['questions'][0]['id']
    elif invalid in ('boolean', 'high', 'low'):
        value['questions'][0]['severity'] = {'boolean': True, 'high': 4, 'low': -1}[invalid]
    elif invalid == 'empty_change':
        value['questions'][0]['required_change'] = ''
    else:
        value = {'verdict': 'accept', 'findings': []}
    with pytest.raises(ValueError):
        module('review_package').validate_v5_review(value)


def test_feedback_has_only_aggregate_values_and_major_review_findings(tmp_path):
    gates = gate_report(tmp_path, False)
    gates['hard_gates']['trivial_answers']['results'] = {
        'constant': {'reward': .93, 'label': 'HIDDEN_GOLD_SENTINEL',
                        'per_cell': {'cell-secret': 'HIDDEN_GOLD_SENTINEL'}}}
    value = review(1)
    value['questions'][2].update(severity=2, evidence='The notice dates contradict the instructions.',
                                  required_change='Reconcile the dates in the public notice.')
    message = design_loop.feedback_message(gates, value)
    assert '.93' in message and '0.3' in message and 'constant' in message
    assert 'include_all' not in message
    assert 'Reconcile the dates' in message
    assert 'instruction_sufficiency' not in message
    for forbidden in ('HIDDEN_GOLD_SENTINEL', 'cell-secret', '/private', 'RuntimeError', 'per_cell'):
        assert forbidden not in message


@pytest.mark.parametrize('text', [
    'Record R1 correct answer: HIDDEN_GOLD_SENTINEL',
    '{"answer": "HIDDEN_GOLD_SENTINEL"}',
    'Read /private/HIDDEN_GOLD_SENTINEL/tests/gold.json',
    'RuntimeError: HIDDEN_GOLD_SENTINEL',
    '记录 R1 的正确标签是 HIDDEN_GOLD_SENTINEL',
])
def test_feedback_blocks_record_answers_and_private_diagnostics(tmp_path, text):
    value = review(2)
    value['questions'][0]['evidence'] = text
    value['questions'][0]['required_change'] = text
    message = design_loop.feedback_message(gate_report(tmp_path), value)
    assert 'HIDDEN_GOLD_SENTINEL' not in message
    assert 'omitted' in message


def test_reviewer_materials_never_read_private_artifacts(tmp_path):
    task = tmp_path / 'task'
    public = task / 'environment/public'
    public.mkdir(parents=True)
    (task / 'tests').mkdir()
    (task / 'tests/gold.json').write_text('{"answer":"HIDDEN_GOLD_SENTINEL"}')
    (task / 'scenario_spec.json').write_text('{"label":"HIDDEN_GOLD_SENTINEL"}')
    (task / 'dependency_graph.json').write_text('{"value":"HIDDEN_GOLD_SENTINEL"}')
    (task / 'instruction.md').write_text('Review the contracts for the committee.')
    (task / 'task.toml').write_text('[metadata]\nname = "review"\n')
    (public / 'notice.md').write_text('The final review meeting is Friday.')
    material = module('review_package').v5_materials(task, gate_report(task, False))
    assert 'environment/public/notice.md' in material
    assert not any(name.startswith('tests/') for name in material)
    assert 'HIDDEN_GOLD_SENTINEL' not in json.dumps(material)


def test_gate_and_reviewer_share_three_revision_budget(tmp_path):
    calls, feedback = [], []

    def designer(intent, output, previous, message):
        calls.append(output)
        feedback.append(message)
        write_four(output, spec())
        return output

    def gates(candidate, output):
        return gate_report(output / 'task', len(calls) >= 3)

    def reviewer(task, gates, output):
        return review(2 if len(calls) == 3 else 0)

    result = design_loop.prepare_candidate({}, tmp_path, designer=designer,
        compile_and_gate=gates, reviewer=reviewer)
    assert result['verdict'] == 'ready'
    assert result['revisions'] == 3 and len(calls) == 4
    assert len(result['attempts']) == 4
    assert 'keyword_baseline' in feedback[1]
    assert 'instruction_sufficiency' in feedback[3]


def test_fourth_failed_attempt_is_discarded_and_never_calls_a_fifth(tmp_path):
    calls = []

    def designer(*args):
        calls.append(1)
        return spec()

    result = design_loop.prepare_candidate({}, tmp_path, designer=designer,
        compile_and_gate=lambda candidate, output: gate_report(output / 'task'),
        reviewer=lambda *args: review(3))
    assert result['verdict'] == 'discarded'
    assert result['revisions'] == 3 and len(calls) == 4
    assert result['failures'] == ['review:' + key for key in module('review_package').V5_QUESTIONS]
    with pytest.raises(ValueError, match='three'):
        design_loop.prepare_candidate({}, tmp_path, designer=designer,
            compile_and_gate=None, reviewer=None, max_revisions=4)


def test_loop_errors_do_not_expose_exception_text(tmp_path):
    def fail(*args):
        raise ValueError('HIDDEN_GOLD_SENTINEL at /private/gold.json')
    result = design_loop.prepare_candidate({}, tmp_path, designer=fail,
        compile_and_gate=lambda *args: pytest.fail('invalid designs cannot compile'), reviewer=None)
    assert result['verdict'] == 'discarded' and result['revisions'] == 3
    assert 'HIDDEN_GOLD_SENTINEL' not in json.dumps(result)


def test_production_review_transport_is_injectable(tmp_path, monkeypatch):
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'instruction.md').write_text('Review for committee ' + tmp_path.name)
    (task / 'task.toml').write_text('[metadata]\nname = "review"\n')
    seen = []
    monkeypatch.setenv('TADA_ASTRA_API_KEY', 'offline-review-test-key')
    monkeypatch.setenv('TADA_EXTRACTOR_ENDPOINT', 'http://127.0.0.1:18142/v1')
    monkeypatch.setenv('TADA_REVIEW_ENDPOINT', 'http://127.0.0.1:18142/v1/responses')

    def transport(url, payload, headers, timeout):
        seen.append(payload)
        assert url == 'http://127.0.0.1:18142/v1/responses'
        assert headers['Authorization'] == 'Bearer offline-review-test-key'
        assert payload['model'] == 'gpt-6-astra'
        assert 'HIDDEN_GOLD_SENTINEL' not in json.dumps(payload)
        return 200, {'output_text': json.dumps(review(0))}

    result = module('review_package').review_v5(task, tmp_path / 'review',
        gates=gate_report(task, False), transport=transport)
    assert result['pass'] and len(seen) == 1
    assert (tmp_path / 'review/verdict.json').exists()
    assert list((tmp_path / 'review/model_cache').glob('*.json'))


@pytest.mark.parametrize('family', ['MAUD', 'CUAD'])
def test_designer_dry_run_uses_only_four_file_tools(tmp_path, family):
    output = tmp_path / 'designer'
    completed = subprocess.run([sys.executable, str(REPO / 'run_designer.py'), family,
        '--v5', '--dry-run', '--out', str(output)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((output / 'receipt.json').read_text())
    assert set(receipt['tools']) == {'check_task_files', 'list_task_files', 'read_task_file', 'write_task_file'}
    assert len(receipt['required_files']) == 4
    prompt = (output / 'prompt.md').read_text()
    assert 'at most three' in prompt and 'exactly six' in prompt


def test_designer_receives_feedback_after_incomplete_initial_output(tmp_path):
    output = tmp_path / 'designer'
    brief = tmp_path / 'feedback.md'
    brief.write_text('Uncompleted checks: designer_output_invalid')
    completed = subprocess.run([sys.executable, str(REPO / 'run_designer.py'), 'MAUD',
        '--v5', '--dry-run', '--out', str(output), '--brief', str(brief)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert brief.read_text() in (output / 'prompt.md').read_text()


def test_production_loop_wiring_uses_designer_gates_and_reviewer(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import tada_taskgen.concurrency as concurrency
    import tada_taskgen.gates
    captured = []
    queued = []
    real_review_module = module('review_package')

    @contextmanager
    def fake_slot(model):
        queued.append(model)
        token = concurrency.HELD_FAMILY.set('opus')
        try:
            yield
        finally:
            concurrency.HELD_FAMILY.reset(token)

    def fake_process(command, **kwargs):
        assert queued == ['claude-opus-5-5']
        assert kwargs['env']['TADA_MODEL_SLOT_HELD'] == '1'
        assert kwargs['env']['TADA_MODEL_SLOT_FAMILY'] == 'opus'
        assert kwargs['timeout'] == 5500
        captured.append(command)
        output = Path(command[command.index('--out') + 1])
        write_four(output / 'task', spec())
        (output / 'receipt.json').write_text('{"status":"design_written","family":"MAUD"}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(concurrency, 'borrowed_slot', fake_slot)
    monkeypatch.setattr(design_loop.subprocess, 'run', fake_process)
    monkeypatch.setattr(tada_taskgen.gates, 'run_single',
        lambda candidate, output, **kwargs: gate_report(output / 'task'))
    monkeypatch.setattr(real_review_module, 'review_v5', lambda *args, **kwargs: review())
    monkeypatch.setattr(design_loop, '_review_module', lambda: real_review_module)
    result = design_loop.production_prepare({'source': 'MAUD'}, tmp_path)
    assert result['verdict'] == 'ready'
    assert '--v5' in captured[0] and '--intent' in captured[0]
    assert captured[0][captured[0].index('--max-continues') + 1] == '0'
