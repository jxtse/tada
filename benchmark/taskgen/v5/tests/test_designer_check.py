"""Designer format check: shown to the designer, so it must catch the compiler contract and stay label-free."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from tada_taskgen.designer_check import check
from tada_taskgen.compiler import compile_task

REPO = Path(__file__).resolve().parents[2]


def designer():
    spec = importlib.util.spec_from_file_location('run_designer', REPO / 'run_designer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(directory, spec, *, embed=True):
    directory.mkdir(parents=True, exist_ok=True)
    body = dict(spec)
    instruction, design = body.pop('instruction', 'Prepare the workbook.'), body.pop('design', 'Design notes.')
    graph = body['dependency_graph']
    if not embed:
        body.pop('dependency_graph')
    (directory / 'scenario_spec.json').write_text(json.dumps(body))
    (directory / 'dependency_graph.json').write_text(json.dumps(graph))
    (directory / 'instruction.md').write_text(instruction)
    (directory / 'design.md').write_text(design)
    return directory


def test_prompt_example_passes_check_and_compiles(tmp_path, monkeypatch):
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    example = designer().V5_EXAMPLE
    task_dir = write(tmp_path / 'task', example, embed=False)
    assert check(task_dir) == {'ok': True, 'error': None}
    from tada_taskgen.schema import load_designer_output
    built = compile_task(load_designer_output(task_dir), 7, tmp_path / 'built', client='fake')
    assert (Path(built) / 'tests/gold.json').exists()


def test_regression_spec_passes_check(tmp_path):
    spec = json.loads((REPO / 'v5/specs/maud_screened.json').read_text())
    assert check(write(tmp_path / 'task', spec))['ok']


@pytest.mark.parametrize('mutate,needle', [
    (lambda s: s['dependency_graph']['nodes'][0].pop('template'), 'template'),
    (lambda s: s['work_items'][1].update(scope_from='w1'), 'scope_from'),
    (lambda s: s['dependency_graph']['nodes'][3].update(fact_key='missing'), 'fact_key'),
    (lambda s: s['dependency_graph'].update(edges=[]), 'edges'),
    (lambda s: s.update(questions=['DP999']), 'unknown question'),
])
def test_contract_errors_are_specific(tmp_path, mutate, needle):
    spec = json.loads(json.dumps(designer().V5_EXAMPLE))
    mutate(spec)
    result = check(write(tmp_path / 'task', spec))
    assert result['ok'] is False and needle in result['error']


def test_check_messages_never_contain_labels(tmp_path):
    from tada_taskgen.adapters import get_adapter
    pool = get_adapter('MAUD').label_pool()
    spec = json.loads(json.dumps(designer().V5_EXAMPLE))
    spec['dependency_graph']['nodes'][0].pop('template')
    message = check(write(tmp_path / 'task', spec))['error']
    rid, _ = next(iter(pool))
    assert rid not in message and 'gold' not in message.lower()


def test_cli_prints_one_json_line(tmp_path):
    task_dir = write(tmp_path / 'task', designer().V5_EXAMPLE)
    out = subprocess.run([str(__import__('local_config').path('datasets_python', required=True)), '-m', 'tada_taskgen.designer_check',
                          str(task_dir)], cwd=REPO, env={'PYTHONPATH': str(REPO / 'v5'), 'PYTHONDONTWRITEBYTECODE': '1',
                                                         'PATH': '/usr/bin:/bin'},
                         capture_output=True, text=True, timeout=120)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {'ok': True, 'error': None}


def test_designer_feedback_carries_format_error(tmp_path):
    from tada_taskgen import design_loop
    spec = json.loads(json.dumps(designer().V5_EXAMPLE))
    spec['work_items'][1]['scope_from'] = 'w1'

    def bad_designer(intent, output, previous, feedback):
        write(output / 'task', spec)
        (output / 'receipt.json').write_text('{}')
        raise ValueError('Designer did not complete four valid artifacts')

    result = design_loop.prepare_candidate({}, tmp_path / 'c', designer=bad_designer,
                                           compile_and_gate=lambda *a: pytest.fail('must not compile'), reviewer=None,
                                           max_revisions=1)
    feedback = (tmp_path / 'c/attempt-1/feedback.md').read_text()
    assert 'scope_from must name the record_scope node id' in feedback
    assert result['verdict'] == 'discarded'
