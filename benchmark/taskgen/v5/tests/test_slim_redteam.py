"""Report-only redteam measurements keep agent inputs separate from private scoring."""
from pathlib import Path

import pytest

from tada_taskgen.compiler import compile_task
from tada_taskgen.redteam import FAKE_SCRIPT, run_redteam
from screened_support import integration_adapters, screened_spec


@pytest.fixture
def task(tmp_path):
    with integration_adapters():
        spec, screening = screened_spec('MAUD')
        return compile_task(spec, 7, tmp_path / 'task', screening_report=screening)


def copy_blank_template(request):
    from openpyxl import load_workbook
    workspace = Path(request['cwd'])
    assert set(request['env']) == {'PATH', 'TMPDIR', 'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE'}
    assert '(deny network*)' in request['sandbox_profile']
    assert '(deny file-read*)' in request['sandbox_profile']
    assert request['command'][0] == '/usr/bin/sandbox-exec'
    book = load_workbook(workspace / 'public/review_template.xlsx')
    book.save(workspace / 'out/review.xlsx')
    book.close()
    return {'returncode': 0}


def test_public_only_fresh_replay_and_aggregate_result(task, tmp_path):
    seen = []
    original = (task / 'instruction.md').read_text()

    def runner(request):
        workspace = Path(request['cwd'])
        assert {p.name for p in workspace.iterdir()} == {'public', 'instruction.md', 'out'}
        assert not list(workspace.rglob('gold.json'))
        assert not list(workspace.rglob('reference.xlsx'))
        assert '--ephemeral' in request['command'] and '--ignore-rules' in request['command']
        # The generator cannot make its altered inputs authoritative for replay.
        (workspace / 'instruction.md').write_text('Altered by the untrusted generator')
        (workspace / 'public/attacker.txt').write_text('untrusted mutation')
        seen.append(workspace)
        return {'script': FAKE_SCRIPT, 'returncode': 0}

    def execute(request):
        replay = Path(request['cwd'])
        assert replay != seen[0]
        assert (replay / 'instruction.md').read_text() == original
        assert not (replay / 'public/attacker.txt').exists()
        return copy_blank_template(request)

    result = run_redteam(task, tmp_path / 'redteam', runner=runner, executor=execute)
    assert result['status'] == 'scored' and result['reward'] == 0
    assert result['fake'] is True and result['script'] == FAKE_SCRIPT
    assert 'cell_correct' not in result and 'gold' not in result
    assert (task / 'instruction.md').read_text() == original


def test_task_specific_scorer_is_never_executed(task, tmp_path):
    (task / 'tests/scoring.py').write_text('raise AssertionError("untrusted scorer ran")')
    result = run_redteam(task, tmp_path / 'redteam', executor=copy_blank_template)
    assert result['status'] == 'failed' and result['reason'] == 'scoring_failed'
    assert result['reward'] is None


def test_private_symlink_in_public_inputs_fails_before_generation(task, tmp_path):
    (task / 'environment/public/private-link.json').symlink_to(task / 'tests/gold.json')
    result = run_redteam(task, tmp_path / 'redteam',
                         runner=lambda _: pytest.fail('Unsafe staging must not launch a generator'))
    assert result['reason'] == 'preparation_failed' and result['reward'] is None
