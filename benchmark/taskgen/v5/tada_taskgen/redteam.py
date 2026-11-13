"""Public-only shortcut measurement, with a separate, sandboxed script replay.

The default generator is an explicitly fake offline fixture. Real generation is
opt-in; all scoring happens in the trusted parent after the child has exited.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from .concurrency import model_slot
from local_config import path, require

REPO = Path(__file__).resolve().parents[2]
PYTHON = path('datasets_python')
MODEL = 'gpt-6-astra'
TIMEOUT = 20 * 60
MAX_SCRIPT_BYTES = 400_000
FAKE_SCRIPT = '''# Offline fixture only; this is not an adversarial model result.
from pathlib import Path
from openpyxl import load_workbook
book = load_workbook('public/review_template.xlsx')
Path('out').mkdir(exist_ok=True)
book.save('out/review.xlsx')
'''


def _regular(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Input must be a regular file')


def _stage(task: Path, workspace: Path):
    """Copy the allowlisted solver inputs, never a task or repository directory."""
    public = task / 'environment/public'
    if public.is_symlink() or not public.is_dir():
        raise ValueError('Public input directory is invalid')
    _regular(task / 'instruction.md')
    files = list(public.rglob('*'))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in files):
        raise ValueError('Public inputs contain links or special files')
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copytree(public, workspace / 'public')
    shutil.copyfile(task / 'instruction.md', workspace / 'instruction.md')
    (workspace / 'out').mkdir()


def _runtime_roots():
    # The interpreter is a uv-managed executable outside the venv. Restrict its
    # allowance to that exact Python installation, not the user's home directory.
    require('datasets_python')
    return [PYTHON.parent.parent.resolve(), PYTHON.resolve().parent.parent]


def codex_command(workspace: Path):
    """Use a restricted workspace permission profile, not global filesystem read.

    Codex's legacy --sandbox workspace-write permits reads outside the workspace
    and overrides named profiles. The current native profile extends :workspace
    and explicitly denies global reads; unsupported profiles fail closed.
    """
    filesystem = {':root': 'deny', ':minimal': 'read', ':tmpdir': 'deny', ':slash_tmp': 'deny'}
    filesystem.update({str(p): 'read' for p in _runtime_roots()})
    fs_toml = ', '.join(f'{json.dumps(k)}={json.dumps(v)}' for k, v in filesystem.items())
    profile = 'permissions.redteam={extends=":workspace",filesystem={' + fs_toml + '},network={enabled=false}}'
    settings = ['default_permissions="redteam"', profile, 'approval_policy="never"',
                'model_reasoning_effort="high"', 'mcp_servers={}', 'web_search="disabled"',
                'project_doc_max_bytes=0', 'shell_environment_policy.inherit="none"',
                'shell_environment_policy.set={PATH="/usr/bin:/bin",PYTHONDONTWRITEBYTECODE="1",'
                'PYTHONNOUSERSITE="1",TMPDIR=' + json.dumps(str(workspace / 'out')) + '}',
                'features.apps=false', 'features.plugins=false', 'features.hooks=false',
                'features.multi_agent=false', 'features.multi_agent_v2=false',
                'features.external_agent_memory_import=false', 'features.skill_search=false',
                'features.skill_mcp_dependency_install=false', 'features.skip_host_skill_discovery=true']
    command = ['codex', 'exec', '--ignore-user-config', '--ignore-rules', '--ephemeral',
               '--skip-git-repo-check', '--model', MODEL, '--cd', str(workspace), '--color', 'never']
    for setting in settings:
        command.extend(['-c', setting])
    return command


def prompt():
    return f'''Create solve.py, one Python script using only the standard library and openpyxl.
Produce out/review.xlsx as accurately as possible from instruction.md, public/ and its template.
You are measuring programmatic shortcuts: do not read agreements semantically by hand.
String, regular-expression and statistical heuristics implemented in the script are allowed.
You may inspect the instructions, public file formats and coding conventions, and run your script.
Use {PYTHON} to run Python. The public contract's /workspace and /output paths map to
this directory and out/ respectively. Do not use network, tools outside this workspace,
other agents, external services or nonstandard Python packages except openpyxl.
Leave your complete runnable source in solve.py. The parent will independently replay it
against pristine public inputs, then measure its output. Do not modify the public inputs.
'''


def _real_runner(request):
    # Always acquire an Astra slot even when called inside a designer that holds
    # an Opus slot. A generic inherited "slot held" flag is not sufficient.
    with model_slot(MODEL, timeout=TIMEOUT):
        result = subprocess.run(request['command'], input=request['prompt'], text=True,
                                cwd=request['cwd'], env=request['env'], capture_output=True,
                                timeout=request['timeout'])
    return {'returncode': result.returncode}


def script_sandbox(workspace: Path):
    """Seatbelt limits even absolute reads to public inputs and Python runtime."""
    def subpath(path):
        return '(subpath ' + json.dumps(str(path)) + ')'
    def literal(path):
        return '(literal ' + json.dumps(str(path)) + ')'
    # These OS paths contain the dynamic loader, system libraries and locale data.
    runtime = [Path('/System'), Path('/usr/lib'), Path('/usr/share'), Path('/dev'),
               Path('/private/etc'), *_runtime_roots()]
    # macOS Python's loader needs the root directory object during startup.
    # A literal allowance does not grant reads to its children or private data.
    readable = [literal(Path('/')), *(subpath(p) for p in runtime), subpath(workspace / 'public'),
                subpath(workspace / 'out'), literal(workspace / 'instruction.md'),
                literal(workspace / 'solve.py')]
    return '\n'.join(['(version 1)', '(allow default)', '(deny network*)',
                      '(deny file-read*)', '(deny file-write*)',
                      '(allow file-read* ' + ' '.join(readable) + ')',
                      '(allow file-write* ' + subpath(workspace / 'out') + ')',
                      '(allow file-write* (literal "/dev/null"))'])


def _execute(request):
    result = subprocess.run(request['command'], cwd=request['cwd'], env=request['env'],
                            capture_output=True, text=True, timeout=request['timeout'])
    return {'returncode': result.returncode}


def _unit(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def run_redteam(task, out, *, runner=None, executor=None, mode='fake'):
    """Return only aggregate scores and generated source, never scorer internals.

    Injected runner/executor receive public-only request dictionaries. A runner
    may write solve.py or return {'script': source, 'returncode': 0}. An executor
    must create out/review.xlsx in its supplied fresh replay workspace.
    """
    if mode not in ('fake', 'real'):
        raise ValueError('Red-team mode must be fake or real')
    task, out = Path(task).resolve(), Path(out).resolve()
    if not out.is_relative_to(REPO / 'v5_runs'):
        raise ValueError('Red-team artifacts must be inside v5_runs')
    out.mkdir(parents=True, exist_ok=True)
    run = out / ('session-' + uuid.uuid4().hex)
    workspace, replay = run / 'workspace', run / 'replay'
    report = {'pass': False, 'status': 'failed', 'reason': 'preparation_failed', 'fake': mode == 'fake',
              'model': MODEL, 'reward': None, 'components': {}, 'per_work_item': {}, 'script': ''}
    try:
        _stage(task, workspace)
        request = {'command': codex_command(workspace), 'prompt': prompt(), 'cwd': str(workspace),
                   'timeout': TIMEOUT, 'env': dict(os.environ, PYTHONDONTWRITEBYTECODE='1')}
        report['reason'] = 'generation_failed'
        result = (runner or (_real_runner if mode == 'real' else lambda _: {'script': FAKE_SCRIPT, 'returncode': 0}))(request)
        if not isinstance(result, dict) or result.get('returncode', 0) != 0:
            raise ValueError('Generation failed')
        if 'script' in result:
            if not isinstance(result['script'], str):
                raise ValueError('Script must be text')
            (workspace / 'solve.py').write_text(result['script'])
        report['reason'] = 'script_missing_or_invalid'
        script = workspace / 'solve.py'
        _regular(script)
        if not 0 < script.stat().st_size <= MAX_SCRIPT_BYTES:
            raise ValueError('Invalid script size')
        source = script.read_text()
        compile(source, 'solve.py', 'exec')
        report['script'] = source
        # Discard any workbook or public-file changes created by the agent.
        _stage(task, replay)
        (replay / 'solve.py').write_text(source)
        profile = script_sandbox(replay)
        (run / 'script.sb').write_text(profile)
        command = ['/usr/bin/sandbox-exec', '-p', profile, str(PYTHON), '-I', str(replay / 'solve.py')]
        env = {'PATH': '/usr/bin:/bin', 'TMPDIR': str(replay / 'out'),
               'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'}
        report['reason'] = 'script_execution_failed'
        execution = (executor or _execute)({'command': command, 'cwd': str(replay), 'env': env,
                                           'timeout': TIMEOUT, 'sandbox_profile': profile})
        if not isinstance(execution, dict) or execution.get('returncode') != 0:
            raise ValueError('Script execution failed')
        deliverable = replay / 'out/review.xlsx'
        _regular(deliverable)
        report['reason'] = 'scoring_failed'
        # A task-specific scorer is never executed. Verify its frozen identity,
        # then call the same trusted implementation in this parent process.
        scorer = Path(__file__).with_name('scoring.py')
        _regular(task / 'tests/scoring.py')
        if hashlib.sha256((task / 'tests/scoring.py').read_bytes()).digest() != hashlib.sha256(scorer.read_bytes()).digest():
            raise ValueError('Frozen scorer mismatch')
        from .scoring import score
        value, details = score(deliverable, json.loads((task / 'tests/gold.json').read_text()))
        if not _unit(value) or 'error' in details:
            raise ValueError('Scoring failed')
        report.update({'pass': True, 'status': 'scored', 'reason': None, 'reward': value,
                       'components': {k: v for k, v in details.get('components', {}).items() if _unit(v)},
                       'per_work_item': {k: v for k, v in details.get('per_work_item', {}).items() if _unit(v)}})
    except (OSError, ValueError, SyntaxError, subprocess.SubprocessError, TimeoutError):
        # Failure is an unavailable measurement, not evidence against the task.
        pass
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def sample_batch(candidates, out, *, runner=None):
    """Deterministic 10% batch sample, rounded down; never a per-task gate.

    Partial batches below ten retain pending status. Persist selected identities
    and measurements so resume cannot launch repeated paid calls.
    """
    rows = sorted(candidates, key=lambda row: hashlib.sha256(str(row['id']).encode()).hexdigest())
    count = len(rows) // 10
    report_path = Path(out) / 'batch_report.json'
    previous = json.loads(report_path.read_text()) if report_path.is_file() else {}
    cached = {r['id']: r for r in previous.get('reports', [])}
    ids = {r['id'] for r in rows}
    selected = [r for r in previous.get('reports', []) if r['id'] in ids][:count]
    chosen = {r['id'] for r in selected}
    selected += [row for row in rows if row['id'] not in chosen][:count - len(selected)]
    reports = []
    for row in selected:
        if row['id'] in cached:
            reports.append(dict(cached[row['id']], measurement_reused=True))
            continue
        try:
            result = (runner or (lambda task, target: run_redteam(task, target)))(
                Path(row['task']), Path(out) / row['id'])
        except Exception as exc:
            result = {'status': 'unavailable', 'error_type': type(exc).__name__}
        reports.append({'id': row['id'], 'measurement': result, 'reported_only': True})
    report = {'reported_only': True, 'nominal_fraction': .10,
              'actual_fraction': count / len(rows) if rows else 0,
              'pending_until_batch_size': 10 if 0 < len(rows) < 10 else None,
              'batch_size': len(rows), 'sample_size': count, 'reports': reports}
    Path(out).mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report
