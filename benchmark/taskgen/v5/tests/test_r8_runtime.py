"""Offline configuration boundaries for launchers and restricted subprocesses."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import local_config

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('trial_config', ROOT / 'solver/trial_config.py')
trial_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial_config)


def test_remote_settings_use_config_without_exposing_secret_in_endpoint(monkeypatch, tmp_path):
    file = tmp_path / 'local.toml'
    file.write_text('remote_trial_host="worker.example"\nremote_trial_root="~/trial root"\n'
                    'remote_docker_config="~/docker config"\nremote_mirror_image="mirror.example/alpine@sha256:abc"\n'
                    'remote_solver_endpoint="http://localhost:9000/v1"\nremote_solver_key_env="TEST_TRIAL_KEY"\n')
    monkeypatch.setenv('TADA_LOCAL_PATHS_FILE', str(file))
    monkeypatch.setenv('TEST_TRIAL_KEY', 'test-value')
    settings = trial_config.settings('remote', 'claude-opus-5-5')
    assert settings['H'] == 'worker.example'
    assert settings['REMOTE_ROOT'] == '"$HOME"/\'trial root\''
    assert settings['KEY'] == 'test-value'
    assert settings['KEY'] not in settings['ENDPOINT']
    assert settings['MIRROR'].startswith('mirror.example/')


def test_solver_endpoint_has_no_provider_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv('TADA_LOCAL_PATHS_FILE', str(tmp_path / 'absent.toml'))
    monkeypatch.delenv('TADA_SOLVER_ENDPOINT', raising=False)
    with pytest.raises(local_config.ConfigurationError, match='solver_endpoint'):
        trial_config.settings('local', 'deepseek-flash')


def test_restricted_build_env_retains_root_configuration_import(monkeypatch, tmp_path):
    import controller
    captured = {}
    def run(args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(controller, 'require', lambda *keys: None)
    monkeypatch.setattr(controller.subprocess, 'run', run)
    controller.run_py(['-c', 'pass'], tmp_path, [tmp_path], tmp_path / 'build.log')
    assert str(ROOT) in captured['env']['PYTHONPATH'].split(':')
    assert 'PYTHONDONTWRITEBYTECODE' in captured['env']


def test_public_v4_designer_fails_with_actionable_message(monkeypatch, tmp_path):
    import run_designer
    monkeypatch.setattr(run_designer, 'HERE', tmp_path)
    with pytest.raises(ValueError, match='use --v5'):
        run_designer.assemble('B')


def test_remote_shell_scripts_parse_without_launching():
    for filename in ('run_trial.sh', 'run_trial_remote.sh'):
        assert subprocess.run(['sh', '-n', str(ROOT / 'solver' / filename)]).returncode == 0


def test_configured_cache_fragment_remains_a_private_marker(monkeypatch, tmp_path):
    import runpy
    file = tmp_path / 'local.toml'
    file.write_text('legacy_cache="' + str(tmp_path / 'legacy/cache') + '"\n')
    monkeypatch.setenv('TADA_LOCAL_PATHS_FILE', str(file))
    monkeypatch.delenv('TADA_LEGACY_CACHE', raising=False)
    module = runpy.run_path(str(ROOT / 'controller.py'), run_name='controller_portability_test')
    assert 'legacy/cache' in module['PRIVATE_STRINGS']
    assert str(tmp_path / 'legacy/cache') in module['PRIVATE_STRINGS']


def test_remote_launcher_uses_config_and_keeps_credentials_off_argv(monkeypatch, tmp_path):
    import json
    import os
    import sys
    file = tmp_path / 'local.toml'
    file.write_text('remote_trial_host="worker.example"\nremote_trial_root="~/trial-root"\n'
                    'remote_docker_config="~/docker-settings"\nremote_mirror_image="mirror.example/image@sha256:abc"\n'
                    'remote_solver_endpoint="http://localhost:9000/v1"\nremote_solver_key_env="TEST_TRIAL_KEY"\n')
    binary = tmp_path / 'bin'; binary.mkdir()
    (binary / 'python3').symlink_to(sys.executable)
    log = tmp_path / 'calls.jsonl'
    program = '''#!PYTHON
import json,os,sys
from pathlib import Path
name=Path(sys.argv[0]).name
row={'program':name,'argv':sys.argv[1:]}
if name=='ssh' and sys.argv[-1].startswith('read -r K;'):
 row['stdin']=sys.stdin.read()
with open(os.environ['STUB_LOG'],'a') as f:f.write(json.dumps(row)+'\\n')
if name=='ssh' and '.exit' in sys.argv[-1] and not sys.argv[-1].startswith('read -r K;'):print('0')
'''.replace('PYTHON', sys.executable)
    for name in ('ssh', 'rsync', 'sleep'):
        target = binary / name; target.write_text(program); target.chmod(0o755)
    task = tmp_path / 'task'; task.mkdir(); (task / 'task.toml').write_text('version="1.0"\n')
    env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ['PATH'], STUB_LOG=str(log),
               TADA_LOCAL_PATHS_FILE=str(file), TADA_MODEL_SLOT_HELD='1', TADA_MODEL_SLOT_FAMILY='opus',
               TEST_TRIAL_KEY='fixture-only-value')
    result = subprocess.run(['sh', str(ROOT / 'solver/run_trial_remote.sh'), str(task), str(tmp_path/'jobs'), 'trial', 'claude-opus-5-5'],
                            env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls and all('fixture-only-value' not in json.dumps(row['argv']) for row in calls)
    launch = next(row for row in calls if row.get('stdin'))
    assert launch['stdin'] == 'fixture-only-value\n'
    assert 'worker.example' in launch['argv']
    assert '"$HOME"/trial-root' in launch['argv'][-1]
    assert '"$HOME"/docker-settings' in launch['argv'][-1]
    assert any('local_config.py' in row['argv'] for row in calls if row['program']=='rsync')
    assert any('worker.example:~/trial-root/solver/sitecustomize.py' in row['argv'] for row in calls)
