"""Resolve trial launch settings from the shared local configuration."""
import os
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_config import get


def remote_path(value):
    return ('"$HOME"/' + shlex.quote(value[2:])) if value.startswith('~/') else shlex.quote(value)


def settings(mode, model):
    remote = mode == 'remote'
    if model not in ('deepseek-flash', 'claude-opus-5-5'):
        raise ValueError('Unsupported trial model')
    opus = model == 'claude-opus-5-5'
    prefix = ('remote_' if remote else 'local_') if opus else ''
    values = {'ENDPOINT': get(prefix + 'solver_endpoint', required=True),
              'KEYVAR': get(prefix + 'solver_key_env', required=True)}
    # Credentials remain in the launching process and travel over SSH stdin.
    values['KEY'] = os.environ.get(values['KEYVAR']) or (get(prefix + 'solver_key', required=True) if opus else '')
    if not values['KEY']:
        raise ValueError('Missing credential environment variable: ' + values['KEYVAR'])
    if not values['KEYVAR'].replace('_', '').isalnum() or values['KEYVAR'][0].isdigit():
        raise ValueError('solver_key_env must be an environment variable name')
    if remote:
        values.update(H=get('remote_trial_host', required=True), R=get('remote_trial_root', required=True),
                      DOCKER_DIR=get('remote_docker_config', required=True), MIRROR=get('remote_mirror_image', required=True))
        values['REMOTE_ROOT'] = remote_path(values['R'])
        values['REMOTE_DOCKER'] = remote_path(values['DOCKER_DIR'])
    else:
        values['HARBOR'] = get('harbor_binary', required=True)
    return values


if __name__ == '__main__':
    try:
        for key, value in settings(*sys.argv[1:]).items():
            print(key + '=' + shlex.quote(str(value)))
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
