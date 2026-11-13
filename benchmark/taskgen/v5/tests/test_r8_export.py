"""The public copy has an explicit source allowlist and fails closed on leaks."""
from pathlib import Path
import json
import pytest
from tools.export_public import allowlist, export, scan


def test_public_allowlist_excludes_live_and_internal_material():
    root = Path(__file__).resolve().parents[2]
    names = {str(p.relative_to(root)) for p in allowlist(root)}
    assert {'controller.py', 'solver/tada_solver_agent.py', 'local_config.py',
            'local_config.ts', 'v5/config.toml', 'tools/export_public.py'} <= names
    assert not any(n.startswith(('v5_runs/', 'runs/', 'private_cache/', 'node_modules/', 'CODEX_')) for n in names)
    assert 'local_paths.toml' not in names and 'v5/IMPLEMENTATION_REPORT.md' not in names
    assert not any('.bak' in n or '__pycache__' in n for n in names)


@pytest.mark.parametrize('text,rule', [
    ('/' + 'Users' + '/someone/private.json', 'absolute_user_or_volume_path'),
    ('/' + 'Volumes' + '/Disk/source', 'absolute_user_or_volume_path'),
    ('gu' + '26', 'private_trial_hostname'),
    ('https://' + 'private' + '.corp/v1', 'private_url'),
    ('https://' + 'username:password' + '@example.com/v1', 'private_url'),
    ('sk-' + 'a' * 30, 'credential_like_string'),
])
def test_scan_rejects_private_strings_without_echoing_them(tmp_path,text,rule):
    (tmp_path/'source.py').write_text(text)
    hits=scan(tmp_path)
    assert any(h['rule']==rule for h in hits)
    assert text not in json.dumps(hits)


@pytest.mark.parametrize('relative', ['__pycache__/x.pyc', 'cache.bak', 'v5_runs/log.md', 'local_paths.toml',
    'cache/data.json', '.cache/data.json', 'model_cache/response.json', 'source_cache/index.json'])
def test_scan_rejects_private_artifacts(tmp_path,relative):
    path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('private')
    assert scan(tmp_path)


def test_scan_accepts_documented_public_routes(tmp_path):
    (tmp_path/'config.txt').write_text('https://github.com/example/repo\nhttp://localhost:3000/v1\nhttps://api.example.org/v1')
    assert scan(tmp_path)==[]


def test_export_refuses_to_overwrite_existing_output(tmp_path):
    (tmp_path/'existing').write_text('keep')
    with pytest.raises(ValueError,match='empty'):
        export(tmp_path)
    assert (tmp_path/'existing').read_text()=='keep'
