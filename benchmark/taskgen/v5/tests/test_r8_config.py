"""Portable local settings work with no data and in restricted environments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import local_config as config


def test_environment_precedes_local_file(tmp_path, monkeypatch):
    file = tmp_path / 'paths.toml'
    file.write_text('datasets_root = "relative-data"\n')
    monkeypatch.setenv('TADA_LOCAL_PATHS_FILE', str(file))
    monkeypatch.delenv('TADA_DATASETS_ROOT', raising=False)
    assert config.path('datasets_root') == config.ROOT / 'relative-data'
    monkeypatch.setenv('TADA_DATASETS_ROOT', str(tmp_path / 'override'))
    assert config.path('datasets_root') == tmp_path / 'override'


def test_missing_settings_are_explicit_but_import_safe(tmp_path, monkeypatch):
    monkeypatch.setenv('TADA_LOCAL_PATHS_FILE', str(tmp_path / 'absent.toml'))
    monkeypatch.delenv('TADA_DATASETS_ROOT', raising=False)
    assert not config.configured('datasets_root')
    assert not config.path('datasets_root').exists()
    with pytest.raises(config.ConfigurationError, match='datasets_root.*TADA_DATASETS_ROOT'):
        config.require('datasets_root')
    with pytest.raises(config.ConfigurationError, match='datasets_root'):
        config.path('datasets_root', required=True)


def test_restricted_subprocess_finds_local_file_by_module_location(tmp_path):
    # Use a copied module and neutral local settings; no inherited settings/env.
    (tmp_path / 'local_config.py').write_bytes(Path(config.__file__).read_bytes())
    (tmp_path / 'local_paths.toml').write_text('datasets_root = "sample-data"\n')
    script = 'import local_config;print(local_config.path("datasets_root", required=True))'
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=tmp_path,
                            env={'PATH': os.environ.get('PATH', ''), 'PYTHONDONTWRITEBYTECODE': '1'},
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == str(tmp_path / 'sample-data')


def test_absent_configuration_allows_catalogue_and_reader_imports(tmp_path):
    env = dict(os.environ, TADA_LOCAL_PATHS_FILE=str(tmp_path / 'missing.toml'), PYTHONDONTWRITEBYTECODE='1')
    for key in list(env):
        if key.startswith('TADA_') and key != 'TADA_LOCAL_PATHS_FILE':
            env.pop(key)
    env['PYTHONPATH'] = str(config.ROOT) + os.pathsep + str(config.ROOT / 'v5')
    code = 'import tada_taskgen;from tada_taskgen.batch1 import IDS;import tada_taskgen.adapters.benchmark_reader;print(len(IDS))'
    result = subprocess.run([sys.executable, '-B', '-c', code], env=env, cwd=tmp_path,
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == '29'


def test_public_provenance_has_no_machine_path(tmp_path, monkeypatch):
    monkeypatch.setenv('TADA_DATASETS_ROOT', str(tmp_path))
    assert config.public_text(str(tmp_path / 'license.md')) == '${datasets_root}/license.md'
