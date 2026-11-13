"""Machine-local settings: environment first, then a root-local TOML file.

Public code contains no machine defaults. Importing modules is data-independent;
operational callers require the settings they use. The file is resolved from this
module, not cwd, so restricted-environment subprocesses use the same settings.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parent


class ConfigurationError(ValueError):
    """A required machine-local setting has not been configured."""


def config_file():
    return Path(os.environ.get('TADA_LOCAL_PATHS_FILE') or ROOT / 'local_paths.toml').expanduser()


def _file_values():
    target = config_file()
    if not target.is_file():
        return {}
    try:
        with target.open('rb') as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError('Cannot read local_paths.toml: ' + str(exc)) from exc


def get(key, default=None, required=False):
    if not re.fullmatch(r'[a-z][a-z0-9_]*', key):
        raise ValueError('Invalid local configuration key: ' + str(key))
    variable = 'TADA_' + key.upper()
    value = os.environ.get(variable)
    if value is None:
        value = _file_values().get(key, default)
    if value is None or value == '':
        if required:
            raise ConfigurationError(f'Missing required configuration {key}: set {variable} or {key} in local_paths.toml')
        return default
    return value


def path(key, default=None, required=False):
    value = get(key, default, required)
    if value is None:
        # An inert, never-created location allows test collection without data.
        return ROOT / '.missing-config' / key
    result = Path(value).expanduser()
    return result if result.is_absolute() else ROOT / result


def configured(key):
    return get(key) is not None


def require(*keys):
    for key in keys:
        get(key, required=True)


def public_text(value):
    """Keep upstream provenance while replacing private local path prefixes."""
    text = str(value)
    for key in ('datasets_root', 'unified_root', 'legacy_cache', 'onet_root', 'quarantine'):
        configured_value = get(key)
        if configured_value:
            text = text.replace(str(configured_value), '${' + key + '}')
    # A source license receipt can reference a different machine's audit path.
    text = re.sub(r'/(?:Volumes|Users|home)/[^\s\"<>]+', '[local source provenance]', text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('key')
    parser.add_argument('--required', action='store_true')
    parser.add_argument('--path', action='store_true')
    args = parser.parse_args()
    try:
        value = path(args.key, required=args.required) if args.path else get(args.key, required=args.required)
    except ConfigurationError as exc:
        parser.exit(2, str(exc) + '\n')
    if value is not None:
        print(value)


if __name__ == '__main__':
    main()
