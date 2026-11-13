#!/usr/bin/env python3
"""Copy the reviewed task-generation source subset and reject private artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'local_config.py', 'local_config.ts', 'local_paths.example.toml', 'tada_taskgen/__init__.py',
    'controller.py', 'weekend_checks.py', 'weekend_replay.py', 'label_flip_check.py',
    'review_package.py', 'run_designer.py', 'harbor_templates.md', 'required_files.json',
    'tada_designer_tools.ts', 'tada_web_search.ts', 'designer_v5.md',
    'v5/README.md', 'v5/config.toml', 'v5/batch1_inventory.json', 'v5/compile_task.py',
    'v5/pyproject.toml', 'v5/uv.lock', 'tools/export_public.py',
)
PATTERNS = (
    'v5/tada_taskgen/*.py', 'v5/tada_taskgen/adapters/*.py',
    'v5/tada_taskgen/adapters/maud_conventions.json', 'v5/schemas/*.json',
    'v5/specs/*.json', 'v5/tests/*.py', 'sources/*.py', 'solver/*.py', 'solver/*.sh',
)
PUBLIC_DOMAINS = {
    'github.com', 'raw.githubusercontent.com', 'api.github.com', 'pypi.org',
    'files.pythonhosted.org', 'json-schema.org', 'www.w3.org', 'platform.openai.com',
    'docs.python.org', 'localhost', '127.0.0.1', '0.0.0.0', 'example.com',
    'provider.example', 'example.invalid', 'host', 'pypi.tuna.tsinghua.edu.cn',
    'scdb.wustl.edu', 'creativecommons.org',
}
FORBIDDEN_PARTS = {'cache', '.cache', 'model_cache', 'source_cache', '__pycache__', '.pytest_cache', '.uv-cache', '.venv', 'node_modules',
                   'private_cache', 'pi_agent_dir', 'v5_runs', 'WEEKEND_20261107', 'directions', 'packs'}
PATTERNS_PRIVATE = {
    'absolute_user_or_volume_path': re.compile(r'/(?:Users|Volumes|home|private/var)/[^\s\"\'<>]+'),
    'private_trial_hostname': re.compile(r'\bgu\d+\b', re.I),
    'credential_like_string': re.compile(r'(?:\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}|\bgh[pousr]_[A-Za-z0-9]{20,}|\bAKIA[0-9A-Z]{16}|\bxox[baprs]-[A-Za-z0-9-]{16,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)'),
}
URL = re.compile(r'https?://[^\s\"\'<>`\)\]}]+')
PUBLIC_IGNORE = '''local_paths.toml
v5_runs/
runs*/
private_cache/
node_modules/
pi_agent_dir/
__pycache__/
*.py[cod]
*.bak*
.venv/
.uv-cache/
.pytest_cache/
*.egg-info/
'''


def allowlist(root=ROOT):
    """Unknown top-level files and all run artifacts are excluded by construction."""
    root = Path(root)
    selected = {root / name for name in FILES}
    selected.update(p for pattern in PATTERNS for p in root.glob(pattern))
    for path in sorted(selected):
        if not path.is_file() or path.is_symlink():
            raise ValueError('Required public source is missing or a symlink: ' + str(path.relative_to(root)))
        if '.bak' in path.name or any(part in FORBIDDEN_PARTS for part in path.relative_to(root).parts):
            raise ValueError('Forbidden artifact matched allowlist: ' + str(path.relative_to(root)))
    return sorted(selected)


def scan(root):
    """Return only locations and rule names, never the matched private value."""
    root = Path(root)
    findings = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if path.is_symlink() or any(part in FORBIDDEN_PARTS or part.startswith('runs') for part in relative.parts) or '.bak' in path.name:
            findings.append({'path': str(relative), 'rule': 'forbidden_artifact'})
        if not path.is_file():
            continue
        if path.name == 'local_paths.toml' or path.name.startswith(('CODEX_', 'lessons_')) or path.name in {'IMPLEMENTATION_REPORT.md', 'PIPELINE_V5_DESIGN.md', 'BENCH_CONSTRUCTION_REVIEW.md', 'designer_pi.md'}:
            findings.append({'path': str(relative), 'rule': 'internal_file'})
        content = path.read_text(encoding='utf-8')
        for rule, pattern in PATTERNS_PRIVATE.items():
            for match in pattern.finditer(content):
                findings.append({'path': str(relative), 'line': content.count('\n', 0, match.start()) + 1, 'rule': rule})
        for match in URL.finditer(content):
            value = match.group().rstrip('.,;:')
            parsed = urlparse(value)
            host = parsed.hostname or ''
            if parsed.username or parsed.password or not (host in PUBLIC_DOMAINS or host.endswith('.example') or host.endswith('.invalid') or host.endswith('.example.org')):
                findings.append({'path': str(relative), 'line': content.count('\n', 0, match.start()) + 1, 'rule': 'private_url'})
    return findings


def export(out, root=ROOT):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or root.is_relative_to(out):
        raise ValueError('Export target must not contain the working source tree')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Export target must be empty; preserve or remove the previous generated export explicitly')
    selected = allowlist(root)
    readme = root / 'taskgen/README.md'
    if not readme.is_file():
        # The exported copy has its public README at its root and can export itself.
        readme = root / 'README.md'
    if not readme.is_file():
        raise ValueError('Public README is missing')
    out.mkdir(parents=True, exist_ok=True)
    for source in selected:
        destination = out / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(readme, out / 'README.md')
    (out / '.gitignore').write_text(PUBLIC_IGNORE)
    hits = scan(out)
    files = [p for p in out.rglob('*') if p.is_file()]
    result = {'files': len(files), 'bytes': sum(p.stat().st_size for p in files), 'scan_hits': hits,
              'top_level': sorted(p.name + ('/' if p.is_dir() else '') for p in out.iterdir())}
    print(json.dumps(result, indent=2))
    if hits:
        raise ValueError('Public export rejected: ' + str(len(hits)) + ' scan findings')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    try:
        export(args.out)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
