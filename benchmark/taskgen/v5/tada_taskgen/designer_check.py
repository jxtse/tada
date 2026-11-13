"""Format check for the four designer files, exposed to the designer as a tool.

Structural only: it never loads labels, records or gold, so its messages are safe to show the designer.
Usage: python -m tada_taskgen.designer_check <task_dir>  -> one JSON line {"ok": bool, "error": str|null}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def check(directory) -> dict:
    from tada_taskgen.schema import load_designer_output
    try:
        spec = load_designer_output(Path(directory))
    except (ValueError, KeyError, TypeError) as exc:
        message = str(exc) if not isinstance(exc, KeyError) else f'missing required field {exc}'
        return {'ok': False, 'error': message}
    try:
        from tada_taskgen.adapters import get_adapter
        from .designer_sources import spec_sources
        from tada_taskgen.adapters.base import disabled_questions
        adapters = {source: get_adapter(source) for source in spec_sources(spec)}
        known = {source: set(a.questions()) - disabled_questions(a.quarantine, a.source)
                 for source, a in adapters.items()}
    except Exception:  # adapter unavailable: structural result only
        known = None
    if known is not None:
        requested = spec['questions']
        if not isinstance(requested, dict):
            requested = {next(iter(known)): requested}
        unknown = [source + ':' + q for source, questions in requested.items()
                   for q in questions if q not in known.get(source, set())]
        if unknown:
            return {'ok': False, 'error': 'unknown question id(s): ' + ', '.join(unknown)}
    return {'ok': True, 'error': None}


if __name__ == '__main__':
    print(json.dumps(check(sys.argv[1])))
