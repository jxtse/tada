"""Unified-reader bridge; metadata extensions never parse source records themselves.

Frozen plans omit DABstep's answered dev subset and Krama's workload questions.
Explicit local descriptors use the same source_adapters.iter_units API and its
safe-root/stat checks; source bytes and the frozen dataset directory stay read-only.
"""
from functools import lru_cache
import hashlib
import json
import os
import tempfile
from pathlib import Path
import sys
from .base import REPO

from local_config import path as local_path, require
DATA = local_path('datasets_root')
UNIFIED = local_path('unified_root')

def _reader(repair=False):
    require('datasets_root', 'unified_root')
    if str(UNIFIED) not in sys.path:
        sys.path.insert(0, str(UNIFIED))
    if repair:
        from source_adapters.repair_reader import iter_units
    else:
        from source_adapters import iter_units
    return iter_units

@lru_cache(maxsize=64)
def plan(dataset, repair=False):
    require('datasets_root', 'unified_root')
    path = (UNIFIED / 'continuations/20261104-data-repair/implementation/contracts' / (dataset+'.json')
            if repair else UNIFIED / 'output/plans' / (dataset+'.json'))
    data = json.loads(path.read_text())
    return data['descriptors'] if repair else data['raw']


def units(dataset, table=None, repair=False, limit=None):
    reader = _reader(repair)
    return reader(dataset, view='raw', table=table, projection='author', limit=limit)


def extended_units(dataset, paths, *, text_fields=(), label_fields=(), format=None):
    """Read explicitly selected omitted files through the canonical unified API."""
    require('datasets_root', 'unified_root')
    entry = json.loads((UNIFIED/'output/frozen/catalog'/(dataset+'.json')).read_text())
    base = Path(entry['local_path'])
    descs = []
    for path in paths:
        path = Path(path).resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError('Extended source must remain inside original dataset')
        stat = path.stat()
        descs.append({'table_id': hashlib.sha256(str(path).encode()).hexdigest()[:16],
                      'path': str(path), 'relative_path': str(path.relative_to(base)),
                      'format': format or ('jsonl' if path.suffix=='.jsonl' else 'json'),
                      'role': 'records', 'encoding': 'utf-8',
                      'observed_bytes': stat.st_size, 'observed_mtime_ns': stat.st_mtime_ns,
                      'text_fields': list(text_fields), 'label_fields': list(label_fields)})
    identity = hashlib.sha256(json.dumps(descs,sort_keys=True).encode()).hexdigest()[:20]
    out = REPO/'v5_runs/repair_r7/unified_extensions'/dataset/identity
    (out/'frozen/catalog').mkdir(parents=True, exist_ok=True)
    (out/'plans').mkdir(exist_ok=True)
    files = {out/'frozen/manifest.json': {'ids':[dataset]},
             out/'frozen/catalog'/(dataset+'.json'): entry,
             out/'plans'/(dataset+'.json'): {'raw':descs,'l0':[]}}
    for dest,value in files.items():
        encoded = json.dumps(value,ensure_ascii=False,sort_keys=True)
        if dest.exists() and dest.read_text()!=encoded: raise ValueError('Unified extension metadata drift')
        if not dest.exists():
            with tempfile.NamedTemporaryFile(mode='w',dir=dest.parent,prefix='.metadata-',delete=False) as f:
                temporary=Path(f.name);f.write(encoded)
            os.replace(temporary,dest)
    yield from _reader()(dataset, view='raw', projection='author', output=out)
