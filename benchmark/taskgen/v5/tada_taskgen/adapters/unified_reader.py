"""Read-only access to the delivered unified reader, with repository-owned scratch.

No source parser is replaced. MDB extraction is delegated to the same iter_units
API in an isolated process; only its temporary output directory is relocated.
Private normalized rows never enter a solver workspace through this helper.
"""
import json
import os
import fcntl
import hashlib
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from local_config import path as local_path, require
UNIFIED = local_path('unified_root')
PYTHON = local_path('datasets_python')


def iter_author(dataset, **kwargs):
    require('datasets_root', 'unified_root')
    if str(UNIFIED) not in sys.path:
        sys.path.insert(0, str(UNIFIED))
    from source_adapters import iter_units
    yield from iter_units(dataset, projection='author', **kwargs)


def ntsb_relations():
    require('datasets_root', 'unified_root', 'datasets_python')
    scratch = REPO / 'v5_runs' / 'repair_r7' / 'tmp' / 'unified'
    scratch.mkdir(parents=True, exist_ok=True)
    cache = REPO / 'v5_runs' / 'source_cache' / 'ntsb_unified'
    cache.mkdir(parents=True, exist_ok=True)
    identity = ntsb_identity()
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path, receipt = cache / (key + '.jsonl'), cache / (key + '.json')
    with (cache / 'read.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved.get('identity') != identity or saved.get('sha256') != digest_file(path):
                raise ValueError('Private NTSB normalized cache identity/content mismatch')
        else:
            temp = path.with_suffix('.partial')
            try:
                with temp.open('w') as output:
                    for row in _ntsb_worker(scratch):
                        output.write(json.dumps(row, ensure_ascii=False) + '\n')
                temp.replace(path)
                receipt.write_text(json.dumps({'identity': identity, 'sha256': digest_file(path),
                                               'reader': 'source_adapters.iter_units'}, indent=2) + '\n')
            finally:
                temp.unlink(missing_ok=True)
        with path.open() as handle:
            for line in handle:
                yield json.loads(line)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ntsb_identity():
    require('unified_root')
    plan = UNIFIED / 'output/plans/ntsb_aviation_avall.json'
    descriptor = json.loads(plan.read_text())['raw'][0]
    archive = Path(descriptor['path'])
    stat = archive.stat()
    return {'plan_sha256': digest_file(plan), 'source_path': str(archive),
            'source_bytes': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
            'source_pin_level': 'original descriptor plus fresh stat; no new whole-archive digest claimed',
            'worker_sha256': digest_file(__file__),
            'reader_sha256': {str(p.relative_to(UNIFIED)): digest_file(p) for directory in ('source_adapters', 'tada_unified')
                              for p in sorted((UNIFIED / directory).glob('*.py'))}}


def _ntsb_worker(scratch):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(scratch))
    with subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--ntsb', str(scratch)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env) as process:
        try:
            for line in process.stdout:
                yield json.loads(line)
            error = process.stderr.read()
            if process.wait():
                raise RuntimeError('Unified NTSB reader failed: ' + error[-2000:])
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--ntsb':
        raise SystemExit('This worker reads only the original NTSB relational view')
    sys.path.insert(0, str(UNIFIED))
    # Import the API first, preserving its original source-plan/output defaults.
    from source_adapters import iter_units
    import source_adapters.integrated_reader
    import tada_unified.source_readers
    import tada_unified.common
    # mdb_rows imports common.OUT at call time solely for temporary extraction.
    # The process is isolated; source roots, descriptors, parsers and projections
    # are untouched, and no existing output receipt is overwritten.
    tada_unified.common.OUT = Path(sys.argv[2]).resolve()
    fields = {'events': ['ev_id', 'ev_highest_injury', 'light_cond', 'wx_cond_basic'],
              'aircraft': ['ev_id', 'Aircraft_Key', 'damage'],
              'narratives': ['ev_id', 'Aircraft_Key', 'narr_accp'],
              'Occurrences': ['ev_id', 'Aircraft_Key', 'Occurrence_No', 'Phase_of_Flight'],
              'eADMSPUB_DataDictionary': ['Column', 'Table', 'code_iaids', 'meaning', 'Question_Def']}
    for unit in iter_units('ntsb_aviation_avall', projection='author'):
        relation = unit['source_locator'].get('source_relation')
        if relation in fields:
            row = unit['raw_row']
            print(json.dumps({'relation': relation, 'unit_id': unit['unit_id'],
                              'row': {key: row.get(key) for key in fields[relation]}}, ensure_ascii=False))
