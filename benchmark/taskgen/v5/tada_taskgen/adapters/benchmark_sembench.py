"""SemBench original Movie benchmark items, with original gold SQL kept private.

The local original sf_2000 workload supplies 116 movies and 2000 reviews. All
source rows, original natural-language questions and original SQL are read via
source_adapters.iter_units using explicit source-bound descriptors. SQL executes
with the original DuckDB engine against the original private coded columns.
Visible assets contain only movie id/title and review id/movie id/reviewText;
scoreSentiment, originalScore, reviewState, audienceScore and tomatoMeter remain
hidden. No platform rating is promoted to an expert record-level label.

Cells decode to the original query's complete result table (JSON rows). Checker
source: src/scenario/movie/evaluation/evaluate.py, src/evaluator/generic_evaluator.py,
src/aggregate_table_generator.py in the delivered SemBench repository. Oversize
answers remain subject to the unchanged compiler cell limit; no gold rows are
truncated to make an item eligible.
"""
from functools import lru_cache
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from local_config import public_text
import tempfile

from .benchmark_reader import DATA, UNIFIED, extended_units
from .base import REPO

SOURCE = DATA / 'raw/analytics_benchmarks/sembench/repo'
MOVIE = SOURCE / 'files/movie'
CHECKER_SOURCE = ('sembench/repo/src/scenario/movie/evaluation/evaluate.py; '
                  'sembench/repo/src/evaluator/generic_evaluator.py; '
                  'sembench/repo/src/aggregate_table_generator.py:extract_metrics')


def _text(path):
    rows = list(extended_units('sembench', [path], text_fields=('text',), format='text'))
    if len(rows) != 1:
        raise ValueError('Original query must be exactly one text unit')
    return rows[0]['raw_row']['text']


def _public_asset(name, rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows({f: row[f] for f in fields} for row in rows)
    content = stream.getvalue().encode()
    # Public content alone determines this materialization identity.
    folder = REPO / 'v5_runs/source_cache/sembench_public' / hashlib.sha256(content).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=folder, prefix='projection-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    elif path.read_bytes() != content:
        raise ValueError('SemBench public projection cache drift')
    return {'source_path': str(path), 'path': name, 'bytes': len(content)}


@lru_cache(maxsize=1)
def _load():
    import duckdb
    import pandas as pd
    source_rows = {}
    for name in ('Movies', 'Reviews'):
        path = MOVIE / 'data/sf_2000' / (name + '.csv')
        source_rows[name] = [u['raw_row'] for u in extended_units('sembench', [path], format='csv')]
    assets = [_public_asset('Movies.csv', source_rows['Movies'], ['id', 'title']),
              _public_asset('Reviews.csv', source_rows['Reviews'], ['id', 'reviewId', 'reviewText'])]
    # Match the original pandas CSV type inference after the canonical reader,
    # using an in-memory representation; source CSVs are never parsed ad hoc.
    connection = duckdb.connect(':memory:', config={'threads': 2, 'memory_limit': '1GB'})
    tables = []
    for name, rows in source_rows.items():
        table = pd.DataFrame(rows)
        buffer = io.StringIO()
        table.to_csv(buffer, index=False)
        buffer.seek(0)
        original_types = pd.read_csv(buffer)
        tables.append(original_types)
        connection.register(name, original_types)
    result = []
    try:
        for query in range(1, 11):
            question = _text(MOVIE / f'query/natural_language/Q{query}.txt')
            sql = _text(MOVIE / f'query/gold_sql/Q{query}.sql')
            answer = [list(row) for row in connection.execute(sql).fetchall()]
            # Reject NaN/non-JSON answers; do not fabricate missing ground truth.
            json.dumps(answer, allow_nan=False)
            result.append({'id': f'movie:sf_2000:Q{query}',
                           'text': question + '\nOriginal workload: movie sf_2000. Inputs: Movies.csv and Reviews.csv. '
                                   'Return a JSON array of result rows in the requested column order.',
                           'answer': answer, 'checker': {'kind': 'sembench_movie', 'query': query},
                           'assets': assets})
    finally:
        connection.close()
    return tuple(result)


def iter_items(dataset_id='sembench'):
    if dataset_id != 'sembench':
        raise ValueError('Only SemBench is supported')
    yield from _load()


def metadata(dataset_id='sembench'):
    profile = json.loads((UNIFIED / 'output/cards/sembench/profile.json').read_text())
    return {'status': 'available', 'checker_source': CHECKER_SOURCE,
            'licensing_notes': public_text(profile.get('license', '')),
            'license_evidence': public_text(profile.get('license_evidence', '')),
            'visible_fields': ['original query', 'Movies.id/title', 'Reviews.id/reviewId/reviewText'],
            'hidden_fields': ['gold SQL/result', 'scoreSentiment', 'originalScore', 'reviewState',
                              'audienceScore', 'tomatoMeter'],
            'limitations': ['Original local movie sf_2000 workload only; other scenarios lack an equivalent local answer binding.',
                            'Complete original query results exceeding 2000 cell characters are excluded by the common adapter.',
                            'Original quality formulas are retained, including 0.5 for invalid aggregation tables.']}
