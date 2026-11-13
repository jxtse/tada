"""Dataset combinations, admission quotas, and read-only legacy-state accounting."""
import hashlib
import itertools
import json
import random
from pathlib import Path


def normalize(sources):
    from .adapters import canonical_source
    return sorted({canonical_source(s) for s in sources if s})


def key(sources):
    return '+'.join(normalize(sources))


def enumerate_spaces(sources, maximum=3):
    if not 1 <= maximum <= 3:
        raise ValueError('Combination size must be within 1..3')
    ids = normalize(sources)
    rows = [list(row) for size in range(1, min(maximum, len(ids)) + 1)
            for row in itertools.combinations(ids, size)]
    payload = {'available_sources': ids, 'max_size': maximum,
               'combinations': [{'key': key(row), 'sources': row} for row in rows]}
    payload['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def verdict_sources(verdict, base=None):
    if verdict.get('sources'):
        return normalize(verdict['sources'])
    if verdict.get('combination'):
        return normalize(verdict['combination'].split('+'))
    spec = verdict.get('spec', {})
    candidates = []
    if base and verdict.get('id'):
        candidates.append(Path(base) / verdict['id'] / 'scenario_spec.json')
    if verdict.get('task'):
        candidates.append(Path(verdict['task']) / 'tests/scenario_spec.json')
    for path in candidates:
        if not (spec.get('sources') or spec.get('source')) and path.is_file():
            spec = json.loads(path.read_text())
    return normalize(spec.get('sources') or [spec.get('source') or verdict.get('source')])


def counts(state, enumeration, base=None):
    rows = {r['key']: {**r, 'admitted': 0, 'attempts': 0, 'reserved': 0} for r in enumeration['combinations']}
    for verdict in state.get('final_verdicts', []):
        combination = key(verdict_sources(verdict, base))
        if combination in rows:
            rows[combination]['attempts'] += 1
            rows[combination]['admitted'] += verdict.get('verdict') == 'admitted'
    for reservation in state.get('reservations', {}).values():
        combination = reservation.get('combination')
        if combination in rows:
            rows[combination]['attempts'] += 1
            rows[combination]['reserved'] += 1
    # Attempts are persisted independently, including infrastructure retries.
    for combination, attempted in state.get('combination_attempts', {}).items():
        if combination in rows:
            rows[combination]['attempts'] = max(rows[combination]['attempts'], attempted)
    return rows


def unmapped_verdicts(state, enumeration, base=None):
    keys = {row['key'] for row in enumeration['combinations']}
    unresolved = []
    for verdict in state.get('final_verdicts', []):
        sources = verdict_sources(verdict, base)
        if key(sources) not in keys:
            unresolved.append({'id': verdict.get('id'), 'verdict': verdict.get('verdict'),
                               'sources': sources, 'reason': 'unknown_source' if not sources else 'outside_enumeration'})
    return unresolved


def choose(rows, quota=2, seed=7, prefer_unused=False):
    eligible = [r for r in rows.values() if r['admitted'] + r['reserved'] < quota]
    if not eligible:
        return None
    order = sorted(rows)
    random.Random(seed).shuffle(order)
    tie = {name: index for index, name in enumerate(order)}
    used = {source for row in rows.values() if row['attempts'] or row['admitted'] or row.get('reserved')
            for source in row['sources']}
    return min(eligible, key=lambda r: ((-sum(s not in used for s in r['sources']),) if prefer_unused else ())
               + (r['admitted'], r['attempts'], tie[r['key']]))


def report(state, sources, maximum=3, quota=2, base=None):
    enumeration = enumerate_spaces(sources, maximum)
    return {**enumeration, 'quota_per_combination': quota, 'counts': list(counts(state, enumeration, base).values()),
            'unmapped_verdicts': unmapped_verdicts(state, enumeration, base), 'read_only': True}
