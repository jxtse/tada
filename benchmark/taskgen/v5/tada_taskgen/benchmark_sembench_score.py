"""Dependency-free port of SemBench's original Movie result-table evaluation.

Sources: repo/src/scenario/movie/evaluation/evaluate.py; src/evaluator/
generic_evaluator.py; src/aggregate_table_generator.py:extract_metrics.
Retrieval uses official F1 (limit-aware where specified); aggregation uses the
official 1/(1+relative_error); ranking uses max(0, Spearman correlation).
JSON rows are the transport for the original CSV result tables.
"""
import json
import math


def _rows(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        return [[value]]
    return [list(row.values()) if isinstance(row, dict) else row if isinstance(row, list) else [row]
            for row in value]


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _f1(predicted, expected, limit=None):
    intersection = len(predicted & expected)
    precision = intersection / len(predicted) if predicted else 0.0
    denominator = min(limit, len(expected)) if limit else len(expected)
    recall = intersection / denominator if denominator else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            result[order[k]] = (i + j - 1) / 2 + 1
        i = j
    return result


def score_answer(predicted, expected, spec, root=None):
    """Return the original quality metric; malformed tables fail closed."""
    try:
        p, e = _rows(predicted), _rows(expected)
        q = int(spec['query'])
        if q in (1, 2):
            # IDs are native CSV columns; numeric ID text and integers have the
            # same CSV representation, as in pandas' upstream result loading.
            ps = {str(row[0]) for row in p[:5] if row and row[0] is not None}
            es = {str(row[0]) for row in e if row and row[0] is not None}
            return _f1(ps, es, 5)
        if q in (5, 6, 7):
            def pairs(rows):
                return {(str(r[0]), tuple(sorted((str(r[1]), str(r[2]))))) for r in rows
                        if len(r) >= 3 and all(v is not None for v in r[:3])}
            limit = 10 if q in (5, 6) else None
            return _f1(pairs(p[:limit] if limit else p), pairs(e), limit)
        if q in (3, 4):
            if len(p) != 1 or len(e) != 1:
                return 0.5  # Official invalid aggregate: relative_error = 1.
            pv = next((x for x in map(_number, p[0]) if x is not None), None)
            ev = next((x for x in map(_number, e[0]) if x is not None), None)
            if pv is None or ev is None:
                return 0.5
            error = abs(pv - ev) / abs(ev) if ev else (0.0 if pv == 0 else math.inf)
            return 1 / (1 + error)
        if q == 8:
            def counts(rows):
                return {str(r[0]).strip().upper(): _number(r[1]) for r in rows
                        if len(r) >= 2 and r[0] is not None and _number(r[1]) is not None}
            pc, ec = counts(p), counts(e)
            if not pc or not ec:
                return 0.5
            errors = []
            for label in pc.keys() | ec.keys():
                pv, ev = pc.get(label, 0.0), ec.get(label, 0.0)
                if ev:
                    errors.append(abs(pv - ev) / abs(ev))
                elif pv:
                    errors.append(1.0)
            return 1 / (1 + sum(errors) / len(errors)) if errors else 1.0
        if q in (9, 10):
            def scores(rows):
                return {str(r[0]): _number(r[1]) for r in rows
                        if len(r) >= 2 and r[0] is not None and _number(r[1]) is not None}
            ps, es = scores(p), scores(e)
            ids = sorted(ps.keys() & es.keys())
            if len(ids) < 2:
                return 0.0
            x, y = _ranks([ps[i] for i in ids]), _ranks([es[i] for i in ids])
            xm, ym = sum(x) / len(x), sum(y) / len(y)
            covariance = sum((a - xm) * (b - ym) for a, b in zip(x, y))
            scale = math.sqrt(sum((a - xm) ** 2 for a in x) * sum((b - ym) ** 2 for b in y))
            return max(0.0, min(1.0, covariance / scale)) if scale else 0.0
        raise ValueError('Unsupported original SemBench query')
    except (ValueError, TypeError, IndexError, KeyError, OverflowError):
        return 0.0
