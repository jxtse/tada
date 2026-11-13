"""SemBench original result-table metrics and label-free item assets."""
import ast
import dataclasses
import json
from pathlib import Path
import pytest

from tada_taskgen.benchmark_sembench_score import score_answer


@pytest.mark.parametrize('query,answer,bad', [
    (1, [[1], [2]], [[99]]), (2, [[1], [2]], [[99]]),
    (3, [[14]], [[15]]), (4, [[0.25]], [[0.5]]),
    (5, [['movie', 1, 2], ['movie', 1, 3]], [['other', 1, 2]]),
    (6, [['movie', 1, 2], ['movie', 1, 3]], [['other', 1, 2]]),
    (7, [['movie', 1, 2], ['movie', 1, 3]], [['other', 1, 2]]),
    (8, [['POSITIVE', 14], ['NEGATIVE', 106]], [['POSITIVE', 20]]),
    (9, [[1, 1], [2, 2], [3, 2]], [[1, 3], [2, 2], [3, 1]]),
    (10, [['a', 30], ['b', 10], ['c', 50]], [['a', 30], ['b', 50], ['c', 10]]),
])
def test_gold_and_perturbation(query, answer, bad):
    spec = {'kind': 'sembench_movie', 'query': query}
    assert score_answer(answer, answer, spec) == 1.0
    assert score_answer(bad, answer, spec) < 1.0


def test_original_invalid_aggregation_and_limit_rules():
    assert score_answer([], [[14]], {'query': 3}) == 0.5
    assert score_answer([[0]], [[0]], {'query': 3}) == 1.0
    assert score_answer([[1]], [[0]], {'query': 3}) == 0.0
    assert score_answer([[1]] * 5, [[1], [2], [3], [4], [5]], {'query': 2}) == pytest.approx(1 / 3)
    assert score_answer([['m', 2, 1]], [['m', 1, 2]], {'query': 7}) == 1.0


def test_all_local_items_keep_gold_out_of_assets():
    from tada_taskgen.adapters.benchmark_sembench import iter_items
    items = list(iter_items())
    assert len(items) == 10
    assert {v['id'] for v in items} == {f'movie:sf_2000:Q{q}' for q in range(1, 11)}
    eligible = [v for v in items if len(json.dumps(v['answer'], separators=(',', ':'))) <= 2000]
    assert [v['checker']['query'] for v in eligible] == [2, 3, 4, 8]
    for value in eligible:
        assert score_answer(value['answer'], value['answer'], value['checker']) == 1.0
        assert score_answer('__perturbed__', value['answer'], value['checker']) < 1.0
        for asset in value['assets']:
            header = Path(asset['source_path']).read_text().splitlines()[0]
            assert not any(secret in header for secret in ['scoreSentiment', 'originalScore', 'reviewState',
                                                           'audienceScore', 'tomatoMeter'])


def test_port_matches_original_upstream_evaluator():
    """Execute only upstream pure metric definitions; no source writes/imports."""
    import pandas as pd
    from tada_taskgen.adapters.benchmark_sembench import SOURCE
    scope = {'dataclass': dataclasses.dataclass, 'pd': pd}
    source = ast.parse((SOURCE / 'src/evaluator/generic_evaluator.py').read_text())
    keep = [n for n in source.body if isinstance(n, ast.ClassDef) and n.name in
            {'QueryMetricAggregation', 'QueryMetricRetrieval', 'QueryMetricRank'}]
    # Future annotations remove irrelevant imported typing/dependency requirements.
    code = 'from __future__ import annotations\n' + '\n'.join(ast.unparse(n) for n in keep)
    exec(code, scope)
    methods = []
    for path, cname in [('src/evaluator/generic_evaluator.py', 'GenericEvaluator'),
                        ('src/scenario/movie/evaluation/evaluate.py', 'MovieEvaluator')]:
        tree = ast.parse((SOURCE / path).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cname)
        methods += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                    {'_generic_aggregation_evaluation', '_generic_retrieval_limit_evaluation',
                     '_evaluate_sentiment_counts', '_evaluate_review_pairs', '_evaluate_review_pairs_with_limit'}]
    for method in methods:
        exec('from __future__ import annotations\n' + ast.unparse(method), scope)
    cases = [(3, [[12]], [[14]], '_generic_aggregation_evaluation', {}),
             (3, [], [[14]], '_generic_aggregation_evaluation', {}),
             (2, [[1], [9]], [[1], [2], [3]], '_generic_retrieval_limit_evaluation', {'limit': 5}),
             (8, [['positive', 8], ['NEGATIVE', 7]], [['POSITIVE', 10], ['NEGATIVE', 5]], '_evaluate_sentiment_counts', {}),
             (6, [['m', 1, 2]], [['m', 2, 1], ['m', 1, 3]], '_evaluate_review_pairs_with_limit', {'limit': 10})]
    for q, p, e, method, kwargs in cases:
        original = scope[method](None, pd.DataFrame(p), pd.DataFrame(e), **kwargs)
        expected = 1 / (1 + original.relative_error) if hasattr(original, 'relative_error') else original.f1_score
        assert score_answer(p, e, {'query': q}) == pytest.approx(expected)
