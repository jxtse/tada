"""Independent finite-population protocol, paired evaluation and budget checks."""
from collections import Counter
import gzip
import hashlib
import itertools
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from pydantic import ValidationError

from tada_operators import retention_evaluation as evaluation
from tada_runtime.retention import NumericBudgetExceeded, RetentionConfig, paired_recovery


def tiny_config(**kwargs):
    return evaluation.EvaluationConfig(populations=[1000], regimes=['separated'], trials=3,
        planner=RetentionConfig(outer_draws=2, inner_draws=2, r_max=10, no_tail_budget=10), **kwargs)


def metric(r, successes, trials=100):
    low, high = evaluation.wilson_interval(successes, trials)
    return evaluation.RecoveryMetric(r=r, successes=successes, trials=trials,
        recovery=successes/trials, wilson_95_low=low, wilson_95_high=high,
        candidate_volume=2*r, mean_retained_fraction_by_category=[1.0])


def test_all_six_exact_weight_families_and_largest_remainder():
    expected_gaps = [0.061891, 0.002293, 0.001963, 0.001634, 0.001306, 0.001175]
    for regime, gap in zip(evaluation.REGIMES, expected_gaps):
        population = evaluation.fixed_population(regime, 32000)
        assert len(population.counts) == 20
        assert sum(population.counts) == 32000
        assert population.weights[:5] == [2.00, 1.75, 1.50, 1.25, 1.00]
        assert abs(population.probabilities[4] - population.probabilities[5] - gap) < 0.0000006
        exact = np.asarray(population.probabilities) * 32000
        assert np.all(np.abs(np.asarray(population.counts) - exact) < 1)
        # Fractional remainders selected for an extra row dominate every omitted one.
        rounded_up = np.asarray(population.counts) > np.floor(exact)
        fractions = exact - np.floor(exact)
        assert fractions[rounded_up].min() >= fractions[~rounded_up].max()
    assert evaluation.regime_weights('separated')[6:] == [0.30*0.85**j for j in range(14)]
    assert evaluation.regime_weights('narrow_5')[5:8] == [.982, .972, .962]
    assert evaluation.regime_weights('narrow_1')[8:] == [.75*.88**j for j in range(12)]
    assert evaluation._largest_remainder([.25]*4, 2) == [1, 1, 0, 0]
    with pytest.raises(ValueError, match='unknown'):
        evaluation.fixed_population('invented', 32000)


def test_uniform_without_replacement_sampler_matches_enumerated_small_population():
    # Every two-row subset is equiprobable. Repeated labels have the expected
    # hypergeometric multiplicity; the second chunk is the exact remainder.
    labels = [0, 0, 1, 2]
    exact = Counter(tuple(labels[i] for i in selected) for selected in itertools.combinations(range(4), 2))
    expected_counts = Counter()
    for labels_in_chunk, multiplicity in exact.items():
        expected_counts[tuple(labels_in_chunk.count(c) for c in range(3))] += multiplicity
    rng = np.random.default_rng(174)
    observed = Counter()
    trials = 6000
    for _ in range(trials):
        table = evaluation.sample_fixed_partition([2, 1, 1], [2, 2], rng)
        assert table.sum(axis=0).tolist() == [2, 1, 1]
        assert table.sum(axis=1).tolist() == [2, 2]
        observed[tuple(table[0])] += 1
    assert set(observed) == set(expected_counts)
    for counts, combinations in expected_counts.items():
        assert abs(observed[counts]/trials - combinations/6) < .025


def test_balanced_count_sampling_reproducible_and_margins_fixed():
    counts = [100, 20, 0, 11]
    sizes = [33, 33, 33, 32]
    a = evaluation.sample_fixed_partition(counts, sizes, np.random.default_rng(12))
    b = evaluation.sample_fixed_partition(counts, sizes, np.random.default_rng(12))
    assert np.array_equal(a, b)
    assert a.sum(axis=0).tolist() == counts
    assert a.sum(axis=1).tolist() == sizes
    assert evaluation.sample_fixed_partition([1, 0], [1], np.random.default_rng(9)).tolist() == [[1, 0]]
    with pytest.raises(ValidationError):
        evaluation.sample_fixed_partition([True, 2], [3], np.random.default_rng())
    with pytest.raises(ValidationError, match='exactly cover'):
        evaluation.sample_fixed_partition([1, 2], [2], np.random.default_rng())


def test_independent_retention_accumulator_matches_literal_reference_and_planner():
    table = np.asarray([[10, 9, 9, 8, 6, 7, 3, 0], [9, 8, 6, 5, 8, 6, 1, 1],
                        [8, 8, 6, 7, 7, 6, 0, 2]], dtype=np.int64)
    candidates = [5, 6, 7, 8]
    reference = np.sort(np.argsort(-table.sum(axis=0), kind='stable')[:5])
    successes, retained = evaluation._paired_results(table, reference, 5, candidates)
    planner = paired_recovery(table, 5, candidates)
    for i, r in enumerate(candidates):
        sums = [0] * table.shape[1]
        for row in table.tolist():
            for category in sorted(range(len(row)), key=lambda c: (-row[c], c))[:r]:
                sums[category] += row[category]
        top = sorted(sorted(range(len(sums)), key=lambda c: (-sums[c], c))[:5])
        assert sums == retained[i].tolist()
        assert bool(successes[i]) == (top == reference.tolist()) == planner[r]
    # Grid order/subsetting changes neither count table nor results for shared r.
    subset, _ = evaluation._paired_results(table, reference, 5, [8, 5])
    assert subset.tolist() == [successes[-1], successes[0]]
    assert successes[-1]


def test_stable_tail_oracle_is_not_individual_threshold_crossing():
    curve = [metric(5, 96), metric(6, 90), metric(7, 97), metric(8, 100)]
    assert min(m.r for m in curve if m.recovery >= .95) == 5
    assert evaluation.stable_tail_oracle(curve) == 7
    assert evaluation.stable_tail_oracle([metric(5, 90)]) is None
    assert evaluation.stable_tail_oracle([]) is None
    with pytest.raises(ValueError, match='repeat'):
        evaluation.stable_tail_oracle([metric(5, 96), metric(5, 96)])


def test_named_seeds_and_uncertainty_conventions():
    seeds = [evaluation.named_seed(9, 'same-case', stage) for stage in ('pilot', 'planner', 'evaluation')]
    assert len(set(seeds)) == 3
    assert seeds[0] == evaluation.named_seed(9, 'same-case', 'pilot')
    assert seeds[0] != evaluation.named_seed(9, 'other-case', 'pilot')
    low, high = evaluation.wilson_interval(5000, 5000)
    assert .999 < low < 1 and high == 1
    low, high = evaluation.wilson_interval(0, 5000)
    assert low == 0 and 0 < high < .001
    low, high = evaluation.wilson_interval(2500, 5000)
    assert abs(low + high - 1) < 1e-12
    for n in (1, 2, 3, 10, 956, 5000):
        assert evaluation.wilson_interval(n, n)[1] == 1.0
        assert evaluation.wilson_interval(0, n)[0] == 0.0


def test_actual_small_run_pilot_only_planner_and_replayable_fixed_tables(tmp_path, monkeypatch):
    config = tiny_config()
    original = evaluation.plan_retention
    seen = []
    def spy(pilot, **kwargs):
        seen.append((pilot, set(kwargs)))
        return original(pilot, **kwargs)
    monkeypatch.setattr(evaluation, 'plan_retention', spy)
    report = evaluation.run_evaluation(config, tmp_path)
    assert report.status == 'completed' and len(report.cases) == 1
    case = report.cases[0]
    assert len(seen) == 1
    assert seen[0][1] == {'population_size', 'target_chunk_rows', 'k', 'config', 'check_cancel'}
    assert seen[0][0] == case.pilot == case.planner_decision.request.pilot
    assert sum(case.pilot.counts) == config.pilot_rows
    assert all(m.recovery == 1 for m in case.recovery_curve)  # B=1 exact case
    assert case.comparisons.selected_minus_naive == 0
    assert case.python_peak_allocated_bytes > 0 and case.process_peak_rss_bytes > 0
    raw = gzip.decompress((tmp_path / case.evidence_file).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == case.count_tables_sha256
    records = [evaluation.TrialRecord.model_validate_json(line, context={
        'counts': case.population.counts, 'sizes': case.geometry.sizes, 'candidates': config.evaluation_candidates})
        for line in raw.splitlines()]
    assert [record.trial for record in records] == list(range(config.trials))
    assert evaluation.EvaluationReport.model_validate_json((tmp_path / 'report.json').read_text()) == report
    with pytest.raises(FileExistsError):
        evaluation.run_evaluation(config, tmp_path)


def test_count_cell_budget_and_configuration_are_strict():
    with pytest.raises(ValidationError, match='count-cell budget'):
        evaluation.EvaluationConfig(max_count_cells=10)
    with pytest.raises(ValidationError):
        evaluation.EvaluationConfig(trials=True)
    with pytest.raises(ValidationError, match='complete unsmoothed'):
        evaluation.EvaluationConfig(evaluation_candidates=[5, 10, 20])
    with pytest.raises(ValidationError, match='online budget'):
        evaluation.EvaluationConfig(planner=RetentionConfig(r_max=20))


def test_planner_sees_sampled_pilot_not_full_population(tmp_path, monkeypatch):
    seen = []
    original = evaluation.plan_retention
    def spy(pilot, **kwargs):
        seen.append(pilot.model_dump())
        assert sum(pilot.counts) == 1000 < kwargs['population_size']
        assert 'counts' not in kwargs and 'reference' not in kwargs and 'gold' not in kwargs
        return original(pilot, **kwargs)
    monkeypatch.setattr(evaluation, 'plan_retention', spy)
    config = evaluation.EvaluationConfig(populations=[2000], regimes=['narrow_5'], trials=1,
        planner=RetentionConfig(outer_draws=2, inner_draws=2, r_max=10, no_tail_budget=10))
    report = evaluation.run_evaluation(config, tmp_path)
    assert report.status == 'completed'
    case = report.cases[0]
    assert seen == [case.pilot.model_dump()]
    assert case.pilot.counts != case.population.counts


def test_cpu_budget_and_wall_budget_are_distinct_and_prevent_work(tmp_path, monkeypatch):
    budget = evaluation.EvaluationBudget(1.0, 100.0)
    monkeypatch.setattr(evaluation.time, 'process_time', lambda: budget.cpu_started + 2)
    with pytest.raises(NumericBudgetExceeded, match='CPU'):
        budget.check()
    monkeypatch.undo()
    budget = evaluation.EvaluationBudget(100.0, 1.0)
    monkeypatch.setattr(evaluation.time, 'perf_counter', lambda: budget.wall_started + 2)
    with pytest.raises(NumericBudgetExceeded, match='wall-time'):
        budget.check()
    monkeypatch.undo()
    report = evaluation.run_evaluation(tiny_config(max_cpu_seconds=1e-12), tmp_path)
    assert report.status == 'budget_exhausted'
    assert report.cases[0].completed_trials == 0
    assert report.cases[0].planner_decision is None
    assert report.cases[0].evidence_file is None


def test_cancel_midstream_preserves_partial_trial_evidence_and_stops_sampler(tmp_path, monkeypatch):
    stop = threading.Event()
    calls = 0
    original = evaluation._sample_partition
    def sampling(*args):
        nonlocal calls
        calls += 1
        result = original(*args)
        if calls == 2:
            stop.set()
        return result
    def cancel():
        if stop.is_set():
            raise evaluation.EvaluationCancelled('test cancellation')
    monkeypatch.setattr(evaluation, '_sample_partition', sampling)
    report = evaluation.run_evaluation(tiny_config(), tmp_path, check_cancel=cancel)
    assert report.status == 'cancelled'
    assert calls == 2 and report.cases[0].completed_trials == 2
    case = report.cases[0]
    records = gzip.decompress((tmp_path / case.evidence_file).read_bytes()).splitlines()
    assert len(records) == 2
    assert all(m.trials == 2 for m in case.recovery_curve)
    assert case.error == 'test cancellation'


def test_trial_margin_validator_rejects_a_resampled_population():
    with pytest.raises(ValidationError, match='changed the fixed population'):
        evaluation.TrialRecord.model_validate({'case_id': 'test', 'trial': 0,
            'counts': [[1, 2], [1, 2]], 'successes': [True]},
            context={'sizes': [3, 3], 'counts': [3, 3], 'candidates': [5]})


def test_cli_signal_cancel_writes_partial_receipt_and_closes_stream(tmp_path):
    destination = tmp_path / 'signal-cancel'
    process = subprocess.Popen([sys.executable, '-m', 'tada_operators.retention_evaluation',
        '--output', str(destination), '--populations', '2000', '--regimes', 'separated',
        '--trials', '100000', '--max-cpu-seconds', '30', '--max-wall-seconds', '45'],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        until = time.monotonic() + 10
        evidence = destination / 'separated-N2000-n1000.count-tables.jsonl.gz'
        while time.monotonic() < until:
            if evidence.exists() and evidence.stat().st_size > 0:
                break
            assert process.poll() is None
            time.sleep(.02)
        else:
            pytest.fail('evaluation did not begin streaming evidence')
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, stderr
        report = evaluation.EvaluationReport.model_validate_json((destination / 'report.json').read_text())
        assert report.status == 'cancelled'
        case = report.cases[0]
        assert 0 < case.completed_trials < case.requested_trials
        assert len(gzip.decompress(evidence.read_bytes()).splitlines()) == case.completed_trials
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
