"""Bounded, provider-free evaluation of fixed-population count retention.

The six weight families and evaluation geometry follow the frozen manuscript.
Seeds, largest-remainder ties, confidence intervals and the count-only sampler
are disclosed engineering choices. This is not a reproduction of author seeds
or a test of open-ended semantic discovery quality.

Run ``python -m tada_operators.retention_evaluation --output DIR``. Each pilot is
sampled independently of held-out partitions; only pilot counts enter the online
planner. Compressed count tables retain every evaluated trial without keeping the
simulation history in memory. No model/provider integration is imported here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import math
from pathlib import Path
import platform
import resource
import signal
import sys
import threading
import time
import tracemalloc
from typing import Callable, Literal

import numpy as np
from pydantic import Field, StrictInt, ValidationInfo, model_validator

from tada_runtime.audit_contracts import AuditModel
from tada_runtime.retention import (
    BalancedGeometry, NumericBudgetExceeded, PilotCounts, RetentionConfig,
    RetentionDecision, balanced_geometry, plan_retention,
)

VERSION = 'fixed-population-retention-evaluation-v1'
REGIMES = ['separated', 'narrow_1', 'narrow_2', 'narrow_3', 'narrow_4', 'narrow_5']


class FixedPopulation(AuditModel):
    regime: str
    population_size: int = Field(ge=1, le=1_000_000_000)
    category_ids: list[str]
    weights: list[float]
    probabilities: list[float]
    counts: list[StrictInt]
    rounding: Literal['largest-remainder-canonical-tie-v1'] = 'largest-remainder-canonical-tie-v1'

    @model_validator(mode='after')
    def population_contract(self):
        C = len(self.category_ids)
        if not C or len(set(self.category_ids)) != C or any(not c for c in self.category_ids):
            raise ValueError('canonical population IDs must be nonempty and unique')
        if len(self.weights) != C or len(self.probabilities) != C or len(self.counts) != C:
            raise ValueError('population arrays must align')
        if any(w <= 0 for w in self.weights) or any(p <= 0 for p in self.probabilities):
            raise ValueError('weights and probabilities must be positive')
        if any(n < 0 for n in self.counts) or sum(self.counts) != self.population_size:
            raise ValueError('fixed counts must be nonnegative and sum to N')
        expected = np.asarray(self.weights, dtype=float) / sum(self.weights)
        if not np.allclose(self.probabilities, expected, rtol=0, atol=1e-14):
            raise ValueError('probabilities must be normalized weights')
        if self.counts != _largest_remainder(self.probabilities, self.population_size):
            raise ValueError('integer population differs from disclosed rounding policy')
        return self


class PartitionRequest(AuditModel):
    counts: list[StrictInt] = Field(min_length=1)
    sizes: list[StrictInt] = Field(min_length=1)

    @model_validator(mode='after')
    def margins(self):
        if any(n < 0 for n in self.counts) or any(n <= 0 for n in self.sizes):
            raise ValueError('counts must be nonnegative and chunk sizes positive')
        if sum(self.counts) != sum(self.sizes):
            raise ValueError('chunk sizes must exactly cover the fixed population')
        return self


class EvaluationConfig(AuditModel):
    populations: list[StrictInt] = Field(default_factory=lambda: [2000, 8000, 32000])
    regimes: list[str] = Field(default_factory=lambda: list(REGIMES))
    target_chunk_rows: int = Field(default=1000, ge=1)
    pilot_rows: int = Field(default=1000, ge=1)
    k: Literal[5] = 5
    evaluation_candidates: list[StrictInt] = Field(default_factory=lambda: list(range(5, 21)))
    trials: int = Field(default=5000, ge=1, le=100000)
    target: float = Field(default=0.95, gt=0, le=1)
    master_seed: int = Field(default=20261110, ge=0, le=2**63-1)
    planner: RetentionConfig = Field(default_factory=lambda: RetentionConfig(r_max=10, no_tail_budget=10))
    max_cpu_seconds: float = Field(default=600.0, gt=0, le=7200)
    max_wall_seconds: float = Field(default=900.0, gt=0, le=10800)
    max_count_cells: int = Field(default=100_000_000, ge=1, le=1_000_000_000)
    version: Literal['fixed-population-retention-evaluation-v1'] = VERSION
    sampler: Literal['sequential-multivariate-hypergeometric-v1'] = 'sequential-multivariate-hypergeometric-v1'
    seed_policy: Literal['sha256-named-independent-PCG64-v1'] = 'sha256-named-independent-PCG64-v1'
    tie_policy: Literal['canonical-domain-order-v1'] = 'canonical-domain-order-v1'

    @model_validator(mode='after')
    def bounded_domain(self):
        if not self.populations or len(set(self.populations)) != len(self.populations):
            raise ValueError('population scales must be nonempty and unique')
        if any(n < self.pilot_rows or n > 1_000_000_000 for n in self.populations):
            raise ValueError('each scale must cover the pilot and remain bounded')
        if not self.regimes or len(set(self.regimes)) != len(self.regimes) or any(r not in REGIMES for r in self.regimes):
            raise ValueError('unknown or duplicate weight regime')
        if self.evaluation_candidates != list(range(self.k, 21)):
            raise ValueError('evaluation must include the complete unsmoothed r=5..20 grid')
        if self.planner.r_max != 10 or self.planner.no_tail_budget != 10:
            raise ValueError('online budget must be 5..10, separate from evaluation 5..20')
        if self.planner.candidates not in (None, list(range(5, 11))):
            raise ValueError('online candidate grid must be 5..10')
        if self.planner.target != self.target:
            raise ValueError('online and held-out threshold must match')
        cells = sum(((n + self.target_chunk_rows - 1) // self.target_chunk_rows) * 20
                    for n in self.populations) * len(self.regimes) * self.trials
        if cells > self.max_count_cells:
            raise ValueError('requested fixed-population evaluation exceeds count-cell budget')
        return self


class TrialRecord(AuditModel):
    case_id: str
    trial: int = Field(ge=0)
    counts: list[list[StrictInt]]
    successes: list[bool]

    @model_validator(mode='after')
    def fixed_margins(self, info: ValidationInfo):
        if not self.counts or not self.counts[0]:
            raise ValueError('trial count table must be nonempty')
        C = len(self.counts[0])
        if any(len(row) != C or any(n < 0 for n in row) for row in self.counts):
            raise ValueError('trial count table must be rectangular and nonnegative')
        context = info.context or {}
        if 'sizes' in context and [sum(row) for row in self.counts] != context['sizes']:
            raise ValueError('trial row totals do not match balanced chunk sizes')
        if 'counts' in context and [sum(row[j] for row in self.counts) for j in range(C)] != context['counts']:
            raise ValueError('trial column totals changed the fixed population')
        if 'candidates' in context and len(self.successes) != len(context['candidates']):
            raise ValueError('trial must have one paired result for every candidate')
        return self


class RecoveryMetric(AuditModel):
    r: int = Field(ge=5, le=20)
    successes: int = Field(ge=0)
    trials: int = Field(ge=1)
    recovery: float = Field(ge=0, le=1)
    wilson_95_low: float = Field(ge=0, le=1)
    wilson_95_high: float = Field(ge=0, le=1)
    candidate_volume: int = Field(ge=1)
    mean_retained_fraction_by_category: list[float]

    @model_validator(mode='after')
    def metric_contract(self):
        if self.successes > self.trials or self.recovery != self.successes / self.trials:
            raise ValueError('recovery must be the empirical success frequency')
        if not self.wilson_95_low <= self.recovery <= self.wilson_95_high:
            raise ValueError('uncertainty interval must contain the point estimate')
        if any(not 0 <= value <= 1 for value in self.mean_retained_fraction_by_category):
            raise ValueError('retained fractions must lie in [0,1]')
        return self


class PairedComparison(AuditModel):
    selected_r: int = Field(ge=5, le=10)
    naive_r: Literal[5] = 5
    retain_all_r: Literal[20] = 20
    selected_minus_naive: float = Field(ge=-1, le=1)
    paired_standard_error: float = Field(ge=0)
    normal_95_low: float
    normal_95_high: float
    selected_only_successes: int = Field(ge=0)
    naive_only_successes: int = Field(ge=0)
    note: str = 'Normal Monte Carlo interval on paired trial differences; conditional on this one pilot and fixed population.'


class EvaluationCase(AuditModel):
    case_id: str
    status: Literal['completed', 'cancelled', 'budget_exhausted', 'failed']
    error: str | None = None
    population: FixedPopulation
    geometry: BalancedGeometry
    pilot: PilotCounts
    seeds: dict[str, StrictInt]
    planner_decision: RetentionDecision | None
    full_population_top_k: list[str]
    probability_cutoff_gap: float
    integer_population_cutoff_gap: float
    requested_trials: int = Field(ge=1)
    completed_trials: int = Field(ge=0)
    recovery_curve: list[RecoveryMetric]
    comparisons: PairedComparison | None
    held_out_stable_tail_r: int | None = Field(default=None, ge=5, le=20)
    oracle_scope: Literal['held-out-evaluation-only-never-planner-input'] = 'held-out-evaluation-only-never-planner-input'
    evidence_file: str | None
    count_tables_sha256: str | None
    timings_seconds: dict[str, float]
    python_peak_allocated_bytes: int = Field(ge=0)
    process_peak_rss_bytes: int = Field(ge=0)
    memory_note: str = 'tracemalloc measures Python allocations in this case; ru_maxrss is cumulative process peak RSS, not per-case RSS.'

    @model_validator(mode='after')
    def result_contract(self):
        if self.completed_trials > self.requested_trials:
            raise ValueError('completed count cannot exceed requested trials')
        if self.status == 'completed' and self.completed_trials != self.requested_trials:
            raise ValueError('complete status requires all requested partitions')
        if any(m.trials != self.completed_trials for m in self.recovery_curve):
            raise ValueError('curve trials must agree with actual completed count')
        if self.completed_trials and [m.r for m in self.recovery_curve] != list(range(5, 21)):
            raise ValueError('completed trial curves must retain all r=5..20')
        if self.recovery_curve and self.recovery_curve[-1].recovery != 1.0:
            raise ValueError('retain-all must exactly recover the full-population reference')
        if self.planner_decision and self.planner_decision.request.pilot != self.pilot:
            raise ValueError('planner decision must identify the actual pilot')
        if any(value < 0 for value in self.timings_seconds.values()):
            raise ValueError('timings must be nonnegative')
        return self


class EvaluationReport(AuditModel):
    version: Literal['fixed-population-retention-evaluation-v1'] = VERSION
    status: Literal['completed', 'cancelled', 'budget_exhausted', 'failed']
    config: EvaluationConfig
    cases: list[EvaluationCase]
    remaining_case_ids: list[str]
    elapsed_wall_seconds: float = Field(ge=0)
    elapsed_cpu_seconds: float = Field(ge=0)
    environment: dict[str, str]
    notes: list[str]

    @model_validator(mode='after')
    def complete_means_complete(self):
        if self.status == 'completed' and (self.remaining_case_ids or any(c.status != 'completed' for c in self.cases)):
            raise ValueError('incomplete work cannot be reported as complete')
        expected = [f'{regime}-N{n}-n{self.config.target_chunk_rows}'
                    for n in self.config.populations for regime in self.config.regimes]
        if [case.case_id for case in self.cases] + self.remaining_case_ids != expected:
            raise ValueError('report must account for every requested case in order')
        return self


class EvaluationCancelled(RuntimeError):
    pass


class EvaluationBudget:
    def __init__(self, max_cpu_seconds: float, max_wall_seconds: float,
                 check_cancel: Callable[[], None] | None = None):
        self.cpu_started = time.process_time()
        self.wall_started = time.perf_counter()
        self.max_cpu_seconds = max_cpu_seconds
        self.max_wall_seconds = max_wall_seconds
        self.check_cancel = check_cancel

    def check(self):
        if self.check_cancel:
            self.check_cancel()
        if time.process_time() - self.cpu_started > self.max_cpu_seconds:
            raise NumericBudgetExceeded('evaluation CPU budget exceeded')
        if time.perf_counter() - self.wall_started > self.max_wall_seconds:
            raise NumericBudgetExceeded('evaluation wall-time budget exceeded')


def regime_weights(regime: str) -> list[float]:
    """Exact appendix weight families, in canonical category order."""
    head = [2.00, 1.75, 1.50, 1.25, 1.00]
    if regime == 'separated':
        return head + [0.40] + [0.30 * 0.85**j for j in range(14)]
    triples = {
        'narrow_1': [0.965, 0.955, 0.945],
        'narrow_2': [0.970, 0.960, 0.950],
        'narrow_3': [0.975, 0.965, 0.955],
        'narrow_4': [0.980, 0.970, 0.960],
        'narrow_5': [0.982, 0.972, 0.962],
    }
    if regime not in triples:
        raise ValueError('unknown fixed-population weight regime')
    return head + triples[regime] + [0.75 * 0.88**j for j in range(12)]


def _largest_remainder(probabilities: list[float], population_size: int) -> list[int]:
    exact = np.asarray(probabilities, dtype=float) * population_size
    counts = np.floor(exact).astype(np.int64)
    remaining = population_size - int(counts.sum())
    order = np.argsort(-(exact - counts), kind='stable')
    counts[order[:remaining]] += 1
    return counts.tolist()


def fixed_population(regime: str, population_size: int) -> FixedPopulation:
    weights = regime_weights(regime)
    probabilities = [w / sum(weights) for w in weights]
    return FixedPopulation(regime=regime, population_size=population_size,
        category_ids=[f'category_{i:02d}' for i in range(1, 21)], weights=weights,
        probabilities=probabilities, counts=_largest_remainder(probabilities, population_size))


def named_seed(master_seed: int, case_id: str, stage: str) -> int:
    digest = hashlib.sha256(f'{VERSION}\0{master_seed}\0{case_id}\0{stage}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') & ((1 << 63) - 1)


def _sample_partition(counts: np.ndarray, sizes: list[int], rng: np.random.Generator) -> np.ndarray:
    # Conditional hypergeometric draws exactly equal a uniform shuffle cut into
    # chunks, while avoiding row materialization. The final chunk is the remainder.
    remaining = counts.copy()
    table = np.empty((len(sizes), len(counts)), dtype=np.int64)
    for b, size in enumerate(sizes[:-1]):
        table[b] = rng.multivariate_hypergeometric(remaining, size)
        remaining -= table[b]
    table[-1] = remaining
    return table


def sample_fixed_partition(counts: list[int], sizes: list[int], rng: np.random.Generator) -> np.ndarray:
    request = PartitionRequest(counts=counts, sizes=sizes)
    return _sample_partition(np.asarray(request.counts, dtype=np.int64), request.sizes, rng)


def _paired_results(table: np.ndarray, reference: np.ndarray, k: int,
                    candidates: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Independent evaluation accumulator; gold is fixed population totals."""
    local_order = np.argsort(-table, axis=1, kind='stable')
    retained = np.zeros(table.shape[1], dtype=np.int64)
    successes = np.zeros(len(candidates), dtype=bool)
    totals = np.zeros((len(candidates), table.shape[1]), dtype=np.int64)
    indices = {r: i for i, r in enumerate(candidates)}
    for rank in range(max(candidates)):
        ids = local_order[:, rank]
        np.add.at(retained, ids, table[np.arange(len(table)), ids])
        if rank + 1 in indices:
            i = indices[rank + 1]
            reduced = np.sort(np.argsort(-retained, kind='stable')[:k])
            successes[i] = np.array_equal(reduced, reference)
            totals[i] = retained
    return successes, totals


def stable_tail_oracle(curve: list[RecoveryMetric], target: float = 0.95) -> int | None:
    """Held-out diagnostic only: every tested budget at/above r must qualify."""
    if not 0 < target <= 1:
        raise ValueError('recovery target must lie in (0,1]')
    ordered = sorted(curve, key=lambda metric: metric.r)
    if len({metric.r for metric in ordered}) != len(ordered):
        raise ValueError('stable-tail curve cannot repeat a retention budget')
    return next((metric.r for i, metric in enumerate(ordered)
                 if all(tail.recovery >= target for tail in ordered[i:])), None)


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError('Wilson interval needs positive trials and valid success count')
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / denominator
    radius = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denominator
    # Preserve the exact boundary endpoints: floating cancellation may otherwise
    # produce high=1-epsilon for an all-success partial run (or low=epsilon for 0).
    return (0.0 if successes == 0 else max(0.0, center-radius),
            1.0 if successes == trials else min(1.0, center+radius))


def _rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


def evaluate_case(config: EvaluationConfig, regime: str, population_size: int,
                  output: Path, budget: EvaluationBudget) -> EvaluationCase:
    case_id = f'{regime}-N{population_size}-n{config.target_chunk_rows}'
    population = fixed_population(regime, population_size)
    geometry = balanced_geometry(population_size, config.target_chunk_rows)
    seeds = {stage: named_seed(config.master_seed, case_id, stage) for stage in ('pilot', 'planner', 'evaluation')}
    population_counts = np.asarray(population.counts, dtype=np.int64)
    reference = np.sort(np.argsort(-population_counts, kind='stable')[:config.k])
    started, cpu_started = time.perf_counter(), time.process_time()
    stage_times = {key: 0.0 for key in ('pilot', 'planner', 'partition_sampling', 'paired_retention', 'evidence_write')}
    pilot_started = time.perf_counter()
    pilot_rng = np.random.Generator(np.random.PCG64(seeds['pilot']))
    pilot = PilotCounts(category_ids=population.category_ids,
        counts=pilot_rng.multivariate_hypergeometric(population_counts, config.pilot_rows).tolist(), taxonomy_scope='fixed')
    stage_times['pilot'] = time.perf_counter() - pilot_started
    planner_config = RetentionConfig.model_validate({**config.planner.model_dump(), 'seed': seeds['planner']})
    decision = None
    completed = 0
    successes = np.zeros(len(config.evaluation_candidates), dtype=np.int64)
    retained_sums = np.zeros((len(config.evaluation_candidates), 20), dtype=np.int64)
    selected_only = naive_only = 0
    status = 'completed'
    error = None
    evidence_name = f'{case_id}.count-tables.jsonl.gz'
    digest = hashlib.sha256()
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        budget.check()
        planner_started = time.perf_counter()
        try:
            # No fixed population counts, gold reference or held-out trials are
            # passed into this call. It sees exactly the sampled pilot and geometry.
            decision = plan_retention(pilot, population_size=population_size,
                target_chunk_rows=config.target_chunk_rows, k=config.k,
                config=planner_config, check_cancel=budget.check)
        finally:
            stage_times['planner'] = time.perf_counter() - planner_started
        if decision.selected_r is None:
            raise ValueError('nonempty evaluation needs a planner budget')
        selected_index = config.evaluation_candidates.index(decision.selected_r)
        eval_rng = np.random.Generator(np.random.PCG64(seeds['evaluation']))
        context = {'counts': population.counts, 'sizes': geometry.sizes, 'candidates': config.evaluation_candidates}
        with (output / evidence_name).open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0, filename='') as evidence:
                for trial in range(config.trials):
                    budget.check()
                    tick = time.perf_counter()
                    table = _sample_partition(population_counts, geometry.sizes, eval_rng)
                    stage_times['partition_sampling'] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    result, retained = _paired_results(table, reference, config.k, config.evaluation_candidates)
                    stage_times['paired_retention'] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    record = TrialRecord.model_validate({'case_id': case_id, 'trial': trial,
                        'counts': table.tolist(), 'successes': result.tolist()}, context=context)
                    encoded = (record.model_dump_json() + '\n').encode()
                    evidence.write(encoded)
                    digest.update(encoded)
                    stage_times['evidence_write'] += time.perf_counter() - tick
                    # Include a trial in aggregate counts only after evidence is written.
                    successes += result
                    retained_sums += retained
                    selected_only += int(result[selected_index] and not result[0])
                    naive_only += int(result[0] and not result[selected_index])
                    completed += 1
    except EvaluationCancelled as exc:
        status, error = 'cancelled', str(exc)
    except NumericBudgetExceeded as exc:
        status, error = 'budget_exhausted', str(exc)
    except Exception as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    curve = []
    comparison = None
    if completed:
        for i, r in enumerate(config.evaluation_candidates):
            low, high = wilson_interval(int(successes[i]), completed)
            fractions = np.divide(retained_sums[i], population_counts * completed,
                out=np.zeros(20, dtype=float), where=population_counts > 0)
            curve.append(RecoveryMetric(r=r, successes=int(successes[i]), trials=completed,
                recovery=int(successes[i])/completed, wilson_95_low=low, wilson_95_high=high,
                candidate_volume=geometry.B*r, mean_retained_fraction_by_category=fractions.tolist()))
        gain = (selected_only - naive_only) / completed
        variance = ((selected_only + naive_only) - completed * gain**2) / (completed-1) if completed > 1 else 0.0
        se = math.sqrt(max(0.0, variance)/completed)
        comparison = PairedComparison(selected_r=decision.selected_r,
            selected_minus_naive=gain, paired_standard_error=se,
            normal_95_low=max(-1.0, gain-1.959963984540054*se),
            normal_95_high=min(1.0, gain+1.959963984540054*se),
            selected_only_successes=selected_only, naive_only_successes=naive_only)
    stage_times['total_wall'] = time.perf_counter() - started
    stage_times['total_cpu'] = time.process_time() - cpu_started
    result = EvaluationCase(case_id=case_id, status=status, error=error,
        population=population, geometry=geometry, pilot=pilot, seeds=seeds,
        planner_decision=decision, full_population_top_k=[population.category_ids[i] for i in reference],
        probability_cutoff_gap=population.probabilities[4]-population.probabilities[5],
        integer_population_cutoff_gap=(population.counts[4]-population.counts[5])/population_size,
        requested_trials=config.trials, completed_trials=completed, recovery_curve=curve,
        comparisons=comparison, held_out_stable_tail_r=stable_tail_oracle(curve, config.target) if curve else None,
        evidence_file=evidence_name if (output / evidence_name).exists() else None,
        count_tables_sha256=digest.hexdigest() if (output / evidence_name).exists() else None, timings_seconds=stage_times,
        python_peak_allocated_bytes=peak, process_peak_rss_bytes=_rss_bytes())
    (output / f'{case_id}.json').write_text(result.model_dump_json(indent=2) + '\n')
    return result


def run_evaluation(config: EvaluationConfig, output: Path,
                   check_cancel: Callable[[], None] | None = None,
                   progress: Callable[[EvaluationCase], None] | None = None) -> EvaluationReport:
    output.mkdir(parents=True, exist_ok=True)
    cases_to_run = [(regime, n) for n in config.populations for regime in config.regimes]
    case_ids = [f'{regime}-N{n}-n{config.target_chunk_rows}' for regime, n in cases_to_run]
    # Never overwrite prior numerical evidence, even on a cancelled/partial run.
    if (output / 'report.json').exists() or any((output / f'{case_id}.json').exists() or
            (output / f'{case_id}.count-tables.jsonl.gz').exists() for case_id in case_ids):
        raise FileExistsError('evaluation destination already contains evidence; choose a fresh directory')
    budget = EvaluationBudget(config.max_cpu_seconds, config.max_wall_seconds, check_cancel)
    cases = []
    status = 'completed'
    for regime, n in cases_to_run:
        case = evaluate_case(config, regime, n, output, budget)
        cases.append(case)
        if progress:
            progress(case)
        if case.status != 'completed':
            status = case.status
            break
    report = EvaluationReport(status=status, config=config, cases=cases,
        remaining_case_ids=case_ids[len(cases):],
        elapsed_wall_seconds=time.perf_counter()-budget.wall_started,
        elapsed_cpu_seconds=time.process_time()-budget.cpu_started,
        environment={'python': platform.python_version(), 'numpy': np.__version__,
            'platform': platform.system(), 'machine': platform.machine()},
        notes=[
            'Each trial partitions the same largest-remainder integer population without replacement; it is not a fresh multinomial population.',
            'All evaluation r share each sampled count table. No monotonicity smoothing is applied.',
            'Pilot, posterior planner and held-out evaluation have independent named PCG64 streams.',
            'The planner receives pilot counts only; the stable-tail oracle is computed after held-out evaluation.',
            'Wilson intervals and paired standard errors quantify Monte Carlo error conditional on one sampled pilot per case; they do not cover pilot-selection variation or semantic model errors.',
            'The configured CPU/wall deadlines include Python allocation tracking and streamed evidence serialization. RSS is a cumulative process maximum.',
            'This uses the published protocol with disclosed engineering RNG/rounding choices, not recovered author seeds or claimed reproduction of published numerical values.',
        ])
    (output / 'report.json').write_text(report.model_dump_json(indent=2) + '\n')
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=5000)
    parser.add_argument('--populations', type=int, nargs='+', default=[2000, 8000, 32000])
    parser.add_argument('--regimes', nargs='+', default=list(REGIMES), choices=REGIMES)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--max-cpu-seconds', type=float, default=600.0)
    parser.add_argument('--max-wall-seconds', type=float, default=900.0)
    args = parser.parse_args(argv)
    config = EvaluationConfig(trials=args.trials, populations=args.populations,
        regimes=args.regimes, master_seed=args.seed,
        max_cpu_seconds=args.max_cpu_seconds, max_wall_seconds=args.max_wall_seconds)
    cancelled = threading.Event()
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def cancel(_signum, _frame):
        cancelled.set()
    def check_cancel():
        if cancelled.is_set():
            raise EvaluationCancelled('evaluation cancelled by signal')
    def progress(case):
        print(f'{case.case_id}: {case.status}; partitions={case.completed_trials}/{case.requested_trials}; '
              f'selected_r={case.planner_decision.selected_r if case.planner_decision else None}; '
              f'wall_seconds={case.timings_seconds["total_wall"]:.3f}', file=sys.stderr, flush=True)
    for sig in old_handlers:
        signal.signal(sig, cancel)
    try:
        report = run_evaluation(config, args.output, check_cancel, progress)
        print(report.model_dump_json())
        return 0 if report.status == 'completed' else (130 if report.status == 'cancelled' else 1)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
