"""Pilot-only paired retention guidance for the fixed-taxonomy count abstraction.

The protocol follows the frozen manuscript; PCG64, stream separation and the
inverse-ECDF quantile are disclosed engineering choices, not recovered author seeds.
No provider calls, semantic frequency invention, or evaluation oracle occurs here.
"""
from __future__ import annotations
import hashlib
import math
import time
from typing import Callable, Literal
import numpy as np
from pydantic import Field, StrictInt, StrictStr, model_validator
from tada_runtime.audit_contracts import AuditModel

VERSION = 'paired-jeffreys-retention-v1'
MAX_BLOCKS = 100_000


class RetentionConfig(AuditModel):
    outer_draws: int = Field(default=100, ge=1, le=10000)
    inner_draws: int = Field(default=20, ge=1, le=10000)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    target: float = Field(default=0.95, gt=0, le=1)
    r_max: int = Field(default=32, ge=1, le=10000)
    no_tail_budget: int = Field(default=32, ge=1, le=10000)
    candidates: list[StrictInt] | None = None
    seed: int = Field(default=0, ge=0, le=2**63-1)
    max_seconds: float = Field(default=30.0, gt=0, le=600)
    max_table_cells: int = Field(default=1_000_000, ge=1, le=100_000_000)
    max_simulation_cells: int = Field(default=50_000_000, ge=1, le=10_000_000_000)
    max_work_units: int = Field(default=1_000_000_000, ge=1, le=100_000_000_000)
    prior_half_count: Literal[0.5] = 0.5
    quantile_method: Literal['inverse_ecdf'] = 'inverse_ecdf'
    sampler: Literal['per-inner-iid-multinomial-balanced-v1'] = 'per-inner-iid-multinomial-balanced-v1'
    rng: Literal['numpy-PCG64-seedsequence-v1'] = 'numpy-PCG64-seedsequence-v1'
    tie_policy: Literal['canonical-domain-order-v1'] = 'canonical-domain-order-v1'
    version: Literal['paired-jeffreys-retention-v1'] = VERSION

    @model_validator(mode='after')
    def candidate_domain(self):
        if self.candidates is not None:
            if not self.candidates or len(set(self.candidates)) != len(self.candidates):
                raise ValueError('candidate grid must be nonempty and unique')
            if any(r < 1 or r > self.r_max for r in self.candidates):
                raise ValueError('candidate grid must be positive and bounded by r_max')
        return self


class PilotCounts(AuditModel):
    category_ids: list[StrictStr]
    counts: list[StrictInt]
    taxonomy_scope: Literal['observed', 'fixed'] = 'observed'

    @model_validator(mode='after')
    def aligned(self):
        if len(self.category_ids) != len(self.counts) or len(set(self.category_ids)) != len(self.category_ids):
            raise ValueError('pilot counts require aligned unique canonical IDs')
        if any(not value for value in self.category_ids) or any(value < 0 for value in self.counts):
            raise ValueError('pilot IDs must be nonempty and counts nonnegative integers')
        if len(self.counts) > 10000:
            raise ValueError('taxonomy exceeds bounded numeric domain')
        return self


class BalancedGeometry(AuditModel):
    population_size: int = Field(ge=0, le=1_000_000_000)
    target_chunk_rows: int = Field(ge=1)
    B: int = Field(ge=0, le=MAX_BLOCKS)
    n_min: int = Field(ge=0)
    sizes: list[StrictInt]

    @model_validator(mode='after')
    def geometry(self):
        n, target = self.population_size, self.target_chunk_rows
        blocks = (n + target - 1) // target
        if blocks > MAX_BLOCKS:
            raise ValueError('balanced partition exceeds bounded block domain')
        q, t = divmod(n, blocks) if blocks else (0, 0)
        if self.B != blocks or self.n_min != q or self.sizes != [q+1]*t + [q]*(blocks-t):
            raise ValueError('balanced geometry does not match ceil/floor partition policy')
        return self


class BalancedPartitionPlan(AuditModel):
    geometry: BalancedGeometry
    seed: int = Field(ge=0, le=2**63-1)
    batches: list[list[StrictStr | StrictInt]]
    policy: Literal['uniform-shuffle-balanced-largest-first-v1'] = 'uniform-shuffle-balanced-largest-first-v1'
    rng: Literal['numpy-PCG64'] = 'numpy-PCG64'

    @model_validator(mode='after')
    def coverage(self):
        rows = [rid for batch in self.batches for rid in batch]
        if len(rows) != self.geometry.population_size or [len(batch) for batch in self.batches] != self.geometry.sizes:
            raise ValueError('partition coverage or sizes are inconsistent')
        if len(set(rows)) != len(rows) or '' in rows or len({type(rid) for rid in rows}) > 1:
            raise ValueError('partition rid domain must be unique, nonempty and type stable')
        return self


class RetentionRequest(AuditModel):
    pilot: PilotCounts
    geometry: BalancedGeometry
    k: int = Field(ge=1, le=10000)
    config: RetentionConfig

    @model_validator(mode='after')
    def input_contract(self):
        if sum(self.pilot.counts) > self.geometry.population_size:
            raise ValueError('pilot size exceeds target population')
        if self.geometry.population_size:
            if not self.pilot.counts or sum(self.pilot.counts) == 0:
                raise ValueError('nonempty target requires a nonempty observed pilot')
            if self.pilot.taxonomy_scope == 'fixed' and self.k > len(self.pilot.counts):
                raise ValueError('requested k exceeds fixed known taxonomy')
        return self


class TheoremDiagnostic(AuditModel):
    empirical_gap: float | None
    raw_bound: float | None
    clipped_bound: float | None
    strict_cutoff_gap: bool
    vacuous: bool
    exact_count_special_case: Literal['retain_all', 'single_chunk'] | None = None
    sufficient_n_min: float | None = None
    sufficient_chunk_condition_met: bool | None = None
    theorem_applicable_to_empirical_profile: bool
    support_source: Literal['pilot_plugin_not_full_population'] = 'pilot_plugin_not_full_population'
    calibrated_deployment_bound: Literal[False] = False
    open_semantic_guarantee: Literal[False] = False
    limitations: list[str]


class CandidateRetentionDiagnostic(AuditModel):
    r: int = Field(ge=1)
    mean_recovery: float = Field(ge=0, le=1)
    lower_quantile: float = Field(ge=0, le=1)
    success_counts: list[StrictInt]
    outer_recoveries: list[float]
    pooled_mc_standard_error: float = Field(ge=0)
    max_inner_binomial_standard_error: float = Field(ge=0)
    outer_profile_standard_error: float = Field(ge=0)
    candidate_volume: int = Field(ge=0)
    theorem: TheoremDiagnostic

    @model_validator(mode='after')
    def mc_shape(self):
        if not self.success_counts or len(self.success_counts) != len(self.outer_recoveries):
            raise ValueError('paired outer success counts and recoveries must align')
        if any(value < 0 for value in self.success_counts) or any(not 0 <= value <= 1 for value in self.outer_recoveries):
            raise ValueError('invalid Monte Carlo result')
        return self


class RetentionDecision(AuditModel):
    status: Literal['selected', 'fallback_no_tail', 'fallback_no_qualified_r', 'empty']
    selected_r: int | None = Field(default=None, ge=1)
    quantile_target_met: bool
    fallback_reason: Literal['no_observed_tail', 'no_qualified_candidate'] | None = None
    request: RetentionRequest
    diagnostics: list[CandidateRetentionDiagnostic]
    simulated_candidates: list[StrictInt]
    observed_positive_categories: int = Field(ge=0)
    simulated_category_count: int = Field(ge=0)
    requested_mapper_budget: int | None = Field(default=None, ge=1)
    paired_draws_sha256: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    posterior_parameters: list[float]
    posterior_profiles: list[list[float]]
    elapsed_seconds: float = Field(ge=0)
    completed_tables: int = Field(ge=0)
    computed_count_cells: int = Field(ge=0)
    estimated_work_units: int = Field(ge=0)
    mc_note: str = 'Finite U/V diagnostics are not calibrated deployment confidence bounds; zero plug-in SE does not establish zero error.'
    semantic_scope: str = 'Conditional fixed-taxonomy exact-count retention guidance; unseen categories, semantic discovery/assignment/consolidation errors remain outside this estimate.'

    @model_validator(mode='after')
    def result_contract(self):
        cfg = self.request.config
        pilot = self.request.pilot
        C = len(pilot.counts)
        if self.observed_positive_categories != sum(n>0 for n in pilot.counts) or self.simulated_category_count != C:
            raise ValueError('decision taxonomy dimensions mismatch pilot')
        if self.posterior_parameters != [n+0.5 for n in pilot.counts]:
            raise ValueError('Jeffreys posterior parameters must be pilot counts plus 0.5')
        if self.status == 'empty':
            if self.request.geometry.population_size or self.selected_r is not None or self.diagnostics or self.quantile_target_met:
                raise ValueError('empty decision cannot claim a selected or certified budget')
        elif not self.request.geometry.population_size:
            raise ValueError('zero population requires empty decision')
        if not self.diagnostics:
            if self.completed_tables or self.computed_count_cells or self.estimated_work_units or self.paired_draws_sha256 or self.posterior_profiles or self.simulated_candidates:
                raise ValueError('unsimulated decision cannot claim draws or work')
        elif self.completed_tables != cfg.outer_draws*cfg.inner_draws or self.computed_count_cells != self.completed_tables*self.request.geometry.B*C or self.paired_draws_sha256 is None:
            raise ValueError('Monte Carlo draws or cells mismatch request')
        if self.status == 'empty':
            if self.requested_mapper_budget is not None or self.fallback_reason is not None:
                raise ValueError('empty decision cannot request a mapper fallback')
            return self
        if self.selected_r is None or self.selected_r < self.request.k or self.requested_mapper_budget != self.selected_r:
            raise ValueError('selected mapper budget must cover requested k')
        if self.status.startswith('fallback') and (self.quantile_target_met or self.fallback_reason is None):
            raise ValueError('fallback must remain explicitly uncertified')
        if self.status == 'fallback_no_tail' and (self.fallback_reason != 'no_observed_tail' or
            self.diagnostics or self.selected_r != cfg.no_tail_budget or pilot.taxonomy_scope != 'observed' or self.observed_positive_categories > self.request.k):
            raise ValueError('no-tail fallback does not match observed pilot and configured budget')
        if self.status == 'fallback_no_qualified_r' and (self.fallback_reason != 'no_qualified_candidate' or
            not self.diagnostics or any(d.lower_quantile >= cfg.target for d in self.diagnostics) or self.selected_r != cfg.r_max):
            raise ValueError('no-qualified fallback requires a completely unsuccessful candidate grid')
        if self.status == 'selected':
            qualified = [d.r for d in self.diagnostics if d.lower_quantile >= self.request.config.target]
            if not qualified or self.selected_r != min(qualified) or not self.quantile_target_met or self.fallback_reason is not None:
                raise ValueError('selected budget must be the smallest empirical quantile crossing')
        if self.simulated_candidates != [d.r for d in self.diagnostics]:
            raise ValueError('candidate diagnostics must match the simulated grid')
        if self.simulated_candidates != sorted(set(self.simulated_candidates)):
            raise ValueError('diagnostic candidate grid must be sorted and unique')
        for d in self.diagnostics:
            if len(d.success_counts) != cfg.outer_draws or any(n > cfg.inner_draws for n in d.success_counts):
                raise ValueError('Monte Carlo diagnostics mismatch U/V')
            expected = [n/cfg.inner_draws for n in d.success_counts]
            if expected != d.outer_recoveries:
                raise ValueError('outer recovery must be successes divided by V')
            if d.mean_recovery != sum(expected)/len(expected) or d.lower_quantile != sorted(expected)[math.ceil(cfg.alpha*len(expected))-1]:
                raise ValueError('candidate mean/quantile mismatch outer recoveries')
            if not self.request.k <= d.r <= C or d.candidate_volume != self.request.geometry.B*d.r:
                raise ValueError('candidate grid or volume mismatch request')
        if self.diagnostics and (len(self.posterior_profiles) != self.request.config.outer_draws or
            any(len(row) != self.simulated_category_count or any(value < 0 for value in row) or
                not math.isclose(sum(row), 1.0, abs_tol=1e-12) for row in self.posterior_profiles)):
            raise ValueError('posterior profile matrix must have U normalized C-category rows')
        return self


class CountTable(AuditModel):
    rows: list[list[StrictInt]]
    k: int = Field(ge=1)
    candidates: list[StrictInt]

    @model_validator(mode='after')
    def table_contract(self):
        if not self.rows or not self.rows[0] or any(len(row) != len(self.rows[0]) for row in self.rows):
            raise ValueError('count table must be nonempty and rectangular')
        if len(self.rows)*len(self.rows[0]) > 1_000_000:
            raise ValueError('external count table exceeds bounded cell domain')
        if any(value < 0 for row in self.rows for value in row) or sum(map(sum,self.rows)) > 1_000_000_000:
            raise ValueError('counts must be nonnegative and total at most one billion')
        C = len(self.rows[0])
        if self.k > C or not self.candidates or len(set(self.candidates)) != len(self.candidates):
            raise ValueError('invalid k or candidate domain')
        if any(not self.k <= r <= C for r in self.candidates):
            raise ValueError('retention candidates must lie in [k,C]')
        return self


class NumericBudgetExceeded(RuntimeError):
    pass


def balanced_geometry(population_size: int, target_chunk_rows: int) -> BalancedGeometry:
    # Validate scalar inputs before integer arithmetic (bool must not masquerade as int).
    class Inputs(AuditModel):
        population_size: int = Field(ge=0, le=1_000_000_000)
        target_chunk_rows: int = Field(ge=1)
    values = Inputs(population_size=population_size, target_chunk_rows=target_chunk_rows)
    N, n = values.population_size, values.target_chunk_rows
    B = (N+n-1)//n
    if B > MAX_BLOCKS:
        raise NumericBudgetExceeded('balanced partition exceeds bounded block domain')
    q, t = divmod(N, B) if B else (0, 0)
    return BalancedGeometry(population_size=N, target_chunk_rows=n, B=B, n_min=q, sizes=[q+1]*t+[q]*(B-t))


def balanced_partition(rids: list[str | int], target_chunk_rows: int, seed: int) -> BalancedPartitionPlan:
    geometry = balanced_geometry(len(rids), target_chunk_rows)
    # Validate identity/seed before passing the values to NumPy.
    offsets = np.cumsum([0]+geometry.sizes)
    preliminary = BalancedPartitionPlan(geometry=geometry, seed=seed,
        batches=[rids[int(a):int(b)] for a,b in zip(offsets[:-1],offsets[1:])])
    order = np.random.Generator(np.random.PCG64(preliminary.seed)).permutation(len(rids))
    shuffled = [rids[int(i)] for i in order]
    offsets = np.cumsum([0]+geometry.sizes)
    return BalancedPartitionPlan(geometry=geometry, seed=seed,
        batches=[shuffled[int(a):int(b)] for a,b in zip(offsets[:-1],offsets[1:])])


def _recover(table: np.ndarray, k: int, candidates: list[int], check: Callable[[],None] | None = None) -> dict[int, bool]:
    # Stable sort fixes all local/reference/reducer ties to canonical domain order.
    reference = np.sort(np.argsort(-table.sum(axis=0), kind='stable')[:k])
    local_order = np.argsort(-table, axis=1, kind='stable')
    retained = np.zeros(table.shape[1], dtype=np.int64)
    result = {}
    wanted = set(candidates)
    for rank in range(max(candidates)):
        if check is not None:
            check()
        columns = local_order[:,rank]
        np.add.at(retained, columns, table[np.arange(len(table)),columns])
        r = rank+1
        if r in wanted:
            result[r] = bool(np.array_equal(np.sort(np.argsort(-retained,kind='stable')[:k]), reference))
    return result


def paired_recovery(table: np.ndarray | list[list[int]], k: int, candidates: list[int]) -> dict[int, bool]:
    """Validate an external exact count table, then evaluate all r on that same table."""
    values = table.tolist() if isinstance(table, np.ndarray) else table
    checked = CountTable(rows=values, k=k, candidates=candidates)
    return _recover(np.asarray(checked.rows,dtype=np.int64), checked.k, checked.candidates)


def inverse_ecdf(values: list[float], alpha: float) -> float:
    class QuantileInput(AuditModel):
        values: list[float] = Field(min_length=1)
        alpha: float = Field(gt=0,lt=1)
    checked = QuantileInput(values=values,alpha=alpha)
    return sorted(checked.values)[max(0,math.ceil(checked.alpha*len(checked.values))-1)]


def _draw_table(rng: np.random.Generator, profile: np.ndarray, geometry: BalancedGeometry,
                check: Callable[[],None]) -> np.ndarray:
    # Independent multinomial chunks, grouped by equal size in bounded blocks.
    table = np.empty((geometry.B,len(profile)),dtype=np.int64)
    offset = 0
    while offset < geometry.B:
        check()
        size = geometry.sizes[offset]
        end = min(offset+1024,geometry.B)
        while end > offset and geometry.sizes[end-1] != size:
            end -= 1
        table[offset:end] = rng.multinomial(size,profile,size=end-offset)
        offset = end
    return table


def theorem_diagnostic(counts: list[int], geometry: BalancedGeometry, k: int, r: int, target: float) -> TheoremDiagnostic:
    checked = CountTable(rows=[counts],k=k,candidates=[r])
    if sum(counts) == 0:
        raise ValueError('pilot plug-in theorem needs positive observed support')
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target) or not 0 < target <= 1:
        raise ValueError('target must lie in (0,1]')
    if geometry.population_size == 0:
        raise ValueError('pilot plug-in theorem needs a nonempty population')
    supports = sorted((value/sum(counts) for value in checked.rows[0]), reverse=True)
    C = len(supports)
    cutoff = k == C or supports[k-1] > supports[k]
    exact = 'retain_all' if r == C else ('single_chunk' if geometry.B == 1 else None)
    gap = supports[k-1] - (supports[r] if r<C else 0.0)
    raw = 1.0 if exact else 1.0-geometry.B*(C-r)*sum(math.exp(-geometry.n_min*(p-supports[r])**2/2) for p in supports[:k])
    sufficient = None
    if cutoff and r<C and gap>0 and target<1:
        sufficient = (2/gap**2)*math.log(geometry.B*k*(C-r)/(1-target))
    limitations = ['Pilot supports are noisy plug-in values, not known full-population frequencies.',
                   'Fixed taxonomy, partition-invariant one-label assignment, exact local counts and canonical retained-count consolidation are required.']
    if not cutoff:
        limitations.append('Strict global cutoff gap is absent; the general preservation theorem is inapplicable.')
    if raw <= 0:
        limitations.append('The un-clipped plug-in theorem bound is vacuous.')
    return TheoremDiagnostic(empirical_gap=gap,raw_bound=raw,clipped_bound=min(1.0,max(0.0,raw)),
        strict_cutoff_gap=cutoff,vacuous=raw<=0,exact_count_special_case=exact,
        sufficient_n_min=sufficient,sufficient_chunk_condition_met=geometry.n_min>=sufficient if sufficient is not None else None,
        theorem_applicable_to_empirical_profile=cutoff,limitations=limitations)


def plan_retention(pilot: PilotCounts, *, population_size: int, target_chunk_rows: int, k: int,
                   config: RetentionConfig, check_cancel: Callable[[],None] | None = None) -> RetentionDecision:
    started = time.monotonic()
    request = RetentionRequest(pilot=pilot,geometry=balanced_geometry(population_size,target_chunk_rows),k=k,config=config)
    pilot, geometry, cfg = request.pilot, request.geometry, request.config
    C = len(pilot.counts)
    positive = sum(n>0 for n in pilot.counts)
    posterior = [value+0.5 for value in pilot.counts]
    def check():
        if check_cancel is not None:
            check_cancel()
        if time.monotonic()-started > cfg.max_seconds:
            raise NumericBudgetExceeded('retention planner walltime budget exceeded')
    profiles = []
    work_units = 0
    def decision(status, r=None, diagnostics=None, reason=None, draw_hash=None, tables=0):
        check()
        diagnostics = diagnostics or []
        return RetentionDecision(status=status,selected_r=r,quantile_target_met=status=='selected',fallback_reason=reason,
            request=request,diagnostics=diagnostics,simulated_candidates=[d.r for d in diagnostics],
            observed_positive_categories=positive,simulated_category_count=C,requested_mapper_budget=r,
            paired_draws_sha256=draw_hash,posterior_parameters=posterior,posterior_profiles=profiles,
            elapsed_seconds=time.monotonic()-started,
            completed_tables=tables,computed_count_cells=tables*geometry.B*C,estimated_work_units=work_units)
    check()
    if population_size == 0:
        return decision('empty')
    if pilot.taxonomy_scope == 'observed' and positive <= k:
        if cfg.no_tail_budget < k:
            raise ValueError('configured no-tail mapper budget is smaller than requested k')
        return decision('fallback_no_tail',cfg.no_tail_budget,reason='no_observed_tail')
    candidates = sorted(cfg.candidates if cfg.candidates is not None else range(k,min(C,cfg.r_max)+1))
    if not candidates or any(not k <= r <= C for r in candidates):
        raise ValueError('simulated candidate grid must be nonempty and within [k,C]')
    if cfg.r_max < k:
        raise ValueError('configured maximum mapper budget is smaller than k')
    cells = geometry.B*C
    if cells > cfg.max_table_cells or cells*cfg.outer_draws*cfg.inner_draws > cfg.max_simulation_cells:
        raise NumericBudgetExceeded('retention count-table computation exceeds configured cell budget')
    # A conservative static accounting includes sorting and every r evaluation,
    # even though local ranking is cached and retained sums are updated incrementally.
    work_units = cfg.outer_draws*cfg.inner_draws*cells*(max(candidates)+len(candidates)+math.ceil(math.log2(max(2,C))))
    if work_units > cfg.max_work_units:
        raise NumericBudgetExceeded('retention ranking work exceeds configured work budget')
    profiles_seed, tables_seed = np.random.SeedSequence(cfg.seed).spawn(2)
    profile_rng = np.random.Generator(np.random.PCG64(profiles_seed))
    table_rng = np.random.Generator(np.random.PCG64(tables_seed))
    successes = {r:[] for r in candidates}
    digest = hashlib.sha256()
    for _ in range(cfg.outer_draws):
        check()
        profile = profile_rng.dirichlet(np.asarray(posterior,dtype=float))
        profiles.append(profile.tolist())
        digest.update(np.asarray(profile,dtype='<f8').tobytes())
        row = {r:0 for r in candidates}
        for _ in range(cfg.inner_draws):
            check()
            table = _draw_table(table_rng,profile,geometry,check)
            digest.update(np.asarray(table,dtype='<i8').tobytes())
            for r, success in _recover(table,k,candidates,check=check).items():
                row[r] += int(success)
        for r in candidates:
            successes[r].append(row[r])
    diagnostics = []
    for r in candidates:
        check()
        recovery = [n/cfg.inner_draws for n in successes[r]]
        mean = sum(recovery)/cfg.outer_draws
        diagnostics.append(CandidateRetentionDiagnostic(r=r,mean_recovery=mean,
            lower_quantile=inverse_ecdf(recovery,cfg.alpha),success_counts=successes[r],outer_recoveries=recovery,
            pooled_mc_standard_error=math.sqrt(mean*(1-mean)/(cfg.outer_draws*cfg.inner_draws)),
            max_inner_binomial_standard_error=max(math.sqrt(p*(1-p)/cfg.inner_draws) for p in recovery),
            outer_profile_standard_error=float(np.std(recovery,ddof=1)/math.sqrt(cfg.outer_draws)) if cfg.outer_draws>1 else 0.0,
            candidate_volume=geometry.B*r,theorem=theorem_diagnostic(pilot.counts,geometry,k,r,cfg.target)))
    qualified = [d.r for d in diagnostics if d.lower_quantile>=cfg.target]
    return decision('selected' if qualified else 'fallback_no_qualified_r',min(qualified) if qualified else cfg.r_max,
        diagnostics,reason=None if qualified else 'no_qualified_candidate',draw_hash=digest.hexdigest(),tables=cfg.outer_draws*cfg.inner_draws)
