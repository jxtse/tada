"""Numeric protocol regressions; no provider or full-population oracle is used."""
import math

import numpy as np
import pytest
from pydantic import ValidationError

from tada_runtime import retention as rt


def pilot(counts=(8, 5, 3, 2), scope='fixed'):
    return rt.PilotCounts(category_ids=[f'c{i}' for i in range(len(counts))], counts=list(counts), taxonomy_scope=scope)


def config(**updates):
    return rt.RetentionConfig(outer_draws=4, inner_draws=5, r_max=4, no_tail_budget=9, **updates)


def plan(*, counts=(8, 5, 3, 2), scope='fixed', cfg=None, k=1, N=40, n=10, check_cancel=None):
    return rt.plan_retention(pilot(counts, scope), population_size=N, target_chunk_rows=n,
        k=k, config=cfg or config(), check_cancel=check_cancel)


@pytest.mark.parametrize('N,n,sizes', [(0, 4, []), (1, 100, [1]), (10, 4, [4, 3, 3]), (11, 4, [4, 4, 3]), (32, 10, [8]*4)])
def test_exact_balanced_geometry(N, n, sizes):
    result = rt.balanced_geometry(N, n)
    assert result.sizes == sizes
    assert result.B == math.ceil(N/n)
    assert result.n_min == (min(sizes) if sizes else 0)
    assert sum(sizes) == N


def test_partition_uniform_seeded_coverage_and_no_quadratic_prefix_sums():
    rids = [f'row{i}' for i in range(10001)]
    a = rt.balanced_partition(rids, 7, 82)
    b = rt.balanced_partition(rids, 7, 82)
    c = rt.balanced_partition(rids, 7, 83)
    assert a == b and a.batches != c.batches
    assert sorted(rid for batch in a.batches for rid in batch) == sorted(rids)
    assert max(map(len,a.batches))-min(map(len,a.batches)) <= 1
    assert [len(batch) for batch in a.batches] == a.geometry.sizes
    order = np.random.Generator(np.random.PCG64(82)).permutation(len(rids))
    assert [rid for batch in a.batches for rid in batch] == [rids[int(i)] for i in order]


@pytest.mark.parametrize('rids', [[1, 1], ['a', ''], [1, '2'], [True, False]])
def test_partition_rejects_invalid_identity_domain(rids):
    with pytest.raises(ValueError):
        rt.balanced_partition(rids, 2, 0)


@pytest.mark.parametrize('N,n', [(True, 1), (1, False), (-1, 2), (4, 0), ('4', 2)])
def test_geometry_scalar_authority(N, n):
    with pytest.raises(ValidationError):
        rt.balanced_geometry(N, n)


def test_geometry_refuses_extreme_allocation_before_building_sizes():
    with pytest.raises(rt.NumericBudgetExceeded, match='block domain'):
        rt.balanced_geometry(1_000_000_000, 1)
    with pytest.raises(ValidationError, match='block domain'):
        rt.BalancedGeometry(population_size=1_000_000_000,target_chunk_rows=1,B=1,n_min=1,sizes=[1])


@pytest.mark.parametrize('updates', [dict(prior_half_count=1.0),dict(outer_draws=True),dict(inner_draws=0),
    dict(candidates=[1,1]),dict(candidates=[]),dict(candidates=[33]),dict(alpha=0.0),dict(alpha=float('nan')),
    dict(target=1.1),dict(seed=-1),dict(unknown=True)])
def test_strict_configuration(updates):
    with pytest.raises(ValidationError):
        rt.RetentionConfig(**updates)


@pytest.mark.parametrize('counts', [[True,1],[-1,1],[1.0,1],['1',1]])
def test_counts_are_nonnegative_strict_integers(counts):
    with pytest.raises(ValidationError):
        pilot(counts)


def test_counts_require_unique_aligned_canonical_ids():
    for ids, counts in [(['a','a'],[1,2]), (['a'],[1,2]), ([''],[1])]:
        with pytest.raises(ValidationError):
            rt.PilotCounts(category_ids=ids, counts=counts)


def test_jeffreys_half_count_is_used_by_actual_rng():
    d = plan(counts=(12,3,0,0),cfg=config(seed=47))
    params = [12.5,3.5,0.5,0.5]
    expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence(47).spawn(2)[0])).dirichlet(params)
    assert d.posterior_parameters == params
    np.testing.assert_array_equal(d.posterior_profiles[0], expected)
    assert all(len(row)==4 and math.isclose(sum(row),1) for row in d.posterior_profiles)


def test_every_r_shares_profiles_tables_and_rng_independent_of_grid_order():
    a = plan(cfg=config(seed=313,candidates=[4,1,3,2]))
    b = plan(cfg=config(seed=313,candidates=[1,3]))
    c = plan(cfg=config(seed=314,candidates=[1,3]))
    assert a.paired_draws_sha256 == b.paired_draws_sha256
    assert a.posterior_profiles == b.posterior_profiles
    assert a.paired_draws_sha256 != c.paired_draws_sha256
    assert {d.r:d.success_counts for d in a.diagnostics if d.r in (1,3)} == {d.r:d.success_counts for d in b.diagnostics}
    assert a.simulated_candidates == [1,2,3,4]


def test_realized_reference_and_retained_counts_not_candidate_union():
    # B is the realized winner under canonical ties. B survives in one mapper,
    # but retained A=9 beats retained B=4; summing full counts or candidate-union
    # containment would incorrectly report recovery.
    table = [[3,2,3,2],[6,2,2,0],[0,4,5,1],[2,4,2,2]]
    assert np.argmax(np.sum(table,axis=0)) == 1
    assert rt.paired_recovery(table,1,[1,4]) == {1:False,4:True}


def test_reference_is_each_realized_table_not_pilot_profile_rank(monkeypatch):
    # Pilot strongly favors c0, but the count table realizes c1 as the winner.
    table = np.array([[1,9],[1,9]],dtype=np.int64)
    monkeypatch.setattr(rt,'_draw_table',lambda *args: table)
    d = plan(counts=(18,2),cfg=rt.RetentionConfig(outer_draws=2,inner_draws=3,r_max=2),N=20,n=10)
    assert d.selected_r == 1
    assert d.diagnostics[0].success_counts == [3,3]


def test_nonmonotone_recovery_is_preserved_and_minimum_crossing_selected(monkeypatch):
    table = np.array([[2,2,4,2],[2,2,4,2],[4,2,3,1],[3,3,3,1]],dtype=np.int64)
    assert rt.paired_recovery(table,1,[1,2,3,4]) == {1:True,2:False,3:True,4:True}
    monkeypatch.setattr(rt,'_draw_table',lambda *args: table)
    d = plan()
    assert [x.lower_quantile for x in d.diagnostics] == [1.0,0.0,1.0,1.0]
    assert d.selected_r == 1
    assert d.status == 'selected' and d.quantile_target_met


def test_outer_quantile_is_not_pooled_bernoulli_quantile(monkeypatch):
    # Two independent outer profiles achieve 2/4 and 4/4. Q_.5 over the outer
    # recoveries is .5. A pooled Bernoulli Q_.5 would be 1; pooled mean is .75.
    outcomes = iter([True,True,False,False,True,True,True,True])
    monkeypatch.setattr(rt,'_recover',lambda *args,**kwargs:{1:next(outcomes)})
    cfg = rt.RetentionConfig(outer_draws=2,inner_draws=4,alpha=.5,target=.6,r_max=1)
    d = plan(counts=(8,5),cfg=cfg)
    assert d.diagnostics[0].success_counts == [2,4]
    assert d.diagnostics[0].outer_recoveries == [.5,1.0]
    assert d.diagnostics[0].lower_quantile == .5
    assert d.diagnostics[0].mean_recovery == .75
    assert d.status == 'fallback_no_qualified_r'


def test_inverse_ecdf_convention_has_no_interpolation():
    assert rt.inverse_ecdf([0.0,.5,1.0,1.0],.5) == .5
    assert rt.inverse_ecdf([0.0,.5,1.0,1.0],.51) == 1.0
    assert rt.inverse_ecdf([i/100 for i in range(100)],.05) == .04
    with pytest.raises(ValidationError):
        rt.inverse_ecdf([], .5)


def test_no_tail_fallback_is_uncertified_and_never_clipped_to_pilot_C():
    d = plan(counts=(18,0,0,0),scope='observed',k=2)
    assert d.status == 'fallback_no_tail' and d.fallback_reason == 'no_observed_tail'
    assert d.selected_r == d.requested_mapper_budget == 9 > d.simulated_category_count
    assert not d.quantile_target_met and d.completed_tables == 0
    assert d.diagnostics == d.posterior_profiles == []


def test_no_tail_k_can_exceed_observed_C_but_not_configured_fallback():
    assert plan(counts=(8,),scope='observed',k=3).selected_r == 9
    with pytest.raises(ValueError,match='smaller than requested k'):
        plan(counts=(8,),scope='observed',k=10)


def test_known_fixed_taxonomy_zero_counts_do_not_invent_unseen_categories():
    d = plan(counts=(18,0,0,0),scope='fixed')
    assert d.status != 'fallback_no_tail'
    assert d.simulated_category_count == 4 and len(d.posterior_parameters) == 4
    assert d.completed_tables == 20
    with pytest.raises(ValidationError,match='exceeds fixed'):
        plan(counts=(8,),scope='fixed',k=2)


def test_no_qualified_fallback_is_separate_and_uncertified(monkeypatch):
    monkeypatch.setattr(rt,'_recover',lambda *args,**kwargs:{1:False,2:False})
    d = plan(cfg=rt.RetentionConfig(outer_draws=2,inner_draws=3,r_max=2,no_tail_budget=9))
    assert d.status == 'fallback_no_qualified_r'
    assert d.fallback_reason == 'no_qualified_candidate' and d.selected_r == 2
    assert not d.quantile_target_met and d.completed_tables == 6


def test_empty_is_noop_allzero_positive_population_is_error():
    d = plan(counts=(),N=0)
    assert d.status == 'empty' and d.selected_r is None and not d.quantile_target_met
    with pytest.raises(ValidationError,match='nonempty observed pilot'):
        plan(counts=(0,0))
    with pytest.raises(ValidationError,match='exceeds target population'):
        plan(counts=(10,5),N=10)


@pytest.mark.parametrize('table,k', [([[5,5,0],[0,5,5]],1), ([[0,0,0],[0,0,0]],2), ([[5,5,5]],2)])
def test_retain_all_and_single_chunk_ties_are_exact(table,k):
    C = len(table[0])
    assert rt.paired_recovery(table,k,[C])[C]
    if len(table)==1:
        assert all(rt.paired_recovery(table,k,list(range(k,C+1))).values())


def test_theorem_formula_gap_and_sufficient_chunk_bound():
    g = rt.balanced_geometry(32000,1000)
    d = rt.theorem_diagnostic([500,300,150,50],g,2,2,.95)
    expected = 1-32*2*(math.exp(-1000*(.5-.15)**2/2)+math.exp(-1000*(.3-.15)**2/2))
    assert d.raw_bound == pytest.approx(expected)
    assert d.clipped_bound == pytest.approx(max(0,expected))
    assert d.empirical_gap == pytest.approx(.15)
    assert d.sufficient_n_min == pytest.approx(2/(.15**2)*math.log(32*2*2/.05))
    assert d.theorem_applicable_to_empirical_profile and d.strict_cutoff_gap
    assert not d.calibrated_deployment_bound and not d.open_semantic_guarantee


def test_full_cutoff_tie_invalidates_theorem_despite_positive_rank_r_gap():
    d = rt.theorem_diagnostic([40,30,30,0],rt.balanced_geometry(100,20),2,3,.95)
    assert d.empirical_gap == .3
    assert not d.strict_cutoff_gap and not d.theorem_applicable_to_empirical_profile
    assert d.sufficient_n_min is None
    exact = rt.theorem_diagnostic([40,30,30,0],rt.balanced_geometry(100,20),2,4,.95)
    assert exact.exact_count_special_case == 'retain_all' and exact.raw_bound == 1.0
    assert not exact.theorem_applicable_to_empirical_profile


def test_theorem_vacuous_and_exact_special_cases():
    d = rt.theorem_diagnostic([3,2,1],rt.balanced_geometry(6,2),1,1,1.0)
    assert d.raw_bound < 0 and d.clipped_bound == 0 and d.vacuous
    assert d.sufficient_n_min is None
    for g,k,r,special in [(rt.balanced_geometry(6,6),1,1,'single_chunk'),(rt.balanced_geometry(6,2),3,3,'retain_all')]:
        d = rt.theorem_diagnostic([3,2,1],g,k,r,.95)
        assert d.raw_bound == 1 and d.clipped_bound == 1 and not d.vacuous
        assert d.exact_count_special_case == special


@pytest.mark.parametrize('change,match', [({'max_table_cells':1},'cell budget'),({'max_simulation_cells':1},'cell budget'),({'max_work_units':1},'work budget')])
def test_resource_budgets_refuse_before_sampling(monkeypatch,change,match):
    monkeypatch.setattr(rt.np.random,'SeedSequence',lambda *args:pytest.fail('sampling must not start'))
    with pytest.raises(rt.NumericBudgetExceeded,match=match):
        plan(cfg=config(**change))


def test_cancellation_before_sampling_and_during_retained_ranks(monkeypatch):
    class Cancelled(Exception): pass
    def immediate(): raise Cancelled()
    with pytest.raises(Cancelled): plan(check_cancel=immediate)
    calls = []
    original = rt._recover
    def wrapped(table,k,candidates,check=None):
        def during_rank():
            calls.append(1)
            if len(calls)==2: raise Cancelled()
            check()
        return original(table,k,candidates,check=during_rank)
    monkeypatch.setattr(rt,'_recover',wrapped)
    with pytest.raises(Cancelled): plan()
    assert len(calls)==2


def test_walltime_bound_before_draw(monkeypatch):
    times = iter([0.0,31.0])
    monkeypatch.setattr(rt.time,'monotonic',lambda:next(times))
    with pytest.raises(rt.NumericBudgetExceeded,match='walltime'):
        plan()


@pytest.mark.parametrize('rows,k,candidates', [([[True,1]],1,[1]), ([[1.0,2]],1,[1]),
    ([[1,-1]],1,[1]), ([[1],[1,2]],1,[1]), ([[1,2]],3,[3]), ([[1,2]],1,[1,1]),
    ([[1,2]],1,[3]), ([[2**63,1]],1,[1])])
def test_external_count_table_authority(rows,k,candidates):
    with pytest.raises(ValidationError): rt.paired_recovery(rows,k,candidates)


def test_decision_serialization_rejects_tampered_certification_and_statistics():
    original = plan().model_dump()
    assert rt.RetentionDecision.model_validate(original).model_dump() == original
    for mutate in [lambda d:d.update(posterior_parameters=[1.0]*4),
        lambda d:d.update(simulated_category_count=3),
        lambda d:d.update(completed_tables=19),
        lambda d:d['diagnostics'][0].update(mean_recovery=.123),
        lambda d:d['diagnostics'][0].update(candidate_volume=10000),
        lambda d:d.update(posterior_profiles=[[.1,.1,.1,.1]]*4)]:
        import copy
        data = copy.deepcopy(original)
        mutate(data)
        with pytest.raises(ValidationError): rt.RetentionDecision.model_validate(data)
    fallback = plan(counts=(18,),scope='observed').model_dump()
    fallback['quantile_target_met'] = True
    with pytest.raises(ValidationError): rt.RetentionDecision.model_validate(fallback)


def test_all_new_models_generate_strict_json_schema():
    for model in [rt.RetentionConfig,rt.PilotCounts,rt.BalancedGeometry,rt.BalancedPartitionPlan,
        rt.RetentionRequest,rt.TheoremDiagnostic,rt.CandidateRetentionDiagnostic,rt.RetentionDecision,rt.CountTable]:
        assert model.model_json_schema()['additionalProperties'] is False
