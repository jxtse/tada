"""Combination coverage, reservation races, and actual pipeline overlap."""
import json
import os
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tada_taskgen import combinations
from tada_taskgen.concurrency import borrowed_slot, model_slot, slot_is_held, subprocess_environment
from tada_taskgen.scheduler import Hooks, Scheduler, TRIAL_MODELS, production_trial, qualified_exclusions


def intents(sources=('MAUD', 'CUAD', 'NDA')):
    return [{'id': 'intent-' + source, 'sources': [source], 'combination': source,
             'quota_key': source, 'onet_tasks': []} for source in sources]


def fixture_hooks(tmp_path):
    def prepare(intent, root):
        task = root / 'task'
        (task / 'tests').mkdir(parents=True)
        (task / 'tests/gold.json').write_text(json.dumps({'cells': [{'id': 'c', 'answer': 'Yes'}]}))
        return {'verdict': 'ready', 'spec': {'sources': intent['sources'], 'category': 'parallel_coding'},
                'task': str(task), 'gates': {'pass': True, 'eligible_instances': [7],
                    'instances': [{'seed': 7, 'task': str(task), 'keyword': .2}]}, 'review': {},
                'revisions': 0, 'failures': []}

    none = lambda *a, **k: {}
    return Hooks(design=none, gates=none, probe=none, difficulty=none, review=none, prepare=prepare,
                 trial=lambda task, model, index, out: {'valid': True, 'score': .7, 'index': index, 'wrong_cells': []},
                 adjudicate=lambda *a: {'confirmed_exclusions': []})


def test_enumeration_is_canonical_bounded_and_hashed():
    first = combinations.enumerate_spaces(['NDA', 'MAUD', 'CUAD', 'CUADX', 'B', 'PREC'])
    second = combinations.enumerate_spaces(['PREC', 'NDA', 'MAUD', 'C2', 'B', 'CUADX'])
    assert first == second and len(first['combinations']) == 41
    assert len({r['key'] for r in first['combinations']}) == 41
    assert all(1 <= len(r['sources']) <= 3 for r in first['combinations'])
    assert combinations.enumerate_spaces(['MAUD', 'CUAD'], 1)['sha256'] != combinations.enumerate_spaces(['MAUD', 'CUAD'], 2)['sha256']
    with pytest.raises(ValueError):
        combinations.enumerate_spaces(['MAUD'], 4)


def test_admission_quota_attempts_and_seeded_tie_break():
    space = combinations.enumerate_spaces(['MAUD', 'CUAD', 'NDA'], 1)
    state = {'final_verdicts': [
        {'sources': ['MAUD'], 'verdict': 'admitted'}, {'sources': ['MAUD'], 'verdict': 'admitted'},
        {'sources': ['CUAD'], 'verdict': 'discarded'}]}
    rows = combinations.counts(state, space)
    assert rows['MAUD']['admitted'] == 2 and rows['CUAD']['admitted'] == 0
    assert combinations.choose(rows)['key'] == 'NDA'
    state['final_verdicts'].append({'sources': ['NDA'], 'verdict': 'discarded'})
    rows = combinations.counts(state, space)
    assert combinations.choose(rows, seed=7) == combinations.choose(rows, seed=7)
    assert combinations.choose(rows)['key'] in {'CUAD', 'NDA'}
    rows['CUAD']['reserved'] = rows['NDA']['reserved'] = 2
    assert combinations.choose(rows) is None


def test_legacy_migration_reads_spec_without_changing_any_file(tmp_path):
    candidate = tmp_path / 'candidate-001'
    candidate.mkdir()
    (candidate / 'scenario_spec.json').write_text('{"source":"MAUD"}')
    state = {'dry_run': False, 'final_verdicts': [{'id': 'candidate-001', 'verdict': 'admitted'}]}
    path = tmp_path / 'state.json'
    path.write_text(json.dumps(state))
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    report = combinations.report(state, ['MAUD', 'CUAD'], base=tmp_path)
    assert next(r for r in report['counts'] if r['key'] == 'MAUD')['admitted'] == 1
    after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after and report['read_only']


def test_report_cli_never_creates_state_or_swallows_missing_path(tmp_path, monkeypatch, capsys):
    from tada_taskgen.scheduler import main
    monkeypatch.setattr('sys.argv', ['scheduler', '--combinations-report'])
    main()
    report = json.loads(capsys.readouterr().out)
    from tada_taskgen.adapters import available_sources
    expected = combinations.enumerate_spaces(available_sources())
    assert report['read_only'] and len(report['counts']) == len(expected['combinations'])
    assert report['available_sources'] == expected['available_sources']
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr('sys.argv', ['scheduler', '--combinations-report', '--state', str(tmp_path / 'missing.json')])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and 'State file does not exist' in capsys.readouterr().err


def test_unmapped_legacy_admission_is_explicit_and_blocks_new_reservations(tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    state = {'dry_run': True, 'final_verdicts': [{'id': 'candidate-001', 'verdict': 'admitted'}],
             'trial_sets': 0, 'coverage': {}, 'stages': {}, 'tokens': 0, 'cost_usd': 0, 'discarded_tokens': 0}
    (out / 'state.json').write_text(json.dumps(state))
    report = combinations.report(state, ['MAUD'], base=out)
    assert report['unmapped_verdicts'][0]['reason'] == 'unknown_source'
    scheduler = Scheduler(out, fixture_hooks(tmp_path), dry_run=True)
    with pytest.raises(ValueError, match='cannot be mapped'):
        scheduler._reservation(intents()[:1][0])


def test_enumeration_or_hash_drift_cannot_rewrite_existing_state(tmp_path):
    out = tmp_path / 'run'
    hooks = fixture_hooks(tmp_path)
    Scheduler(out, hooks, dry_run=True, available_sources=['MAUD']).run([], loops=0)
    path = out / 'state.json'
    before = path.read_bytes()
    with pytest.raises(ValueError, match='space/hash differs'):
        Scheduler(out, hooks, dry_run=True, available_sources=['MAUD', 'CUAD'])
    assert path.read_bytes() == before
    state = json.loads(before)
    state['combination_space']['sha256'] = 'tampered'
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='space/hash differs'):
        Scheduler(out, hooks, dry_run=True, available_sources=['MAUD'])
    assert path.read_bytes() == before


def test_legacy_prepared_candidate_resumes_saved_intent_before_new_combination(tmp_path):
    hooks = fixture_hooks(tmp_path)
    out = tmp_path / 'run'
    root = out / 'candidate-001'
    root.mkdir(parents=True)
    saved_intent = intents()[:1][0]
    (root / 'intent.json').write_text(json.dumps(saved_intent))
    prepared = hooks.prepare(saved_intent, root)
    (root / 'design_loop.json').write_text(json.dumps(prepared))
    hooks.prepare = lambda *a: pytest.fail('Legacy prepared task must not be redesigned')
    result = Scheduler(out, hooks, dry_run=True).run(intents(('CUAD',)), loops=1)
    assert result['verdicts'][0]['intent'] == saved_intent['id']
    assert result['verdicts'][0]['combination'] == 'MAUD'
    assert json.loads((root / 'intent.json').read_text()) == saved_intent


@pytest.mark.parametrize('entrypoint', ['run', 'run_one'])
def test_prepared_task_without_saved_intent_cannot_be_reassigned(tmp_path, entrypoint):
    hooks = fixture_hooks(tmp_path)
    out = tmp_path / 'run'
    root = out / 'candidate-001'
    root.mkdir(parents=True)
    prepared = hooks.prepare(intents()[:1][0], root)
    saved = root / 'design_loop.json'
    saved.write_text(json.dumps(prepared))
    before = saved.read_bytes()
    scheduler = Scheduler(out, hooks, dry_run=True)
    with pytest.raises(RuntimeError, match='missing intent.json'):
        if entrypoint == 'run':
            scheduler.run(intents(), loops=1)
        else:
            scheduler.run_one(intents()[:1][0])
    assert not (root / 'intent.json').exists() and saved.read_bytes() == before
    assert scheduler.state['final_verdicts'] == [] and scheduler.state['trial_sets'] == 0


def test_four_trials_overlap_with_model_caps_and_preserve_order(tmp_path):
    hooks = fixture_hooks(tmp_path)
    lock, ds_pair, first_opus = threading.Lock(), threading.Event(), threading.Event()
    active = {'deepseek-flash': 0, 'claude-opus-5-5': 0}
    maximum = dict(active)
    finished = []

    def trial(task, model, index, out):
        with lock:
            active[model] += 1
            maximum[model] = max(maximum[model], active[model])
            if active['deepseek-flash'] == 2:
                ds_pair.set()
            if model == 'claude-opus-5-5':
                first_opus.set()
        assert ds_pair.wait(3) and first_opus.wait(3)
        if index == 1:
            time.sleep(.08)
        with lock:
            finished.append(index)
            active[model] -= 1
        return {'valid': True, 'score': .6, 'index': index, 'wrong_cells': []}

    hooks.trial = trial
    scheduler = Scheduler(tmp_path / 'run', hooks, dry_run=True)
    verdict = scheduler.run(intents()[:1])['verdicts'][0]
    assert [t['index'] for t in verdict['trials']] == [1, 2, 3, 4]
    assert [t['model'] for t in verdict['trials']] == list(TRIAL_MODELS)
    assert maximum == {'deepseek-flash': 2, 'claude-opus-5-5': 1}
    assert finished.index(2) < finished.index(1)


def test_next_candidate_prepares_while_previous_trials_run(tmp_path):
    hooks = fixture_hooks(tmp_path)
    original = hooks.prepare
    trial_started, next_prepared = threading.Event(), threading.Event()

    def prepare(intent, root):
        if root.name == 'candidate-002':
            assert trial_started.wait(5)
            result = original(intent, root)
            next_prepared.set()
            return result
        return original(intent, root)

    def trial(task, model, index, out):
        if task.parent.name == 'candidate-001':
            trial_started.set()
            assert next_prepared.wait(5)
        return {'valid': True, 'score': .6, 'wrong_cells': []}

    hooks.prepare, hooks.trial = prepare, trial
    scheduler = Scheduler(tmp_path / 'run', hooks, dry_run=True)
    result = scheduler.run(intents(), loops=3)
    assert result['completed'] == 3 and result['max_inflight_observed'] == 2
    assert len({v['combination'] for v in result['verdicts']}) == 3
    assert scheduler.state['reservations'] == {}


def test_verdict_reservation_atomic_between_scheduler_instances(tmp_path):
    out = tmp_path / 'run'
    schedulers = [Scheduler(out, fixture_hooks(tmp_path), dry_run=True, max_verdicts=1) for _ in range(2)]
    barrier = threading.Barrier(2)

    def reserve(scheduler):
        barrier.wait(timeout=3)
        return scheduler._reservation(intents()[:1][0])

    with ThreadPoolExecutor(max_workers=2) as pool:
        reservations = list(pool.map(reserve, schedulers))
    assert sum(r is not None for r in reservations) == 1
    assert len(json.loads((out / 'state.json').read_text())['reservations']) == 1


def test_parallel_candidate_bound_is_shared_between_scheduler_instances(tmp_path):
    out = tmp_path / 'run'
    schedulers = [Scheduler(out, fixture_hooks(tmp_path), dry_run=True) for _ in range(3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        reservations = list(pool.map(lambda pair: pair[0]._reservation(pair[1]), zip(schedulers, intents())))
    assert sum(r is not None for r in reservations) == 2


def test_trial_set_reservation_atomic_and_all_infra_refund(tmp_path):
    hooks = fixture_hooks(tmp_path)
    ready = threading.Barrier(2)
    hooks.trial_preflight = lambda: ready.wait(timeout=3) and None
    # Barrier.wait may return zero; both zero and None mean preflight success.
    scheduler = Scheduler(tmp_path / 'run', hooks, dry_run=True, max_trial_sets=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(scheduler._trials, tmp_path, tmp_path / str(i)) for i in range(2)]
        results = [f.result() for f in futures]
    assert scheduler.state['trial_sets'] == 1
    assert sorted(len(r[0]) for r in results) == [0, 4]
    assert sum(bool(r[1]) for r in results) == 1

    hooks.trial_preflight = None
    hooks.trial = lambda *a: {'valid': False, 'infra': True, 'score': None}
    refunded = Scheduler(tmp_path / 'refund', hooks, dry_run=True, max_trial_sets=1)
    with pytest.raises(RuntimeError, match='infrastructure'):
        refunded._trials(tmp_path, tmp_path / 'refund/trials')
    assert refunded.state['trial_sets'] == 0


def test_failed_candidate_is_not_resubmitted_before_failed_future_returns(tmp_path, monkeypatch):
    import tada_taskgen.scheduler as module
    hooks = fixture_hooks(tmp_path)
    prepare = hooks.prepare
    failure_published, release_failed_future, retry_allowed = threading.Event(), threading.Event(), threading.Event()

    def saved_prepare(intent, root):
        if root.name == 'candidate-002':
            assert failure_published.wait(5)
        prepared = prepare(intent, root)
        (root / 'design_loop.json').write_text(json.dumps(prepared))
        return prepared

    def trial(task, model, index, out):
        if not retry_allowed.is_set():
            if task.parent.name == 'candidate-001':
                return {'valid': False, 'infra': True}
            assert failure_published.wait(5)
        return {'valid': True, 'score': .7, 'wrong_cells': []}

    class DelayedFailureScheduler(Scheduler):
        def run_one(self, *args, **kwargs):
            try:
                return super().run_one(*args, **kwargs)
            except RuntimeError:
                failure_published.set()
                assert release_failed_future.wait(5)
                raise

    calls, actual_wait = [], module.wait

    def observe_wait(futures, **kwargs):
        calls.append(len(futures))
        if len(calls) == 2:
            release_failed_future.set()
            assert len(futures) == 1, 'Pending failed candidate was resubmitted before its error was observed'
        return actual_wait(futures, **kwargs)

    hooks.prepare, hooks.trial = saved_prepare, trial
    monkeypatch.setattr(module, 'wait', observe_wait)
    scheduler = DelayedFailureScheduler(tmp_path / 'run', hooks, dry_run=True)
    with pytest.raises(RuntimeError, match='infrastructure'):
        scheduler.run(intents(), loops=3)
    assert calls == [2, 1]
    assert scheduler.state['stages']['intent'] == 2 and scheduler.state['trial_sets'] == 1
    assert len(scheduler.state['final_verdicts']) == 1 and len(scheduler.state['reservations']) == 1
    retry_allowed.set()
    result = scheduler.run(intents(), loops=1)
    assert result['completed'] == 2 and result['trial_sets'] == 2
    assert not scheduler.state['reservations'] and scheduler.state['stages']['resume_prepared'] == 1


def test_quota_reservation_blocks_third_and_failed_candidate_does_not_close(tmp_path):
    scheduler = Scheduler(tmp_path / 'run', fixture_hooks(tmp_path), dry_run=True, max_parallel_candidates=3)
    chosen = intents()[:1][0]
    first, second = scheduler._reservation(chosen), scheduler._reservation(chosen)
    assert first and second and scheduler._reservation(chosen) is None
    with scheduler.transaction():
        scheduler.state['final_verdicts'].append({'id': first['id'], 'sources': chosen['sources'], 'verdict': 'discarded'})
        del scheduler.state['reservations'][first['id']]
    assert scheduler._reservation(chosen) is not None


def test_legacy_intent_without_source_cannot_bypass_final_admission_quota(tmp_path):
    hooks = fixture_hooks(tmp_path)
    original = hooks.prepare
    hooks.prepare = lambda intent, root: original(dict(intent, sources=['MAUD']), root)
    scheduler = Scheduler(tmp_path / 'run', hooks, dry_run=True)
    result = scheduler.run([{'id': 'legacy', 'quota_key': 'old-occupation-quota'}], loops=3,
                           quotas={'old-occupation-quota': 10})
    assert [v['verdict'] for v in result['verdicts']].count('admitted') == 2
    # Admission is serialized in completion order, while returned verdicts are
    # sorted by candidate ID. A slower earlier candidate can finish last.
    discarded = [v for v in result['verdicts'] if v['verdict'] == 'discarded']
    assert len(discarded) == 1
    assert discarded[0]['failures'][0]['gate'] == 'combination_admission_quota'


def test_model_slot_inheritance_is_thread_local_and_explicit(monkeypatch):
    monkeypatch.delenv('TADA_MODEL_SLOT_HELD', raising=False)
    monkeypatch.delenv('TADA_MODEL_SLOT_FAMILY', raising=False)
    before = dict(os.environ)
    with borrowed_slot('deepseek-flash'):
        assert slot_is_held('deepseek-flash')
        assert subprocess_environment('deepseek-flash')['TADA_MODEL_SLOT_FAMILY'] == 'deepseek'
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(slot_is_held, 'deepseek-flash').result() is False
        assert dict(os.environ) == before
    assert not slot_is_held('deepseek-flash') and dict(os.environ) == before


def test_borrowed_slot_forwards_an_explicit_synthetic_lock_root(tmp_path, monkeypatch):
    import tada_taskgen.concurrency as module
    seen = []

    @contextmanager
    def fake_slot(model, *, lock_root=None):
        seen.append((model, lock_root))
        yield

    monkeypatch.delenv('TADA_MODEL_SLOT_HELD', raising=False)
    monkeypatch.delenv('TADA_MODEL_SLOT_FAMILY', raising=False)
    monkeypatch.setattr(module, 'model_slot', fake_slot)
    with borrowed_slot('opus', lock_root=tmp_path):
        assert slot_is_held('opus')
    assert seen == [('opus', tmp_path)]


@pytest.mark.parametrize('dry_run', [True, False])
def test_only_dry_trials_use_isolated_synthetic_slots(tmp_path, monkeypatch, dry_run):
    import tada_taskgen.scheduler as module
    seen = []

    @contextmanager
    def capture_slot(model, *, lock_root=None):
        seen.append((model, lock_root))
        yield

    monkeypatch.setattr(module, 'borrowed_model_slot', capture_slot)
    scheduler = Scheduler(tmp_path / 'run', fixture_hooks(tmp_path), dry_run=dry_run)
    trials, failures = scheduler._trials(tmp_path, tmp_path / 'trials')
    expected = scheduler.out / 'fake_model_slots' if dry_run else None
    assert len(trials) == 4 and not failures
    assert sorted(model for model, _ in seen) == sorted(TRIAL_MODELS)
    assert all(root == expected for _, root in seen)


def test_default_slot_queue_can_wait_longer_than_one_hour(tmp_path, monkeypatch):
    import tada_taskgen.concurrency as module
    elapsed, attempts = [0.0], [0]

    def flock(stream, operation):
        if operation & module.fcntl.LOCK_NB:
            attempts[0] += 1
            if attempts[0] < 3:
                raise BlockingIOError('Another legitimate long trial still holds the slot')

    monkeypatch.setattr(module.fcntl, 'flock', flock)
    monkeypatch.setattr(module.time, 'monotonic', lambda: elapsed[0])
    monkeypatch.setattr(module.time, 'sleep', lambda _: elapsed.__setitem__(0, elapsed[0] + 7200))
    with model_slot('opus', lock_root=tmp_path):
        assert elapsed[0] == 14400 and attempts[0] == 3


@pytest.mark.parametrize('source,confirmed,expected', [
    ('MAUD', 'MAUD:r:q', True), ('CUAD', 'MAUD:r:q', False),
    ('MAUD', 'r:q', False), ('NDA', 'NDA:r:q', False)])
def test_multisource_qualified_exclusions_use_full_original_identity(source, confirmed, expected):
    audit = {'sources': ['MAUD', 'CUAD'], 'confirmed_exclusions': [confirmed], 'proposals': [{
        'source': source, 'record': 'r', 'question': 'q', 'original_answer': 'No', 'action': 'exclude',
        'readings': [{'answer': 'Yes', 'confidence': 'high'}] * 2,
        'structural_evidence': {'kind': 'pool_majority', 'total': 10, 'support': 9, 'answer': 'Yes'}}]}
    assert qualified_exclusions(audit) is expected
    audit.pop('sources')
    audit['confirmed_exclusions'] = ['r:q']
    assert qualified_exclusions(audit)


def test_remote_trial_names_distinguish_identical_tasks_in_different_candidate_outputs(tmp_path):
    from types import SimpleNamespace
    task = tmp_path / 'task'
    (task / 'tests').mkdir(parents=True)
    (task / 'tests/gold.json').write_text('{"cells":[{"id":"c"}]}')
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1, stdout='', stderr='Synthetic launcher result')

    for candidate in ('candidate-001', 'candidate-002'):
        production_trial(task, 'deepseek-flash', 1, tmp_path / candidate / 'trials-initial', runner=runner)
    assert calls[0][0][4] != calls[1][0][4]
    assert all(call[0][4].startswith('trial-1-deepseek-flash-') and call[1]['timeout'] == 34200 for call in calls)
