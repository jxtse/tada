"""Quota-driven slim pilot; dry-run fixtures and real execution are explicit.

Trials label difficulty. Only mechanical gates and four-question review gate
admission; the batch red team is always a report-only measurement.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
import threading
import time
import tomllib
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .clients import FakeClient
from .concurrency import borrowed_slot, subprocess_environment
from . import combinations

REPO = Path(__file__).resolve().parents[2]
SEEDS = (7,)
TRIAL_MODELS = ('deepseek-flash', 'deepseek-flash', 'claude-opus-5-5', 'claude-opus-5-5')
PASS_REWARD = .80


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


borrowed_model_slot = borrowed_slot


def scheduler_settings():
    settings = tomllib.loads((REPO / 'v5/config.toml').read_text())['scheduler']
    if settings.get('max_combination_size', settings['combination_max_size']) != settings['combination_max_size']:
        raise ValueError('Legacy max_combination_size must match combination_max_size')
    return settings


@dataclass
class Hooks:
    design: Callable
    gates: Callable
    probe: Callable
    difficulty: Callable
    review: Callable
    trial: Callable
    adjudicate: Callable
    repair: Callable | None = None
    rebuild: Callable | None = None
    prepare: Callable | None = None
    quarantine: Callable | None = None
    redteam: Callable | None = None
    trial_preflight: Callable | None = None


def common_errors(trials: list[dict]) -> list[str]:
    if len(trials) != 4 or any(t.get('valid') is not True for t in trials):
        return []
    return sorted(set.intersection(*(set(t.get('wrong_cells', [])) for t in trials)))


def valid_score(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def difficulty_label(trials: list[dict]) -> dict:
    """Difficulty is a stratum, never an admission threshold."""
    if len(trials) != 4 or Counter(t.get('model') for t in trials) != Counter(TRIAL_MODELS):
        raise ValueError('Difficulty requires exactly DeepSeek x2 and Opus x2')
    if any(t.get('valid') is not True or not valid_score(t.get('score', t.get('reward'))) for t in trials):
        raise ValueError('Difficulty requires four valid rewards in [0, 1]')
    rewards = [t.get('score', t.get('reward')) for t in trials]
    passes = sum(r >= PASS_REWARD for r in rewards)
    return {'label': 'easy' if passes >= 3 else 'medium' if passes == 2 else 'hard',
            'pass_reward': PASS_REWARD, 'passes': passes, 'trial_count': 4,
            'per_model_rewards': {m: [r for t, r in zip(trials, rewards) if t['model'] == m]
                                  for m in dict.fromkeys(TRIAL_MODELS)}}


def review_failures(review):
    """Admission blockers only: severity 3 (or legacy 'blocker'). Severity 2 drives revisions upstream
    and, once the revision budget is spent, is recorded as a note rather than blocking (owner policy)."""
    findings = review.get('findings', review.get('questions', []))
    if isinstance(findings, dict):
        findings = list(findings.values())
    return [f for f in findings if f.get('severity') == 'blocker'
            or isinstance(f.get('severity'), int) and not isinstance(f.get('severity'), bool) and f['severity'] >= 3]


def auto_verdict(gates: dict, probe: dict, review: dict, trials: list[dict], keyword: float) -> dict:
    trials = [dict(t, score=t.get('score', t.get('reward'))) for t in trials]
    failures = []
    if not valid_score(keyword):
        failures.append({'gate': 'valid_keyword_score', 'margin': None})
    invalid = [i for i, trial in enumerate(trials) if trial.get('valid') is True and not valid_score(trial.get('score'))]
    if invalid:
        failures.append({'gate': 'valid_trial_scores', 'margin': len(invalid), 'trial_indices': invalid})
    if gates.get('pass') is not True:
        failures.append({'gate': 'hard_gates', 'margin': None, 'details': gates.get('failures', [])})
    for finding in review_failures(review):
        failures.append({'gate': 'review', 'margin': None, 'details': finding})
    ds = [t['score'] for t in trials if t.get('valid') is True and valid_score(t.get('score')) and t.get('model') == 'deepseek-flash']
    opus = [t['score'] for t in trials if t.get('valid') is True and valid_score(t.get('score')) and t.get('model') == 'claude-opus-5-5']
    if len(ds) != 2 or len(opus) != 2:
        failures.append({'gate': 'four_valid_trials', 'margin': 4 - len(ds) - len(opus)})
    difficulty = None
    try:
        difficulty = difficulty_label(trials)
    except ValueError:
        if not any(f['gate'] == 'four_valid_trials' for f in failures):
            failures.append({'gate': 'four_valid_trials', 'margin': None})
    return {'verdict': 'discarded' if failures else 'admitted', 'failures': failures,
            'difficulty': difficulty,
            'reported_only': {'spec_probe': probe, 'multi_seed': gates.get('reported_only', {}),
                              'deepseek_advantage': statistics.mean(ds) - keyword if len(ds) == 2 and valid_score(keyword) else None,
                              'opus_max_reward': max(opus) if opus else None}}


def gate_failures(gates: dict) -> list[dict]:
    failures = []
    if gates.get('hard_gates'):
        return [{'gate': name, 'details': row, 'value': row.get('value'), 'limit': row.get('limit'), 'margin': None}
                for name, row in gates['hard_gates'].items() if row.get('pass') is not True]
    instances = gates.get('instances', [])
    if isinstance(instances, dict):
        instances = list(instances.values())
    for instance in instances:
        for error in instance.get('errors', []):
            failures.append({'gate': 'controller', 'seed': instance.get('seed'), 'margin': None, 'details': error})
        for name, metric in instance.get('metrics', {}).items():
            if metric.get('pass') is not True:
                value, limit = metric.get('value'), metric.get('limit')
                failures.append({'gate': name, 'seed': instance.get('seed'), 'value': value,
                                 'limit': limit, 'margin': max(0, value - limit)
                                 if isinstance(value, (int, float)) and isinstance(limit, (int, float)) else None})
        for name, passed in instance.get('engineering', {}).items():
            if passed is not True:
                failures.append({'gate': name, 'seed': instance.get('seed'), 'margin': None,
                                 'details': 'Engineering/correctness gate failed'})
    return failures or [{'gate': 'hard_gates', 'margin': None,
                         'details': gates.get('failures', 'No eligible instance or incomplete gate report')}]


def selected_instance(gates: dict) -> dict:
    eligible = gates.get('eligible_instances', [])
    if not eligible:
        raise ValueError('No eligible instance')
    selected = eligible[0]
    if isinstance(selected, dict):
        return selected
    instances = gates.get('instances', [])
    if isinstance(instances, dict):
        instances = list(instances.values())
    instance = next((i for i in instances if i.get('seed') == selected or i.get('id') == selected), None)
    if instance is None:
        raise ValueError('Selected eligible instance is absent from the gate report')
    return instance


def harbor_verified(gates: dict) -> bool:
    """Verify the exact instance selected for trials, not another passing seed."""
    try:
        instance = selected_instance(gates)
    except ValueError:
        return False
    if instance.get('harbor_status') is True:
        return True
    report = instance.get('report')
    if report and Path(report).is_file():
        measured = json.loads(Path(report).read_text())
        return measured.get('verdict', {}).get('harbor_oracle_nop') is True
    return False


def qualified_exclusions(audit: dict) -> bool:
    proposals = audit.get('proposals', [])
    confirmed = audit.get('confirmed_exclusions', [])
    if not confirmed:
        return True
    matched = set()
    for item in proposals:
        readings = item.get('readings', [])
        structural = item.get('structural_evidence', {}) or {}
        agreeing = (len(readings) == 2 and readings[0].get('answer') is not None
                    and readings[0].get('answer') == readings[1].get('answer')
                    and readings[0]['answer'] != item.get('original_answer')
                    and all(r.get('confidence') == 'high' for r in readings))
        structure_ok = ((structural.get('kind') == 'pool_majority'
                         and structural.get('total', 0) >= 5
                         and structural.get('support', 0) / structural['total'] >= .8
                         and structural.get('answer') == readings[0].get('answer')) if agreeing else False)
        structure_ok |= (structural.get('kind') == 'citable_clause'
                         and len(structural.get('quotes', [])) == 2
                         and all(len(q.strip()) >= 25 for q in structural['quotes'])
                         and len(structural.get('offsets', [])) == 2)
        if item.get('action') == 'exclude' and agreeing and structure_ok:
            key = str(item.get('record')) + ':' + str(item.get('question'))
            if audit.get('sources'):
                if item.get('source') not in audit['sources']:
                    continue
                key = str(item['source']) + ':' + key
            matched.add(key)
    return set(confirmed) <= matched


def _eligible(gates: dict) -> list:
    return gates.get('eligible_instances', [])


def _keyword(instance: dict):
    for key in ('keyword', 'keyword_score'):
        if key in instance and instance[key] is not None:
            return instance[key]
    numeric = instance.get('metrics', instance.get('numeric', {}))
    value = numeric.get('keyword', numeric.get('keyword_baseline'))
    if isinstance(value, dict):
        value = value.get('value', value.get('score'))
    return value


def _task_for(gates: dict, root: Path) -> tuple[Path, float]:
    instance = selected_instance(gates)
    path = instance.get('task') or instance.get('task_path') or instance.get('package')
    if not path and instance.get('report'):
        path = Path(instance['report']).parent / 'task'
    if not path:
        seed = instance.get('seed')
        path = root / 'multiseed' / f'seed-{seed}' / 'task'
    return Path(path), _keyword(instance)


def scored_gold(task: Path):
    """Compare only values and identities consumed by the frozen scorer."""
    path = task / 'tests/gold.json'
    if not path.is_file():
        raise ValueError('Cannot compare scored gold without private gold.json')
    gold = json.loads(path.read_text())
    return {name: gold.get(name, []) for name in ('cells', 'facts', 'scopes', 'consistency')}


class Scheduler:
    def __init__(self, out: Path, hooks: Hooks, *, dry_run=False, max_verdicts=10, max_trial_sets=15,
                 max_parallel_candidates=None, available_sources=None):
        if not 1 <= max_verdicts <= 10 or not 1 <= max_trial_sets <= 15:
            raise ValueError('Pilot caps may not exceed 10 verdicts / 15 trial sets')
        self.out, self.hooks, self.dry_run = Path(out), hooks, dry_run
        self.max_verdicts, self.max_trial_sets = max_verdicts, max_trial_sets
        settings = scheduler_settings()
        self.max_parallel_candidates = (settings['max_parallel_candidates'] if max_parallel_candidates is None
                                        else max_parallel_candidates)
        if not 1 <= self.max_parallel_candidates <= 10:
            raise ValueError('max_parallel_candidates must be within 1..10')
        self.quota = settings['quota_per_combination']
        if self.quota != 2:
            raise ValueError('Owner policy fixes quota_per_combination at 2 admitted tasks')
        self.seed = settings['combination_seed']
        if available_sources is None:
            from .adapters import available_sources as registry_sources
            available_sources = registry_sources()
        self.enumeration = combinations.enumerate_spaces(available_sources, settings['combination_max_size'])
        self._mutex, self._local = threading.RLock(), threading.local()
        self._abort = threading.Event()
        self.owner = uuid.uuid4().hex
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = self.out / 'state.json'
        self.state = self._load()

    def _load(self):
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state['dry_run'] != self.dry_run:
                raise ValueError('Do not mix dry-run and production state')
            recorded = state.get('combination_space')
            if recorded is not None and recorded != self.enumeration:
                raise ValueError('Recorded combination space/hash differs from the configured enumeration; '
                                 'resume requires the original task space')
        else:
            state = {'dry_run': self.dry_run, 'final_verdicts': [], 'trial_sets': 0, 'coverage': {},
                     'stages': {}, 'cost_usd': 0.0, 'tokens': 0, 'discarded_tokens': 0}
        state.setdefault('reservations', {})
        state.setdefault('combination_attempts', {})
        return state

    @contextmanager
    def transaction(self):
        """Reload and commit under both a thread lock and the existing process lock.

        Model calls never hold this lock. Nested mutations share one snapshot.
        """
        with self._mutex:
            if getattr(self._local, 'transaction_depth', 0):
                yield
                return
            with (self.out / 'scheduler.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self._local.transaction_depth = 1
                try:
                    self.state = self._load()
                    yield
                    self.save()
                finally:
                    self._local.transaction_depth = 0
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def save(self):
        dump(self.path, self.state)

    def stage(self, event, details=None):
        with self.transaction():
            self.state['stages'][event] = self.state['stages'].get(event, 0) + 1
            if details:
                self.state['tokens'] += int(details.get('tokens', 0))
                self.state['cost_usd'] += float(details.get('cost_usd', 0))
                candidate = getattr(self._local, 'candidate', None)
                if candidate in self.state['reservations']:
                    reservation = self.state['reservations'][candidate]
                    reservation['tokens'] = reservation.get('tokens', 0) + int(details.get('tokens', 0))
                    reservation['cost_usd'] = reservation.get('cost_usd', 0) + float(details.get('cost_usd', 0))

    @staticmethod
    def intent_sources(intent):
        sources = intent.get('sources') or (intent['combination'].split('+') if intent.get('combination') else [])
        return combinations.normalize(sources or [intent.get('source')])

    def combination_counts(self):
        unresolved = [row for row in combinations.unmapped_verdicts(self.state, self.enumeration, self.out)
                      if row['verdict'] == 'admitted']
        if unresolved:
            raise ValueError('Admitted verdicts cannot be mapped to available combinations: ' + json.dumps(unresolved))
        return combinations.counts(self.state, self.enumeration, self.out)

    def _adopt_prepared_candidates(self):
        """Migrate only scheduler bookkeeping; preserve prepared tasks and intents."""
        finalized = {row['id'] for row in self.state['final_verdicts']}
        for path in sorted(self.out.glob('candidate-*/design_loop.json')):
            root = path.parent
            if root.name in finalized or root.name in self.state['reservations']:
                continue
            intent_path = root / 'intent.json'
            prepared = json.loads(path.read_text())
            if prepared.get('verdict') != 'ready':
                continue
            if not intent_path.is_file():
                raise RuntimeError(f'{root.name} is prepared but missing intent.json; cannot safely resume')
            if len(finalized) + len(self.state['reservations']) >= self.max_verdicts:
                break
            intent = json.loads(intent_path.read_text())
            spec = prepared.get('spec') or {}
            sources = self.intent_sources(intent) or combinations.normalize(spec.get('sources') or [spec.get('source')])
            if sources:
                intent.update(sources=sources, combination=combinations.key(sources))
            self.state['reservations'][root.name] = {'id': root.name, 'intent': intent,
                'sources': sources, 'combination': combinations.key(sources), 'phase': 'pending',
                'pid': 0, 'owner': None, 'tokens': 0, 'cost_usd': 0.0, 'migrated_prepared': True}

    def _set_phase(self, candidate, phase):
        with self.transaction():
            self.state['reservations'][candidate]['phase'] = phase
            self.state.setdefault('pipeline_events', []).append(
                {'candidate': candidate, 'phase': phase, 'time_ns': time.time_ns()})

    def _reservation(self, intent):
        """Reserve a verdict slot before any candidate work, including in flight."""
        with self.transaction():
            pending = [r for r in self.state['reservations'].values()
                       if r['intent'].get('id') == intent.get('id') and r.get('phase') == 'pending']
            if pending:
                if sum(r.get('phase') != 'pending' for r in self.state['reservations'].values()) >= self.max_parallel_candidates:
                    return None
                reservation = min(pending, key=lambda r: r['id'])
                reservation.update(owner=self.owner, pid=os.getpid(), phase='reserved')
                return copy.deepcopy(reservation)
            if self.stop_reason() or len(self.state['final_verdicts']) + len(self.state['reservations']) >= self.max_verdicts:
                return None
            if sum(r.get('phase') != 'pending' for r in self.state['reservations'].values()) >= self.max_parallel_candidates:
                return None
            sources = self.intent_sources(intent)
            combination = combinations.key(sources)
            if combination:
                row = self.combination_counts().get(combination)
                if row is None:
                    raise ValueError('Intent combination is outside the recorded task space: ' + combination)
                if row['admitted'] + row['reserved'] >= self.quota:
                    return None
            existing = [v['id'] for v in self.state['final_verdicts']] + list(self.state['reservations'])
            index = max([int(name.rsplit('-', 1)[-1]) for name in existing if name.startswith('candidate-')] or [0]) + 1
            name = f'candidate-{index:03d}'
            reservation = {'id': name, 'intent': copy.deepcopy(intent), 'combination': combination,
                           'sources': sources, 'owner': self.owner, 'pid': os.getpid(), 'phase': 'reserved',
                           'tokens': 0, 'cost_usd': 0.0}
            self.state['reservations'][name] = reservation
            if combination:
                attempts = self.state['combination_attempts']
                attempts[combination] = self.combination_counts()[combination]['attempts']
            self.state['combination_space'] = self.enumeration
            self.state['max_parallel_candidates'] = self.max_parallel_candidates
            self.state['max_inflight_observed'] = max(self.state.get('max_inflight_observed', 0),
                                                     len(self.state['reservations']))
            return copy.deepcopy(reservation)

    def _choose_intent(self, intents, quotas=None):
        # A refunded/prepared candidate is resumed only with its saved intent.
        for reservation in self.state['reservations'].values():
            if reservation.get('phase') == 'pending':
                return copy.deepcopy(reservation['intent'])
        grouped = {}
        for intent in intents:
            combination = combinations.key(self.intent_sources(intent))
            if combination:
                grouped.setdefault(combination, []).append(intent)
        if grouped:
            rows = {k: v for k, v in self.combination_counts().items() if k in grouped}
            selected = combinations.choose(rows, self.quota, self.seed, prefer_unused=True)
            if selected:
                choices = sorted(grouped[selected['key']], key=lambda i: i['id'])
                return choices[selected['attempts'] % len(choices)]
            return None
        # Compatibility for injected legacy intents with no dataset field.
        from .intents import select_undercovered
        coverage = dict(self.state['coverage'])
        for reservation in self.state['reservations'].values():
            key = reservation['intent'].get('quota_key', reservation['intent']['id'])
            coverage[key] = coverage.get(key, 0) + 1
        legacy = [dict(i, quota_key=i.get('quota_key', i['id'])) for i in intents]
        return select_undercovered(legacy, coverage, quotas)

    def stop_reason(self):
        if self.state.get('policy_stop'):
            return self.state['policy_stop']
        if len(self.state['final_verdicts']) >= self.max_verdicts:
            return 'pilot_verdict_cap'
        if self.state['trial_sets'] >= self.max_trial_sets:
            return 'pilot_trial_set_cap'
        return None

    def run(self, intents: list[dict], *, loops=1, quotas=None) -> dict:
        if loops < 0:
            raise ValueError('loops must be nonnegative')
        self._abort.clear()
        submitted, futures, errors = 0, {}, []
        with self.transaction():
            self.state['combination_space'] = self.enumeration
            self._adopt_prepared_candidates()
            # Only reclaim reservations whose owning process has actually exited.
            for reservation in self.state['reservations'].values():
                if reservation.get('phase') == 'pending':
                    continue
                try:
                    os.kill(reservation.get('pid', 0), 0)
                except ProcessLookupError:
                    reservation['phase'] = 'pending'
        with ThreadPoolExecutor(max_workers=self.max_parallel_candidates) as pool:
            while futures or submitted < loops:
                while not errors and not self._abort.is_set() and submitted < loops and len(futures) < self.max_parallel_candidates:
                    with self.transaction():
                        if self._abort.is_set() or self.stop_reason():
                            break
                        intent = self._choose_intent(intents, quotas)
                        reservation = self._reservation(intent) if intent is not None else None
                    if reservation is None:
                        break
                    future = pool.submit(self.run_one, intent, _reservation=reservation)
                    futures[future] = reservation['id']
                    submitted += 1
                if not futures:
                    break
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    futures.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append(exc)
        with self.transaction():
            from .redteam import sample_batch
            admitted = [v for v in self.state['final_verdicts'] if v['verdict'] == 'admitted']
            self.write_summary()
            result = {'completed': len(self.state['final_verdicts']), 'trial_sets': self.state['trial_sets'],
                      'stop_reason': self.stop_reason(), 'dry_run': self.dry_run,
                      'state': str(self.path), 'verdicts': copy.deepcopy(self.state['final_verdicts']),
                      'combination_space': self.enumeration,
                      'combination_counts': list(self.combination_counts().values()),
                      'max_inflight_observed': self.state.get('max_inflight_observed', 0)}
        if errors:
            raise errors[0]
        if self.hooks.redteam is not None:
            sample_batch(admitted, self.out / 'redteam', runner=self.hooks.redteam)
        return result

    def _prepare(self, intent, root):
        if self.hooks.prepare:
            return self.hooks.prepare(intent, root)
        with borrowed_model_slot('opus'):
            spec = self.hooks.design(intent)
        self.stage('design')
        gates, review, task, failures = {}, {}, None, []
        for revision in range(4):
            attempt = root / f'revision-{revision}'
            gates = self.hooks.gates(spec, attempt, SEEDS)
            dump(attempt / 'gates.json', gates)
            self.stage('gates')
            failures = gate_failures(gates) if gates.get('pass') is not True or not _eligible(gates) else []
            # Review any compiled task, even when a hard gate failed, to give one
            # consolidated round of feedback. Model hooks own their own slots.
            try:
                task, _ = _task_for(gates, attempt)
            except ValueError:
                task = Path(gates['task']) if gates.get('task') else None
            review = self.hooks.review(task) if task else {'findings': []}
            self.stage('independent_review', review)
            failures += [{'gate': 'review', 'details': f, 'margin': None} for f in review_failures(review)]
            if not failures:
                break
            if revision == 3 or self.hooks.repair is None:
                break
            from .design_loop import feedback_message
            feedback = feedback_message(gates, review, spec=spec)
            with borrowed_model_slot('opus'):
                spec = self.hooks.repair(copy.deepcopy(spec), feedback)
            self.stage('revision')
        return {'verdict': 'discarded' if failures else 'ready', 'spec': spec,
                'task': str(task) if task else None, 'gates': gates, 'review': review,
                'revisions': revision, 'failures': failures}

    def _trials(self, task, out):
        with self.transaction():
            if self.state['trial_sets'] >= self.max_trial_sets:
                return [], [{'gate': 'pilot_trial_set_cap', 'margin': 0}]
        if self.hooks.trial_preflight is not None:
            problem = self.hooks.trial_preflight()
            if problem:
                raise RuntimeError('trial infrastructure unavailable: ' + problem)
        # Reserve before launch, so crashes cannot reset the paid-trial budget.
        candidate = getattr(self._local, 'candidate', None)
        with self.transaction():
            if self.state['trial_sets'] >= self.max_trial_sets:
                return [], [{'gate': 'pilot_trial_set_cap', 'margin': 0}]
            self.state['trial_sets'] += 1
            if candidate:
                self._set_phase(candidate, 'trials')

        def trial(index, model):
            self._local.candidate = candidate
            # Synthetic rewards do not consume the live machine's model slots.
            # Real trials always retain the fixed global lock directory.
            lock_root = self.out / 'fake_model_slots' if self.dry_run else None
            with borrowed_model_slot(model, lock_root=lock_root):
                result = self.hooks.trial(task, model, index, out)
            result = dict(result, model=model)
            result['score'] = result.get('score', result.get('reward'))
            if not valid_score(result['score']):
                result.update(valid=False, error='Reward must be finite and in [0, 1]')
                result.pop('score', None)
            self.stage('trial', result)
            return result

        # Submit all four now; cross-process model slots enforce 3/1/2 caps.
        # Read futures in submission order, regardless of their completion order.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(trial, index, model) for index, model in enumerate(TRIAL_MODELS, 1)]
            trials = [future.result() for future in futures]
        dump(out / 'trials.json', trials)
        if trials and all(t.get('infra') for t in trials):
            # No trial reached the solver: an infrastructure failure, not a task result. Refund the set.
            with self.transaction():
                self.state['trial_sets'] -= 1
            raise RuntimeError('all trials failed before reaching the solver (infrastructure)')
        return trials, []

    def run_one(self, intent: dict, *, _reservation=None):
        reservation = _reservation or self._reservation(intent)
        if reservation is None:
            raise RuntimeError(self.stop_reason() or 'combination_or_inflight_cap')
        candidate = reservation['id']
        intent = copy.deepcopy(reservation['intent'])
        self._local.candidate = candidate
        try:
            return self._execute_candidate(intent, self.out / candidate)
        except Exception:
            # Publish the abort before making the reservation resumable. A
            # different completed future must not resubmit this candidate in
            # the short interval before this failed future is marked done.
            self._abort.set()
            with self.transaction():
                if candidate in self.state['reservations']:
                    self.state['reservations'][candidate]['phase'] = 'pending'
            raise
        finally:
            self._local.candidate = None

    def _execute_candidate(self, intent, root):
        root.mkdir(exist_ok=True)
        if (root / 'design_loop.json').exists() and not (root / 'intent.json').is_file() and \
                json.loads((root / 'design_loop.json').read_text()).get('verdict') == 'ready':
            raise RuntimeError(f'{root.name} is prepared but missing intent.json; cannot safely resume')
        if (root / 'design_loop.json').exists() and (root / 'intent.json').exists() and \
                json.loads((root / 'intent.json').read_text()).get('id') != intent.get('id'):
            raise RuntimeError(f'{root.name} holds a prepared candidate for another intent; resolve manually')
        if not (root / 'intent.json').exists():
            dump(root / 'intent.json', intent)
        self.stage('intent')
        self._set_phase(root.name, 'preparing')
        saved = root / 'design_loop.json'
        saved_intent = root / 'intent.json'
        prepared = json.loads(saved.read_text()) if saved.exists() else None
        same_intent = saved_intent.exists() and json.loads(saved_intent.read_text()).get('id') == intent.get('id')
        if prepared and same_intent and prepared.get('verdict') == 'ready' and prepared.get('task') and Path(prepared['task']).is_dir():
            self.stage('resume_prepared')  # earlier run stopped after preparation (e.g. trial infrastructure)
        else:
            prepared = self._prepare(intent, root)
        if prepared.get('verdict') == 'infra_failed':
            # Not a task verdict: keep the pilot slot, leave evidence, stop for the operator.
            dump(root / 'infra_failure.json', {k: prepared.get(k) for k in ('failures', 'revisions', 'task')})
            self.stage('infra_failure')
            raise RuntimeError('infrastructure failure during candidate preparation: ' + ', '.join(prepared['failures']))
        spec, gates, review = prepared.get('spec') or {}, prepared.get('gates', {}), prepared.get('review', {})
        dump(root / 'scenario_spec.json', spec)
        failures = [f if isinstance(f, dict) else {'gate': str(f), 'margin': None}
                    for f in prepared.get('failures', [])]
        assigned = self.intent_sources(intent)
        actual = combinations.normalize(spec.get('sources') or [spec.get('source')])
        if assigned and actual != assigned:
            failures.append({'gate': 'fixed_source_combination', 'assigned': assigned, 'actual': actual})
        if prepared.get('verdict') == 'discarded' and not failures:
            failures = [{'gate': 'design_loop', 'details': 'Revision budget exhausted'}]
        task = Path(prepared['task']) if prepared.get('task') else None
        keyword = _keyword(selected_instance(gates)) if gates.get('eligible_instances') else None
        if not failures and (task is None or not valid_score(keyword)):
            failures.append({'gate': 'valid_keyword_score', 'margin': None})
        trials, audit, rebuilds = [], {}, 0
        if not failures:
            trials, failures = self._trials(task, root / 'trials-initial')
        common = common_errors(trials)
        if common and not failures:
            audit = self.hooks.adjudicate(task, common)
            self.stage('common_error_adjudication', audit)
            dump(root / 'common_error_audit.json', audit)
            if not qualified_exclusions(audit):
                failures.append({'gate': 'unqualified_exclusion', 'margin': None,
                                 'details': 'Two high-confidence agreeing readings and structural evidence required'})
            elif audit.get('confirmed_exclusions'):
                if self.hooks.quarantine is None or self.hooks.rebuild is None:
                    failures.append({'gate': 'quarantine_unavailable', 'margin': None})
                else:
                    # Quarantine writes and pilot policy stops share the state lock.
                    with self.transaction():
                        applied = self.hooks.quarantine(audit, root / 'notify_owner.json')
                    dump(root / 'quarantine_result.json', applied)
                    if applied.get('status') != 'applied':
                        failures.append({'gate': 'quarantine_policy_stop', 'details': applied})
                        if applied.get('status') == 'stopped_notify_richard':
                            with self.transaction():
                                self.state['policy_stop'] = 'quarantine_cap_stop'
                            dump(root / 'notify_owner.json', applied)
                    else:
                        before = scored_gold(task)
                        spec = self.hooks.rebuild(copy.deepcopy(spec), audit)
                        rebuilds = 1
                        self.stage('rebuild')
                        attempt = root / 'quarantine-rebuild'
                        gates = self.hooks.gates(spec, attempt, SEEDS)
                        dump(attempt / 'gates.json', gates)
                        self.stage('gates')
                        if gates.get('pass') is not True or not _eligible(gates):
                            failures.extend(gate_failures(gates))
                        else:
                            task, keyword = _task_for(gates, attempt)
                            changed = before != scored_gold(task)
                            dump(root / 'gold_change.json', {'scored_cells_changed': changed})
                            if changed:
                                trials, failures = self._trials(task, root / 'trials-rebuilt')
                                remaining = common_errors(trials)
                                if remaining and not failures:
                                    recheck = self.hooks.adjudicate(task, remaining)
                                    self.stage('common_error_adjudication', recheck)
                                    dump(root / 'common_error_recheck.json', recheck)
                                    if recheck.get('confirmed_exclusions') or not qualified_exclusions(recheck):
                                        failures.append({'gate': 'confirmed_label_conflict_after_rebuild',
                                                         'details': 'One permitted recompile already used'})
                            # The policy permits one recompile, not an unbounded
                            # exclusion/rebuild loop. Retain the original audit.
        if failures:
            verdict = {'verdict': 'discarded', 'failures': failures, 'difficulty': None}
        else:
            verdict = auto_verdict(gates, {}, review, trials, keyword)
        verdict.update({'id': root.name, 'intent': intent['id'], 'dry_run': self.dry_run,
                        'sources': actual, 'combination': combinations.key(actual),
                        'trial_scores_are_fixtures': self.dry_run,
                        'gate_measurements_are_fixtures': bool(gates.get('gate_measurements_are_fixtures')),
                        'rebuilds': rebuilds,
                        'repairs': prepared.get('revisions', 0), 'task': str(task) if task else None,
                        'keyword': keyword, 'trials': trials, 'review': review})
        with self.transaction():
            reservation = self.state['reservations'][root.name]
            if verdict['verdict'] == 'admitted':
                row = self.combination_counts().get(verdict['combination'])
                if row is None or row['admitted'] >= self.quota:
                    verdict.update(verdict='discarded', difficulty=None)
                    verdict['failures'].append({'gate': 'combination_admission_quota', 'combination': verdict['combination']})
            verdict.update(tokens=reservation.get('tokens', 0), cost_usd=round(reservation.get('cost_usd', 0), 12))
            if verdict['verdict'] != 'admitted':
                self.state['discarded_tokens'] += verdict['tokens']
            dump(root / 'verdict.json', verdict)
            self.state['final_verdicts'].append(verdict)
            self.state['final_verdicts'].sort(key=lambda v: v['id'])
            key = intent.get('quota_key', intent['id'])
            self.state['coverage'][key] = self.state['coverage'].get(key, 0) + 1
            self._set_phase(root.name, 'final')
            del self.state['reservations'][root.name]
            self.stage('final_verdict')
            self.write_card(root, intent, spec, gates, verdict)
            self.write_manifest()
        return verdict

    def write_card(self, root, intent, spec, gates, verdict):
        from .adapters import get_adapter
        from .batch1 import public_metadata
        source_notes = []
        for source in spec.get('sources', [spec.get('source')]):
            if source:
                metadata = public_metadata(get_adapter(source))
                source_notes.extend(metadata['limitations'])
                if metadata['licensing_notes']:
                    source_notes.append(source + ': ' + metadata['licensing_notes'])
        notes_path = root / 'design_loop.json'
        notes = json.loads(notes_path.read_text()).get('admitted_with_review_notes', []) if notes_path.exists() else []
        card = {'intent': intent, 'category': spec.get('category'), 'review_notes': notes,
                'occupation': intent.get('occupation', intent.get('soc')), 'onet_tasks': intent.get('onet_tasks', []),
                'dependency_graph': spec.get('dependency_graph'), 'source': spec.get('source'),
                'sources': spec.get('sources', [spec.get('source')]),
                'gates': gates, 'review': verdict.get('review'), 'difficulty': verdict.get('difficulty'), 'verdict': verdict,
                'limitations': ['Public dataset memorisation by models remains a risk.'] + source_notes +
                (['Offline dry-run: model judgements, trial rewards and explicitly marked gate fixtures are simulated.'] if self.dry_run else [])}
        dump(root / 'task_card.json', card)
        (root / 'task_card.md').write_text('# ' + intent['id'] + '\n\n' +
            f"Verdict: {verdict['verdict']}; dry run: {self.dry_run}.\n\n" +
            '```json\n' + json.dumps(card, ensure_ascii=False, indent=2) + '\n```\n')

    def write_manifest(self):
        admitted = []
        for item in self.state['final_verdicts']:
            if item['verdict'] != 'admitted':
                continue
            task = Path(item['task'])
            card_path = self.out / item['id'] / 'task_card.json'
            card = json.loads(card_path.read_text())
            hashes = {str(p.relative_to(task)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(task.rglob('*')) if p.is_file()}
            evaluator = self.out / item['id'] / 'evaluator_package.json'
            dump(evaluator, {'task': str(task), 'private_tests': str(task / 'tests'),
                             'reference': str(task / 'solution'), 'sha256': hashes})
            admitted.append({'id': item['id'], 'task': str(task), 'sha256': hashes,
                             'difficulty': item['difficulty'], 'review': item.get('review'),
                             **{key: card[key] for key in ('occupation', 'onet_tasks', 'dependency_graph',
                                                         'sources', 'gates', 'limitations')},
                             'measurements_are_fixtures': item['dry_run'],
                             'evaluator_package': str(evaluator),
                             'task_card': str(card_path)})
        dump(self.out / 'benchmark_manifest.json', {'dry_run': self.dry_run, 'tasks': admitted})

    def write_summary(self):
        verdicts = self.state['final_verdicts']
        defects = Counter(f['gate'] for v in verdicts for f in v['failures'])
        summary = {'verdicts': len(verdicts), 'admitted': sum(v['verdict'] == 'admitted' for v in verdicts),
                   'discarded': sum(v['verdict'] == 'discarded' for v in verdicts),
                   'trial_sets': self.state['trial_sets'], 'stages': self.state['stages'],
                   'defects': dict(defects), 'cost_usd': self.state['cost_usd'],
                   'tokens': self.state['tokens'], 'discarded_token_ratio':
                   self.state['discarded_tokens'] / max(1, self.state['tokens']), 'dry_run': self.dry_run,
                   'stop_reason': self.stop_reason()}
        dump(self.out / 'pilot_report.json', summary)
        # A notification hook only: never sends messages to Feishu.
        (self.out / 'daily_feishu_summary.md').write_text('# TADA pilot daily summary\n\n' +
            'Offline simulation; no model charges.\n\n' * self.dry_run +
            '```json\n' + json.dumps(summary, indent=2, ensure_ascii=False) + '\n```\n')


def production_trial(task, model, index, out, *, runner=subprocess.run):
    """Use the existing Harbor entry point; called only by explicit real runs."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # The remote launcher also uses this name. Different candidate directories
    # can contain identical task bytes and launch within the same second.
    output_key = hashlib.sha256(str(out).encode()).hexdigest()[:12]
    name = f'trial-{index}-{model}-{output_key}'
    # Trials run on the configured remote host. The task budget is 8 h; allow sync/poll overhead on top.
    command = ['sh', str(REPO / 'solver/run_trial_remote.sh'), str(Path(task).resolve()), str(out), name, model]
    try:
        result = runner(command, cwd=REPO, capture_output=True, text=True, timeout=34200,
                        env=subprocess_environment(model))
        (out / (name + '.launcher.log')).write_text(result.stdout + result.stderr)
        job = out / name
        results = [p for p in job.rglob('result.json') if p.parent != job]
        rows = [json.loads(p.read_text()) for p in results]
        scores = [r.get('verifier_result', {}).get('rewards', {}).get('reward') for r in rows
                  if isinstance(r.get('verifier_result'), dict)]
        detail_paths = list(results[0].parent.rglob('details.json')) if len(results) == 1 else []
        details = [json.loads(p.read_text()) for p in detail_paths]
        cells = [d['cell_correct'] for d in details if isinstance(d.get('cell_correct'), dict)]
        expected = {c['id'] for c in json.loads((Path(task) / 'tests/gold.json').read_text())['cells']}
        valid = (result.returncode == 0 and len(rows) == len(scores) == len(cells) == 1
                 and len(details) == 1 and bool(expected) and set(cells[0]) == expected
                 and all(type(v) is bool for v in cells[0].values())
                 and valid_score(scores[0]) and valid_score(details[0].get('reward'))
                 and abs(details[0]['reward'] - scores[0]) <= 1e-6
                 and not details[0].get('error') and not rows[0].get('exception_info'))
        # Harbor never produced a trial (sync/launch/docker failure): infrastructure, not a solver result.
        infra = not rows
        return {'valid': valid, 'score': scores[0] if len(scores) == 1 else None, 'infra': infra,
                'wrong_cells': sorted(k for k, v in cells[0].items() if v is False) if len(cells) == 1 else [],
                'job': str(job), 'fake': False,
                'error': None if valid else 'Missing or failed Harbor trial/reward/cell details'}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.SubprocessError) as exc:
        return {'valid': False, 'error': type(exc).__name__, 'job': str(out / name), 'fake': False}


def docker_preflight():
    """Configured trial host reachable, Docker up, Harbor venv present."""
    from local_config import get
    import shlex
    host = get('remote_trial_host', required=True)
    root = get('remote_trial_root', required=True)
    remote = ('"$HOME"/' + shlex.quote(root[2:])) if root.startswith('~/') else shlex.quote(root)
    try:
        done = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', host,
                               "docker info --format '{{.ServerVersion}}' && test -x " + remote + '/.venv/bin/harbor'],
                              capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.SubprocessError) as exc:
        return type(exc).__name__
    return None if done.returncode == 0 and done.stdout.strip() else 'Remote trial Docker/Harbor not reachable'


def production_hooks(spec: dict) -> Hooks:
    from .common_error import BlindReadingClient, adjudicate_common_errors, quarantine_common_errors
    from .design_loop import production_prepare
    from .gates import run_single
    from .redteam import run_redteam
    from .adapters.base import _load_source

    def audit(task, errors):
        client = BlindReadingClient.from_config('extractor', task.parent / 'common_error/model_calls')
        return adjudicate_common_errors(task, errors, client=client,
                                       out=task.parent / 'common_error/adjudication.json')

    def rebuild(candidate, report):
        # A new adapter must observe the newly approved exclusion snapshot.
        _load_source.cache_clear()
        return candidate

    return Hooks(design=lambda intent: copy.deepcopy(spec),
                 gates=lambda candidate, out, seeds: run_single(candidate, out, seed=seeds[0], client='real'),
                 probe=lambda task: {}, difficulty=lambda task, keyword: {}, review=lambda task: {},
                 trial=production_trial, adjudicate=audit, rebuild=rebuild,
                 quarantine=quarantine_common_errors,
                 prepare=lambda intent, root: production_prepare(intent, root,
                     family=combinations.key(Scheduler.intent_sources(intent)) or spec.get('source'), seed=7),
                 redteam=lambda task, out: run_redteam(task, out, mode='real'),
                 trial_preflight=docker_preflight)


def offline_hooks(spec: dict | None = None) -> Hooks:
    def design(intent):
        sources = Scheduler.intent_sources(intent)
        if sources:
            from .multisource import example_spec
            return example_spec(sources)
        if spec is None:
            raise ValueError('Offline design requires a fixed source combination')
        return copy.deepcopy(spec)

    def gates(candidate, out, seeds):
        from .gates import run_single
        measured = run_single(candidate, out / 'measured', seed=seeds[0], client='fake')
        dump(out / 'actual_gate_measurements.json', measured)
        from .design_loop import HARD_GATES
        if set(measured.get('hard_gates', {})) != set(HARD_GATES) or measured.get('failure'):
            raise ValueError('Lifecycle fixture requires all six real gate measurements; build failure cannot be admitted')
        # The lifecycle smoke test must traverse admission even if the specified
        # regression spec fails a real gate. Measurements are retained intact;
        # fixture booleans are explicitly marked, never called real acceptance.
        fixture = copy.deepcopy(measured)
        fixture.update({'pass': True, 'fake_gates': True, 'gate_measurements_are_fixtures': True,
                        'eligible_instances': [seeds[0]],
                        'reported_only': {'actual_hard_gates': measured.get('hard_gates'),
                                          'actual_pass': measured.get('pass')},
                        'hard_gates': {name: {'pass': True, 'fixture': True}
                                       for name in measured.get('hard_gates', {})}})
        task = measured.get('task') or str(out / 'measured/task')
        fixture['instances'] = [{'seed': seeds[0], 'task': task, 'keyword': .2, 'all_pass': True,
                                 'fixture': True, 'measurement_report': str(out / 'actual_gate_measurements.json')}]
        if not Path(task).is_dir():
            raise ValueError('Offline lifecycle requires a successfully compiled task')
        return fixture

    def review(task):
        return {'questions': [{'id': name, 'severity': 0, 'evidence': 'Offline lifecycle fixture',
                               'required_change': ''} for name in ('instruction_sufficiency', 'scoring_fairness',
                                                                  'realism_factual_correctness', 'text_analysis_difficulty')],
                'max_severity': 0, 'revision_required': False, 'fake': True}

    from .design_loop import prepare_candidate
    prepare = lambda intent, root: prepare_candidate(
        intent, root, designer=lambda *args: design(intent),
        compile_and_gate=lambda candidate, out: gates(candidate, out, (7,)),
        reviewer=lambda task, measured, out: review(task))
    return Hooks(design=design, gates=gates,
                 probe=lambda task: {'reported_only': True}, difficulty=lambda *args: {'reported_only': True},
                 review=review, prepare=prepare,
                 trial=lambda task, model, index, out: {'valid': True, 'score': .65 if model == 'deepseek-flash' else .80,
                                                       'wrong_cells': [], 'fake': True, 'tokens': 0, 'cost_usd': 0},
                 adjudicate=lambda task, errors: {'confirmed_exclusions': [], 'common_errors': errors, 'fake': True},
                 redteam=lambda task, out: {'status': 'fixture', 'reward': .20, 'fake': True, 'reported_only': True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', help='Optional legacy seed spec; combinations remain fixed by the scheduler')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--loops', type=int, default=1)
    parser.add_argument('--combinations-report', action='store_true', help='Read-only enumeration and admission counts')
    parser.add_argument('--datasets-report', action='store_true', help='Read-only inventory of all 29 batch-1 IDs')
    parser.add_argument('--state', type=Path, help='Existing state.json to inspect in --combinations-report mode')
    args = parser.parse_args()
    from .adapters import available_sources, get_adapter, REFUSED_SOURCES
    settings = scheduler_settings()
    sources = available_sources()
    if args.datasets_report:
        from .adapters.batch1_registry import inventory
        print(json.dumps({'datasets': inventory(), 'available_sources': sources,
                          'combination_count': len(combinations.enumerate_spaces(sources,
                              settings['combination_max_size'])['combinations'])}, ensure_ascii=False, indent=2))
        return
    if args.combinations_report:
        path = args.state or (args.out / 'state.json' if args.out else None)
        if path is not None and not path.is_file():
            parser.error('State file does not exist: ' + str(path))
        state = json.loads(path.read_text()) if path and path.is_file() else {}
        report = combinations.report(state, sources, settings['combination_max_size'],
                                     settings['quota_per_combination'], path.parent if path else None)
        report['refused_sources'] = REFUSED_SOURCES
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.state:
        parser.error('--state is only available with --combinations-report')
    if args.out is None:
        parser.error('--out is required for candidate execution')
    spec = json.loads(Path(args.spec).read_text()) if args.spec else {}
    from .intents import generate_combination_intents
    enumeration = combinations.enumerate_spaces(sources, settings['combination_max_size'])
    intents = generate_combination_intents([get_adapter(source) for source in sources], enumeration,
                                          review=FakeClient('fake-intent-review') if args.dry_run else None)
    if not intents:
        raise SystemExit('No reviewed intents available')
    hooks = offline_hooks(spec) if args.dry_run else production_hooks(spec)
    result = Scheduler(args.out, hooks, dry_run=args.dry_run, available_sources=sources).run(intents, loops=args.loops)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
