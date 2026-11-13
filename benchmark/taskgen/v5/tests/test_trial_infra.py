"""Trial infrastructure failures (e.g. Docker down) are not task verdicts and do not consume pilot budget."""
import json
from pathlib import Path

import pytest

from tada_taskgen.scheduler import Hooks, Scheduler

REPO = Path(__file__).resolve().parents[2]


def hooks(tmp_path, *, preflight=None, infra=False):
    task = tmp_path / 'task'
    task.mkdir(exist_ok=True)

    def prepare(intent, root):
        result = {'verdict': 'ready', 'spec': {'category': 'parallel_coding'}, 'task': str(task),
                  'gates': {'eligible_instances': [7], 'instances': [{'seed': 7, 'keyword': .2, 'all_pass': True}]},
                  'review': {}, 'revisions': 0, 'failures': []}
        (root / 'design_loop.json').write_text(json.dumps(result))
        return result

    trial = lambda task, model, index, out: ({'valid': False, 'infra': True, 'error': 'x'} if infra else
                                             {'valid': True, 'score': .5, 'wrong_cells': []})
    none = lambda *a, **k: {}
    return Hooks(design=none, gates=none, probe=none, difficulty=none, review=none, trial=trial,
                 adjudicate=lambda *a: {'confirmed_exclusions': []}, prepare=prepare, trial_preflight=preflight)


def test_preflight_failure_stops_before_any_trial_or_budget(tmp_path):
    s = Scheduler(tmp_path / 'out', hooks(tmp_path, preflight=lambda: 'docker daemon not reachable'))
    with pytest.raises(RuntimeError, match='trial infrastructure'):
        s.run_one({'id': 'i1'})
    assert s.state['trial_sets'] == 0 and s.state['final_verdicts'] == []


def test_all_infra_trials_refund_the_set_and_record_no_verdict(tmp_path):
    s = Scheduler(tmp_path / 'out', hooks(tmp_path, infra=True))
    with pytest.raises(RuntimeError, match='infrastructure'):
        s.run_one({'id': 'i1'})
    assert s.state['trial_sets'] == 0 and s.state['final_verdicts'] == []


def test_prepared_candidate_is_resumed_without_redesign(tmp_path):
    calls = []
    h = hooks(tmp_path)
    original = h.prepare
    h.prepare = lambda intent, root: calls.append(1) or original(intent, root)
    s = Scheduler(tmp_path / 'out', h)
    root = tmp_path / 'out/candidate-001'
    root.mkdir(parents=True)
    original({}, root)
    (root / 'intent.json').write_text(json.dumps({'id': 'i1'}))
    s.run_one({'id': 'i1'})
    assert calls == [] and s.state['stages'].get('resume_prepared') == 1
