import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tada_taskgen.concurrency import model_slot
from tada_taskgen.scheduler import Hooks, Scheduler, auto_verdict, common_errors


def spec():
    return {'id': 'scenario', 'source': 'MAUD', 'category': 'parallel_coding',
            'work_items': [{'id': 'a'}, {'id': 'b'}], 'parameters': {}, 'dependency_graph': {'nodes': []}}


def intent():
    return {'id': 'intent', 'quota_key': '23|MAUD|scope-review', 'source': 'MAUD',
            'onet_tasks': [{'id': 'fixture', 'text': 'Test fixture only'}]}


def hooks(tmp_path):
    def gates(candidate, out, seeds):
        task = out / 'task'
        task.mkdir(parents=True)
        (task / 'instruction.md').write_text('Review the agreements.')
        (task / 'tests').mkdir()
        (task / 'tests/gold.json').write_text(json.dumps({'cells': [{'id':'x', 'answer': str(out)}]}))
        return {'pass': True, 'eligible_instances': [7], 'instances': [
            {'seed': 7, 'task': str(task), 'keyword': .2, 'all_pass': True}], 'fake_gates': True}
    return Hooks(design=lambda i: spec(), gates=gates,
                 probe=lambda t: {'findings': []}, difficulty=lambda t, k: {'score': .6, 'valid': True},
                 review=lambda t: {'findings': []},
                 trial=lambda *a: {'valid': True, 'score': .7, 'wrong_cells': ['c'], 'fake': True},
                 adjudicate=lambda *a: {'confirmed_exclusions': [], 'fake': True})


def test_full_offline_loop_manifest_and_summary(tmp_path):
    scheduler = Scheduler(tmp_path, hooks(tmp_path), dry_run=True)
    result = scheduler.run([intent()])
    assert result['completed'] == 1 and result['trial_sets'] == 1
    verdict = result['verdicts'][0]
    assert verdict['verdict'] == 'admitted' and verdict['trial_scores_are_fixtures']
    assert scheduler.state['stages']['trial'] == 4
    assert scheduler.state['stages']['common_error_adjudication'] == 1
    assert json.loads((tmp_path / 'benchmark_manifest.json').read_text())['dry_run']
    assert (tmp_path / 'daily_feishu_summary.md').exists()


def test_failed_gates_never_enter_trials(tmp_path):
    h = hooks(tmp_path)
    h.gates = lambda *a: {'pass': False, 'eligible_instances': [], 'failures': ['keyword']}
    h.trial = lambda *a: pytest.fail('failed instance must not enter trials')
    result = Scheduler(tmp_path, h, dry_run=True).run([intent()])
    assert result['verdicts'][0]['verdict'] == 'discarded'
    assert result['trial_sets'] == 0


def test_pilot_caps_survive_resume(tmp_path):
    h = hooks(tmp_path)
    result = Scheduler(tmp_path, h, dry_run=True, max_verdicts=2).run([intent()], loops=4, quotas={intent()['quota_key']: 10})
    assert result['completed'] == 2 and result['stop_reason'] == 'pilot_verdict_cap'
    resumed = Scheduler(tmp_path, h, dry_run=True, max_verdicts=2).run([intent()], loops=1)
    assert resumed['completed'] == 2
    with pytest.raises(ValueError):
        Scheduler(tmp_path, h, dry_run=False)


def exclusion():
    return {'confirmed_exclusions': ['record:q'], 'proposals': [
        {'record': 'record', 'question': 'q', 'original_answer': 'No', 'action': 'exclude',
         'readings': [{'answer': 'Yes', 'confidence': 'high'}] * 2,
         'structural_evidence': {'kind': 'pool_majority', 'total': 10, 'support': 9, 'answer': 'Yes'}}]}


def test_rebuild_once_discards_scores(tmp_path):
    h = hooks(tmp_path)
    h.adjudicate = lambda *a: exclusion()
    h.rebuild = lambda candidate, audit: candidate
    h.quarantine = lambda *a: {'status': 'applied'}
    result = Scheduler(tmp_path, h, dry_run=True).run([intent()])
    assert result['trial_sets'] == 2
    assert result['verdicts'][0]['rebuilds'] == 1
    assert result['verdicts'][0]['verdict'] == 'discarded'


def test_trial_set_cap_before_rebuild(tmp_path):
    h = hooks(tmp_path)
    h.adjudicate = lambda *a: exclusion()
    h.rebuild = lambda candidate, audit: candidate
    h.quarantine = lambda *a: {'status': 'applied'}
    result = Scheduler(tmp_path, h, dry_run=True, max_trial_sets=1).run([intent()])
    assert result['trial_sets'] == 1
    assert result['stop_reason'] == 'pilot_trial_set_cap'


def test_verdict_measured_margin_and_common_errors():
    trials = [{'model': m, 'valid': True, 'score': .5, 'wrong_cells': ['x', str(i)]}
              for i, m in enumerate(['deepseek-flash'] * 2 + ['claude-opus-5-5'] * 2)]
    verdict = auto_verdict({'pass': True}, {'findings': []}, {'findings': []}, trials, .25)
    assert verdict['verdict'] == 'admitted'
    assert verdict['difficulty']['label'] == 'hard'
    assert verdict['reported_only']['deepseek_advantage'] == .25
    assert common_errors(trials) == ['x']


def test_shared_opus_slot_blocks_another_process(tmp_path):
    code = ('from pathlib import Path; from tada_taskgen.concurrency import model_slot; '
            'ctx=model_slot("opus",timeout=0,lock_root=Path(__import__("sys").argv[1])); ctx.__enter__()')
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    with model_slot('opus', timeout=0, lock_root=tmp_path):
        result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True, env=env)
        assert result.returncode != 0 and 'TimeoutError' in result.stderr
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True, env=env)
    assert result.returncode == 0


def test_deepseek_capacity_three(tmp_path):
    with model_slot('deepseek', timeout=0, lock_root=tmp_path):
        with model_slot('deepseek', timeout=0, lock_root=tmp_path):
            with model_slot('deepseek', timeout=0, lock_root=tmp_path):
                with pytest.raises(TimeoutError):
                    with model_slot('deepseek', timeout=0, lock_root=tmp_path):
                        pass


def test_harbor_status_is_report_only_under_slim_policy(tmp_path):
    h = hooks(tmp_path)
    result = Scheduler(tmp_path, h, dry_run=False).run([intent()])
    assert result['verdicts'][0]['verdict'] == 'admitted'
    assert result['trial_sets'] == 1


def test_failed_numeric_gates_have_exact_margin(tmp_path):
    h = hooks(tmp_path)
    h.gates = lambda *a: {'pass': False, 'eligible_instances': [], 'instances': [
        {'seed': 7, 'metrics': {'keyword': {'pass': False, 'value': .41, 'limit': .30}}}]}
    result = Scheduler(tmp_path, h, dry_run=True).run([intent()])
    failure = result['verdicts'][0]['failures'][0]
    assert failure['gate'] == 'keyword' and abs(failure['margin'] - .11) < 1e-9


def test_unqualified_common_error_is_not_quarantined(tmp_path):
    h = hooks(tmp_path)
    h.adjudicate = lambda *a: {'confirmed_exclusions': ['record:q']}
    h.rebuild = lambda *a: pytest.fail('Model agreement alone must not rebuild')
    result = Scheduler(tmp_path, h, dry_run=True).run([intent()])
    assert result['verdicts'][0]['failures'][0]['gate'] == 'unqualified_exclusion'


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -1, 1.01, None, True])
def test_invalid_trial_scores_cannot_admit(score):
    trials = [{'model': m, 'valid': True, 'score': score}
              for m in ['deepseek-flash'] * 2 + ['claude-opus-5-5'] * 2]
    result = auto_verdict({'pass': True}, {'findings': []}, {'findings': []}, trials, .2)
    assert result['verdict'] == 'discarded'
    assert any(f['gate'] == 'valid_trial_scores' for f in result['failures'])


@pytest.mark.parametrize('keyword', [float('nan'), float('inf'), -1, 1.01, None, True])
def test_invalid_keyword_cannot_admit(keyword):
    trials = [{'model': m, 'valid': True, 'score': .7}
              for m in ['deepseek-flash'] * 2 + ['claude-opus-5-5'] * 2]
    result = auto_verdict({'pass': True}, {'findings': []}, {'findings': []}, trials, keyword)
    assert result['verdict'] == 'discarded'
    assert any(f['gate'] == 'valid_keyword_score' for f in result['failures'])


def test_harbor_bound_to_selected_instance():
    from tada_taskgen.scheduler import harbor_verified, _task_for
    gates = {'eligible_instances': [11, 7], 'instances': [
        {'seed': 7, 'harbor_status': True, 'task': 'seven', 'keyword': .2},
        {'seed': 11, 'harbor_status': False, 'task': 'eleven', 'keyword': .2}]}
    assert _task_for(gates, Path('.'))[0] == Path('eleven')
    assert not harbor_verified(gates)
    gates['instances'][1]['harbor_status'] = True
    assert harbor_verified(gates)


@pytest.mark.parametrize('keyword', [None, True, '0.2'])
def test_scheduler_rejects_missing_or_typed_keyword(tmp_path, keyword):
    h = hooks(tmp_path)
    h.gates = lambda *a: {'pass': True, 'eligible_instances': [7], 'instances': [
        {'seed': 7, 'task': str(tmp_path), 'keyword': keyword, 'all_pass': True}]}
    h.trial = lambda *a: pytest.fail('Invalid keyword must not reach trials')
    result = Scheduler(tmp_path, h, dry_run=True).run([intent()])
    assert result['verdicts'][0]['failures'][0]['gate'] == 'valid_keyword_score'
