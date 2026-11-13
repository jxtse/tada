import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tada_taskgen.common_error import adjudicate_common_errors, quarantine_common_errors
from tada_taskgen.concurrency import model_slot
from tada_taskgen.redteam import sample_batch
from tada_taskgen.scheduler import (TRIAL_MODELS, Hooks, Scheduler, auto_verdict,
                                   difficulty_label, production_trial)


def trials(rewards):
    return [{'model': m, 'valid': True, 'score': r, 'wrong_cells': []}
            for m, r in zip(TRIAL_MODELS, rewards)]


@pytest.mark.parametrize('rewards,label', [([.8, .8, .8, 0], 'easy'), ([1, 1, 1, 1], 'easy'),
                                         ([.8, .8, .799, 0], 'medium'), ([.8, 0, 0, 0], 'hard'),
                                         ([0, 0, 0, 0], 'hard')])
def test_difficulty_is_annotation(rewards, label):
    result = auto_verdict({'pass': True}, {'findings': [{'severity': 'blocker'}]},
                          {'findings': []}, trials(rewards), .2)
    assert result['verdict'] == 'admitted'
    assert result['difficulty']['label'] == label
    assert result['difficulty']['pass_reward'] == .80
    assert result['difficulty']['per_model_rewards']['deepseek-flash'] == rewards[:2]


def test_exact_two_trials_of_each_model():
    rows = trials([.8] * 4)
    rows[-1]['model'] = 'deepseek-flash'
    with pytest.raises(ValueError):
        difficulty_label(rows)


def audit():
    return {'source': 'MAUD', 'confirmed_exclusions': ['r:q'], 'question_pool_sizes': {'q': 100},
            'proposals': [{'record': 'r', 'question': 'q', 'source': 'MAUD',
                           'action': 'exclude', 'original_answer': 'No',
                           'readings': [{'answer': 'Yes', 'confidence': 'high'}] * 2,
                           'structural_evidence': {'kind': 'pool_majority', 'total': 10,
                                                   'support': 9, 'answer': 'Yes'}}]}


def make_hooks(*, changed=False):
    counts = {'gates': 0, 'quarantine': 0, 'audit': 0}

    def gates(spec, out, seeds):
        assert tuple(seeds) == (7,)
        counts['gates'] += 1
        task = out / 'task'
        (task / 'tests').mkdir(parents=True)
        (task / 'instruction.md').write_text('fixture')
        gold = {'cells': [{'id': 'label:r:q', 'answer': 'Yes' if changed and counts['gates'] > 1 else 'No'}]}
        (task / 'tests/gold.json').write_text(json.dumps(gold))
        return {'pass': True, 'eligible_instances': [7], 'instances': [{'seed': 7, 'task': str(task), 'keyword': .2}]}

    def adjudicate(*args):
        counts['audit'] += 1
        return audit() if counts['audit'] == 1 else {'confirmed_exclusions': []}

    def quarantine(report, notify):
        counts['quarantine'] += 1
        return {'status': 'applied'}

    hooks = Hooks(design=lambda i: {'source': 'MAUD'}, gates=gates,
                  probe=lambda *a: pytest.fail('Old spec probe is not a hard gate'),
                  difficulty=lambda *a: pytest.fail('Old difficulty probe must not run'),
                  review=lambda *a: {'questions': [], 'max_severity': 0},
                  trial=lambda *a: {'valid': True, 'score': 1, 'wrong_cells': ['label:r:q']},
                  adjudicate=adjudicate, quarantine=quarantine, rebuild=lambda spec, report: spec)
    return hooks, counts


def intent():
    return {'id': 'i', 'quota_key': 'q', 'occupation': 'Lawyer', 'onet_tasks': [{'id': 'x'}]}


@pytest.mark.parametrize('changed,sets', [(False, 1), (True, 2)])
def test_reruns_only_when_scored_gold_changes(tmp_path, changed, sets):
    hooks, counts = make_hooks(changed=changed)
    result = Scheduler(tmp_path, hooks, dry_run=True).run([intent()])
    row = result['verdicts'][0]
    assert row['verdict'] == 'admitted'
    assert row['difficulty']['label'] == 'easy'
    assert row['rebuilds'] == 1
    assert result['trial_sets'] == sets
    assert counts['gates'] == 2 and counts['quarantine'] == 1
    assert counts['audit'] == sets
    card = json.loads((tmp_path / 'candidate-001/task_card.json').read_text())
    assert card['onet_tasks'] and card['review']
    assert any('memorisation' in x for x in card['limitations'])
    manifest = json.loads((tmp_path / 'benchmark_manifest.json').read_text())['tasks'][0]
    for key in ('occupation', 'onet_tasks', 'dependency_graph', 'sources', 'gates',
                'review', 'difficulty', 'limitations'):
        assert manifest[key] == card[key]


def test_cap_stops_and_notifies_without_apply(tmp_path):
    hooks, _ = make_hooks()
    quarantine = tmp_path / 'scratch_quarantine.json'
    quarantine.write_text('{}')
    report = audit()
    report['question_pool_sizes']['q'] = 19
    hooks.adjudicate = lambda *a: report
    hooks.quarantine = lambda report, notify: quarantine_common_errors(report, notify, quarantine=quarantine)
    hooks.rebuild = lambda *a: pytest.fail('Cap stop must not rebuild')
    result = Scheduler(tmp_path / 'run', hooks, dry_run=True).run([intent()], loops=3)
    assert result['completed'] == 1
    assert result['stop_reason'] == 'quarantine_cap_stop'
    assert result['verdicts'][0]['verdict'] == 'discarded'
    assert quarantine.read_text() == '{}'
    notice = json.loads((tmp_path / 'run/candidate-001/notify_owner.json').read_text())
    assert notice['reason'] == '5_percent_per_question_cap'


def test_safe_quarantine_applies_only_to_scratch(tmp_path):
    report = audit()
    path = tmp_path / 'scratch.json'
    path.write_text('{}')
    result = quarantine_common_errors(report, tmp_path / 'notify.json', quarantine=path)
    assert result['status'] == 'applied'
    assert Path(result['backup']).read_text() == '{}'
    assert json.loads(path.read_text())['MAUD_labels'][0]['contract_id'] == 'r'


def test_blind_rereads_include_definitions_and_no_labels(tmp_path):
    quote = 'The supplier must obtain prior written consent before any assignment.'
    text = '"Assignment" means transfer of any obligation.\n\n' + quote

    class Adapter:
        source = 'MAUD'
        quarantine = tmp_path / 'none.json'
        def label_pool(self):
            return {('r', 'q'): ('No', [quote])}
        def records(self):
            return {'r': {'text': text}}
        def questions(self):
            return {'q': {'question': 'Is consent required?', 'options': ['Yes', 'No'], 'coding_conventions': []}}
        def keyword_predict(self, *args):
            return 'Yes'

    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/gold.json').write_text(json.dumps({'source': 'MAUD',
        'cells': [{'id': 'label:r:q', 'record': 'r', 'question': 'q', 'answer': 'No'}]}))
    payloads = []
    def client(payload):
        payloads.append(copy.deepcopy(payload))
        return {'answer': 'Yes', 'confidence': 'high', 'clause_quote': quote}
    result = adjudicate_common_errors(tmp_path, ['label:r:q'], client=client, adapter=Adapter())
    assert result['confirmed_exclusions'] == ['r:q']
    assert len(payloads) == 2 and [p['read_index'] for p in payloads] == [0, 1]
    assert all(p['definitions'][0]['term'] == 'Assignment' for p in payloads)
    assert not any(k in p for p in payloads for k in ('answer', 'original_answer', 'trial', 'readings'))


def test_redteam_ten_percent_and_failure_report_only(tmp_path):
    rows = [{'id': f'task-{i}', 'task': str(tmp_path), 'verdict': 'admitted'} for i in range(10)]
    before = copy.deepcopy(rows)
    calls = []
    def runner(task, out):
        calls.append(out.name)
        raise ValueError('Unavailable measurement')
    report = sample_batch(rows, tmp_path / 'redteam', runner=runner)
    assert report['sample_size'] == 1 and report['actual_fraction'] == .1
    assert report['reported_only'] and report['reports'][0]['measurement']['status'] == 'unavailable'
    assert rows == before
    again = sample_batch(list(reversed(rows)), tmp_path / 'again', runner=runner)
    assert report['reports'][0]['id'] == again['reports'][0]['id']
    sample_batch(rows, tmp_path / 'redteam', runner=runner)
    assert len(calls) == 2  # original plus the separate "again" batch, no resume call


def test_partial_batch_does_not_run_per_task_redteam(tmp_path):
    rows = [{'id': 'one', 'task': str(tmp_path), 'verdict': 'admitted'}]
    report = sample_batch(rows, tmp_path / 'batch', runner=lambda *a: pytest.fail('No per-task red team'))
    assert report['sample_size'] == 0 and report['pending_until_batch_size'] == 10


@pytest.mark.parametrize('correct,reward,valid', [({'a': False, 'b': True}, .81, True),
                                                 ({}, .81, False), ({'a': False}, .81, False),
                                                 ({'a': False, 'b': 1}, .81, False),
                                                 ({'a': False, 'b': True}, .5, False)])
def test_production_trial_uses_existing_entrypoint_without_running_it(tmp_path, correct, reward, valid):
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/gold.json').write_text(json.dumps({'cells': [{'id': 'a'}, {'id': 'b'}]}))
    def runner(command, **kwargs):
        assert command[:2] == ['sh', str(Path(__file__).resolve().parents[2] / 'solver/run_trial_remote.sh')]
        trial = Path(command[3]) / command[4] / 'trial-instance'
        trial.mkdir(parents=True)
        (trial / 'result.json').write_text(json.dumps({'verifier_result': {'rewards': {'reward': .81}}}))
        (trial / 'details.json').write_text(json.dumps({'cell_correct': correct, 'reward': reward}))
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    result = production_trial(tmp_path, 'deepseek-flash', 1, tmp_path / 'jobs', runner=runner)
    assert result['valid'] is valid
    if valid:
        assert result['score'] == .81 and result['wrong_cells'] == ['a']


def test_astra_capacity_two(tmp_path):
    with model_slot('gpt-6-astra', timeout=0, lock_root=tmp_path):
        with model_slot('astra', timeout=0, lock_root=tmp_path):
            with pytest.raises(TimeoutError):
                with model_slot('astra', timeout=0, lock_root=tmp_path):
                    pass
