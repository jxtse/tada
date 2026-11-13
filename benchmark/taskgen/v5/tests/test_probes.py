import copy
import pytest
from tada_taskgen.clients import FakeClient
from tada_taskgen.probes import difficulty_decision, preserve_work, repair_loop, spec_probe, trial_command


def spec():
    return {'work_items': [{'id': 'a'}, {'id': 'b'}], 'sample_size': 10, 'questions': ['q'],
            'documents': [{'id': 'd', 'facts': {'scope': 1}}], 'parameters': {'cutoff': 1},
            'dependency_graph': {'nodes': [{'id': 'f', 'type': 'construction_fact', 'value': 1}]}}


@pytest.mark.parametrize('score,keyword,expected', [(0.9, 0.2, 'deepen'), (0.3, 0.2, 'repair'), (0.6, 0.2, 'continue')])
def test_difficulty_boundaries(score, keyword, expected):
    assert difficulty_decision(score, keyword) == expected


def test_spec_probe_sees_only_visible_inputs():
    client = FakeClient('probe')
    assert spec_probe('Do the review', {'memo.txt': 'A current memo'}, client)['pass']
    assert set(client.calls[0]['payload']) == {'instruction', 'workspace', 'request'}


def test_nonblockers_do_not_trigger_repairs():
    report = repair_loop(spec(), lambda _: {'findings': [{'severity': 'non-blocker'}]},
                         lambda *_: pytest.fail('must not repair'))
    assert report['status'] == 'passed' and report['rounds'] == 0


def test_only_parameters_and_content_can_change():
    before = spec()
    after = copy.deepcopy(before)
    after['parameters']['cutoff'] = 2
    after['documents'][0]['facts']['scope'] = 2
    preserve_work(before, after)
    after['sample_size'] = 9
    with pytest.raises(ValueError, match='structure'):
        preserve_work(before, after)


def test_repair_budget_and_preservation():
    calls = []
    def repair(current, blockers):
        calls.append(1)
        current['parameters']['cutoff'] += 1
        return current
    result = repair_loop(spec(), lambda _: {'findings': [{'severity': 'blocker', 'defect_class': 'contradiction'}]}, repair)
    assert result['status'] == 'dropped' and len(calls) == 3
    assert result['defect_classes'] == ['contradiction']


def test_trial_hook_is_command_construction_only(tmp_path):
    cmd = trial_command(tmp_path, tmp_path / 'jobs', 'probe', 'deepseek-flash')
    assert cmd[0] == 'sh' and cmd[1].endswith('solver/run_trial_remote.sh')
