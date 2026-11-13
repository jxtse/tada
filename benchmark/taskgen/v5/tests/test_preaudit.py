import copy
import json
import pytest
from tada_taskgen.preaudit import agreement_definitions, apply_proposals, preaudit, find_suspects


TEXT = '“Ordinary Course” means consistent with past practice.\n\nThe Company shall operate in the Ordinary Course of business.'
QUOTE = 'The Company shall operate in the Ordinary Course of business.'


class Adapter:
    source = 'MAUD'
    def questions(self):
        return {'Q': {'question': 'Past practice required?', 'options': ['Yes', 'No'], 'coding_conventions': ['Read defined terms.']}}
    def label_pool(self):
        return {(f'r{i}', 'Q'): ('No' if i == 0 else 'Yes', [QUOTE]) for i in range(20)}
    def records(self):
        return {f'r{i}': {'text': TEXT} for i in range(20)}
    def keyword_predict(self, question, text):
        return 'Yes'


def report(client=None):
    return preaudit(Adapter(), client or (lambda _: {'answer': 'Yes', 'confidence': 'high', 'clause_quote': QUOTE}))


def test_definitions_attached_and_double_readings_blind():
    seen = []
    def fake(payload):
        seen.append(payload)
        assert payload['definitions'][0]['term'] == 'Ordinary Course'
        assert 'original_answer' not in payload and 'readings' not in payload and 'label' not in payload
        return {'answer': 'Yes', 'confidence': 'high', 'clause_quote': QUOTE}
    result = report(fake)
    assert len(seen) == 2 and result['calls'] == 2
    assert len(result['proposals']) == 1
    assert result['proposals'][0]['original_answer'] == 'No'
    assert {x['read_index'] for x in seen} == {0, 1}


@pytest.mark.parametrize('readings', [
    [{'answer': 'Yes', 'confidence': 'medium'}, {'answer': 'Yes', 'confidence': 'high'}],
    [{'answer': 'Yes', 'confidence': 'high'}, {'answer': 'No', 'confidence': 'high'}],
    [{'answer': 'No', 'confidence': 'high'}, {'answer': 'No', 'confidence': 'high'}],
])
def test_no_exclusion_without_agreement_high_confidence_conflict(readings):
    calls = iter(readings)
    assert report(lambda _: next(calls))['proposals'] == []


def test_model_agreement_alone_is_not_structural_evidence():
    suspects = [{'record': 'r0', 'question': 'Q'}]
    result = preaudit(Adapter(), lambda _: {'answer': 'Yes', 'confidence': 'high'}, suspects=suspects)
    assert not result['proposals']


def test_fabricated_clause_quote_does_not_qualify():
    result = preaudit(Adapter(), lambda _: {'answer': 'Yes', 'confidence': 'high', 'clause_quote': 'x' * 90},
                      suspects=[{'record': 'r0', 'question': 'Q'}])
    assert not result['proposals']


def test_exact_clause_quote_can_qualify_without_pool_majority():
    result = preaudit(Adapter(), lambda _: {'answer': 'Yes', 'confidence': 'high', 'clause_quote': QUOTE},
                      suspects=[{'record': 'r0', 'question': 'Q'}])
    assert result['proposals'][0]['structural_evidence']['kind'] == 'citable_clause'


def test_apply_backup_idempotence_and_cap(tmp_path):
    target = tmp_path / 'quarantine.json'
    notice = tmp_path / 'notice.json'
    target.write_text('{"unrelated": [1]}\n')
    result = apply_proposals(report(), target, {'MAUD:Q': 20}, notice)
    assert result['status'] == 'applied'
    assert json.loads(open(result['backup']).read()) == {'unrelated': [1]}
    assert json.loads(target.read_text())['unrelated'] == [1]
    assert len(json.loads(target.read_text())['MAUD_labels']) == 1
    apply_proposals(report(), target, {'MAUD:Q': 20}, notice)
    assert len(json.loads(target.read_text())['MAUD_labels']) == 1
    more = copy.deepcopy(report())
    more['proposals'][0]['record'] = 'r1'
    before = target.read_bytes()
    result = apply_proposals(more, target, {'MAUD:Q': 20}, notice)
    assert result['status'] == 'stopped_notify_richard'
    assert target.read_bytes() == before
    assert json.loads(notice.read_text())['reason'] == '5_percent_per_question_cap'


def test_apply_rejects_unqualified_proposal(tmp_path):
    item = report()
    item['proposals'][0]['readings'][0]['confidence'] = 'low'
    with pytest.raises(ValueError):
        apply_proposals(item, tmp_path / 'q.json', {'Q': 100})
