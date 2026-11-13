import json

import pytest

from tada_taskgen.adapters import get_adapter
from tada_taskgen.adapters.base import SourceAdapter, exclusions
from tada_taskgen.intents import generate_intents, load_onet, select_undercovered, review_intent


@pytest.fixture(scope='module')
def cuad_screening():
    from tada_taskgen.screening import screen_adapter
    adapter = get_adapter('CUAD')
    return adapter, screen_adapter(adapter)


def test_local_onet_real_citations_and_work_chain(screened_fixture):
    adapter, screening = screened_fixture
    intents = generate_intents([adapter], limit=6, review=lambda _: {'approved': True},
                               screening_reports={'CUAD': screening})
    assert len(intents) == 6
    _, tasks, dwa = load_onet()
    references = {(r['O*NET-SOC Code'], r['Task ID']): r['Task'] for r in tasks}
    for intent in intents:
        task = intent['onet_tasks'][0]
        assert references[intent['occupation'], task['id']] == task['text']
        assert 2 <= len(intent['work_chain']) <= 4
        assert intent['clean_label_counts'] and min(intent['clean_label_counts'].values()) > 0
        assert intent['dwa'] == dwa.get((intent['occupation'], task['id']), [])
        assert intent['review']['approved']
        assert intent['back_translation'] and intent['deliverable']
        assert set(intent['questions']) == set(adapter.questions())
        assert intent['screening_report']['used_for_admission'] is False
        assert intent['screening_version'] == screening['version']
    assert len(set(i['work_chain_pattern'] for i in intents)) == 2


def test_intent_review_rejects_before_design(screened_fixture):
    adapter, screening = screened_fixture
    assert generate_intents([adapter], limit=3, review=lambda _: {'approved': False},
                            screening_reports={'CUAD': screening}) == []


def test_quota_selection_prefers_undercoverage_and_stops():
    intents = [{'id': 'a', 'quota_key': '23|MAUD|revision'}, {'id': 'b', 'quota_key': '23|CUAD|scope'}]
    assert select_undercovered(intents, {'23|MAUD|revision': 1})['id'] == 'b'
    assert select_undercovered(intents, {i['quota_key']: 1 for i in intents}) is None
    assert select_undercovered(intents, {i['quota_key']: 1 for i in intents}, {'23|MAUD|revision': 2})['id'] == 'a'


class ScreeningFixtureAdapter(SourceAdapter):
    """A local pool with independently controlled screening failure reasons."""
    source = 'CUAD'

    def questions(self):
        return {q: {'question': q + ' contract provision', 'options': ['No', 'Yes'],
                    'coding_conventions': []}
                for q in ('hard', 'easy', 'rare', 'small', 'single')}

    def records(self):
        return {f'R{i:02d}': {'text': f"Agreement {i}: {'Yes' if i % 2 else 'No'}"} for i in range(40)}

    def label_pool(self):
        excluded, records = exclusions(self.quarantine, self.source)
        pool = {}
        for i, record in enumerate(self.records()):
            for q in self.questions():
                if record in records or (record, q) in excluded or q == 'small' and i >= 20:
                    continue
                answer = ('Yes' if i < 2 else 'No') if q == 'rare' else (
                    'No' if q == 'single' else 'Yes' if (i % 2 if q == 'easy' else i % 4 < 2) else 'No')
                pool[record, q] = answer, []
        return pool

    def keyword_predict(self, question, text):
        return text.rsplit(': ', 1)[1] if question == 'easy' else 'No'


@pytest.fixture
def screened_fixture(tmp_path):
    from tada_taskgen.screening import screen_adapter
    quarantine = tmp_path / 'quarantine.json'
    quarantine.write_text('{}')
    adapter = ScreeningFixtureAdapter(quarantine)
    return adapter, screen_adapter(adapter)


def test_historical_screening_does_not_veto_intents(screened_fixture):
    adapter, report = screened_fixture
    seen = []

    def review(payload):
        seen.append(payload['intent'])
        return {'approved': True}

    intents = generate_intents([adapter], limit=2, review=review, screening_reports={'CUAD': report})
    assert report['eligible_questions'] == ['hard']
    assert intents and seen
    for intent in [*intents, *seen]:
        assert set(intent['questions']) == set(adapter.questions())
        assert set(intent['question_types']) == set(adapter.questions())
        assert intent['clean_label_counts']['hard'] == 40
        assert intent['screening_report']['eligible_questions'] == ['hard']


def test_quarantine_change_invalidates_intent_screening(screened_fixture):
    from tada_taskgen.screening import ScreeningError, screen_adapter
    adapter, before = screened_fixture
    adapter.quarantine.write_text(json.dumps({'CUAD_labels': [{'record': 'R00', 'question': 'hard'}]}))
    stale = generate_intents([adapter], limit=1, screening_reports={'CUAD': before})
    assert stale[0]['screening_report']['status'] == 'unavailable'
    assert stale[0]['clean_label_counts']['hard'] == 39
    after = screen_adapter(adapter)
    assert before['version']['label_pool_sha256'] != after['version']['label_pool_sha256']
    assert before['version']['quarantine_sha256'] != after['version']['quarantine_sha256']
    intents = generate_intents([adapter], limit=1, screening_reports={'CUAD': after})
    assert intents[0]['clean_label_counts']['hard'] == 39
    assert intents[0]['screening_version'] == after['version']


def test_invalid_screening_is_reported_before_intent_review(screened_fixture):
    adapter, _ = screened_fixture
    intents = generate_intents([adapter], limit=1, review=lambda _: {'approved': True},
                              screening_reports={'CUAD': {}})
    assert intents[0]['screening_report']['status'] == 'unavailable'
    assert intents[0]['review']['approved']


def test_real_adapter_intent_coverage_is_scorable(cuad_screening):
    adapter, report = cuad_screening
    intents = generate_intents([adapter], limit=6, screening_reports={'CUAD': report})
    assert len(intents) == 6
    for intent in intents:
        assert set(intent['questions']) == {q for _, q in adapter.label_pool()}
        assert intent['screening_report']['used_for_admission'] is False
