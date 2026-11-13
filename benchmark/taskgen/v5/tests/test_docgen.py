import pytest
from tada_taskgen.clients import FakeClient
from tada_taskgen.docgen import generate_document, generate_documents
from tada_taskgen.roundtrip import verify_document


def document():
    return {'id': 'notice', 'kind': 'email', 'path': 'notice.txt',
            'facts': {'scope': ['A', 'B'], 'cutoff': '2026-09-30', 'amount': 3, 'approved': True}}


def test_roundtrip_blind_and_exact():
    writer, reader = FakeClient('writer'), FakeClient('reader')
    result = generate_document(document(), writer, reader)
    assert result['status'] == 'verified'
    assert len(result['attempts']) == 1
    payload = reader.calls[0]['payload']
    assert set(payload) == {'instruction', 'text', 'fact_names'}
    assert 'facts' not in payload
    assert payload['text'] == result['text']
    altered = result['text'].replace('2026-09-30', '2026-10-01')
    assert not verify_document(altered, document()['facts'], reader)['pass']


def test_max_three_regenerations_drop_document():
    writer, reader = FakeClient('writer'), FakeClient('reader', {'extract_facts': {'facts': {}}})
    result = generate_document(document(), writer, reader)
    assert result['status'] == 'dropped'
    assert len(result['attempts']) == 4
    assert result['text'] == '' and result['facts'] == {}
    assert set(result['dropped_facts']) == set(document()['facts'])
    report = generate_documents([document()], writer, reader)
    assert not report['pass']


def test_retry_until_exact():
    calls = []
    def extract(payload):
        calls.append(payload)
        return {'facts': document()['facts'] if len(calls) == 3 else {}}
    result = generate_document(document(), FakeClient('a'), FakeClient('b', {'extract_facts': extract}))
    assert result['pass'] and len(result['attempts']) == 3


def test_same_model_refused():
    with pytest.raises(ValueError, match='different model'):
        generate_document(document(), FakeClient('same'), FakeClient('same'))


def test_json_equality_is_type_strict():
    result = verify_document('x', {'value': True}, FakeClient('x', {'extract_facts': {'facts': {'value': 1}}}))
    assert not result['pass']


def test_overlong_retry_budget_refused():
    with pytest.raises(ValueError):
        generate_document(document(), max_regenerations=4)
