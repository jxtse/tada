"""Original group-A cells, public projections and exact type-preserving codes."""
import json
import pytest

from tada_taskgen.adapters import batch1_text as text


def unit(key, body, labels, metadata=None):
    return {'unit_id': key, 'text': body, 'labels': {q: {'value': v} for q, v in labels.items()},
            'metadata': metadata or {}, 'raw_row': {'secret': 'DO_NOT_RENDER'}}


@pytest.fixture
def fake_sources(monkeypatch):
    rows = {
        'asap-chinese-restaurant-absa': [unit('a', {'review': '餐厅环境安静，菜品很好。'}, {q: '1' for q in text.ASPECTS}),
                                       unit('b', {'review': '排队很久，菜品不好。'}, {q: '-1' for q in text.ASPECTS})],
        'biodex': [unit('a', {'title': 'Clinical case', 'abstract': 'Public report', 'fulltext': 'Original science'},
                       {'target': 'serious: 1\npatientsex: 2\ndrugs: SECRET_DRUG\n'}),
                  unit('b', {'title': 'Second case', 'abstract': 'Public report', 'fulltext': 'Clinical narrative'},
                       {'target': 'serious: 2\npatientsex: 1\n'})],
        'uda-bench-vldb': [unit('a', {'text': 'Public court opinion'},
                              {'gold': {'case_type': 'Civil', 'verdict': 'Dismissed', 'url': 'PRIVATE'}}, {'scenario': 'legal'}),
                           unit('b', {'text': 'Public criminal opinion'},
                              {'gold': {'case_type': 'Criminal', 'verdict': 'Convicted'}}, {'scenario': 'legal'})]}
    def reader(source, **kwargs):
        # BioDEX table arguments identify distinct physical occurrences.
        for row in rows[source]:
            if source == 'biodex':
                row = dict(row, unit_id=row['unit_id'] + kwargs['table'])
            yield row
    monkeypatch.setattr(text, 'iter_author', reader)
    monkeypatch.setattr(text, '_uda_definitions', lambda: {('legal', 'case_type'): 'What is the case type?', ('legal', 'verdict'): 'What is the verdict?'})
    for fn in (text._asap, text._biodex, text._uda):
        fn.cache_clear()
    yield
    for fn in (text._asap, text._biodex, text._uda):
        fn.cache_clear()


@pytest.mark.parametrize('cls', [text.ASAPAdapter, text.BioDEXAdapter, text.UDAAdapter])
def test_original_codes_public_projection_and_quarantine(fake_sources, cls, tmp_path):
    quarantine = tmp_path / 'quarantine.json'
    adapter = cls(quarantine=quarantine)
    assert adapter.label_pool()
    for (record, question), (answer, evidence) in adapter.label_pool().items():
        assert adapter.decode(question, adapter.encode(question, answer)) == answer
        assert adapter.official_score(question, adapter.encode(question, answer), answer) == 1
        assert adapter.official_score(question, '__perturbed__', answer) == 0
        assert set(adapter.records()[record]) == {'text'}
        assert 'PRIVATE' not in adapter.records()[record]['text']
        assert 'SECRET' not in adapter.records()[record]['text']
        assert adapter.record_id(adapter.visible_id(record)) == record
    first = next(iter(adapter.label_pool()))
    quarantine.write_text(json.dumps({adapter.source + '_labels': [{'record': first[0], 'question': first[1]}]}))
    assert first not in adapter.label_pool()
    assert not adapter.scorable(*first)
    with pytest.raises(ValueError):
        adapter.encode(first[1], True)


@pytest.mark.parametrize('cls', [text.ASAPAdapter, text.BioDEXAdapter, text.UDAAdapter])
def test_permutation_does_not_change_visible_records(fake_sources, cls):
    original, permuted = cls(), cls(permutation_seed=991)
    assert original.records() == permuted.records()
    assert original.questions() == permuted.questions()
    assert original.label_pool().keys() == permuted.label_pool().keys()


def ntsb_rows():
    for field, code, description in [('ev_highest_injury', 'NONE', 'None'), ('light_cond', 'DAYL', 'Daylight'),
                                     ('wx_cond_basic', 'VMC', 'Visual conditions'), ('damage', 'SUBS', 'Substantial'),
                                     ('Phase_of_Flight', 'TAKEOFF', 'Takeoff')]:
        yield {'relation': 'eADMSPUB_DataDictionary', 'row': {'Column': field, 'code_iaids': code,
               'meaning': description, 'Question_Def': 'Select ' + field}}
    for i in range(3):
        event = str(i)
        yield {'relation': 'events', 'row': {'ev_id': event, 'ev_highest_injury': 'NONE',
                                           'light_cond': 'DAYL', 'wx_cond_basic': 'VMC'}}
        yield {'relation': 'narratives', 'row': {'ev_id': event, 'narr_accp': 'Public flight report. ' * 100}}
        yield {'relation': 'aircraft', 'row': {'ev_id': event, 'damage': 'SUBS'}}
        yield {'relation': 'Occurrences', 'row': {'ev_id': event, 'Phase_of_Flight': 'TAKEOFF'}}
    yield {'relation': 'aircraft', 'row': {'ev_id': '1', 'damage': 'DEST'}}
    yield {'relation': 'Occurrences', 'row': {'ev_id': '1', 'Phase_of_Flight': 'CRUISE'}}
    yield {'relation': 'narratives', 'row': {'ev_id': '2', 'narr_accp': 'The probable cause was private.'}}


def test_ntsb_event_codes_never_aggregate_aircraft_or_render_conclusions(monkeypatch, tmp_path):
    monkeypatch.setattr(text, 'ntsb_relations', ntsb_rows)
    text._ntsb.cache_clear()
    try:
        a = text.NTSBAdapter(quarantine=tmp_path / 'q.json')
        assert set(a.records()) == {'0', '1'}
        assert ('0', 'damage') in a.label_pool()
        assert ('0', 'Phase_of_Flight') in a.label_pool()
        assert ('1', 'damage') not in a.label_pool()
        assert ('1', 'Phase_of_Flight') not in a.label_pool()
        for row in a.records().values():
            assert 'DAYL' not in row['text'] and 'SUBS' not in row['text']
        assert a.decode('damage', a.encode('damage', 'SUBS')) == 'SUBS'
        for (record, question), (answer, _) in a.label_pool().items():
            assert a.record_id(a.visible_id(record)) == record
            assert a.official_score(question, a.encode(question, answer), answer) == 1.0
            assert a.official_score(question, '__perturbed__', answer) == 0.0
        a.quarantine.write_text(json.dumps({'NTSB_events': [{'ev_id': '0'}]}))
        assert not any(record == '0' for record, _ in a.label_pool())
        assert not a.scorable('0', 'damage')
    finally:
        text._ntsb.cache_clear()


def test_ntsb_placeholder_and_duplicate_event_fail_closed():
    records, labels = text.ntsb_data([
        {'relation': 'events', 'row': {'ev_id': 'x', 'ev_highest_injury': 'NONE'}},
        {'relation': 'narratives', 'row': {'ev_id': 'x', 'narr_accp': 'See report'}}])
    assert not records and not labels
    with pytest.raises(ValueError, match='Duplicate'):
        text.ntsb_data([{'relation': 'events', 'row': {'ev_id': 'x'}}] * 2)
