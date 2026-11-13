"""Real cached source contracts, including typed labels and visibility under permutation."""
import json

import pytest

from tada_taskgen.adapters import (AVAILABLE_SOURCES, REFUSED_SOURCES, available_sources,
                                  canonical_source, get_adapter)
from tada_taskgen.adapters.base import generic_term_predict, load_source, normalize
from tada_taskgen.adapters.public_records import PublicRecords


NEW = ('CUADX', 'NDA', 'PREC', 'B')


@pytest.fixture(scope='module', params=NEW)
def adapter(request):
    return get_adapter(request.param, permutation_seed='')


def test_supported_registry_is_canonical_and_explicit_about_refusals():
    assert AVAILABLE_SOURCES == ('B', 'CUAD', 'CUADX', 'MAUD', 'NDA', 'PREC')
    assert canonical_source('c2') == canonical_source('cuad') == 'CUAD'
    assert get_adapter('C2').source == 'CUAD'
    assert 'finding' in REFUSED_SOURCES['NTSB_FINDINGS'] and 'private' in REFUSED_SOURCES['NTSB_FINDINGS']
    with pytest.raises(ValueError, match='not bijective'):
        get_adapter('NTSB_FINDINGS')
    assert get_adapter('NTSB').source == 'ntsb_aviation_avall'
    with pytest.raises(ValueError, match='Unsupported'):
        get_adapter('invented')


def test_all_original_options_round_trip_without_type_loss(adapter):
    for question, meta in adapter.questions().items():
        assert meta['question'] and meta['coding_conventions']
        assert set(meta) == {'question', 'options', 'answer_options', 'coding_conventions'}
        encoded = [adapter.encode(question, value) for value in meta['options']]
        assert len(set(encoded)) == len(meta['options'])
        for original, visible in zip(meta['options'], encoded):
            decoded = adapter.decode(question, visible)
            assert decoded == original and type(decoded) is type(original)
            assert adapter.encode(question, decoded) == visible
            assert adapter.official_score(question, visible, original) == 1
        with pytest.raises(ValueError):
            adapter.decode(question, '__INVALID__')


def test_public_mapping_and_private_annotation_boundary(adapter):
    mapping = adapter.record_map()
    assert mapping and len(mapping) == len(set(mapping.values()))
    for record in list(mapping)[:4]:
        assert adapter.record_id(mapping[record]) == record
        assert adapter.visible_id(record) == mapping[record]
        public = adapter.records()[record]
        allowed = {'contract_id', 'document_type', 'organizations', 'text'} if adapter.source == 'CUADX' else {'text'}
        assert set(public) == allowed
        assert public['text']
        assert adapter.pseudonymize(record, public['text']) == public['text']
        for question, meta in adapter.questions().items():
            assert adapter.keyword_predict(question, public['text']) in meta['options']


def test_permutation_preserves_public_catalogue_records_and_eligible_cells(adapter):
    permuted = get_adapter(adapter.source, permutation_seed='1709')
    original, changed = adapter.label_pool(), permuted.label_pool()
    assert set(original) == set(changed)
    assert any(original[key][0] != changed[key][0] for key in original)
    assert adapter.questions() == permuted.questions()
    assert adapter.record_map() == permuted.record_map()
    # Check public records from both label classes and distant positions in the corpus.
    records = sorted(adapter.records())
    sample = records[:2] + records[len(records)//2:len(records)//2+2] + records[-2:]
    for record in sample:
        assert adapter.records()[record] == permuted.records()[record]


def test_original_expert_labels_are_preserved(adapter):
    pool = adapter.label_pool()
    assert pool and not adapter.scorable('__missing__', '__missing__')
    if adapter.source == 'CUADX':
        raw = {row['contract_id']: row for row in load_source('cuadx_sources', '').iter_contracts()}
        expected = lambda r, q: (raw[r]['labels'][q], raw[r]['evidence_spans'][q])
    elif adapter.source == 'NDA':
        expected = load_source('nda_sources', '').label
    elif adapter.source == 'PREC':
        expected = lambda r, q: (load_source('prec_sources', '').scdb_precedent_alteration(r), [])
    else:
        raw = {row['prompt_id']: row for row in load_source('b_sources', '').iter_comparisons()}
        expected = lambda r, q: (raw[r]['direction'], raw[r]['evidence_spans'])
    for (record, question), (answer, evidence) in pool.items():
        original, original_evidence = expected(record, question)
        assert answer == original and type(answer) is type(original)
        assert evidence == original_evidence
    for key in list(pool)[:5]:
        assert adapter.scorable(*key)


def test_selected_evidence_is_grounded_in_original_public_record(adapter):
    pool = adapter.label_pool()
    # PREC has no source span annotations; do not fabricate them.
    if adapter.source == 'PREC':
        assert all(not evidence for _, evidence in pool.values())
        return
    text = {}
    for (record, _), (_, evidence) in pool.items():
        if record not in text:
            text[record] = normalize(adapter.records()[record]['text'])
        assert all(normalize(span) in text[record] for span in evidence)


def test_quarantine_excludes_pair_and_whole_record_without_relabelling(adapter, tmp_path):
    pool = adapter.label_pool()
    first = next(iter(pool))
    other = next(key for key in pool if key[0] != first[0])
    path = tmp_path / 'quarantine.json'
    whole_key = {'CUADX': 'CUADX_contracts', 'NDA': 'NDA_documents', 'PREC': 'PREC_cases', 'B': 'B_comparisons'}[adapter.source]
    path.write_text(json.dumps({adapter.source + '_labels': [{'record': first[0], 'question': first[1]}],
                                whole_key: [other[0]]}))
    filtered = get_adapter(adapter.source, quarantine=path, permutation_seed='')
    got = filtered.label_pool()
    assert first not in got and not filtered.scorable(*first)
    assert all(record != other[0] for record, _ in got)
    assert not filtered.scorable(*other)
    assert got and all(pool[key] == value for key, value in got.items())
    assert filtered.records()[first[0]] == adapter.records()[first[0]]


def test_cuadx_reuses_audited_contract_text_without_new_name_transformations():
    a, c = get_adapter('CUADX'), get_adapter('CUAD')
    shared = set(a.records()) & set(c.records())
    assert len(shared) == 197
    assert not set(a.questions()) & set(c.questions())
    for record in shared:
        assert a.records()[record] == c.records()[record]
        assert len(a.records()[record]['organizations']) >= 2
    assert a.visible_id(next(iter(shared))) != c.visible_id(next(iter(shared)))


def test_nda_keeps_original_template_text_and_legal_forms():
    a = get_adapter('NDA')
    original = {row['doc_id']: row['text'] for row in a.api.iter_documents()}
    assert len(original) == 607
    assert all(public['text'] == original[record] for record, public in a.records().items())
    assert 'retained exactly' in ' '.join(a.limitations)


def test_prec_is_one_to_one_scdb_scope_and_never_loc_absence_negatives():
    a = get_adapter('PREC')
    assert len(a.questions()) == 1
    assert all('#' not in record for record in a.records())
    assert 'LoC overruling pairs are not scored' in ' '.join(a.limitations)
    assert 'not a current-law determination' in ' '.join(a.limitations)
    record = next(iter(a.records()))
    majority = '\n\n'.join(o['text'] for o in a.api.opinions(record) if o['type'] == 'majority')
    assert a.records()[record]['text'].endswith(majority)


def test_b_original_integer_coding_and_source_quarantine_formats(tmp_path):
    a = get_adapter('B')
    assert type(generic_term_predict([-1, 0, 1], 'No numeric tokens')) is int
    for invalid in (True, False, '-1', 2, -1.0):
        with pytest.raises(ValueError):
            a.encode('direction', invalid)
    for invalid in (-1, 0, 1, '+1', '01', ' 1'):
        with pytest.raises(ValueError):
            a.decode('direction', invalid)
    original = a.label_pool()
    first = next(iter(original))
    paper = a._comparisons[first[0]]['pmc']
    other = next(key for key in original if a._comparisons[key[0]]['pmc'] != paper)
    path = tmp_path / 'q.json'
    path.write_text(json.dumps({'B_papers': [{'pmc': paper}], 'B_prompts': [{'PromptID': other[0]}]}))
    filtered = get_adapter('B', quarantine=path)
    assert not filtered.scorable(*first) and not filtered.scorable(*other)
    assert all(a._comparisons[key[0]]['pmc'] != paper and key != other for key in filtered.label_pool())


def test_large_public_corpora_are_lazy_and_bounded():
    loaded = []
    mapping = PublicRecords(['a', 'b'], lambda key: loaded.append(key) or {'text': key})
    assert list(mapping) == ['a', 'b'] and len(mapping) == 2 and loaded == []
    assert mapping['a'] == mapping['a'] == {'text': 'a'} and loaded == ['a']
    with pytest.raises(KeyError):
        mapping['missing']
