import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tada_taskgen.adapters import get_adapter
from tada_taskgen.adapters.base import REPO, normalize
from tada_taskgen.adapters.cuad import jaccard_words


@pytest.fixture(scope='module', params=['MAUD', 'CUAD'])
def adapter(request):
    return get_adapter(request.param)


def test_all_question_options_are_bijective(adapter):
    tested = 0
    for question, metadata in adapter.questions().items():
        options = metadata['options']
        values = [adapter.encode(question, option) for option in options]
        assert len(values) == len(set(values))
        assert metadata['question'] and metadata['coding_conventions']
        for original, visible in zip(options, values):
            assert adapter.decode(question, visible) == original
            assert adapter.encode(question, adapter.decode(question, visible)) == visible
            tested += 1
        with pytest.raises(ValueError):
            adapter.decode(question, '__INVALID_OPTION__')
    assert tested >= 16


def test_frozen_package_used_questions_covered(adapter):
    package = ('MAUD-20261111T061721Z' if adapter.source == 'MAUD' else 'C2-20261106T213701Z')
    build = (REPO / 'runs' / package / 'controller/task/build/build.py').read_text()
    import re
    used = set(re.findall(r'\bDP\d{3}\b', build)) if adapter.source == 'MAUD' else {
        'exclusivity', 'termination_for_convenience', 'change_of_control'}
    assert used <= set(adapter.questions())
    for question in used:
        for option in adapter.questions()[question]['options']:
            assert adapter.decode(question, adapter.encode(question, option)) == option


def test_record_ids_bijective_and_records_label_free(adapter):
    mapping = adapter.record_map()
    assert len(mapping) == len(set(mapping.values()))
    for record, visible in mapping.items():
        assert adapter.record_id(visible) == record
        assert adapter.visible_id(record) == visible
        assert 'labels' not in adapter.records()[record]
        assert 'evidence_spans' not in adapter.records()[record]


def test_pool_preserves_original_labels_and_grounding(adapter):
    pool = adapter.label_pool()
    normalized = {record: normalize(row['text']) for record, row in adapter.records().items()}
    assert pool
    for (record, question), (answer, evidence) in pool.items():
        assert answer in adapter.questions()[question]['options']
        assert all(normalize(segment) in normalized[record] for segment in evidence)
    for key in list(pool)[:5]:
        assert adapter.scorable(*key)
    assert not adapter.scorable('__missing__', '__missing__')


def test_quarantine_excludes_without_relabel(tmp_path):
    adapter = get_adapter('CUAD')
    key, original = next(iter(adapter.label_pool().items()))
    target = tmp_path / 'quarantine.json'
    target.write_text(json.dumps({'CUAD_labels': [{'contract_id': key[0], 'question': key[1]}]}))
    filtered = get_adapter('CUAD', quarantine=target)
    assert key not in filtered.label_pool()
    assert adapter.label_pool()[key] == original


def test_cuad_official_span_hook_separate_from_categorical():
    assert jaccard_words('A clause, here.', 'a clause here') == 1
    assert jaccard_words('a b', 'a c') == pytest.approx(1 / 3)
    adapter = get_adapter('CUAD')
    assert adapter.official_score('exclusivity', 'Yes', 'Yes') == 1
    assert adapter.official_score('exclusivity', 'yes', 'Yes') == 0
    assert adapter.official_score('exclusivity', 'No', 'Yes') == 0


def test_cuad_permutation_preserves_eligible_pairs(monkeypatch):
    original = get_adapter('CUAD')
    keys = set(original.label_pool())
    monkeypatch.setenv('TADA_LABEL_PERMUTE', '91')
    permuted = get_adapter('CUAD')
    assert set(permuted.label_pool()) == keys
    assert permuted.records() == original.records()
    assert any(permuted.label_pool()[k] != original.label_pool()[k] for k in sorted(keys)[:100])


def test_keyword_rules_are_label_free_and_valid(adapter):
    text = next(iter(adapter.records().values()))['text']
    for question, metadata in adapter.questions().items():
        assert adapter.keyword_predict(question, text) in metadata['options']
