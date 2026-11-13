"""R2 frozen-rule and full-pool screening contract, using synthetic labels."""
from collections import Counter
import copy
import json
from pathlib import Path

import pytest

from tada_taskgen.adapters import get_adapter
from tada_taskgen.adapters.base import distinctive_tokens, generic_term_predict
from tada_taskgen.adapters.maud import MAUDAdapter
from tada_taskgen.screening import (
    POLICY, ScreeningError, load_screening, reference_adapter, require_eligible, screen_adapter,
)


def test_generic_term_rule_requires_all_distinctive_tokens():
    options = ['cash only', 'cash plus stock']
    assert distinctive_tokens(options) == [{'only'}, {'plus', 'stock'}]
    assert generic_term_predict(options, 'The payment includes cash plus STOCK.') == options[1]
    assert generic_term_predict(options, 'The payment includes stock.') == options[0]


def test_generic_term_rule_whole_tokens_option_order_and_fallback():
    options = ['strict', 'loose']
    assert generic_term_predict(options, 'loosely constrained') == 'strict'
    assert generic_term_predict(options, 'A LOOSE requirement') == 'loose'
    assert generic_term_predict(options, 'loose and strict alternatives') == 'strict'
    assert generic_term_predict(['common', 'common common'], 'common') == 'common'
    with pytest.raises(ValueError):
        generic_term_predict([], 'text')


def test_maud_non_dedicated_uses_generic_but_dedicated_is_unchanged(tmp_path):
    adapter = MAUDAdapter(quarantine=tmp_path / 'missing.json')
    adapter._questions = {
        'synthetic': {'options': ['strict', 'loose']},
        'DP132': {'options': ['All Cash', 'All Stock', 'Mixed Cash/Stock', 'Mixed Cash/Stock: Election']},
    }
    assert adapter.keyword_predict('synthetic', 'loose obligation') == 'loose'
    # The existing DP132 rule recognizes Exchange Ratio without literal option text.
    assert adapter.keyword_predict('DP132', 'The Exchange Ratio is two shares.') == 'All Stock'


class FixtureAdapter:
    source = 'MAUD'

    def __init__(self, quarantine, questions=None):
        self.quarantine = Path(quarantine)
        self.data = questions or {'eligible': (20, 20, 13, 13)}
        self.calls = []

    def questions(self):
        return {q: {'question': 'Synthetic question ' + q, 'options': ['A', 'B'],
                    'coding_conventions': ['Synthetic fixture only.']} for q in self.data}

    def records(self):
        result = {}
        for q, (a, b, correct_a, correct_b) in self.data.items():
            for i in range(a + b):
                actual = 'A' if i < a else 'B'
                correct = i < correct_a if i < a else i - a < correct_b
                prediction = actual if correct else 'B' if actual == 'A' else 'A'
                result[f'{q}-{i}'] = {'text': prediction}
        return result

    def label_pool(self):
        exclusions = json.loads(self.quarantine.read_text()).get('exclude', []) if self.quarantine.exists() else []
        result = {}
        for q, (a, b, _ca, _cb) in self.data.items():
            for i in range(a + b):
                record = f'{q}-{i}'
                if record not in exclusions:
                    result[record, q] = ('A' if i < a else 'B', [])
        return result

    def keyword_predict(self, question, text):
        self.calls.append((question, text))
        return text


@pytest.fixture
def adapter(tmp_path):
    return FixtureAdapter(tmp_path / 'quarantine.json', {
        'eligible': (20, 20, 13, 13),
        'too_easy': (20, 20, 14, 13),
        'small_pool': (10, 19, 0, 0),
        'small_class': (4, 36, 0, 0),
        'single_class': (40, 0, 0, 0),
        'empty': (0, 0, 0, 0),
    })


def test_full_pool_balanced_accuracy_frozen_boundaries_and_all_question_reasons(adapter):
    result = screen_adapter(adapter)
    assert POLICY == {'minimum_labels': 30, 'minimum_observed_classes': 2,
                      'shortcut_balanced_accuracy_max': .65, 'minimum_class_count': 5}
    assert result['eligible_questions'] == ['eligible']
    assert len(adapter.calls) == len(adapter.label_pool())
    assert set(result['questions']) == set(adapter.questions())
    row = result['questions']['eligible']
    assert row['kw_bal_q'] == .65 and row['maj_q'] == .5
    assert row['minority_count'] == 20 and row['minority_share'] == .5
    assert row['class_recalls'] == {'A': .65, 'B': .65}
    assert any('shortcut_bal_q' in reason for reason in result['questions']['too_easy']['reasons'])
    assert any('label_count' in reason for reason in result['questions']['small_pool']['reasons'])
    assert any('minority_count' in reason for reason in result['questions']['small_class']['reasons'])
    assert any('observed_classes' in reason for reason in result['questions']['single_class']['reasons'])
    assert result['questions']['empty']['kw_bal_q'] is None


def test_imbalanced_classes_use_mean_recall_not_accuracy(tmp_path):
    adapter = FixtureAdapter(tmp_path / 'missing.json', {'q': (35, 5, 35, 0)})
    row = screen_adapter(adapter)['questions']['q']
    assert row['maj_q'] == .875 and row['kw_bal_q'] == .5
    assert row['minority_share'] == .125 and row['eligible']


def test_require_eligible_reports_exact_question_reasons(adapter):
    report = screen_adapter(adapter)
    calls = len(adapter.calls)
    assert require_eligible(adapter, ['eligible'], report) is report
    assert len(adapter.calls) == calls  # Validation never fits or remeasures the baseline.
    with pytest.raises(ScreeningError, match='too_easy.*shortcut_bal_q'):
        require_eligible(adapter, ['too_easy'], report)
    with pytest.raises(ScreeningError, match='unknown question'):
        require_eligible(adapter, ['missing'], report)


def test_missing_report_has_actionable_error(adapter, tmp_path):
    with pytest.raises(ScreeningError, match='Missing screening report'):
        load_screening(adapter, tmp_path / 'missing.json')


def test_quarantine_change_invalidates_even_if_it_excludes_no_current_pair(adapter):
    report = screen_adapter(adapter)
    adapter.quarantine.write_text(json.dumps({'policy_note': 'changed'}))
    with pytest.raises(ScreeningError, match='quarantine_sha256'):
        load_screening(adapter, report)


def test_pool_change_invalidates_cache(adapter):
    report = screen_adapter(adapter)
    adapter.data['eligible'] = (21, 19, 13, 13)
    with pytest.raises(ScreeningError, match='label_pool_sha256'):
        load_screening(adapter, report)


@pytest.mark.parametrize('field', ['rule_version', 'rule_sha256', 'record_text_sha256'])
def test_rule_or_text_version_change_invalidates_cache(adapter, field):
    report = screen_adapter(adapter)
    report['version'][field] = 'stale'
    with pytest.raises(ScreeningError, match=field):
        load_screening(adapter, report)


@pytest.mark.parametrize('mutation', ['policy', 'promoted', 'index', 'metric', 'counts'])
def test_cached_policy_metrics_and_eligibility_cannot_be_overridden(adapter, mutation):
    report = screen_adapter(adapter)
    if mutation == 'policy':
        report['policy']['shortcut_balanced_accuracy_max'] = .9
    elif mutation == 'promoted':
        report['questions']['too_easy']['eligible'] = True
        report['questions']['too_easy']['reasons'] = []
    elif mutation == 'index':
        report['eligible_questions'].append('too_easy')
    elif mutation == 'metric':
        report['questions']['too_easy']['kw_bal_q'] = .1
    else:
        report['questions']['too_easy']['class_counts']['A'] = 200
    with pytest.raises(ScreeningError):
        load_screening(adapter, report)


def test_source_loader_pins_explicit_permutation_and_reference_ignores_environment(monkeypatch):
    original = get_adapter('CUAD', permutation_seed='')
    expected = original.label_pool()
    monkeypatch.setenv('TADA_LABEL_PERMUTE', '91')
    # Explicit reference stays authoritative even if source._data is lazy.
    assert reference_adapter(original).label_pool() == expected
    permuted = get_adapter('CUAD')
    monkeypatch.delenv('TADA_LABEL_PERMUTE')
    assert set(permuted.label_pool()) == set(expected)
    assert permuted.label_pool() != expected
    assert reference_adapter(permuted).label_pool() == expected


def test_source_loader_cache_key_includes_quarantine_digest(tmp_path, monkeypatch):
    from tada_taskgen.adapters import base
    quarantine = tmp_path / 'source-quarantine.json'
    quarantine.write_text('{}')
    monkeypatch.setattr(base, 'QUARANTINE', quarantine)
    monkeypatch.setenv('TADA_QUARANTINE', str(quarantine))
    first = base.load_source('c2_sources', '')
    assert base.load_source('c2_sources', '') is first
    quarantine.write_text('{"policy_note":"changed"}')
    assert base.load_source('c2_sources', '') is not first
