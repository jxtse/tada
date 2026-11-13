"""R2 train-only feature/model fitting and fixed out-of-fold protocol."""
from collections import Counter
import copy
import math

import pytest

from tada_taskgen import screening_nb as nb
from tada_taskgen.screening import ScreeningError, load_screening, screen_adapter
from test_screening import FixtureAdapter


def test_word_unigrams_bigrams_case_counts_and_no_cross_feature_collisions():
    assert nb.word_features('A a B!') == Counter({'u:a': 2, 'u:b': 1, 'b:a\0a': 1, 'b:a\0b': 1})
    assert nb.word_features('') == {}


def test_folds_are_deterministic_stratified_and_partition_every_row_once():
    labels = ['A'] * 23 + ['B'] * 7 + ['C'] * 5
    folds = nb.stratified_folds(labels)
    assert folds == nb.stratified_folds(labels, folds=5, seed=0)
    assert folds != nb.stratified_folds(labels, folds=5, seed=1)
    assert sorted(i for fold in folds for i in fold) == list(range(len(labels)))
    for label in set(labels):
        counts = [sum(labels[i] == label for i in fold) for fold in folds]
        assert max(counts) - min(counts) <= 1
    assert all(set(labels[i] for i in fold) == set(labels) for fold in folds)


def test_laplace_likelihoods_empirical_priors_and_document_frequency_not_term_frequency():
    rows = [(Counter({'common': 1, 'rare': 999}), 'A'),
            (Counter({'common': 1}), 'A'), (Counter({'other': 1}), 'B')]
    model = nb.fit_nb(rows, max_features=1)
    assert model.vocabulary == ('common',)
    assert model.class_log_prior == {'A': math.log(2/3), 'B': math.log(1/3)}
    assert model.feature_log_probability == {'A': {'common': 0.0}, 'B': {'common': 0.0}}
    model = nb.fit_nb(rows)
    assert model.feature_log_probability['B']['common'] == math.log(1/4)
    assert model.feature_log_probability['B']['other'] == math.log(2/4)


def test_top_5000_features_lexical_ties_train_only():
    features = Counter({f'u:{i:05}': 1 for i in range(5100)})
    model = nb.fit_nb([(features, 'A'), (Counter({'u:05099': 1}), 'B')])
    assert len(model.vocabulary) == 5000
    assert model.vocabulary[0] == 'u:05099'  # DF=2 beats DF=1.
    assert 'u:04998' in model.vocabulary and 'u:04999' not in model.vocabulary
    assert model.predict(Counter({'heldout_only': 1000})) == 'A'


def test_nb_deterministic_oof_and_no_heldout_text_or_label_passed_to_fit(monkeypatch):
    labels = ['A'] * 10 + ['B'] * 10
    features = [nb.word_features(f'unique{i} shared {labels[i]}') for i in range(20)]
    expected_folds = nb.stratified_folds(labels)
    original = nb.fit_nb
    calls = []

    def guarded_fit(rows, **kwargs):
        held_out = set(expected_folds[len(calls) % 5])
        identities = {id(features[i]): i for i in range(len(features))}
        indices = [identities[id(feature)] for feature, _ in rows]
        assert set(indices) == set(range(len(features))) - held_out
        assert [label for _, label in rows] == [labels[i] for i in indices]
        model = original(rows, **kwargs)
        assert all('u:unique' + str(i) not in model.vocabulary for i in held_out)
        calls.append(indices)
        return model

    monkeypatch.setattr(nb, 'fit_nb', guarded_fit)
    predictions, audit = nb.oof_predict(features, labels)
    assert (predictions, audit) == nb.oof_predict(features, labels)
    assert len(calls) == 10 and predictions == labels
    for row in audit:
        assert set(row['train_indices']).isdisjoint(row['test_indices'])


def test_changing_heldout_labels_cannot_change_their_own_predictions_with_fixed_folds(monkeypatch):
    labels = ['A'] * 10 + ['B'] * 10
    features = [nb.word_features(f'contract {'cash' if i % 2 else 'stock'}') for i in range(20)]
    folds = nb.stratified_folds(labels)
    # Stratification legitimately sees labels; freeze its result to isolate fit/predict.
    monkeypatch.setattr(nb, 'stratified_folds', lambda *args, **kwargs: folds)
    before, _ = nb.oof_predict(features, labels)
    changed = list(labels)
    for i in folds[0]:
        changed[i] = 'HELDOUT-ONLY-LABEL'
    after, _ = nb.oof_predict(features, changed)
    assert [before[i] for i in folds[0]] == [after[i] for i in folds[0]]


def test_empty_small_folds_and_prior_only_empty_documents():
    assert nb.oof_predict([], []) == (None, [])
    assert nb.oof_predict([Counter()], ['A']) == (None, [])
    predictions, folds = nb.oof_predict([Counter(), Counter()], ['A', 'B'])
    assert predictions == ['B', 'A'] and len(folds) == 2
    with pytest.raises(ValueError):
        nb.fit_nb([])


def test_nb_stronger_than_constant_generic_is_used_for_screening(tmp_path):
    class Separable(FixtureAdapter):
        def records(self):
            return {f'q-{i}': {'text': 'cash cash cash' if i < 20 else 'stock stock stock'} for i in range(40)}

        def keyword_predict(self, question, text):
            return 'A'

    adapter = Separable(tmp_path / 'q.json', {'q': (20, 20, 0, 0)})
    result = screen_adapter(adapter)
    row = result['questions']['q']
    assert row['dedicated_bal_q'] is None  # No MAUD dedicated rule for synthetic q.
    assert row['generic_bal_q'] == row['kw_bal_q'] == .5
    assert row['nb_oof_bal_q'] == row['shortcut_bal_q'] == 1
    assert not row['eligible']
    assert load_screening(adapter, result) is result


@pytest.mark.parametrize('mutation', ['nb_score', 'max', 'training_indices', 'test_indices', 'nb_policy'])
def test_cached_nb_metrics_and_fold_isolation_are_validated(tmp_path, mutation):
    adapter = FixtureAdapter(tmp_path / 'q.json')
    report = screen_adapter(adapter)
    row = report['questions']['eligible']
    if mutation == 'nb_score':
        row['nb_oof_bal_q'] = .01
    elif mutation == 'max':
        row['shortcut_bal_q'] = .01
    elif mutation == 'training_indices':
        row['nb_oof_folds'][0]['train_indices'].append(row['nb_oof_folds'][0]['test_indices'][0])
    elif mutation == 'test_indices':
        row['nb_oof_folds'][0]['test_indices'].pop()
    else:
        report['nb_policy']['fold_seed'] = 1
    with pytest.raises(ScreeningError):
        load_screening(adapter, report)
