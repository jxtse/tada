"""Frozen R2 local shortcut proxy; standard-library multinomial naive Bayes.

Text featurization is label-free and reusable. Each OOF model receives only its
training rows; vocabulary document frequencies, class priors and likelihoods
are fitted inside that fold. No test labels are passed to fit or predict.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
import re

NB_POLICY = {'folds': 5, 'fold_seed': 0, 'alpha': 1.0, 'max_features': 5000,
             'features': 'lowercase Unicode word unigrams+bigrams',
             'feature_selection': 'training-fold document frequency descending; lexical ties',
             'class_prior': 'empirical training-fold frequency',
             'prediction_ties': 'lexical class order'}


def word_features(text):
    words = re.findall(r'(?u)\b\w+\b', text.lower())
    result = Counter('u:' + word for word in words)
    result.update('b:' + left + '\0' + right for left, right in zip(words, words[1:]))
    return result


def stratified_folds(labels, *, folds=5, seed=0):
    """Partition sorted row indices once; each class differs by at most one/fold."""
    if folds < 2:
        raise ValueError('At least two folds required')
    by_class = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)
    rng = random.Random(seed)
    result = [[] for _ in range(folds)]
    offset = 0
    for label in sorted(by_class):
        rows = list(by_class[label])
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            result[(offset + i) % folds].append(row)
        offset = (offset + len(rows)) % folds
    return [sorted(rows) for rows in result]


@dataclass(frozen=True)
class MultinomialNB:
    vocabulary: tuple
    class_log_prior: dict
    feature_log_probability: dict

    def predict(self, features):
        if not self.class_log_prior:
            raise ValueError('Cannot predict with an empty training fold')
        scores = {}
        for label in sorted(self.class_log_prior):
            weights = self.feature_log_probability[label]
            scores[label] = self.class_log_prior[label] + sum(
                count * weights[feature] for feature, count in features.items() if feature in weights)
        return max(scores, key=scores.get)


def fit_nb(training_rows, *, max_features=5000, alpha=1.0):
    """Fit only (feature Counter, label) training rows; no held-out inputs."""
    if alpha <= 0 or max_features < 1:
        raise ValueError('Positive smoothing and feature limit required')
    training_rows = list(training_rows)
    if not training_rows:
        raise ValueError('Training fold is empty')
    df = Counter()
    classes = Counter()
    for features, label in training_rows:
        df.update(features.keys())
        classes[label] += 1
    vocabulary = tuple(sorted(df, key=lambda term: (-df[term], term))[:max_features])
    totals = {label: Counter() for label in classes}
    for features, label in training_rows:
        totals[label].update({feature: features[feature] for feature in vocabulary if feature in features})
    probabilities = {}
    for label in sorted(classes):
        denominator = sum(totals[label].values()) + alpha * len(vocabulary)
        probabilities[label] = {feature: math.log((totals[label][feature] + alpha) / denominator)
                                for feature in vocabulary}
    return MultinomialNB(vocabulary,
                         {label: math.log(n / len(training_rows)) for label, n in sorted(classes.items())},
                         probabilities)


def oof_predict(feature_rows, labels):
    """Return one held-out prediction per row and reproducible fold diagnostics."""
    if len(feature_rows) != len(labels):
        raise ValueError('Features and labels must have equal length')
    if len(labels) < 2:
        return None, []
    partitions = stratified_folds(labels, folds=NB_POLICY['folds'], seed=NB_POLICY['fold_seed'])
    predictions = [None] * len(labels)
    diagnostics = []
    for fold, test_indices in enumerate(partitions):
        held_out = set(test_indices)
        train_indices = [i for i in range(len(labels)) if i not in held_out]
        if not test_indices:
            continue
        model = fit_nb([(feature_rows[i], labels[i]) for i in train_indices],
                       max_features=NB_POLICY['max_features'], alpha=NB_POLICY['alpha'])
        for i in test_indices:
            predictions[i] = model.predict(feature_rows[i])
        diagnostics.append({'fold': fold, 'train_indices': train_indices, 'test_indices': test_indices,
                            'training_class_counts': dict(sorted(Counter(labels[i] for i in train_indices).items())),
                            'vocabulary_size': len(model.vocabulary)})
    return predictions, diagnostics
