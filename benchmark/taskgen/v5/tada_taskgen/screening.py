"""Frozen R2 full-pool question screening with three shortcut proxies.

This local computation is independent of sampled tasks and model trials. Cached
reports are invalid after label, evidence, text, quarantine, question, or rule
changes; compilation never silently refits the screening models.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .adapters import get_adapter
from .adapters.base import REPO, generic_term_predict
from .adapters.maud_keywords import DEDICATED_QUESTIONS
from .screening_nb import NB_POLICY, oof_predict, word_features

RULE_VERSION = 'r2-three-proxies-nb-oof-v1'
REPORT_VERSION = 2
POLICY = {'minimum_labels': 30, 'minimum_observed_classes': 2,
          'shortcut_balanced_accuracy_max': 0.65, 'minimum_class_count': 5}
DEFAULT_OUTPUT = REPO / 'v5_runs' / 'screening'
BASELINES = ('keyword', 'dedicated', 'generic', 'nb_oof')
METRIC_FIELDS = {'keyword': 'kw_bal_q', 'dedicated': 'dedicated_bal_q',
                 'generic': 'generic_bal_q', 'nb_oof': 'nb_oof_bal_q'}


class ScreeningError(ValueError):
    """Missing, stale, invalid, or failed question screening."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def reference_adapter(adapter):
    """Screen authoritative labels even during a controller permutation build."""
    if hasattr(adapter, 'reference'):
        return adapter.reference()
    if hasattr(adapter, 'unpermuted'):
        return adapter.unpermuted()
    return adapter


def _snapshot(adapter):
    adapter = reference_adapter(adapter)
    pool, records, questions = adapter.label_pool(), adapter.records(), adapter.questions()
    unknown = {q for _, q in pool} - set(questions)
    if unknown:
        raise ScreeningError('Pool contains unknown questions: ' + ', '.join(sorted(unknown)))
    module = Path(__file__).parent
    files = [module / 'adapters' / name for name in ['base.py', 'maud.py', 'maud_pseudo.py', 'maud_keywords.py', 'cuad.py']]
    files += [Path(__file__), module / 'screening_nb.py']
    implementation = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    quarantine = Path(adapter.quarantine)
    version = {
        'label_pool_sha256': digest([[record, q, answer, list(evidence)]
                                    for (record, q), (answer, evidence) in sorted(pool.items())]),
        'quarantine_sha256': hashlib.sha256(quarantine.read_bytes()).hexdigest() if quarantine.exists() else 'missing',
        'record_text_sha256': digest({r: hashlib.sha256(row['text'].encode()).hexdigest()
                                      for r, row in sorted(records.items())}),
        'question_schema_sha256': digest(questions),
        'rule_version': RULE_VERSION,
        'rule_sha256': digest(implementation),
        'nb_policy_sha256': digest(NB_POLICY),
        'pseudonymiser_version': getattr(adapter, 'pseudonymiser_version', 'source-default'),
    }
    return adapter, pool, records, questions, version


def _decision(counts, shortcut):
    size = sum(counts.values())
    reasons = []
    if size < POLICY['minimum_labels']:
        reasons.append(f'label_count {size} < {POLICY["minimum_labels"]}')
    if len(counts) < POLICY['minimum_observed_classes']:
        reasons.append(f'observed_classes {len(counts)} < {POLICY["minimum_observed_classes"]}')
    minority = min(counts.values()) if counts else 0
    if minority < POLICY['minimum_class_count']:
        reasons.append(f'minority_count {minority} < {POLICY["minimum_class_count"]}')
    if shortcut is None:
        reasons.append('shortcut_balanced_accuracy unavailable (empty pool)')
    elif shortcut > POLICY['shortcut_balanced_accuracy_max']:
        reasons.append(f'shortcut_bal_q {shortcut:.12g} > {POLICY["shortcut_balanced_accuracy_max"]}')
    return not reasons, reasons


def _has_dedicated(adapter, question):
    return adapter.source == 'CUAD' or (adapter.source == 'MAUD' and question in DEDICATED_QUESTIONS)


def _metrics(labels, predictions):
    if predictions is None:
        return None
    counts = Counter(labels)
    correct = Counter(y for y, predicted in zip(labels, predictions) if y == predicted)
    recalls = {y: correct[y] / n for y, n in sorted(counts.items())}
    return {'class_correct_counts': {y: correct[y] for y in sorted(counts)},
            'class_recalls': recalls,
            'balanced_accuracy': sum(recalls.values()) / len(recalls) if recalls else None}


def _balanced(metrics):
    return metrics['balanced_accuracy'] if metrics is not None else None


def screen_adapter(adapter):
    adapter, pool, records, questions, version = _snapshot(adapter)
    by_question = defaultdict(list)
    for (record, question), (answer, _evidence) in sorted(pool.items()):
        if answer not in questions[question]['options']:
            raise ScreeningError(f'Original answer not in options: {record}:{question}')
        by_question[question].append((record, answer))
    # Featurization uses pseudonymised full agreement text only, without labels.
    # Vocabulary selection and all learned statistics still happen in each fold.
    features = {record: word_features(records[record]['text']) for record in sorted({r for r, _ in pool})}
    rows = {}
    for question in sorted(questions):
        examples = by_question[question]
        record_ids = [r for r, _ in examples]
        labels = [y for _, y in examples]
        counts = dict(sorted(Counter(labels).items()))
        options = questions[question]['options']
        keyword = [adapter.keyword_predict(question, records[r]['text']) for r in record_ids]
        generic = [generic_term_predict(options, records[r]['text']) for r in record_ids]
        if any(predicted not in options for predicted in keyword + generic):
            raise ScreeningError(f'Keyword prediction not in options: {question}')
        nb_predictions, folds = oof_predict([features[r] for r in record_ids], labels)
        metrics = {'keyword': _metrics(labels, keyword),
                   'dedicated': _metrics(labels, keyword) if _has_dedicated(adapter, question) else None,
                   'generic': _metrics(labels, generic), 'nb_oof': _metrics(labels, nb_predictions)}
        shortcuts = [_balanced(metrics[name]) for name in ('dedicated', 'generic', 'nb_oof')]
        shortcut = max((value for value in shortcuts if value is not None), default=None)
        size = len(labels)
        minority = min(counts.values()) if counts else 0
        eligible, reasons = _decision(counts, shortcut)
        rows[question] = {
            'question': questions[question]['question'], 'n_labels': size,
            'n_observed_classes': len(counts), 'class_counts': counts,
            'class_correct_counts': metrics['keyword']['class_correct_counts'],
            'class_recalls': metrics['keyword']['class_recalls'],
            'maj_q': max(counts.values()) / size if size else None,
            **{field: _balanced(metrics[name]) for name, field in METRIC_FIELDS.items()},
            'shortcut_bal_q': shortcut, 'baseline_metrics': metrics,
            'nb_oof_record_ids': record_ids, 'nb_oof_folds': folds,
            'minority_count': minority, 'minority_share': minority / size if size else None,
            'minority_classes': [answer for answer, count in counts.items() if count == minority],
            'eligible': eligible, 'reasons': reasons,
        }
    eligible = [question for question, row in rows.items() if row['eligible']]
    return {'schema_version': REPORT_VERSION, 'source': adapter.source, 'version': version,
            'policy': dict(POLICY), 'nb_policy': dict(NB_POLICY), 'pool_size': len(pool),
            'question_count': len(questions), 'eligible_count': len(eligible),
            'eligible_questions': eligible, 'questions': rows,
            'keyword_rule': 'Dedicated rules retained. Generic requires all nonempty option-distinctive '
                            'ASCII alphanumeric tokens; option-order ties and first-option fallback. '
                            'Dedicated is null when no dedicated rule exists. kw_bal_q is the unchanged '
                            'deployable keyword baseline; eligibility uses max(dedicated, generic, nb_oof).'}


def load_screening(adapter, report=None):
    adapter, pool, _records, questions, version = _snapshot(adapter)
    if report is None:
        report = DEFAULT_OUTPUT / (adapter.source + '.json')
    if isinstance(report, (str, Path)):
        path = Path(report)
        if not path.is_file():
            raise ScreeningError(f'Missing screening report {path}; run python -m tada_taskgen.screening --source {adapter.source}')
        try:
            report = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            raise ScreeningError(f'Invalid screening report {path}: {exc}') from exc
    if not isinstance(report, dict):
        raise ScreeningError('Screening report must be a JSON object or path')
    if report.get('schema_version') != REPORT_VERSION or report.get('source') != adapter.source:
        raise ScreeningError('Screening report source/schema mismatch')
    stale = [key for key, value in version.items() if report.get('version', {}).get(key) != value]
    if stale:
        raise ScreeningError('Stale screening report: ' + ', '.join(stale) + '; rerun explicit screening before compilation')
    if canonical(report.get('policy')) != canonical(POLICY) or canonical(report.get('nb_policy')) != canonical(NB_POLICY):
        raise ScreeningError('Screening report policy differs from frozen R2 policy')
    if set(report.get('questions', {})) != set(questions):
        raise ScreeningError('Screening report does not cover every adapter question')
    observed = defaultdict(Counter)
    question_records = defaultdict(list)
    for (record, question), (answer, _evidence) in sorted(pool.items()):
        observed[question][answer] += 1
        question_records[question].append(record)
    eligible = []
    for question in sorted(questions):
        row = report['questions'][question]
        counts = dict(sorted(observed[question].items())); size = sum(counts.values())
        if row.get('class_counts') != counts or row.get('n_labels') != size or row.get('n_observed_classes') != len(counts):
            raise ScreeningError(f'Invalid screening counts for {question}')
        metrics = row.get('baseline_metrics', {})
        if set(metrics) != set(BASELINES):
            raise ScreeningError(f'Invalid screening baselines for {question}')
        values = {}
        for name in BASELINES:
            metric = metrics[name]
            unavailable = (name == 'dedicated' and not _has_dedicated(adapter, question)) or (name == 'nb_oof' and size < 2)
            if unavailable:
                if metric is not None:
                    raise ScreeningError(f'Unexpected screening baseline {name} for {question}')
                values[name] = None
                continue
            if not isinstance(metric, dict):
                raise ScreeningError(f'Missing screening baseline {name} for {question}')
            correct = metric.get('class_correct_counts', {})
            if set(correct) != set(counts) or any(type(n) is not int or not 0 <= n <= counts[y] for y, n in correct.items()):
                raise ScreeningError(f'Invalid screening correct counts for {question}:{name}')
            recalls = {y: correct[y] / n for y, n in counts.items()}
            balanced = sum(recalls.values()) / len(recalls) if recalls else None
            if canonical(metric.get('class_recalls')) != canonical(recalls) or metric.get('balanced_accuracy') != balanced:
                raise ScreeningError(f'Invalid screening baseline metrics for {question}:{name}')
            values[name] = balanced
        shortcut = max((values[n] for n in ('dedicated', 'generic', 'nb_oof') if values[n] is not None), default=None)
        minority = min(counts.values()) if counts else 0
        derived = {**{field: values[name] for name, field in METRIC_FIELDS.items()},
                   'class_correct_counts': metrics['keyword']['class_correct_counts'],
                   'class_recalls': metrics['keyword']['class_recalls'], 'shortcut_bal_q': shortcut,
                   'maj_q': max(counts.values()) / size if size else None,
                   'minority_count': minority, 'minority_share': minority / size if size else None,
                   'minority_classes': [y for y, n in counts.items() if n == minority]}
        if any(canonical(row.get(k)) != canonical(v) for k, v in derived.items()):
            raise ScreeningError(f'Invalid screening metrics for {question}')
        _validate_folds(question, row, question_records[question], pool)
        passed, reasons = _decision(counts, shortcut)
        if row.get('eligible') is not passed or row.get('reasons') != reasons:
            raise ScreeningError(f'Invalid screening verdict for {question}')
        if passed:
            eligible.append(question)
    if (report.get('eligible_questions') != eligible or report.get('eligible_count') != len(eligible)
            or report.get('pool_size') != len(pool) or report.get('question_count') != len(questions)):
        raise ScreeningError('Invalid screening eligibility index or totals')
    return report


def _validate_folds(question, row, record_ids, pool):
    from .screening_nb import stratified_folds
    if row.get('nb_oof_record_ids') != record_ids:
        raise ScreeningError(f'Invalid NB row index for {question}')
    labels = [pool[(r, question)][0] for r in record_ids]
    expected = stratified_folds(labels) if len(labels) >= 2 else []
    diagnostics = row.get('nb_oof_folds')
    if not isinstance(diagnostics, list) or len(diagnostics) != sum(bool(ids) for ids in expected):
        raise ScreeningError(f'Invalid NB folds for {question}')
    for diagnostic, (fold, held_out) in zip(diagnostics, ((f, ids) for f, ids in enumerate(expected) if ids)):
        train = sorted(set(range(len(labels))) - set(held_out))
        if (diagnostic.get('fold') != fold or diagnostic.get('test_indices') != held_out
                or diagnostic.get('train_indices') != train
                or diagnostic.get('training_class_counts') != dict(Counter(labels[i] for i in train))
                or type(diagnostic.get('vocabulary_size')) is not int
                or not 0 <= diagnostic['vocabulary_size'] <= NB_POLICY['max_features']):
            raise ScreeningError(f'Invalid NB fold isolation for {question}:{fold}')


def require_eligible(adapter, questions, report=None):
    checked = load_screening(adapter, report)
    failures = []
    for question in questions:
        row = checked['questions'].get(question)
        if row is None:
            failures.append(f'{question}: unknown question')
        elif not row['eligible']:
            failures.append(f'{question}: ' + '; '.join(row['reasons']))
    if failures:
        raise ScreeningError('Question screening failed: ' + ' | '.join(failures))
    return checked


def markdown(report):
    lines = [f'# {report["source"]} R2 question screening', '',
             f'Eligible: {report["eligible_count"]}/{report["question_count"]}; labels: {report["pool_size"]}.', '',
             'Frozen policy: n >= 30; observed classes >= 2; max(dedicated, generic, NB OOF) <= 0.65; smallest class >= 5.', '',
             '| Question | Labels | Classes | Majority accuracy | Dedicated | Generic | NB OOF | Shortcut max | Minority count | Minority share | Verdict / reasons |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    number = lambda value: 'n/a' if value is None else f'{value:.6f}'
    for q, row in report['questions'].items():
        verdict = 'eligible' if row['eligible'] else '; '.join(row['reasons'])
        lines.append(f'| {q} | {row["n_labels"]} | {row["n_observed_classes"]} | {number(row["maj_q"])} | '
                     f'{number(row["dedicated_bal_q"])} | {number(row["generic_bal_q"])} | {number(row["nb_oof_bal_q"])} | '
                     f'{number(row["shortcut_bal_q"])} | {row["minority_count"]} | {number(row["minority_share"])} | {verdict} |')
    lines += ['', 'Version:', '', '```json', json.dumps(report['version'], indent=2), '```', '',
              'NB policy:', '', '```json', json.dumps(report['nb_policy'], indent=2), '```', '', report['keyword_rule'], '']
    return '\n'.join(lines)


def write_report(report, output_dir=DEFAULT_OUTPUT):
    directory = Path(output_dir).resolve()
    if not directory.is_relative_to(REPO.resolve()):
        raise ScreeningError('Screening output must remain inside the repository')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (report['source'] + '.json')
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    path.with_suffix('.md').write_text(markdown(report))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['MAUD', 'CUAD'], action='append', required=True)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for source in args.source:
        adapter = get_adapter(source, permutation_seed='')
        report = screen_adapter(adapter)
        output = write_report(report, args.out_dir)
        print(json.dumps({'source': source, 'questions': report['question_count'], 'eligible': report['eligible_count'],
                          'path': str(output), 'version': report['version']}, sort_keys=True))


if __name__ == '__main__':
    main()
