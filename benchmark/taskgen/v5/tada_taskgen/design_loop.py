"""One initial design plus at most three shared gate/reviewer revision rounds."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARD_GATES = ('oracle_invertibility', 'trivial_answers', 'deterministic_offline',
              'leakage', 'keyword_baseline', 'output_contract')
MAX_REVISIONS = 3


def _review_module():
    module_spec = importlib.util.spec_from_file_location('tada_v5_package_review', REPO / 'review_package.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _mapping(value):
    return value if isinstance(value, dict) else {}


def aggregate_gates(report):
    """Project only named scalar checks; raw reports may contain gold and private paths."""
    hard = _mapping(_mapping(report).get('hard_gates'))
    if not hard and isinstance(report, dict):
        instances = report.get('instances', {})
        rows = instances.values() if isinstance(instances, dict) else instances if isinstance(instances, list) else []
        hard = next((row.get('hard_gates', {}) for row in rows if isinstance(row, dict)
                     and row.get('hard_gates')), {})
    result = {}
    for name in HARD_GATES:
        raw = _mapping(hard.get(name))
        item = {'pass': raw.get('pass') is True, 'required': True}
        if name in ('oracle_invertibility', 'keyword_baseline'):
            item['reward'] = _number(raw.get('reward'))
            item['required' if name == 'oracle_invertibility' else 'limit'] = 1.0 if name == 'oracle_invertibility' else .30
        if name == 'oracle_invertibility':
            item['invertibility'] = {'value': _mapping(raw.get('invertibility')).get('pass') is True, 'required': True}
        elif name == 'trivial_answers':
            item['results'] = {key: {'reward': _number(_mapping(_mapping(raw.get('results')).get(key)).get('reward')),
                                     'limit': .30}
                               for key in ('empty', 'all_majority', 'constant')}
        elif name == 'deterministic_offline':
            item['checks'] = {key: {'value': raw.get(key) is True, 'required': True}
                             for key in ('repeat_identical', 'offline', 'extra_rows_never_raise', 'tail_gap_no_gain')}
        elif name == 'leakage':
            item['checks'] = {key: {'value': _mapping(raw.get(key)).get('pass') is True, 'required': True}
                             for key in ('visible_scan', 'label_permutation')}
        elif name == 'output_contract':
            item['checks'] = {key: {'value': _mapping(raw.get(key)).get('pass') is True, 'required': True}
                             for key in ('contract', 'canary')}
        result[name] = item
    return result


def _review_result(review):
    module = _review_module()
    return module.validate_v5_review({'questions': review['questions']})


def _safe_review_prose(text):
    """Defense in depth: public reviewers must not assert individual correct answers."""
    unsafe = (
        r'gold(?:\.json|[_\s-]*(?:answer|label|value))|wrong_cells|correct_cells|per[_ -]?cell',
        r'["\'](?:answer|label|correctness|expected)["\']\s*:',
        r'(?:record|row|cell|document|contract)\b[^\n]{0,160}\b(?:answer|label|correct|incorrect|wrong)\b',
        r'(?:answer|label)\s*(?:=|:)|(?:记录|单元格|行)[^\n]{0,80}(?:答案|标签|正确|错误)',
        r'(?:^|\s|["\'])/[^\s]+|(?:tests|solution)/(?:gold|reference)|Traceback|\b\w+(?:Error|Exception):',
    )
    if any(re.search(pattern, text, re.I) for pattern in unsafe):
        return 'Record-level assertions or private diagnostic text were omitted; state the issue using public task requirements only.'
    return text


def feedback_message(gates, review, *, spec=None):
    """The only designer feedback channel: hard-check scalars and public review findings."""
    failures = {name: row for name, row in aggregate_gates(gates).items() if not row['pass']}
    try:
        findings = [{**row, 'evidence': _safe_review_prose(row['evidence']),
                     'required_change': _safe_review_prose(row['required_change'])}
                    for row in _review_result(review)['findings']]
    except (KeyError, ValueError, TypeError):
        findings = []
    payload = {'failed_hard_gates': failures, 'review_findings': findings}
    from .compile_errors import safe_diagnostic
    diagnostic = safe_diagnostic(_mapping(gates).get('compile_diagnostic'), spec)
    if diagnostic is not None:
        payload['compile_failure'] = diagnostic
    return ('Revise the four designer files using this consolidated feedback. The values are '
            'aggregate measurements, not record-level answers. Keep the fixed scoring policy and '
            'seed. Only the listed failed hard gates and review findings require changes.\n\n'
            + json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def _format_error(designer_dir):
    """Structural validator message for the designer; recomputed, never raw exception text."""
    task = Path(designer_dir) / 'task'
    if not task.is_dir():
        return None
    from .designer_check import check
    result = check(task)
    if result.get('ok'):
        return None
    message = str(result.get('error') or '')[:600]
    return message if message and not re.search(r'(?:^|\s)/|Traceback|gold', message, re.I) else None


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def prepare_candidate(intent, candidate_root, *, designer, compile_and_gate, reviewer,
                      max_revisions=MAX_REVISIONS):
    """Injectable callbacks; no extra retry is hidden inside this revision budget.

    designer(intent, attempt_dir, previous_designer_dir, feedback_text) returns a
    four-file directory (or a spec for deterministic test clients).
    compile_and_gate(spec, output_dir) returns a report with task and hard_gates.
    reviewer(task_path, gate_report, review_dir) returns the four-question JSON.
    """
    from .schema import load_designer_output, validate_spec
    if type(max_revisions) is not int or not 0 <= max_revisions <= MAX_REVISIONS:
        raise ValueError('At most three revision rounds are allowed')
    root = Path(candidate_root).resolve()
    if not root.is_relative_to(REPO / 'v5_runs'):
        raise ValueError('Design loop output must stay under v5_runs')
    root.mkdir(parents=True, exist_ok=True)
    history, previous, feedback = [], None, ''
    spec, task, gates, review = None, None, {}, {}
    for revision in range(max_revisions + 1):
        attempt = root / f'attempt-{revision}'
        attempt.mkdir()
        if feedback:
            (attempt / 'feedback.md').write_text(feedback + '\n')
        failures = []
        gates, review, task = {}, {}, None
        try:
            designed = designer(intent, attempt / 'designer', previous, feedback)
            if isinstance(designed, dict):
                spec = validate_spec(designed)
            else:
                directory = Path(designed)
                spec = load_designer_output(directory / 'task' if (directory / 'task').is_dir() else directory)
                previous = directory
            from .designer_sources import validate_assignment
            validate_assignment(spec, intent)
        except Exception as exc:
            failures.append('designer_output_invalid')
            if (attempt / 'designer/receipt.json').is_file():
                previous = attempt / 'designer'
            _write(attempt / 'error.json', {'stage': 'designer', 'error_type': type(exc).__name__})
        if not failures:
            try:
                gates = compile_and_gate(spec, attempt / 'gates')
                task = gates.get('task')
                if not task:
                    raise ValueError('Gate report has no compiled task')
                if not all(row['pass'] for row in aggregate_gates(gates).values()):
                    failures.extend(name for name, row in aggregate_gates(gates).items() if not row['pass'])
            except Exception as exc:
                failures.append('compile_or_gates_unavailable')
                from .compile_errors import from_exception
                diagnostic = from_exception(exc, spec)
                if diagnostic is not None:
                    gates['compile_diagnostic'] = diagnostic
                _write(attempt / 'error.json', {'stage': 'compile_and_gate', 'error_type': type(exc).__name__})
        if task:
            try:
                review = _review_result(reviewer(Path(task), gates, attempt / 'review'))
                if not review['pass']:
                    failures.extend('review:' + row['id'] for row in review['findings'])
            except Exception as exc:
                failures.append('review_unavailable_or_invalid')
                _write(attempt / 'review_error.json', {'stage': 'reviewer', 'error_type': type(exc).__name__})
                # Infrastructure failure, not a design defect: stop without spending designer revisions.
                history.append({'revision': revision, 'pass': False, 'failures': failures, 'feedback': ''})
                _write(attempt / 'result.json', history[-1])
                result = {'verdict': 'infra_failed', 'spec': spec, 'task': str(task), 'gates': gates, 'review': {},
                          'revisions': revision, 'failures': failures, 'attempts': history}
                _write(root / 'design_loop.json', result)
                return result
        feedback = feedback_message(gates, review, spec=spec)
        format_error = _format_error(attempt / 'designer') if 'designer_output_invalid' in failures else None
        if format_error:
            feedback += '\n\nFormat check error (from the same check as check_task_files): ' + format_error
        if any(name in failures for name in ('designer_output_invalid', 'compile_or_gates_unavailable',
                                              'review_unavailable_or_invalid')):
            # No raw exception or private path reaches the designer.
            feedback += '\n\nUncompleted checks: ' + ', '.join(name for name in failures if name in (
                'designer_output_invalid', 'compile_or_gates_unavailable', 'review_unavailable_or_invalid'))
        history.append({'revision': revision, 'pass': not failures, 'failures': failures,
                        'feedback': feedback})
        _write(attempt / 'result.json', history[-1])
        if not failures:
            break
    # Owner policy (2026-11-12): after the revision budget, remaining severity-2 review findings are
    # minor and do not block; they are recorded on the task card. Hard gates and severity 3 still block.
    notes = []
    if failures and task and all(name.startswith('review:') for name in failures):
        if all(row.get('severity', 0) < 3 for row in review.get('findings', [])):
            notes, failures = [dict(row) for row in review.get('findings', [])], []
    result = {'verdict': 'ready' if not failures else 'discarded', 'spec': spec,
              'task': str(task) if task else None, 'gates': gates, 'review': review,
              'revisions': revision, 'failures': failures, 'attempts': history,
              'admitted_with_review_notes': notes}
    _write(root / 'design_loop.json', result)
    return result


def production_prepare(intent, candidate_root, *, family=None, seed=7, fake=False):
    """Live wiring, called only by an explicitly non-dry scheduler run."""
    if fake:
        raise ValueError('Dry runs must inject deterministic designer and reviewer callbacks')
    from .gates import run_single
    root = Path(candidate_root).resolve()
    if not root.is_relative_to(REPO / 'v5_runs'):
        raise ValueError('Design loop output must stay under v5_runs')
    root.mkdir(parents=True, exist_ok=True)
    intent_path = root / 'intent.json'
    _write(intent_path, intent)
    from .designer_sources import assigned_sources
    family = '+'.join(assigned_sources(intent, family))

    def designer(current_intent, output, previous, feedback):
        cmd = [sys.executable, str(REPO / 'run_designer.py'), family, '--v5',
               '--intent', str(intent_path), '--out', str(output), '--max-continues', '0']
        if previous is not None:
            cmd.extend(['--revise', str(previous)])
        if feedback:
            cmd.extend(['--brief', str(output.parent / 'feedback.md')])
        from .concurrency import borrowed_slot, subprocess_environment
        # Queue before starting the existing process timeout. An Opus solver can
        # legitimately hold this shared slot longer than the designer timeout.
        with borrowed_slot('claude-opus-5-5'):
            env = dict(subprocess_environment('claude-opus-5-5'), PYTHONDONTWRITEBYTECODE='1')
            with (output.parent / 'designer.log').open('w') as log:
                completed = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=5500)
        if completed.returncode:
            raise ValueError('Designer process failed')
        receipt = json.loads((output / 'receipt.json').read_text())
        if receipt.get('status') != 'design_written':
            raise ValueError('Designer did not complete four valid artifacts')
        return output

    def compile_and_gate(spec, output):
        return run_single(spec, output, seed=seed, client='real')

    def reviewer(task, gates, output):
        return _review_module().review_v5(task, output, gates=gates)

    return prepare_candidate(intent, root, designer=designer, compile_and_gate=compile_and_gate, reviewer=reviewer)
