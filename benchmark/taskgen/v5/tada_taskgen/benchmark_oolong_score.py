"""Dependency-free deterministic Oolong/SQE verification.

Oolong: abertsch72/oolong, src/eval/eval_helpers.py (2026-11-12).
SQE: experiment_runner/evaluation.py, exact_match_accuracy / f1_score.
JSON transport unwraps original answers before the upstream answer comparison;
response decoration is accepted but never required for an already typed answer.
"""
import ast
from collections import Counter
from datetime import datetime
import json
import re


def _synth_gold(value):
    if not isinstance(value, str):
        return value
    if 'datetime.date(' in value:
        return datetime.strptime(value, '[datetime.date(%Y, %m, %d)]')
    parsed = ast.literal_eval(value)
    return parsed[0] if isinstance(parsed, list) else parsed


def _synth_response(value):
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    if not isinstance(value, str):
        return value
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list) and len(parsed) == 1:
            return parsed[0]
    except (ValueError, SyntaxError):
        pass
    if ':' not in value:
        return value if len(value) < 20 else value.split()[-1]
    answer = value.split(':')[-1].strip().replace('*', '').replace('[', '').replace(']', '')
    if len(answer) >= 20:
        for phrase in ('more common', 'less common', 'same frequency'):
            if phrase in answer:
                return phrase
    return answer


def _dnd(value):
    if not isinstance(value, str):
        return value
    match = re.search(r'\\boxed\{\\text\{([^}]*)\}\}', value) or re.search(r'\\boxed[\{]+([^}]*)[\}]+', value)
    value = match.group(1) if match else value
    try:
        return int(value)
    except ValueError:
        return [part.strip() for part in value.split(',') if part.strip()] if ',' in value else value


def score_oolong(predicted, expected, spec):
    kind = spec['kind']
    if kind == 'oolong_synth':
        if predicted == expected:
            return 1.0
        gold, answer = _synth_gold(expected), _synth_response(predicted)
        if str(answer) == str(gold):
            return 1.0
        if str(answer) in ('more common', 'less common', 'same frequency'):
            return float(str(answer) in str(gold))
        if spec['answer_type'] == 'ANSWER_TYPE.NUMERIC':
            return 0.75 ** abs(int(gold) - int(answer))
        if spec['answer_type'] == 'ANSWER_TYPE.DATE':
            from dateutil.parser import parse
            try:
                candidate = _synth_gold(answer) if 'datetime.date(' in str(answer) else parse(str(answer))
                return float(candidate == gold)
            except (ValueError, TypeError):
                return 0.0
        return 0.0
    if kind == 'oolong_real':
        gold, answer = _dnd(expected), _dnd(predicted)
        if type(gold) is int and type(answer) is int:
            return 0.75 ** abs(gold - answer)
        if isinstance(gold, str) and isinstance(answer, str):
            return float(gold.strip().lower() == answer.strip().lower())
        if isinstance(gold, list) and isinstance(answer, list):
            return len(set(gold) & set(answer)) / len(gold) if gold else 0.0
        return 0.0
    if kind == 'sqe_derivation':
        if not isinstance(predicted, list) or len(predicted) != len(expected):
            return 0.0
        return sum(str(p) == str(e) for p, e in zip(predicted, expected)) / len(expected) if expected else 0.0
    if kind == 'sqe_set_f1':
        encode = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
        p, e = Counter(map(encode, predicted)), Counter(map(encode, expected))
        # Match pandas merge multiplicity from the official evaluators.
        tp = sum(count*p[key] for key,count in e.items())
        fn = sum(count for key,count in e.items() if not p[key])
        if spec.get('mode') == 'selection':
            negatives = Counter(map(encode, spec['negative_rows']))
            fp = sum(count*p[key] for key,count in negatives.items())
        else:
            fp = sum(count for key,count in p.items() if not e[key])
        return 2*tp/(2*tp+fp+fn) if tp else 0.0
    raise ValueError('Unsupported Oolong/SQE checker')
