"""Allowlisted structural compile diagnostics; private error details stay private."""
from __future__ import annotations

import re


class CompilationFailure(ValueError):
    def __init__(self, message, details=None, *, code=None, context=None):
        super().__init__(message)
        self.details = details or {}
        self.code = code
        self.context = context or {}


REASONS = {
    'carrier_roundtrip_failed': 'A required fact carrier failed blind round-trip verification. Revise the carrier declaration or facts so every named field can be recovered exactly.',
    'class_guard_failed': 'Questions failed the fixed sampled-instance class guard. Each retained question requires at least two present classes and at least two cells in each present class.',
    'too_few_retained_questions': 'Too few questions remain after the fixed class guard. Revise the question selection without changing the fixed seed or sampling rules.',
    'insufficient_scorable_records': 'The source pool has too few records scorable for the requested sample. Revise the question selection or requested sample size.',
}


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.:-]{0,79}', value) is not None


def safe_diagnostic(value, spec=None):
    """Reconstruct known reasons; never copy arbitrary prose, values or record IDs.

    Document/field/question identifiers must already occur in the designer's
    specification. Auto-generated scope carrier keys are per-record facts and
    are deliberately excluded. Counts are aggregates, never class distributions.
    """
    if not isinstance(value, dict) or not isinstance(value.get('code'), str) or value['code'] not in REASONS:
        return None
    code = value['code']
    context = value.get('context', value)
    if not isinstance(context, dict):
        context = {}
    spec = spec if isinstance(spec, dict) else {}
    result = {'code': code, 'reason': REASONS[code]}
    if code == 'carrier_roundtrip_failed':
        docs = {d['id']: d for d in spec.get('documents', [])
                if isinstance(d, dict) and _identifier(d.get('id'))}
        name = context.get('document_id')
        if _identifier(name) and name in {*docs, 'scope_revision', 'scope_prior'}:
            result['document_id'] = name
            declared = docs.get(name, {}).get('facts', {})
            declared = declared if isinstance(declared, dict) else {}
            keys = context.get('fact_keys', [])
            result['fact_keys'] = sorted({key for key in keys if _identifier(key) and key in declared}) if isinstance(keys, list) else []
    elif code in ('class_guard_failed', 'too_few_retained_questions'):
        declared = {q for q in spec.get('questions', []) if _identifier(q)}
        questions = context.get('question_ids', [])
        result['question_ids'] = sorted({q for q in questions if _identifier(q) and q in declared}) if isinstance(questions, list) else []
        for key in ('retained_count', 'minimum_required'):
            number = context.get(key)
            if type(number) is int and 0 <= number <= 1000000:
                result[key] = number
    elif code == 'insufficient_scorable_records':
        for key in ('available_count', 'required_count', 'minimum_scorable_questions'):
            number = context.get(key)
            if type(number) is int and 0 <= number <= 1000000:
                result[key] = number
    return result


def from_exception(error, spec):
    if not isinstance(error, CompilationFailure):
        return None
    return safe_diagnostic({'code': error.code, 'context': error.context}, spec)
