"""Native decision backend for closed-label Tagging: Jev through TypeSafe SystemOne.

One HTTP attempt per call, the same budget ledger and audit events as generative operator
requests, and strict validation of the typed answers: exact question keys, choices inside
the supplied criteria, finite probabilities. Credentials come only from the configured
environment variable and never enter request blobs.
"""
from __future__ import annotations
import json
import math
import threading
import time
import uuid
import httpx
from pydantic import Field, JsonValue
from .contracts import StrictModel, JSON, TextBlock, Usage, ErrorInfo
from .cancellation import ExecutionCancelled, cancellation_token
from .providers import ProviderFailure, _CancellationTransport

MAX_CHOICE_OPTIONS = 255


class DecisionContractError(ProviderFailure):
    """The service answered, but not with exactly the requested typed answers (retryable)."""
    def __init__(self, message):
        super().__init__(message, retryable=True)


class DecisionAnswer(StrictModel):
    type: str
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    noul: float | None = None
    confidence: float | None = None


def _finite_unit(value, what):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise DecisionContractError(f'decision response contract: {what} must be a finite number in [0, 1]')
    return float(value)


def validate_questions(questions: dict) -> None:
    if not isinstance(questions, dict) or not questions:
        raise ValueError('a decision request needs at least one question')
    for key, question in questions.items():
        if not isinstance(key, str) or not key:
            raise ValueError('decision question keys must be nonempty strings')
        kind = question.get('type')
        if kind == 'choice':
            criteria = question.get('criteria')
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
                raise ValueError(f'a Choice question needs 2..{MAX_CHOICE_OPTIONS} options')
        elif kind != 'noul':
            raise ValueError('only choice and noul decision questions are supported')


def parse_answers(payload, questions: dict, model: str) -> dict[str, DecisionAnswer]:
    if not isinstance(payload, dict):
        raise DecisionContractError('decision response contract: response is not a JSON object')
    if payload.get('model') != model:
        # Identity drift is not retried: a different served model is not the configured backend.
        raise ProviderFailure(f'decision backend served an unexpected model identity: {payload.get("model")!r}')
    raw = payload.get('answers')
    if not isinstance(raw, dict) or set(raw) != set(questions):
        raise DecisionContractError('decision response contract: answers do not match the requested question keys')
    answers = {}
    for key, question in questions.items():
        answer = raw[key]
        if not isinstance(answer, dict) or answer.get('type') != question['type']:
            raise DecisionContractError('decision response contract: answer type differs from the question type')
        if question['type'] == 'choice':
            options = set(question['criteria'])
            choice = answer.get('choice')
            if not isinstance(choice, str) or choice not in options:
                raise DecisionContractError('decision response contract: choice outside the supplied options')
            probabilities = answer.get('probabilities')
            if probabilities is not None:
                if not isinstance(probabilities, dict) or not set(probabilities) <= options:
                    raise DecisionContractError('decision response contract: probabilities reference unknown options')
                probabilities = {name: _finite_unit(value, 'probability') for name, value in probabilities.items()}
            confidence = _finite_unit(answer['confidence'], 'confidence') if answer.get('confidence') is not None else None
            answers[key] = DecisionAnswer(type='choice', choice=choice, probabilities=probabilities, confidence=confidence)
        else:
            answers[key] = DecisionAnswer(type='noul', noul=_finite_unit(answer.get('noul'), 'noul'))
    return answers


class DecisionClient:
    def __init__(self, config, budget, collector, *, transport=None, concurrency: int = 1):
        self.config, self.budget, self.collector, self.transport = config, budget, collector, transport
        self._slots = threading.BoundedSemaphore(max(1, min(2, concurrency)))
        self._local = threading.local()

    @property
    def last_request_id(self) -> str | None:
        return getattr(self._local, 'request_id', None)

    def decide(self, state: JsonValue, questions: dict, *, role: str, parent_id: str | None = None,
               cancellation=None, retry_of: str | None = None) -> dict[str, DecisionAnswer]:
        cancellation = cancellation_token(cancellation)
        validate_questions(questions)
        body = {'model': self.config.model, 'state': state, 'questions': questions}
        body = JSON.validate_python(body)
        size = len(json.dumps(body, ensure_ascii=False))
        if size > self.config.max_request_chars:
            raise ValueError(f'decision request of {size} characters exceeds decision.max_request_chars; no request was sent')
        while True:
            if cancellation:
                cancellation.check()
            if self._slots.acquire(timeout=.05):
                break
        try:
            return self._decide(body, questions, role=role, parent_id=parent_id, cancellation=cancellation, retry_of=retry_of)
        finally:
            self._slots.release()

    def _decide(self, body, questions, *, role, parent_id, cancellation, retry_of):
        cfg = self.config
        rid = uuid.uuid4().hex
        self._local.request_id = rid
        ids = {'request_id': rid, 'attempt_id': rid + ':1', 'parent_id': parent_id, 'source': 'operator'}
        request_ref = self.collector.blob(body)
        self.collector.emit('model/request', {'role': role, 'requested_model': cfg.model, 'protocol': cfg.protocol,
            'request_ref': request_ref.model_dump(), 'parameters': cfg.model_dump(mode='json', exclude={'api_key_env', 'auth_placeholder'})}, **ids)
        started, status, reserved, raw_ref, usage = time.monotonic(), 'failed', False, None, Usage()
        try:
            if cancellation:
                cancellation.check()
            key = cfg.api_key()
            self.budget.reserve(role, cfg.model, rid, output_cap=0)
            reserved, status = True, 'unknown'
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
            if key:
                headers['Authorization'] = 'Bearer ' + key
            transport = self.transport
            if cancellation:
                transport = _CancellationTransport(transport if transport is not None else httpx.HTTPTransport(), cancellation)
            with httpx.Client(timeout=cfg.timeout_seconds, transport=transport, trust_env=False, follow_redirects=False) as client:
                response = client.post(cfg.base_url.rstrip('/') + '/systemone', json=body, headers=headers)
            raw_bytes = response.content
            raw_ref = self.collector.blob_bytes(raw_bytes, media_type='application/json')
            if response.is_error:
                status = 'failed'
                code = response.status_code
                raise ProviderFailure(f'decision HTTP {code}; see raw response artifact', retryable=code in (408, 429) or code >= 500)
            status = 'failed'
            try:
                payload = JSON.validate_json(raw_bytes)
            except Exception:
                raise DecisionContractError('decision response contract: body is not valid JSON') from None
            raw_usage = payload.get('usage') if isinstance(payload, dict) else None
            if isinstance(raw_usage, dict):
                usage = Usage(input_tokens=raw_usage['input_tokens'] if type(raw_usage.get('input_tokens')) is int and raw_usage['input_tokens'] >= 0 else None,
                              output_tokens=raw_usage['output_tokens'] if type(raw_usage.get('output_tokens')) is int and raw_usage['output_tokens'] >= 0 else None)
            answers = parse_answers(payload, questions, cfg.model)
            self.collector.emit('model/response', {'requested_model': cfg.model, 'reported_model': payload.get('model'),
                'protocol': cfg.protocol, 'content': [TextBlock(text=json.dumps({k: v.model_dump(exclude_none=True) for k, v in answers.items()},
                ensure_ascii=False)).model_dump()], 'response_ref': raw_ref.model_dump(), 'usage': usage.model_dump(),
                'finish_reason': 'completed'}, **ids)
            status = 'success'
            self.collector.emit('model/attempt_end', {'status': status, 'elapsed_seconds': time.monotonic() - started, 'retry_of': retry_of}, **ids)
            return answers
        except Exception as exc:
            if isinstance(exc, ExecutionCancelled):
                status = 'cancelled'
            attempt_status = 'schema_error' if isinstance(exc, DecisionContractError) else status
            error = ErrorInfo(kind=type(exc).__name__, message=str(exc), raw_ref=raw_ref)
            self.collector.emit('model/attempt_end', {'status': attempt_status, 'error': error.model_dump(),
                'elapsed_seconds': time.monotonic() - started, 'retry_of': retry_of}, **ids)
            raise
        finally:
            if reserved:
                self.budget.settle(rid, 'success' if status == 'success' else ('cancelled' if status == 'cancelled' else ('unknown' if status == 'unknown' else 'failed')), usage)
