"""Fixes from the cecd4b8 merger-review trajectory: transient provider failures, actionable reasons,
intent memo for repeated queries, and budget visibility in every observation."""
import json
import httpx
import pandas as pd
import pytest
from tada_operators import Engine
from tada_runtime.audit import AuditLog, read_events
from tada_runtime.state import Store
from tada_runtime.config import Config, BudgetConfig
from tada_runtime.budget import BudgetLedger
from tada_runtime.contracts import IntentEnvelope
from tada_runtime.providers import Provider, ProviderFailure, terminal_reason

FAILED_STREAM = [
    {'type': 'response.output_text.delta', 'delta': '{"items":'},
    {'type': 'response.failed', 'response': {'model': 'm', 'status': 'failed', 'output': [],
        'error': {'code': 'invalid_prompt', 'message': 'Invalid prompt: your prompt was flagged as potentially violating our usage policy.'}}},
]


def config(retries):
    return Config.model_validate({'models': {role: {'base_url': 'http://localhost:9999/v1', 'model': 'exact',
        'protocol': 'openai-responses', 'auth_mode': 'optional'} for role in ['agent', 'operators']},
        'runtime': {'batch_retries': retries}})


def ok_response(rids):
    text = json.dumps({'items': [{'rid': rid, 'value': 'Yes'} for rid in rids]})
    return {'model': 'm', 'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}],
            'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}}


def make_engine(tmp_path, retries, responses):
    queue = iter(responses)
    def handler(request):
        kind, payload = next(queue)
        if kind == 'stream':
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                  text=''.join('data: ' + json.dumps(e) + '\n\n' for e in payload))
        return httpx.Response(200, json=payload)
    cfg = config(retries)
    audit = AuditLog(tmp_path / 'session')
    budget = BudgetLedger(BudgetConfig(ledger_path=str(tmp_path / 'ledger.json'), max_requests=10, deadline_seconds=60.0), audit)
    provider = Provider(cfg.models.operators, budget, audit, transport=httpx.MockTransport(handler))
    return Engine(cfg, provider, Store(audit), audit), audit, budget


TEXTS = pd.Series(['a', 'b'], index=['r1', 'r2'])


def test_failed_response_reason_is_reported_and_retried(tmp_path):
    engine, audit, budget = make_engine(tmp_path, 1, [('stream', FAILED_STREAM), ('json', ok_response(['r1', 'r2']))])
    assert engine.tag(TEXTS, 'q', labels=['Yes', 'No']).tolist() == ['Yes', 'Yes']
    events, _ = read_events(audit.root)
    attempts = [e for e in events if e.type == 'model/attempt_end']
    assert [a.payload.status for a in attempts] == ['failed', 'success']
    assert 'invalid_prompt' in attempts[0].payload.error.message
    assert attempts[1].payload.retry_of == attempts[0].request_id
    assert len(budget.snapshot().reservations) == 2
    audit.close()


def test_failed_response_without_retry_surfaces_the_upstream_reason(tmp_path):
    engine, audit, _ = make_engine(tmp_path, 0, [('stream', FAILED_STREAM)])
    with pytest.raises(ProviderFailure, match=r'did not complete: failed \(invalid_prompt: Invalid prompt'):
        engine.tag(TEXTS, 'q', labels=['Yes', 'No'])
    audit.close()


def test_incomplete_response_is_retried(tmp_path):
    incomplete = dict(ok_response(['r1', 'r2']), status='incomplete', incomplete_details={'reason': 'max_output_tokens'})
    engine, audit, _ = make_engine(tmp_path, 1, [('json', incomplete), ('json', ok_response(['r1', 'r2']))])
    assert engine.tag(TEXTS, 'q', labels=['Yes', 'No']).tolist() == ['Yes', 'Yes']
    audit.close()


def test_http_errors_retry_only_when_transient(tmp_path):
    assert ProviderFailure('x', retryable=True).retryable and not ProviderFailure('x').retryable
    assert terminal_reason({'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}}) == 'incomplete reason: max_output_tokens'
    assert terminal_reason([{'type': 'x'}]) == ''


class CountingProvider:
    def __init__(self):
        self.roles = []
    def call(self, prompt, schema, *, role, parent_id, context=None, cancellation=None):
        self.roles.append(role)
        if schema is IntentEnvelope:
            return schema.model_validate({'mode': 'independent', 'domain_query': None})
        return schema.model_validate({'items': [{'rid': r, 'value': 'v'} for r in context['rids']]}, context=context)


def test_identical_query_is_classified_once_per_session(tmp_path):
    cfg = config(0)
    cfg.runtime.reuse = 'off'
    audit = AuditLog(tmp_path / 's')
    provider = CountingProvider()
    engine = Engine(cfg, provider, Store(audit), audit)
    for rid in ['d1', 'd2', 'd3']:
        engine.tag(pd.Series(['text'], index=[rid]), 'Code DP114 for this agreement')
    assert provider.roles.count('intent') == 1 and provider.roles.count('tag-batch') == 3
    engine.tag(pd.Series(['text'], index=['d4']), 'Code DP114 for this agreement', fresh=True)
    engine.tag(pd.Series(['text'], index=['d5']), 'A different question')
    assert provider.roles.count('intent') == 3
    events, _ = read_events(audit.root)
    states = [e for e in events if e.type == 'state/commit']
    assert states
    audit.close()


def test_observation_reports_budget(tmp_path):
    from tada_runtime.supervisor import Supervisor
    supervisor = Supervisor.__new__(Supervisor)
    audit = AuditLog(tmp_path / 'b')
    supervisor.budget = BudgetLedger(BudgetConfig(ledger_path=str(tmp_path / 'l.json'), max_requests=7, deadline_seconds=100.0), audit)
    supervisor.budget.reserve('agent', 'm', 'req-1')
    status = supervisor.budget_status()
    assert status['requests_used'] == 1 and status['requests_remaining'] == 6 and status['requests_limit'] == 7
    assert 0 < status['seconds_remaining'] <= 100
    from tada_runtime.rpc_contracts import Observation
    assert 'budget' in Observation.model_fields
    audit.close()
