"""runtime.batch_retries: bounded, audited retries of a tag batch after an output-contract failure."""
import json
import httpx
import pandas as pd
import pytest
from pydantic import ValidationError
from tada_operators import Engine
from tada_runtime.audit import AuditLog, read_events
from tada_runtime.state import Store
from tada_runtime.config import Config, BudgetConfig
from tada_runtime.budget import BudgetLedger
from tada_runtime.contracts import IntentEnvelope
from tada_runtime.providers import Provider, ProviderFailure

LABELS = ['Yes', 'No']


class FlakyProvider:
    """Returns an extra rid for the first `bad` tag-batch requests, then a valid answer."""
    def __init__(self, bad=1, failure=None):
        self.bad, self.failure = bad, failure
        self.roles, self.prompts = [], []

    def call(self, prompt, schema, *, role, parent_id, context=None, cancellation=None):
        self.roles.append(role)
        self.prompts.append(prompt)
        if schema is IntentEnvelope:
            return schema.model_validate({'mode': 'independent', 'domain_query': None})
        if self.failure:
            raise self.failure
        rids = list(context['rids'])
        if self.bad > 0:
            self.bad -= 1
            rids = rids + [str(rids[-1]) + '-neighbour']
        return schema.model_validate({'items': [{'rid': rid, 'value': 'Yes'} for rid in rids]}, context=context)


def make(tmp_path, retries, provider):
    cfg = Config.model_validate({'models': {role: {'base_url': 'http://localhost:9999/v1', 'model': 'exact',
        'protocol': 'openai-responses', 'auth_mode': 'optional'} for role in ['agent', 'operators']},
        'runtime': {'batch_retries': retries}})
    audit = AuditLog(tmp_path / f'session{retries}{id(provider)}')
    return Engine(cfg, provider, Store(audit), audit), audit


TEXTS = pd.Series(['a', 'b'], index=['r1', 'r2'])


def test_default_is_one_retry():
    assert Config.model_validate({'models': {role: {'base_url': 'http://localhost:9999/v1', 'model': 'exact',
        'protocol': 'openai-responses', 'auth_mode': 'optional'} for role in ['agent', 'operators']}}).runtime.batch_retries == 1


def test_zero_keeps_one_dispatch_per_batch(tmp_path):
    provider = FlakyProvider(bad=1)
    engine, audit = make(tmp_path, 0, provider)
    with pytest.raises(ValidationError):
        engine.tag(TEXTS, 'q', labels=LABELS)
    assert provider.roles == ['tag-batch']
    audit.close()


def test_one_retry_recovers_and_names_the_rejection(tmp_path):
    provider = FlakyProvider(bad=1)
    engine, audit = make(tmp_path, 1, provider)
    assert engine.tag(TEXTS, 'q', labels=LABELS).tolist() == ['Yes', 'Yes']
    assert provider.roles == ['tag-batch', 'tag-batch-retry']
    assert 'rejected by the output contract (Value error, output rid coverage mismatch)' in provider.prompts[1]
    audit.close()


def test_retries_are_bounded(tmp_path):
    provider = FlakyProvider(bad=5)
    engine, audit = make(tmp_path, 2, provider)
    with pytest.raises(ValidationError):
        engine.tag(TEXTS, 'q', labels=LABELS)
    assert provider.roles == ['tag-batch', 'tag-batch-retry', 'tag-batch-retry']
    audit.close()


def test_non_retryable_provider_failures_are_not_retried(tmp_path):
    provider = FlakyProvider(failure=ProviderFailure('stream disconnected without terminal response; result unknown'))
    engine, audit = make(tmp_path, 2, provider)
    with pytest.raises(ProviderFailure):
        engine.tag(TEXTS, 'q', labels=LABELS)
    assert provider.roles == ['tag-batch']
    audit.close()


def test_retryable_provider_failures_are_retried_within_the_bound(tmp_path):
    provider = FlakyProvider(failure=ProviderFailure('provider did not complete: failed', retryable=True))
    engine, audit = make(tmp_path, 2, provider)
    with pytest.raises(ProviderFailure):
        engine.tag(TEXTS, 'q', labels=LABELS)
    assert provider.roles == ['tag-batch', 'tag-batch-retry', 'tag-batch-retry']
    audit.close()


def test_retry_setting_is_part_of_reuse_identity(tmp_path):
    provider = FlakyProvider(bad=0)
    engine, audit = make(tmp_path, 0, provider)
    engine.tag(TEXTS, 'q', labels=LABELS)
    engine.tag(TEXTS, 'q', labels=LABELS)
    assert len(provider.roles) == 1
    engine.config.runtime.batch_retries = 1
    engine.tag(TEXTS, 'q', labels=LABELS)
    assert len(provider.roles) == 2
    audit.close()


def test_real_provider_retry_is_budgeted_and_linked(tmp_path):
    answers = iter([
        {'items': [{'rid': 'r1', 'value': 'Yes'}, {'rid': 'r2', 'value': 'Yes'}, {'rid': 'r3', 'value': 'Yes'}]},
        {'items': [{'rid': 'r1', 'value': 'Yes'}, {'rid': 'r2', 'value': 'No'}]},
    ])
    def handler(request):
        text = json.dumps(next(answers))
        return httpx.Response(200, json={'model': 'exact', 'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}],
            'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}})
    cfg = Config.model_validate({'models': {role: {'base_url': 'http://localhost:9999/v1', 'model': 'exact',
        'protocol': 'openai-responses', 'auth_mode': 'optional'} for role in ['agent', 'operators']},
        'runtime': {'batch_retries': 1}})
    audit = AuditLog(tmp_path / 'session')
    budget = BudgetLedger(BudgetConfig(ledger_path=str(tmp_path / 'ledger.json'), max_requests=5, deadline_seconds=60.0), audit)
    provider = Provider(cfg.models.operators, budget, audit, transport=httpx.MockTransport(handler))
    engine = Engine(cfg, provider, Store(audit), audit)
    assert engine.tag(TEXTS, 'q', labels=LABELS).tolist() == ['Yes', 'No']
    events, _ = read_events(audit.root)
    attempts = [e for e in events if e.type == 'model/attempt_end']
    assert [a.payload.status for a in attempts] == ['schema_error', 'success']
    assert attempts[1].payload.retry_of == attempts[0].request_id
    assert len(budget.snapshot().reservations) == 2
    audit.close()
