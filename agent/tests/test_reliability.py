"""Operator reliability: transport retries, fast failure, batch cache, deadline slicing, truncation split,
budget preflight/dry-run, locate mode, row context, intent skipping, images and deliverable checks."""
import json
import re
import threading
import time

import httpx
import pandas as pd
import pytest
from pydantic import ValidationError

from tada_operators import Engine
from tada_operators.engine import OperatorIncomplete, PreflightRefused
from tada_runtime.audit import AuditLog, read_events
from tada_runtime.budget import BudgetLedger, BudgetStatus
from tada_runtime.cancellation import ExecutionCancellation, ExecutionCancelled, LinkedCancellation
from tada_runtime.config import BudgetConfig, Config
from tada_runtime.contracts import IntentEnvelope, OperatorIdentity, TextRow, TextSnapshot
from tada_runtime.kernel_client import check_deliverable
from tada_runtime.providers import (OutputTruncated, Provider, ProviderFailure, RequestImage, TransientProviderFailure,
                                    prepare_request)
from tada_runtime.state import Store

LABELS = ['Yes', 'No']
PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 32


def config(runtime=None):
    return Config.model_validate({'models': {role: {'base_url': 'http://localhost:9999/v1', 'model': 'exact',
        'protocol': 'openai-responses', 'auth_mode': 'optional'} for role in ['agent', 'operators']},
        'runtime': {'batch_rows': 1, 'transport_backoff_seconds': 0.0, **(runtime or {})}})


class Scripted:
    """Answers every tag batch with 'Yes'; `fail` maps a 1-based call number to an exception to raise."""
    def __init__(self, fail=None, intent='independent'):
        self.fail, self.intent = dict(fail or {}), intent
        self.calls, self.prompts, self.images = [], [], []
        self.lock = threading.Lock()

    def call(self, prompt, schema, *, role, parent_id, context=None, retry_of=None, images=None):
        with self.lock:
            self.calls.append(role)
            self.prompts.append(prompt)
            self.images.append(images)
            number = len(self.calls)
        if number in self.fail:
            raise self.fail[number]
        if schema is IntentEnvelope:
            return schema.model_validate({'mode': self.intent, 'domain_query': None})
        return schema.model_validate({'items': [{'rid': rid, 'value': 'Yes'} for rid in context['rids']]}, context=context)


@pytest.fixture
def make(tmp_path):
    audits = []
    def build(provider, runtime=None):
        audit = AuditLog(tmp_path / f'session{len(audits)}')
        audits.append(audit)
        return Engine(config(runtime), provider, Store(audit), audit), audit
    yield build
    for audit in audits:
        audit.close()


TEXTS = pd.Series(['a', 'b', 'c', 'd'], index=['r1', 'r2', 'r3', 'r4'])


def test_transient_failure_is_retried_with_audit_link(make):
    provider = Scripted(fail={1: TransientProviderFailure('provider HTTP 500')})
    engine, audit = make(provider)
    assert engine.tag(TEXTS[:1], 'q', labels=LABELS).tolist() == ['Yes']
    assert provider.calls == ['tag-batch', 'tag-batch-transport-retry']


def test_transient_retries_are_bounded(make):
    provider = Scripted(fail={n: TransientProviderFailure('500') for n in range(1, 10)})
    engine, audit = make(provider, {'transport_retries': 1})
    with pytest.raises(TransientProviderFailure):
        engine.tag(TEXTS[:1], 'q', labels=LABELS)
    assert provider.calls == ['tag-batch', 'tag-batch-transport-retry']


def test_first_failure_stops_remaining_batches(make):
    provider = Scripted(fail={2: ProviderFailure('provider HTTP 400')})
    engine, audit = make(provider)
    with pytest.raises(ProviderFailure):
        engine.tag(TEXTS, 'q', labels=LABELS)
    assert len(provider.calls) == 2  # batches 3 and 4 were never dispatched


def test_repeated_call_reuses_cached_batches_after_failure(make):
    provider = Scripted(fail={3: ProviderFailure('provider HTTP 400')})
    engine, audit = make(provider)
    with pytest.raises(ProviderFailure):
        engine.tag(TEXTS, 'q', labels=LABELS)
    assert len(provider.calls) == 3
    assert engine.tag(TEXTS, 'q', labels=LABELS).tolist() == ['Yes'] * 4
    # Only the failed and the never-started batches are requested again.
    assert len(provider.calls) == 5
    events, _ = read_events(audit.root)
    assert sum(e.type == 'atom/batch_commit' for e in events) == 4
    plans = [e.payload.preflight for e in events if e.type == 'atom/plan']
    assert plans[-1]['cached_batches'] == 2 and plans[-1]['planned_requests'] == 2


def test_batch_cache_survives_resume(tmp_path):
    provider = Scripted(fail={3: ProviderFailure('provider HTTP 400')})
    audit = AuditLog(tmp_path / 'session')
    engine = Engine(config(), provider, Store(audit), audit)
    with pytest.raises(ProviderFailure):
        engine.tag(TEXTS, 'q', labels=LABELS)
    audit.close()
    resumed = AuditLog(tmp_path / 'session', resume=True)
    engine = Engine(config(), provider, Store(resumed), resumed)
    assert engine.tag(TEXTS, 'q', labels=LABELS).tolist() == ['Yes'] * 4
    assert len(provider.calls) == 5
    resumed.close()


def test_fresh_and_disabled_cache_do_not_reuse(make):
    provider = Scripted(fail={2: ProviderFailure('x')})
    engine, audit = make(provider, {'batch_cache': False})
    with pytest.raises(ProviderFailure):
        engine.tag(TEXTS, 'q', labels=LABELS)
    engine.tag(TEXTS, 'q', labels=LABELS)
    assert len(provider.calls) == 2 + 4


def test_deadline_stops_dispatch_and_repeat_continues(make):
    provider = Scripted()
    engine, audit = make(provider)
    with pytest.raises(OperatorIncomplete, match=r'0/4 batches finished'):
        engine.tag(TEXTS, 'q', labels=LABELS, deadline=time.monotonic() - 1)
    assert provider.calls == []
    original = engine._call_batch
    def slow(*args, **kwargs):
        result = original(*args, **kwargs)
        time.sleep(0.3)
        return result
    engine._call_batch = slow
    with pytest.raises(OperatorIncomplete, match=r'/4 batches finished and cached'):
        engine.tag(TEXTS, 'q', labels=LABELS, deadline=time.monotonic() + 0.45)
    done = len(provider.calls)
    assert 1 <= done < 4
    engine._call_batch = original
    assert engine.tag(TEXTS, 'q', labels=LABELS).tolist() == ['Yes'] * 4
    assert len(provider.calls) == 4
    events, _ = read_events(audit.root)
    assert any(e.type == 'atom/end' and e.payload.status == 'incomplete' for e in events)


def test_truncated_batch_is_split(make):
    class Truncating(Scripted):
        def call(self, prompt, schema, *, role, parent_id, context=None, retry_of=None, images=None):
            if context and len(context.get('rids', [])) > 1:
                self.calls.append(role)
                raise OutputTruncated('provider did not complete: incomplete (reason: max_output_tokens)')
            return super().call(prompt, schema, role=role, parent_id=parent_id, context=context)
    provider = Truncating()
    engine, audit = make(provider, {'batch_rows': 4})
    assert engine.tag(TEXTS, 'q', labels=LABELS).tolist() == ['Yes'] * 4
    assert len(provider.calls) == 1 + 2 + 4  # 4 rows -> 2+2 -> 1+1+1+1


class Budget:
    def __init__(self, left):
        self.left = left
    def remaining(self):
        return BudgetStatus(used=10 - self.left, limit=10, left=self.left, seconds_to_deadline=100.0)


def test_budget_preflight_refuses_before_any_request(make):
    provider = Scripted()
    provider.budget = Budget(left=3)
    engine, audit = make(provider)
    with pytest.raises(PreflightRefused, match='needs about 4 more requests'):
        engine.tag(TEXTS, 'q', labels=LABELS)
    assert provider.calls == []
    provider.budget.left = 4
    assert engine.tag(TEXTS, 'q', labels=LABELS).tolist() == ['Yes'] * 4


def test_dry_run_estimates_without_requests(make):
    provider = Scripted()
    provider.budget = Budget(left=2)
    engine, audit = make(provider)
    estimate = engine.tag(TEXTS, 'q', labels=LABELS, dry_run=True)
    assert estimate['planned_requests'] == 4 and estimate['fits_budget'] is False and provider.calls == []
    estimate = engine.tag(TEXTS, 'q', dry_run=True)
    assert estimate['intent_request'] == 1 and estimate['planned_requests'] == 5
    events, _ = read_events(audit.root)
    assert not any(e.type.startswith('atom/') for e in events)


def long_document():
    paragraphs = [f'Paragraph {i}: boilerplate text about nothing in particular. ' * 20 for i in range(40)]
    paragraphs[25] = 'The Intervening Event means any material event unknown at signing. ' * 5
    return '\n\n'.join(paragraphs)


def test_locate_reads_only_matching_segments(make):
    class SegmentProvider(Scripted):
        def call(self, prompt, schema, *, role, parent_id, context=None, retry_of=None, images=None):
            if role == 'tag-segment-batch':
                self.calls.append(role)
                self.prompts.append(prompt)
                return schema.model_validate({'items': [{'rid': rid, 'value': '[]'} for rid in context['rids']]}, context=context)
            return super().call(prompt, schema, role=role, parent_id=parent_id, context=context)
    text = long_document()
    full = SegmentProvider()
    engine, _ = make(full, {'batch_rows': 32, 'batch_chars': 16000})
    engine.tag(pd.Series([text], index=['DOC']), 'q', labels=LABELS, multi=True)
    located = SegmentProvider()
    engine, audit = make(located, {'batch_rows': 1, 'batch_chars': 16000})
    engine.tag(pd.Series([text], index=['DOC']), 'q', labels=LABELS, multi=True,
               long_rows={'locate': [r'intervening\s+event'], 'locate_window': 0})
    assert 1 <= located.calls.count('tag-segment-batch') < 4 <= sum(json.loads(p.split('\nRows: ', 1)[1]).__len__()
                                                                    for p in full.prompts if 'SEGMENT' in p)
    assert all('Intervening Event' in json.loads(p.split('\nRows: ', 1)[1])[0]['text'] for p in located.prompts)
    events, _ = read_events(audit.root)
    plan = [e for e in events if e.type == 'atom/plan'][-1]
    assert plan.payload.long_rows['located_segments_per_row'][0][1] < plan.payload.long_rows['located_segments_per_row'][0][2]


def test_locate_miss_policy(make):
    provider = Scripted()
    engine, audit = make(provider, {'batch_rows': 32})
    result = engine.tag(pd.Series([long_document()], index=['DOC']), 'q', labels=LABELS, multi=True,
                        long_rows={'locate': ['no such phrase'], 'locate_on_miss': 'none'})
    assert result.tolist() == ['[]'] and provider.calls == []


def test_invalid_locate_pattern_is_refused(make):
    engine, audit = make(Scripted())
    with pytest.raises(ValidationError, match='invalid locate pattern'):
        engine.tag(pd.Series([long_document()], index=['DOC']), 'q', labels=LABELS, multi=True, long_rows={'locate': ['(']})


def test_row_context_reaches_every_row_and_skips_intent(make):
    provider = Scripted()
    engine, audit = make(provider, {'batch_rows': 4})
    contexts = pd.Series(['compare A vs B', 'compare C vs D'], index=['r1', 'r2'])
    engine.tag(TEXTS[:2], 'extract the result for this row', row_context=contexts)
    assert provider.calls == ['tag-batch']
    rows = json.loads(provider.prompts[0].split('\nRows: ', 1)[1])
    assert [row['context'] for row in rows] == ['compare A vs B', 'compare C vs D']
    assert 'context' in provider.prompts[0].split('\nRows: ', 1)[0]


def test_unset_extensions_keep_identity_and_prompt_bytes():
    row = TextRow(rid='r1', text='a')
    assert row.model_dump() == {'rid': 'r1', 'text': 'a'}
    snapshot = TextSnapshot(rows=[row])
    assert '"context"' not in snapshot.model_dump_json() and '"image"' not in snapshot.model_dump_json()
    identity = OperatorIdentity.model_json_schema()
    assert 'intent_mode' in identity['properties']


def test_independent_mode_skips_intent_and_intent_is_cached(make):
    provider = Scripted()
    engine, audit = make(provider, {'batch_rows': 4})
    engine.tag(TEXTS, 'summarize', mode='independent')
    assert provider.calls == ['tag-batch']
    engine.tag(TEXTS[:2], 'classify tone')
    engine.tag(TEXTS[2:], 'classify tone')
    assert provider.calls.count('intent') == 1


def test_images_are_attached_per_batch(make, tmp_path):
    path = tmp_path / 'item.png'
    path.write_bytes(PNG)
    provider = Scripted()
    engine, audit = make(provider, {'batch_rows': 4, 'max_images_per_batch': 1})
    engine.tag(TEXTS[:2], 'what is shown', labels=LABELS, images=pd.Series([str(path), str(path)], index=['r1', 'r2']))
    assert len(provider.calls) == 2  # one image per batch
    assert all(len(images) == 1 and images[0].media_type == 'image/png' for images in provider.images)
    assert 'IMAGE for rid=' in provider.images[0][0].label


def test_prepare_request_with_images_uses_native_parts():
    cfg = config().models.operators
    image = RequestImage(label='IMAGE for rid=1', media_type='image/png', data_base64='AAAA')
    body, _ = prepare_request(cfg, 'hello', IntentEnvelope, [image])
    parts = body['input'][0]['content']
    assert [p['type'] for p in parts] == ['input_text', 'input_text', 'input_image']
    assert parts[2]['image_url'] == 'data:image/png;base64,AAAA'
    plain, _ = prepare_request(cfg, 'hello', IntentEnvelope)
    assert isinstance(plain['input'], str)


def provider_with(tmp_path, handler):
    audit = AuditLog(tmp_path / 'p')
    budget = BudgetLedger(BudgetConfig(ledger_path=str(tmp_path / 'ledger.json'), max_requests=10, deadline_seconds=60.0), audit)
    return Provider(config().models.operators, budget, audit, transport=httpx.MockTransport(handler), concurrency=4), audit, budget


@pytest.mark.parametrize('status,error', [(500, TransientProviderFailure), (429, TransientProviderFailure), (400, ProviderFailure)])
def test_provider_classifies_http_failures(tmp_path, status, error):
    provider, audit, _ = provider_with(tmp_path, lambda request: httpx.Response(status, json={'error': 'x'}))
    with pytest.raises(error) as raised:
        provider.call('q', IntentEnvelope, role='intent')
    assert (type(raised.value) is TransientProviderFailure) == (error is TransientProviderFailure)
    audit.close()


def test_provider_reports_truncation_reason(tmp_path):
    payload = {'model': 'exact', 'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}, 'output': []}
    provider, audit, _ = provider_with(tmp_path, lambda request: httpx.Response(200, json=payload))
    with pytest.raises(OutputTruncated, match='max_output_tokens'):
        provider.call('q', IntentEnvelope, role='intent')
    audit.close()


def test_provider_transport_error_is_transient(tmp_path):
    def handler(request):
        raise httpx.ConnectError('refused')
    provider, audit, budget = provider_with(tmp_path, handler)
    with pytest.raises(TransientProviderFailure):
        provider.call('q', IntentEnvelope, role='intent')
    assert budget.snapshot().reservations[0].status == 'unknown'
    audit.close()


def test_budget_remaining_is_read_only(tmp_path):
    budget = BudgetLedger(BudgetConfig(ledger_path=str(tmp_path / 'ledger.json'), max_requests=5, deadline_seconds=60.0))
    budget.reserve('agent', 'm', 'r1')
    status = budget.remaining()
    assert (status.used, status.limit, status.left) == (1, 5, 4) and status.seconds_to_deadline > 0


def test_linked_cancellation_follows_parent_and_explicit_cancel():
    parent = ExecutionCancellation()
    child = LinkedCancellation(parent)
    child.check()
    parent.set()
    with pytest.raises(ExecutionCancelled):
        child.check()
    lone = LinkedCancellation(None)
    lone.set()
    with pytest.raises(ExecutionCancelled):
        lone.admit_dispatch()


def test_check_deliverable_flags_contract_problems(tmp_path):
    path = tmp_path / 'review.xlsx'
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({'Record_ID': ['a', 'b', 'c', 'd'],
                      'Answer': ['41, 68.33%', 'x' * 2100, '41 seats, representing 68.33% of the total seats in the assembly, which is a clear majority of the chamber overall today and more', 'nan']}
                     ).to_excel(writer, sheet_name='Coding', index=False)
    report = check_deliverable(path, {'max_cell_chars': 2000, 'short_answer_max_words': 20})
    assert report['ok'] is False
    text = '\n'.join(report['issues'])
    assert '2100 characters' in text and 'words; give the canonical short answer' in text and 'non-finite' in text
    assert not any('row 2' in issue for issue in report['issues'])


def test_concurrency_limit_accepts_more_workers():
    cfg = config({'operator_concurrency': 4})
    assert cfg.runtime.operator_concurrency == 4
    with pytest.raises(ValidationError):
        config({'operator_concurrency': 9})
