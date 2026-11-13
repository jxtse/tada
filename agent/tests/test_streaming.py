"""Secrets cannot be reconstructed from independently retained stream fragments."""
import base64
import json
import httpx
import pytest
from tada_runtime.streaming import StreamingRedactor
from tada_runtime.audit import AuditLog, read_events
from tada_runtime.providers import Provider, ProviderFailure
from tada_runtime.budget import BudgetLedger
from tada_runtime.config import ProviderConfig, BudgetConfig
from tada_runtime.contracts import TagEnvelope
from tada_runtime.supervisor import Supervisor
from tada_runtime.rpc_contracts import Stream


def test_redactor_bounded_crossed_overlapping_fragments():
    secret='sensitive-credential'
    redactor=StreamingRedactor([secret,'credential'])
    fragments=['before sens','itive-','cre','dential after']
    parts=[]
    for fragment in fragments:
        parts.extend(redactor.feed(fragment))
        assert len(redactor._text)<=len(secret)-1
    parts.extend(redactor.finish())
    joined=''.join(part.text for part in parts)
    assert secret not in joined and 'credential' not in joined
    assert joined.startswith('before ') and joined.endswith(' after')
    assert any(part.redacted for part in parts)
    for index,fragment in enumerate(fragments):
        assert {part.fragment_length for part in parts if part.fragment_index==index}=={len(fragment)}
        offsets=[part.fragment_offset for part in parts if part.fragment_index==index]
        assert offsets==sorted(offsets) and offsets[0]==0


def test_redactor_unicode_and_partial_failure_flush():
    redactor=StreamingRedactor(['秘密key'])
    parts=redactor.feed('keep 秘')+redactor.feed('密key tail')+redactor.finish()
    text=''.join(part.text for part in parts)
    assert text.startswith('keep ') and text.endswith(' tail') and '秘密key' not in text
    assert redactor.finish()==[]


def test_operator_failed_partial_stream_scrubs_reconstructable_secret(tmp_path):
    secret='fixture-sensitive-credential'
    fragments=['before fixture-sens','itive-cred','ential after']
    events=[{'type':'response.output_text.delta','delta':part} for part in fragments]
    raw=''.join('data: '+json.dumps(event)+'\n\n' for event in events)
    audit=AuditLog(tmp_path/'session',secrets=[secret])
    budget=BudgetLedger(BudgetConfig(ledger_path=str(tmp_path/'ledger.json'),max_requests=2,deadline_seconds=60.0),audit)
    provider=Provider(ProviderConfig(base_url='http://localhost:9999/v1',model='exact',protocol='openai-responses',auth_mode='optional'),budget,audit,
        transport=httpx.MockTransport(lambda request:httpx.Response(200,headers={'content-type':'text/event-stream'},text=raw)))
    with pytest.raises(ProviderFailure):provider.call('fixture',TagEnvelope,role='tag-batch')
    logged=read_events(audit.root)[0]
    text=''.join(event.payload.text for event in logged if event.type=='model/stream' and event.payload.channel=='text')
    assert text.startswith('before ') and text.endswith(' after') and secret not in text
    assert '[REDACTED]' in text
    raw_refs=[event.payload.raw_ref for event in logged if event.type=='model/stream' and event.payload.raw_ref]
    assert raw_refs and all(ref.redacted for ref in raw_refs)
    # No separately retained raw event exposes the original secret pieces.
    for ref in raw_refs:
        content=(audit.root/ref.path).read_bytes()
        assert b'fixture-sens' not in content and b'itive-cred' not in content and b'ential after' not in content
    assert budget.snapshot().reservations[0].status=='unknown'
    audit.close()


def test_supervisor_typed_stream_and_raw_transport_redaction(tmp_path):
    secret='private-test-secret'
    owner=Supervisor()
    owner.audit=AuditLog(tmp_path/'session',secrets=[secret])
    for i,text in enumerate(['visible private-','test-','secret end']):
        owner._stream_event(Stream(request_id='r1',channel='text',text=text,provider_sequence=i))
        raw={'encoding':'base64','data':base64.b64encode(('data: '+json.dumps({'delta':text})).encode()).decode(),'transport_chunk':True}
        owner._stream_event(Stream(request_id='r1',channel='provider_raw',provider_raw=raw,provider_sequence=i))
    owner._flush_streams('r1')
    events=read_events(owner.audit.root)[0]
    text=''.join(e.payload.text for e in events if e.type=='model/stream' and e.payload.channel=='text')
    assert secret not in text and text.startswith('visible ') and text.endswith(' end')
    raw_events=[e for e in events if e.type=='model/stream' and e.payload.channel=='provider_raw']
    assert all(e.payload.raw_ref.redacted for e in raw_events)
    assert not owner.stream_redactors
    owner.audit.close()


def test_supervisor_no_secret_retains_exact_wire_bytes(tmp_path):
    owner=Supervisor()
    owner.audit=AuditLog(tmp_path/'session')
    raw=b'event: message\ndata: {"delta":"hello"}\n\n'
    owner._stream_event(Stream(request_id='r1',channel='provider_raw',provider_raw={
        'encoding':'base64','data':base64.b64encode(raw).decode(),'transport_chunk':True}))
    event=read_events(owner.audit.root)[0][-1]
    assert (owner.audit.root/event.payload.raw_ref.path).read_bytes()==raw
    assert not event.payload.raw_ref.redacted
    owner.audit.close()
