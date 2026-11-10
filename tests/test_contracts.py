import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
import httpx
import pytest
from pydantic import ValidationError
from tada_runtime.contracts import (ExecuteArgs, TagEnvelope, TextSnapshot, CategoryEnvelope,
    IntentEnvelope, NormalizedResponse)
from tada_runtime.config import Config, ProviderConfig, BudgetConfig, load_config
from tada_runtime.budget import BudgetLedger, BudgetExceeded
from tada_runtime.providers import Provider, ProviderFailure
from tada_runtime.audit import AuditLog, read_events

@pytest.mark.parametrize('value', [1, True, None, [], {}, 1.2])
def test_tag_values_are_strict_strings(value):
    with pytest.raises(ValidationError):
        TagEnvelope.model_validate({'items': [{'rid': 'r1', 'value': value}]})

@pytest.mark.parametrize('items', [[], [{'rid': 'wrong', 'value':'a'}], [{'rid':'r1','value':'a'},{'rid':'r1','value':'b'}]])
def test_tag_exact_coverage(items):
    with pytest.raises(ValidationError):
        TagEnvelope.model_validate({'items':items}, context={'rids':['r1']})

@pytest.mark.parametrize('data', [{'code':'x','other':0}, {}, {'code':True}])
def test_execute_boundary(data):
    with pytest.raises(ValidationError):
        ExecuteArgs.model_validate(data)

@pytest.mark.parametrize('rows', [[{'rid':True,'text':'a'}], [{'rid':'','text':'a'}], [{'rid':'a','text':None}],
    [{'rid':'a','text':'a'}, {'rid':'a','text':'b'}], [{'rid':'a','text':'a'},{'rid':1,'text':'b'}]])
def test_input_contract(rows):
    with pytest.raises(ValidationError):
        TextSnapshot.model_validate({'rows':rows})


def test_tag_category_references_and_json_string():
    with pytest.raises(ValidationError):
        TagEnvelope.model_validate({'items':[{'rid':'r1','value':'missing'}]},context={'allowed_values':['c1']})
    result=TagEnvelope.model_validate_json('{"items":[{"rid":"r1","value":"{\\"x\\":1}"}]}')
    assert json.loads(result.items[0].value)=={'x':1}

@pytest.mark.parametrize('data', [{'categories':[{'name':'a'},{'name':'a'}]}, {'categories':[{'name':''}]},
    {'categories':[{'name':'a','cid':'invented'}]}, {'categories':[{'name':'a','description':4}]}])
def test_categories_reject_invalid(data):
    with pytest.raises(ValidationError):
        CategoryEnvelope.model_validate(data)


def test_intent_and_invalid_json():
    for data in [{'mode':'oops'}, {'mode':'joint'}, {'mode':'independent','domain_query':'bad'}]:
        with pytest.raises(ValidationError): IntentEnvelope.model_validate(data)
    with pytest.raises(ValidationError): TagEnvelope.model_validate_json('not JSON')
    with pytest.raises(ValidationError): NormalizedResponse.model_validate({'schema_version':'2.0'})


def test_budget_cumulative_concurrent(tmp_path):
    cfg=BudgetConfig(ledger_path=str(tmp_path/'ledger.json'),max_requests=3,deadline_seconds=60.0)
    ledger=BudgetLedger(cfg)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i: ledger.reserve('operator','exact',str(i)),range(3)))
    ledger.settle('0','unknown')
    ledger.settle('0','unknown')
    again=BudgetLedger(cfg)
    with pytest.raises(BudgetExceeded): again.reserve('main','exact','4')
    with pytest.raises(ValueError): again.reserve('main','exact','0')
    assert len(again.snapshot().reservations)==3
    assert again.snapshot().reservations[0].status in ['unknown','reserved']


def test_budget_deadline_and_boolean(tmp_path):
    with pytest.raises(ValidationError): BudgetConfig(ledger_path='x',max_requests=True,deadline_seconds=60.0)
    with pytest.raises(ValidationError): BudgetConfig(ledger_path='x',max_requests=1,deadline_seconds=math.inf)
    cfg=BudgetConfig(ledger_path=str(tmp_path/'ledger.json'),max_requests=2,deadline_seconds=0.001)
    ledger=BudgetLedger(cfg)
    time.sleep(.01)
    with pytest.raises(BudgetExceeded): ledger.reserve('main','x','a')


def test_provider_url_auth_and_config_precedence(tmp_path):
    with pytest.raises(ValidationError):
        ProviderConfig(base_url='https://example.org/v1',model='exact',protocol='openai-chat',auth_mode='optional')
    path=tmp_path/'config.toml'
    path.write_text('''[models.agent]\nbase_url="http://localhost:9999/v1"\nmodel="main"\nprotocol="anthropic-messages"\nauth_mode="optional"\n[models.operators]\nbase_url="http://localhost:9998/v1"\nmodel="operator"\nprotocol="openai-responses"\nauth_mode="optional"\n''')
    cfg=load_config(path,user_path=tmp_path/'absent',overrides={'models':{'agent':{'model':'override'}}})
    assert cfg.models.agent.model=='override'
    with pytest.raises(ValueError):cfg.require_budget()


def make_provider(tmp_path, handler, protocol='openai-responses', concurrency=1):
    audit=AuditLog(tmp_path/'session')
    budget=BudgetLedger(BudgetConfig(ledger_path=str(tmp_path/'ledger.json'),max_requests=10,deadline_seconds=60.0),audit)
    provider=Provider(ProviderConfig(base_url='http://localhost:9999/v1',model='exact',protocol=protocol,auth_mode='optional'),budget,audit,transport=httpx.MockTransport(handler),concurrency=concurrency)
    return provider,budget,audit


def test_responses_partial_stream_and_exact_route(tmp_path):
    def handler(request):
        assert request.url.path=='/v1/responses'
        body=json.loads(request.content)
        assert body['model']=='exact' and body['stream'] is True
        events=[{'type':'response.reasoning_summary_text.delta','delta':'summary'},
            {'type':'response.output_text.delta','delta':'{"items":[]}'},
            {'type':'response.completed','response':{'model':'reported-alias','status':'completed','output':[
                {'type':'reasoning','summary':[{'type':'summary_text','text':'summary'}]},
                {'type':'message','content':[{'type':'output_text','text':'{"items":[]}'}]}],
                'usage':{'input_tokens':2,'output_tokens':3,'total_tokens':5},'vendor_extension':'retain me'}}]
        return httpx.Response(200,headers={'content-type':'text/event-stream'},text=''.join('data: '+json.dumps(e)+'\n\n' for e in events))
    provider,budget,audit=make_provider(tmp_path,handler)
    assert provider.call('test',TagEnvelope,role='tag-batch').items==[]
    assert provider.last_response.requested_model=='exact'
    assert provider.last_response.reported_model=='reported-alias'
    assert budget.snapshot().reservations[0].usage.total_tokens==5
    events,_=read_events(audit.root)
    assert {e.payload.channel for e in events if e.type=='model/stream'} >= {'text','reasoning','provider_raw'}
    audit.close()


def test_disconnected_stream_preserves_unknown_and_no_retry(tmp_path):
    calls=[]
    def handler(request):
        calls.append(1)
        return httpx.Response(200,headers={'content-type':'text/event-stream'},text='data: {"type":"response.output_text.delta","delta":"partial"}\n\n')
    provider,budget,audit=make_provider(tmp_path,handler)
    with pytest.raises(ProviderFailure):provider.call('test',TagEnvelope,role='tag-batch')
    assert len(calls)==1 and budget.snapshot().reservations[0].status=='unknown'
    assert any(e.type=='model/stream' and e.payload.text=='partial' for e in read_events(audit.root)[0])
    audit.close()


def test_schema_failure_raw_retained(tmp_path):
    def handler(request):
        return httpx.Response(200,json={'model':'exact','status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'{"items":[{"rid":1,"value":{}}]}'}]}]})
    provider,budget,audit=make_provider(tmp_path,handler)
    with pytest.raises(ValidationError):provider.call('test',TagEnvelope,role='tag-batch')
    events=read_events(audit.root)[0]
    attempt=next(e for e in events if e.type=='model/attempt_end')
    assert attempt.payload.status=='schema_error' and attempt.payload.error.raw_ref
    assert budget.snapshot().reservations[0].status=='failed'
    audit.close()


def test_chat_explicit_adapter(tmp_path):
    def handler(request):
        assert request.url.path=='/v1/chat/completions'
        events=[{'model':'exact','choices':[{'index':0,'delta':{'content':'{"items":[]}'},'finish_reason':None}]},
                {'model':'exact','choices':[{'index':0,'delta':{},'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':2,'total_tokens':3}}]
        return httpx.Response(200,headers={'content-type':'text/event-stream'},text=''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n')
    provider,budget,audit=make_provider(tmp_path,handler,'openai-chat')
    assert provider.call('test',TagEnvelope,role='tag-batch').items==[]
    assert budget.snapshot().reservations[0].usage.total_tokens==3
    audit.close()

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_normalized_nested_nonfinite_rejected(value):
    with pytest.raises(ValidationError):
        NormalizedResponse.model_validate({'requested_model':'exact','reported_model':'exact','protocol':'openai-responses',
            'content':[],'usage':{},'finish_reason':'completed','reasoning_availability':'not_provided',
            'provider_raw':{'nested':[{'number':value}]}})


def test_discriminator_and_contract_roundtrip():
    base={'requested_model':'exact','reported_model':'exact','protocol':'openai-responses','content':[{'type':'unknown','text':'x'}],
        'usage':{},'finish_reason':'completed','reasoning_availability':'not_provided','provider_raw':None}
    with pytest.raises(ValidationError):NormalizedResponse.model_validate(base)
    for model, data in [(CategoryEnvelope,{'categories':[{'name':'x'}]}),
                        (TagEnvelope,{'items':[{'rid':1,'value':'x'}]}),
                        (IntentEnvelope,{'mode':'independent'})]:
        first=model.model_validate(data)
        assert model.model_validate_json(first.model_dump_json())==first


def test_denied_budget_records_terminal_without_network(tmp_path):
    calls=[]
    provider,budget,audit=make_provider(tmp_path,lambda request: calls.append(request))
    for i in range(10):budget.reserve('operator','exact',str(i))
    with pytest.raises(BudgetExceeded):provider.call('test',TagEnvelope,role='tag-batch')
    assert calls==[]
    attempt=[e for e in read_events(audit.root)[0] if e.type=='model/attempt_end'][-1]
    assert attempt.payload.status=='failed'
    audit.close()


def test_failed_audit_blocks_dispatch(tmp_path,monkeypatch):
    calls=[]
    provider,budget,audit=make_provider(tmp_path,lambda request: calls.append(request))
    monkeypatch.setattr(audit,'flush',lambda: (_ for _ in ()).throw(OSError('injected fsync failure')))
    with pytest.raises(OSError):provider.call('test',TagEnvelope,role='tag-batch')
    assert calls==[] and len(budget.snapshot().reservations)==0
    audit.close()


def test_nonempty_categories_and_null_description_rejected():
    with pytest.raises(ValidationError):CategoryEnvelope.model_validate({'categories':[]},context={'nonempty':True})
    with pytest.raises(ValidationError):CategoryEnvelope.model_validate({'categories':[{'name':'x','description':None}]})


def test_global_operator_concurrency_bound(tmp_path):
    import threading
    active=0
    peak=0
    lock=threading.Lock()
    def handler(request):
        nonlocal active,peak
        with lock:
            active+=1
            peak=max(peak,active)
        time.sleep(.03)
        with lock:active-=1
        return httpx.Response(200,json={'model':'exact','status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'{"items":[]}'}]}]})
    provider,budget,audit=make_provider(tmp_path,handler,concurrency=2)
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(lambda i:provider.call('test',TagEnvelope,role='tag-batch'),range(5)))
    assert peak==2 and len(budget.snapshot().reservations)==5
    audit.close()


def test_stream_keeps_raw_bytes_before_utf8_failure(tmp_path):
    raw=b': comment\nevent: malformed\ndata: \xff\n\n'
    provider,budget,audit=make_provider(tmp_path,lambda request:httpx.Response(200,headers={'content-type':'text/event-stream'},content=raw))
    with pytest.raises(UnicodeDecodeError):provider.call('test',TagEnvelope,role='tag-batch')
    events=read_events(audit.root)[0]
    streams=[e for e in events if e.type=='model/stream' and e.payload.raw_ref and e.payload.raw_ref.media_type=='text/event-stream']
    assert len(streams)==1
    assert (audit.root/streams[0].payload.raw_ref.path).read_bytes()==raw
    assert budget.snapshot().reservations[0].status=='unknown'
    audit.close()

@pytest.mark.parametrize('status',[401,429,500])
def test_http_failure_never_falls_back_or_retries(tmp_path,status):
    calls=[]
    def handler(request):
        calls.append((str(request.url),json.loads(request.content)['model']))
        return httpx.Response(status,json={'error':{'message':'fixture failure'}})
    provider,budget,audit=make_provider(tmp_path,handler)
    with pytest.raises(ProviderFailure):provider.call('test',TagEnvelope,role='tag-batch')
    assert calls==[('http://localhost:9999/v1/responses','exact')]
    assert budget.snapshot().reservations[0].status=='failed'
    audit.close()


def test_cli_overrides_validate_after_merge():
    from tada_runtime.config import load_config,parse_overrides
    config=load_config('examples/config.toml',overrides=parse_overrides(['runtime.batch_rows=7','models.agent.model="exact-custom-id"']))
    assert config.runtime.batch_rows==7 and config.models.agent.model=='exact-custom-id'
    with pytest.raises(ValidationError):
        load_config('examples/config.toml',overrides=parse_overrides(['runtime.batch_rows=true']))
    with pytest.raises(ValidationError):
        load_config('examples/config.toml',overrides=parse_overrides(['models.agent.api_key="forbidden-plaintext"']))
