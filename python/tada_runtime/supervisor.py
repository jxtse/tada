"""Single session owner, audited JSONL RPC dispatcher and kernel operator service."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
import pandas as pd
from pydantic import ValidationError, JsonValue, Field
from .contracts import StrictModel, Usage
from .config import Config, load_config, parse_overrides, _merge
from .audit import AuditLog, atomic_write, canonical
from .audit_contracts import ErrorInfo, BlobRef
from .state import Store, snapshot, restore
from .storage_contracts import SNAPSHOT_ADAPTER
from .rpc_contracts import *
from .budget import BudgetLedger
from .providers import Provider
from .cancellation import cancellation_token
from .streaming import StreamingRedactor, RawTransportChunk, opaque_raw_fragment
from tada_operators.engine import Engine
from .tool_boundary import ToolBoundary

class SessionHeader(StrictModel):
    schema_version: Literal['1.0'] = '1.0'
    session_id: str
    config: Config
    revision: str
    versions: dict[str,str]

class PiTextBlock(StrictModel):
    type: Literal['text']
    text: str

class PiThinkingBlock(StrictModel):
    type: Literal['thinking']
    thinking: str

class PiToolBlock(StrictModel):
    type: Literal['toolCall']
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: JsonValue

PiBlock = Annotated[PiTextBlock | PiThinkingBlock | PiToolBlock, Field(discriminator='type')]
PI_BLOCK = TypeAdapter(PiBlock)

class PiProjection(StrictModel):
    """Normalized projection, extracted from the vendor raw message."""
    model: str | None = None
    content: list[PiBlock] = Field(default_factory=list)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    stop_reason: str = ''

def normalize_pi(raw):
    # Pi's complete original is retained separately. These fields are validated domain data.
    raw = TypeAdapter(dict[str,JsonValue]).validate_python({} if raw is None else raw)
    content=[]
    for item in TypeAdapter(list[dict[str,JsonValue]]).validate_python(raw.get('content',[])):
        content.append(PI_BLOCK.validate_python({k:item[k] for k in ('type','text','thinking','id','name','arguments') if k in item}))
    usage=TypeAdapter(dict[str,JsonValue]).validate_python(raw.get('usage',{}))
    return PiProjection.model_validate({'model':raw.get('model'), 'content':content,
        'input_tokens':usage.get('input'),'output_tokens':usage.get('output'),
        'total_tokens':usage.get('totalTokens'),'stop_reason':raw.get('stopReason','')})

class Supervisor:
    def __init__(self):
        self.tool_boundary=ToolBoundary(self)
        self.audit=None; self.kernel=None; self.http=None; self.store=None
        self.atom_lock=threading.Lock()
        self.namespace=[]
        self.requests={}
        self.stream_redactors={}
        self.compaction_before=None
        self.compaction_id=None

    def initialize(self, params):
        if self.audit:
            raise RuntimeError('session already initialized')
        try:
            return self._initialize(params)
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass  # Preserve the initialization error after releasing all owned resources.
            raise

    def _initialize(self, params):
        if self.audit: raise RuntimeError('session already initialized')
        path=Path(params.session_dir).resolve() if params.session_dir else Path('.tada/sessions',uuid.uuid4().hex).resolve()
        if params.resume:
            if not path.exists(): path=Path('.tada/sessions',str(params.session_dir)).resolve()
            header=SessionHeader.model_validate_json((path/'session.json').read_bytes())
            config=load_config(params.config_path,overrides=parse_overrides(params.overrides)) if params.config_path else Config.model_validate(_merge(header.config.model_dump(),parse_overrides(params.overrides)))
        else:
            config=load_config(params.config_path,overrides=parse_overrides(params.overrides))
        config.require_budget()
        secrets=[key for provider in (config.models.agent,config.models.operators,config.models.decision) if provider is not None and (key:=provider.api_key()) and key!=provider.auth_placeholder]
        self.config=config
        self.audit=AuditLog(path,secrets=secrets,resume=params.resume)
        for folder in ('inputs','outputs','blobs','states','pi-checkpoints'): (path/folder).mkdir(exist_ok=True,mode=0o700)
        self.store=Store(self.audit)
        self.budget=BudgetLedger(config.require_budget(),self.audit)
        self.provider=Provider(config.models.operators,self.budget,self.audit)
        self.decision=None
        if config.models.decision is not None:
            from .decision import DecisionClient
            self.decision=DecisionClient(config.models.decision,self.budget,self.audit,concurrency=config.runtime.operator_concurrency)
        self.engine=Engine(config,self.provider,self.store,self.audit,decision=self.decision)
        checkpoint=self.audit.load_checkpoint() if params.resume else None
        projection=checkpoint.provider_raw if checkpoint else {}
        projection=TypeAdapter(dict[str,JsonValue]).validate_python(projection)
        if not params.resume:
            revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
            versions={name:importlib.metadata.version(name) for name in ('pydantic','pandas','numpy','jupyter-client','ipykernel')}
            header=SessionHeader(session_id=self.audit.session_id,config=config,revision=revision,versions=versions)
            atomic_write(path/'session.json',canonical(header))
            self.audit.append('session/start',{'status':'open','revision':revision,'versions':versions,'producer_redaction_tail_chars':max((len(s)-1 for s in secrets),default=0),'producer_timeout_seconds':max(config.models.agent.timeout_seconds,config.models.operators.timeout_seconds),'config_ref':self.audit.blob(config.model_dump(mode='json')).model_dump()})
            for spec in params.inputs:
                if spec.name in {'pd','np','json','tag','categorize','state','preview','validate_category_references','labels_frame'}: raise ValueError('reserved input name')
                source=Path(spec.path)
                if source.suffix.lower()!='.csv': raise ValueError('input currently requires CSV; other tables may be explicitly loaded in Python')
                frame=pd.read_csv(source)
                if 'rid' in frame: frame=frame.set_index('rid',verify_integrity=True)
                else: frame.index=pd.Index([f'r{i+1}' for i in range(len(frame))],name='rid')
                # Validate immutable row identities before accepting the dataset.
                from .contracts import TextSnapshot,TextRow
                TextSnapshot(rows=[TextRow(rid=rid,text='') for rid in frame.index.tolist()])
                meta=self.store.put(spec.name,frame,op='input')
                atomic_write(path/'inputs'/f'{spec.name}.json',snapshot(frame).model_dump_json().encode())
        else:
            previous=[]
            for event in self.audit.events:
                if event.type=='cell/end': previous=event.payload.namespace_summary
            self.audit.append('session/resume',{'status':'open','reason':'Only committed registered values and inputs are restored; ordinary side effects are not rolled back.','transient_variables_lost':previous,'boundary_seq':checkpoint.event_seq,'config_ref':self.audit.blob(config.model_dump(mode='json')).model_dump()})
            # Preserve uncertain charged attempts, never silently dispatch them again.
            for reservation in self.budget.snapshot().reservations:
                if reservation.status=='reserved' and any(e.request_id==reservation.request_id for e in self.audit.events): self.budget.settle(reservation.request_id,'unknown')
        owner=self
        token=uuid.uuid4().hex
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                try:
                    if self.headers.get('Authorization')!=f'Bearer {token}': raise ValueError('invalid kernel bridge authentication')
                    size=int(self.headers.get('Content-Length','0'))
                    if size<=0 or size>32*1024*1024: raise ValueError('invalid kernel request size')
                    request=KernelRequest.model_validate_json(self.rfile.read(size))
                    value=owner.kernel_call(request)
                    body=KernelResult(value=value).model_dump_json().encode(); status=200
                except Exception as error:
                    body=RpcFailure(id='kernel',error=owner.error(error)).model_dump_json().encode(); status=400
                self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers()
                try:self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError):pass
        self.http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=self.http.serve_forever,daemon=True).start()
        from .kernel import Kernel
        self.kernel=Kernel(self.audit,self.store,config.runtime,f'http://127.0.0.1:{self.http.server_port}',token)
        self.kernel.namespace=self.namespace
        return {'config':config.model_dump(mode='json'),'session_dir':str(path),'session_id':self.audit.session_id,
                'agent_api_key':config.models.agent.api_key(), 'tool_schema':ExecuteArgs.model_json_schema(),
                'restored_messages':projection.get('messages',[]),'restored_entries':projection.get('pi_entries',[]),
                'namespace_summary':'\n'.join(self.namespace)}

    def kernel_call(self,request):
        if request.op=='inventory':
            self.namespace=request.namespace
            if self.kernel:self.kernel.namespace=request.namespace
            return None
        if request.op=='state.list':return [s.model_dump(mode='json') for s in self.store.list()]
        if request.op=='state.get':return snapshot(self.store.get(request.key)).model_dump(mode='json')
        if request.op=='state.put':
            value=restore(SNAPSHOT_ADAPTER.validate_python(request.snapshot))
            return self.store.put(request.name,value).model_dump(mode='json')
        # Capture this execution's token before waiting for another atom. A
        # later cell gets a different token and cannot revive this queued call.
        cancellation=cancellation_token(self.kernel.cancelled) if self.kernel else None
        execution_id=self.kernel.active_execution if self.kernel else None
        with self.atom_lock:
            if cancellation:
                cancellation.check()
            value=restore(SNAPSHOT_ADAPTER.validate_python(request.snapshot))
            options={}
            if request.op=='tag':
                options={'labels':[c.model_dump() for c in request.labels] if request.labels is not None else None,'multi':request.multi,
                    'long_rows':request.long_rows.model_dump(mode='json') if request.long_rows is not None else None}
            result=getattr(self.engine,request.op)(value,request.query,name=request.name,fresh=request.fresh,
                parent_id=execution_id,cancellation=cancellation,cardinality=request.cardinality,**options)
            return snapshot(result).model_dump(mode='json')

    def budget_status(self):
        """Requests used/remaining and seconds left in the cumulative budget (agent and operator calls)."""
        state=self.budget.snapshot()
        used=len(state.reservations)
        return {'requests_used':used,'requests_limit':state.max_requests,'requests_remaining':max(0,state.max_requests-used),
                'seconds_remaining':max(0,int(state.deadline_at-time.time()))}

    def error(self,error):
        text=str(error)
        if self.audit:text=self.audit._redact(text)[0]
        if isinstance(error,ValidationError):
            errors=error.errors(include_input=False,include_url=False)
            # The kernel only surfaces the message, so name the failed contract rules (never the input values).
            reasons=list(dict.fromkeys(e['msg'] for e in errors))[:3]
            message='Contract validation failed'+(': '+'; '.join(reasons) if reasons else '')
            if self.audit:message=self.audit._redact(message)[0]
            return ErrorInfo(kind='ValidationError',message=message,details=[str(e['loc'])+': '+e['type'] for e in errors])
        return ErrorInfo(kind=type(error).__name__,message=text)

    def _emit_stream_parts(self, request_id, parts):
        events=[]
        for part in parts:
            payload=dict(part.metadata)
            payload.update(text=part.text, block_index=part.fragment_index,
                fragment_offset=part.fragment_offset, fragment_length=part.fragment_length, redacted=part.redacted or payload.get('redacted',False))
            events.append(self.audit.append('model/stream',payload,request_id=request_id,source='agent'))
        return events

    def _flush_streams(self, request_id=None):
        for key in list(self.stream_redactors):
            if request_id is None or key[0]==request_id:
                redactor=self.stream_redactors.pop(key)
                self._emit_stream_parts(key[0],redactor.finish())

    def _stream_event(self, params):
        payload=params.model_dump(exclude={'request_id','provider_raw'},exclude_none=True)
        raw=params.provider_raw
        if raw is not None:
            if self.audit._secrets:
                ref=self.audit.blob({'redacted':True,'reason':'Raw stream may contain secrets split across encoded fields; see sanitized typed content'})
                ref=BlobRef.model_validate({**ref.model_dump(),'redacted':True})
                payload['redacted']=True
            else:
                try:
                    transport=RawTransportChunk.model_validate(raw)
                except ValidationError:
                    ref=self.audit.blob(raw)
                else:
                    ref=self.audit.blob_bytes(transport.decode(),media_type='application/octet-stream')
                    payload['fragment_length']=len(transport.decode())
            payload['raw_ref']=ref.model_dump()
        if params.channel=='provider_raw':
            # Raw text can likewise contain independently encoded secret pieces.
            if self.audit._secrets and payload['text']:
                payload['text']='[REDACTED RAW STREAM]'
                payload['redacted']=True
            self.audit.append('model/stream',payload,request_id=params.request_id,source='agent')
            return {'buffered':False,'emitted_parts':1}
        key=(params.request_id,params.channel)
        redactor=self.stream_redactors.setdefault(key,StreamingRedactor(self.audit._secrets))
        parts=redactor.feed(params.text,payload)
        events=self._emit_stream_parts(params.request_id,parts)
        return {'buffered':not bool(events),'emitted_parts':len(events)}

    def dispatch(self,request):
        params=PARAMS[request.method].model_validate(request.params)
        if request.method=='receipt':
            receipt=params.model_dump(mode='json')
            return self.audit._redact(receipt)[0] if self.audit else receipt
        if request.method=='initialize':return self.initialize(params)
        if not self.audit:raise RuntimeError('initialize first')
        if request.method=='prepare_tool':return self.tool_boundary.prepare(params)
        if request.method=='execute_tool':return self.tool_boundary.execute(params)
        if request.method=='tool_observation':return self.tool_boundary.observe(params)
        if request.method=='execute':
            with self.atom_lock: pass
            observation=self.kernel.execute(params.code)
            observation=observation.model_copy(update={'budget':self.budget_status()})
            ref=self.audit.blob(observation.model_dump(mode='json'))
            self.audit.append('tool/result',{'status':observation.status,'observation_ref':ref.model_dump(),'full_output_ref':observation.full_output_ref.model_dump(),'truncated':observation.truncated},execution_id=observation.execution_id,tool_call_id=params.tool_call_id,source='kernel')
            return observation.model_dump(mode='json')
        if request.method=='inventory':return {'namespace_summary':self.namespace,'states':[s.model_dump(mode='json') for s in self.store.list()],'outputs':[str(p.relative_to(self.audit.root)) for p in (self.audit.root/'outputs').iterdir() if p.is_file()]}
        if request.method=='cancel':
            self.kernel.cancel();return {'status':'cancelling'}
        if request.method=='event':return self.audit.append(params.type,params.payload,**params.ids.model_dump(exclude_none=True)).model_dump(mode='json')
        if request.method=='checkpoint':return self.audit.checkpoint(params.model_dump(mode='json')).model_dump(mode='json')
        if request.method=='model_reserve':
            self.audit.assert_healthy()
            request_id=uuid.uuid4().hex
            body=self.audit.blob(params.provider_raw)
            self.audit.append('model/request',{'role':params.role,'requested_model':self.config.models.agent.model,'protocol':self.config.models.agent.protocol,'request_ref':body.model_dump(),'parameters':self.config.models.agent.model_dump(mode='json')},request_id=request_id,turn_id=params.turn_id,step_id=params.step_id,source='agent')
            try:
                self.budget.reserve(params.role,self.config.models.agent.model,request_id,
                                    output_cap=self.config.models.agent.max_output_tokens)
            except Exception as error:
                self.audit.append('model/attempt_end', {'status':'failed','error':self.error(error).model_dump()},request_id=request_id,source='agent')
                raise
            self.audit.flush();self.requests[request_id]=time.monotonic()
            return {'request_id':request_id}
        if request.method=='model_stream':
            return self._stream_event(params)
        if request.method=='model_settle':
            self._flush_streams(params.request_id)
            normalized=normalize_pi(params.provider_raw)
            content=[]
            for block in normalized.content:
                if block.type=='text':content.append({'type':'text','text':block.text or ''})
                elif block.type=='thinking':content.append({'type':'reasoning','text':block.thinking or '', 'availability':'exposed' if block.thinking else 'opaque'})
                elif block.type=='toolCall':content.append({'type':'tool_call','call_id':block.id or '', 'name':block.name or '', 'arguments':canonical(block.arguments).decode()})
            usage=Usage(input_tokens=normalized.input_tokens,output_tokens=normalized.output_tokens,total_tokens=normalized.total_tokens)
            ref=self.audit.blob(params.provider_raw)
            self.audit.append('model/response',{'requested_model':self.config.models.agent.model,'reported_model':params.reported_model,'protocol':self.config.models.agent.protocol,'content':content,'response_ref':ref.model_dump(),'usage':usage.model_dump(),'finish_reason':normalized.stop_reason,'reasoning_availability':'exposed' if any(b['type']=='reasoning' for b in content) else 'not_provided','provider_raw':params.provider_raw},request_id=params.request_id,source='agent')
            self.audit.append('model/attempt_end',{'status':params.status,'error':ErrorInfo(kind='ProviderError',message=params.error).model_dump() if params.error else None,'response_ref':ref.model_dump(),'elapsed_seconds':time.monotonic()-self.requests.pop(params.request_id,time.monotonic())},request_id=params.request_id,source='agent')
            self.budget.settle(params.request_id,params.status if params.status in ('success','cancelled','unknown') else 'failed',usage)
            return {'status':params.status}
        if request.method=='compaction':
            if params.phase=='start' and self.compaction_id is not None:
                raise ValueError('a compaction is already in progress')
            if params.phase=='end' and self.compaction_id is None:
                raise ValueError('compaction end requires an active compaction')
            ref=self.audit.blob(params.messages)
            if params.phase=='start':
                self.compaction_before=ref
                self.compaction_id=uuid.uuid4().hex
            payload={'summary':params.summary,'status':params.status,'before_ref':(self.compaction_before or ref).model_dump()}
            if params.phase=='end':payload['after_ref']=ref.model_dump()
            ids={'source':'agent','compaction_id':self.compaction_id}
            if params.summary:self.audit.append('compaction/summary',payload,**ids)
            event=self.audit.append('compaction/'+params.phase,payload,**ids)
            if params.phase=='end':
                self.compaction_id=None
                self.compaction_before=None
            return event.model_dump(mode='json')
        if request.method=='close':self.close();return {'status':'closed'}
        raise ValueError('unsupported method')

    def close(self):
        if self.audit:
            self._flush_streams()
        failure = None
        try:
            if self.kernel:
                self.kernel.close()
        except BaseException as error:
            failure = error
        finally:
            self.kernel = None
        # Do not release ownership while an already dispatched operator is still settling.
        with self.atom_lock:
            pass
        try:
            if self.http:
                self.http.shutdown()
                self.http.server_close()
        except BaseException as error:
            failure = failure or error
        finally:
            self.http = None
        audit = self.audit
        if audit:
            try:
                audit.assert_healthy()
                for request_id in list(self.requests):
                    audit.append('model/attempt_end',{'status':'unknown'},request_id=request_id,source='agent')
                    self.budget.settle(request_id,'unknown')
                outputs = audit.root/'outputs'
                for path in outputs.iterdir() if outputs.exists() else ():
                    if path.is_file():
                        ref = audit.blob_bytes(path.read_bytes())
                        audit.append('artifact/register',{'artifact':ref.model_dump(),'name':path.name})
                audit.append('session/end',{'status':'closed'})
            except BaseException as error:
                failure = failure or error
            finally:
                self.requests.clear()
                try:
                    audit.close()
                except BaseException as error:
                    failure = failure or error
                self.audit = None
        if failure:
            raise failure

def main():
    owner=Supervisor();write_lock=threading.Lock()
    def handle(raw):
        request=None
        try:
            request=RpcRequest.model_validate_json(raw)
            result=RESULTS[request.method].validate_python(owner.dispatch(request))
            response=RpcSuccess(id=request.id,result=result.model_dump(mode='json'))
        except Exception as error:
            response=RpcFailure(id=request.id if request else 'invalid',error=owner.error(error))
        with write_lock:
            sys.stdout.write(response.model_dump_json()+'\n');sys.stdout.flush()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for raw in sys.stdin:
                # Cancellation can reach a running cell; all lifecycle calls remain owner driven.
                try:
                    request=RpcRequest.model_validate_json(raw)
                except ValidationError:
                    handle(raw)
                    continue
                if request.method in ('execute','execute_tool'):pool.submit(handle,raw)
                else:handle(raw)
    finally:owner.close()

if __name__=='__main__':main()
