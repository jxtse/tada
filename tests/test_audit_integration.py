from datetime import datetime, timezone
from queue import Empty
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from tada_runtime.audit import AuditLog, read_events, validate_session
from tada_runtime.audit_contracts import ErrorInfo
from tada_runtime.budget import BudgetLedger
from tada_runtime.config import BudgetConfig, Config
from tada_runtime.kernel import Kernel, wire_json
from tada_runtime.rpc_contracts import Initialize, InputSpec, RpcRequest
from tada_runtime.supervisor import Supervisor, normalize_pi


def config(tmp_path):
    return Config.model_validate({'models':{
        'agent':{'base_url':'http://localhost:9999/v1','model':'fixture-agent','protocol':'anthropic-messages','auth_mode':'optional'},
        'operators':{'base_url':'http://localhost:9998/v1','model':'fixture-operator','protocol':'openai-responses','auth_mode':'optional'}},
        'budget':{'ledger_path':str(tmp_path/'ledger.json'),'max_requests':1,'deadline_seconds':600.0}})


def test_partial_initialize_releases_owner(tmp_path,monkeypatch):
    monkeypatch.setattr('tada_runtime.supervisor.load_config',lambda *_,**__:config(tmp_path))
    monkeypatch.setattr('tada_runtime.supervisor.importlib.metadata.version',lambda *_:'fixture')
    owner=Supervisor()
    path=tmp_path/'session'
    with pytest.raises(FileNotFoundError):
        owner.initialize(Initialize(session_dir=str(path),inputs=[InputSpec(name='reviews',path=str(tmp_path/'missing.csv'))]))
    assert owner.audit is None and owner.kernel is None and owner.http is None
    with AuditLog(path,resume=True):
        pass


def test_binary_artifact_registration_and_close_cleanup(tmp_path):
    owner=Supervisor()
    owner.audit=AuditLog(tmp_path,secrets=['private-fixture'])
    (tmp_path/'outputs').mkdir()
    (tmp_path/'outputs'/'result.bin').write_bytes(b'\x00private-fixture\xff')
    owner.close()
    assert owner.audit is None
    event=next(event for event in read_events(tmp_path)[0] if event.type=='artifact/register')
    assert (tmp_path/event.payload.artifact.path).read_bytes()==b'\x00[REDACTED]\xff'
    validate_session(tmp_path)
    with AuditLog(tmp_path,resume=True):
        pass


def test_usage_unknown_and_invalid_not_zero():
    projection=normalize_pi({'content':[]})
    assert projection.input_tokens is None and projection.output_tokens is None and projection.total_tokens is None
    with pytest.raises(ValidationError):
        normalize_pi({'usage':{'input':True}})


def test_denied_budget_has_terminal_attempt_and_no_dispatch(tmp_path):
    owner=Supervisor()
    owner.audit=AuditLog(tmp_path/'session')
    owner.config=config(tmp_path)
    owner.budget=BudgetLedger(owner.config.budget,owner.audit)
    owner.budget.reserve('fixture','fixture','consumed')
    with pytest.raises(Exception,match='limit exhausted'):
        owner.dispatch(RpcRequest(id='deny',method='model_reserve',params={'role':'agent','provider_raw':{'model':'fixture-agent'}}))
    attempts=[event for event in owner.audit.events if event.type=='model/attempt_end']
    assert len(attempts)==1 and attempts[0].payload.status=='failed'
    assert owner.requests=={}
    assert validate_session(owner.audit.root).unknown_requests==[]
    owner.close()


def message(kind,parent,content):
    return {'header':{'msg_id':'reply-'+kind,'msg_type':kind,'date':datetime.now(timezone.utc)},
            'parent_header':{'msg_id':parent},'content':content,'buffers':[]}


def test_jupyter_parent_correlation_and_late_outputs_fixture(tmp_path):
    # Fault fixture for protocol correlation; real Jupyter is tested separately.
    iopub=iter([message('status','previous',{'execution_state':'idle'}),
        message('stream','previous',{'name':'stdout','text':'late'}),
        message('stream','current',{'name':'stdout','text':'actual'}),
        message('status','current',{'execution_state':'idle'})])
    shell=iter([message('execute_reply','previous',{'status':'ok'}),message('execute_reply','current',{'status':'ok'})])
    def get(iterator):
        try:return next(iterator)
        except StopIteration:raise Empty from None
    with AuditLog(tmp_path) as log:
        kernel=Kernel.__new__(Kernel)
        kernel.closed=kernel.poisoned=False
        kernel.cancel_time=None
        kernel.audit=log
        kernel.kernel_id='fixture-kernel'
        kernel.config=SimpleNamespace(cell_timeout_seconds=1.0)
        kernel.client=SimpleNamespace(execute=lambda *_a,**_k:'current',get_iopub_msg=lambda **_:get(iopub),get_shell_msg=lambda **_:get(shell))
        stdout,_,_,_,raw,_=kernel._cell('print("actual")','exec')
        assert stdout=='actual'
        assert any(event.type=='cell/stream' and event.payload.late for event in log.events)
        assert all(isinstance(item['header']['date'],str) for item in raw)
        log.blob({'provider_raw':raw})


@pytest.mark.parametrize('content',[
    [{'type':'image','data':'opaque'}],
    [{'type':'text'}],
    [{'type':'thinking'}],
    [{'type':'toolCall','id':'call','name':'execute_python'}],
    [{'type':'toolCall','id':'','name':'execute_python','arguments':{}}],
])
def test_pi_normalization_rejects_unknown_or_incomplete_blocks(content):
    with pytest.raises(ValidationError):
        normalize_pi({'content':content})


def test_stream_results_are_authoritative_for_all_channels(tmp_path):
    import base64
    from tada_runtime.rpc_contracts import RESULTS
    owner=Supervisor()
    owner.audit=AuditLog(tmp_path/'session',secrets=['fixture-secret'])
    for channel,raw,text in [('text',None,'hello'),('provider_raw',{'encoding':'base64','data':base64.b64encode(b'raw').decode(),'transport_chunk':True},'')]:
        result=owner.dispatch(RpcRequest(id='stream',method='model_stream',params={'request_id':'request','channel':channel,'text':text,'provider_raw':raw}))
        checked=RESULTS['model_stream'].validate_python(result)
        assert checked.emitted_parts>=0 and isinstance(checked.buffered,bool)
    owner.close()


def test_doctor_receipt_and_schema_manifest_are_strict():
    from tada_runtime.cli import DoctorReceipt
    from tada_runtime.schemas import SchemaManifest
    with pytest.raises(ValidationError):
        DoctorReceipt(dependencies={},endpoints={},inference_requests=1)
    with pytest.raises(ValidationError):
        DoctorReceipt(dependencies={},endpoints={'agent':{'kind':'http','status':True,'requested_model':'x','protocol':'anthropic-messages'}})
    with pytest.raises(ValidationError):
        SchemaManifest(schema_version='other',hashes={})
    with pytest.raises(ValidationError):
        SchemaManifest(hashes={'config':'not-a-sha256'})
