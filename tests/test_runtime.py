"""Protocol fault tests and a real Jupyter lifecycle test when installed."""
from queue import Empty
import threading
import pytest
from pydantic import ValidationError
from tada_runtime.audit import AuditLog
from tada_runtime.config import RuntimeConfig
from tada_runtime.rpc_contracts import RpcRequest, PARAMS
from tada_runtime.kernel import Kernel
from tada_runtime.schemas import generate


def test_schema_drift():
    generate(check=True)


def test_rpc_rejects_extra_and_arguments():
    with pytest.raises(ValidationError):RpcRequest.model_validate({'id':1,'method':'oops','params':{}})
    with pytest.raises(ValidationError):PARAMS['execute'].model_validate({'code':True})
    with pytest.raises(ValidationError):PARAMS['execute'].model_validate({'code':'pass','timeout':100})
    with pytest.raises(ValidationError):RpcRequest.model_validate({'schema_version':'0','id':1,'method':'cancel'})


class FakeClient:
    def __init__(self):
        self.messages=[
            {'header':{'msg_type':'status'},'parent_header':{'msg_id':'previous'},'content':{'execution_state':'idle'}},
            {'header':{'msg_type':'stream'},'parent_header':{'msg_id':'previous'},'content':{'name':'stdout','text':'late'}},
            {'header':{'msg_type':'stream'},'parent_header':{'msg_id':'current'},'content':{'name':'stdout','text':'mine'}},
            {'header':{'msg_type':'status'},'parent_header':{'msg_id':'current'},'content':{'execution_state':'idle'}},
        ]
        self.shells=[{'header':{'msg_type':'execute_reply'},'parent_header':{'msg_id':'previous'},'content':{'status':'ok'}},{'header':{'msg_type':'execute_reply'},'parent_header':{'msg_id':'current'},'content':{'status':'ok'}}]
    def execute(self,*args,**kwargs):return 'current'
    def get_iopub_msg(self,**kwargs):
        if self.messages:return self.messages.pop(0)
        raise Empty
    def get_shell_msg(self,**kwargs):
        if self.shells:return self.shells.pop(0)
        raise Empty


def test_jupyter_parent_correlation_and_late_output(tmp_path):
    with AuditLog(tmp_path/'audit') as audit:
        kernel=Kernel.__new__(Kernel);kernel.client=FakeClient();kernel.audit=audit
        kernel.closed=False;kernel.poisoned=False;kernel.closing=False;kernel.cancel_time=None
        kernel.kernel_id='fake';kernel.config=RuntimeConfig();kernel.cancelled=threading.Event()
        out,err,display,error,raw,timeout=kernel._cell('print("mine")','exec1')
        assert out=='mine' and not timeout
        late=[event for event in audit.events if event.type=='cell/stream' and event.payload.late]
        assert len(late)==1 and late[0].payload.parent_message_id=='previous'
        assert not kernel.client.messages and not kernel.client.shells


def test_single_cell_inflight():
    kernel=Kernel.__new__(Kernel);kernel.closed=False;kernel.closing=False;kernel.poisoned=False;kernel.lock=threading.Lock();kernel.lock.acquire()
    with pytest.raises(RuntimeError,match='one cell'):kernel.execute('pass')


def test_actual_persistent_kernel_error_interrupt_and_output(tmp_path):
    pytest.importorskip('jupyter_client',reason='real integration requires Jupyter client dependency')
    pytest.importorskip('ipykernel',reason='real integration requires ipykernel dependency')
    from tada_runtime.supervisor import Supervisor
    from tada_runtime.rpc_contracts import Initialize
    config=tmp_path/'config.toml'
    config.write_text('''[models.agent]
base_url="http://127.0.0.1:1/v1"
model="offline"
protocol="anthropic-messages"
auth_mode="optional"
auth_placeholder="explicit-nonsecret"
[models.operators]
base_url="http://127.0.0.1:1/v1"
model="offline"
protocol="openai-responses"
auth_mode="optional"
[budget]
ledger_path="'''+str(tmp_path/'ledger.json')+'''"
max_requests=1
deadline_seconds=60.0
''')
    owner=Supervisor()
    try:
        owner.initialize(Initialize(config_path=str(config),session_dir=str(tmp_path/'session')))
        assert owner.kernel.execute('answer = 42\ndef twice(x): return x * 2').status=='ok'
        assert owner.kernel.execute('raise ValueError("recoverable")').status=='error'
        assert owner.kernel.execute('print(twice(answer))').stdout.strip()=='84'
        worker=threading.Thread(target=lambda:owner.kernel.execute('import time; time.sleep(60)'))
        worker.start()
        import time
        time.sleep(.5);owner.kernel.cancel();worker.join(timeout=15)
        assert not worker.is_alive()
        assert owner.kernel.execute('print(answer)').stdout.strip()=='42'
        assert len(owner.budget.snapshot().reservations)==0
    finally:owner.close()
