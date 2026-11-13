"""Exercise the actual stdio main and method response validation, offline.

Only Jupyter and the loopback listener are replaced with explicit fixtures;
RPC parsing, dispatch, budget, audit, state and response contracts are real.
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import select
import subprocess
import sys
from tada_runtime.rpc_contracts import RpcRequest, RpcSuccess, RESULTS
from tada_runtime.audit import read_events, validate_session

BOOTSTRAP = r'''
import threading
import tada_runtime.kernel as kernel
import tada_runtime.supervisor as supervisor
from tada_runtime.rpc_contracts import Observation
class FakeHTTP:
    server_port=1
    def __init__(self,*args,**kwargs):pass
    def serve_forever(self):pass
    def shutdown(self):pass
    def server_close(self):pass
class FakeKernel:
    def __init__(self,audit,*args,**kwargs):
        self.audit=audit;self.namespace=[];self.cancelled=threading.Event()
    def execute(self,code):
        return Observation(execution_id='fixture-cell',status='ok',stdout='42\n',stderr='',
            full_output_ref=self.audit.blob({'fixture':True,'stdout':'42\n'}))
    def cancel(self):self.cancelled.set()
    def close(self):pass
kernel.Kernel=FakeKernel
supervisor.ThreadingHTTPServer=FakeHTTP
supervisor.importlib.metadata.version=lambda name: 'fixture-dependency'
supervisor.main()
'''


def test_all_method_responses_cross_real_stdio_main(tmp_path):
    config=tmp_path/'config.toml'
    config.write_text('''[models.agent]
base_url="http://localhost:9999/v1"
model="fixture-main"
protocol="anthropic-messages"
auth_mode="optional"
api_key_env="TADA_STDIO_TEST_KEY"
[models.operators]
base_url="http://localhost:9998/v1"
model="fixture-operator"
protocol="openai-responses"
auth_mode="optional"
[budget]
ledger_path="'''+str(tmp_path/'ledger.json')+'''"
max_requests=2
deadline_seconds=60.0
''')
    repo=Path(__file__).resolve().parents[1]
    env={**os.environ,'PYTHONPATH':str(repo/'python'),'TADA_STDIO_TEST_KEY':'private-stdio-secret'}
    child=subprocess.Popen([sys.executable,'-c',BOOTSTRAP],cwd=repo,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
    counter=0
    def call(method,params=None):
        nonlocal counter
        counter+=1
        request=RpcRequest(id=str(counter),method=method,params=params or {})
        child.stdin.write(request.model_dump_json()+'\n');child.stdin.flush()
        readable,_,_=select.select([child.stdout],[],[],15)
        assert readable, f'{method}: no stdio response within 15 seconds'
        line=child.stdout.readline()
        assert line, f'{method}: child terminated with {child.poll()}'
        decoded=json.loads(line)
        assert 'error' not in decoded, (method,decoded)
        envelope=RpcSuccess.model_validate_json(line)
        assert envelope.id==str(counter)
        return RESULTS[method].validate_python(envelope.result)
    session=tmp_path/'session'
    try:
        assert call('receipt',{'session_id':None,'status':'failed','error':'fixture-before-initialize'}).status=='failed'
        result=call('initialize',{'config_path':str(config),'session_dir':str(session)})
        assert result.session_id==session.name
        receipt=call('receipt',{'session_id':session.name,'status':'turn_completed','message':{'role':'assistant','text':'private-stdio-secret'}})
        assert 'private-stdio-secret' not in receipt.model_dump_json()
        assert call('inventory').states==[]
        assert call('execute',{'code':'print(42)','tool_call_id':'fixture-tool'}).stdout=='42\n'
        call('event',{'type':'turn/start','payload':{'status':'open'},'ids':{'turn_id':'fixture-turn'}})
        reservation=call('model_reserve',{'role':'agent','provider_raw':{'model':'fixture-main','messages':[]}})
        request_id=reservation.request_id
        first=call('model_stream',{'request_id':request_id,'channel':'text','text':'private-','provider_sequence':0})
        assert first.emitted_parts==0 and first.buffered
        call('model_stream',{'request_id':request_id,'channel':'text','text':'stdio-secret end','provider_sequence':1})
        raw=base64.b64encode(b'data: {"delta":"private-stdio-secret"}\n\n').decode()
        call('model_stream',{'request_id':request_id,'channel':'provider_raw','provider_raw':{'encoding':'base64','data':raw,'transport_chunk':True}})
        call('model_settle',{'request_id':request_id,'status':'success','reported_model':'fixture-main',
            'provider_raw':{'content':[{'type':'text','text':'private-stdio-secret end'}],'model':'fixture-main',
                'usage':{'input':1,'output':1,'totalTokens':2},'stopReason':'stop'}})
        call('compaction',{'phase':'start','messages':[],'status':'started'})
        call('compaction',{'phase':'end','messages':[],'summary':'fixture summary','status':'success'})
        call('event',{'type':'turn/end','payload':{'status':'success'},'ids':{'turn_id':'fixture-turn'}})
        call('checkpoint',{'messages':[],'pi_entries':[]})
        assert call('cancel').status=='cancelling'
        assert call('close').status=='closed'
        child.stdin.close()
        child.wait(timeout=10)
        assert child.returncode==0,child.stderr.read()
    finally:
        if child.poll() is None:
            child.kill();child.wait(timeout=10)
    events,_=read_events(session)
    text=''.join(e.payload.text for e in events if e.type=='model/stream' and e.payload.channel=='text')
    assert 'private-stdio-secret' not in text and text.endswith(' end') and '[REDACTED]' in text
    assert validate_session(session).valid
