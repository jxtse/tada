"""R2: real supervisor/engine/provider cancellation with offline HTTP fixtures."""
from __future__ import annotations
import json
import threading
from types import SimpleNamespace
import httpx
import pandas as pd
import pytest
from tada_runtime.audit import AuditLog, read_events
from tada_runtime.budget import BudgetLedger
from tada_runtime.config import Config
from tada_runtime.providers import Provider
from tada_runtime.rpc_contracts import KernelRequest
from tada_runtime.state import Store, snapshot
from tada_runtime.supervisor import Supervisor
from tada_operators.engine import Engine


def make_owner(tmp_path, concurrency, handler, cancelled=None):
    cfg=Config.model_validate({'models':{role:{'base_url':'http://127.0.0.1:1/v1','model':'offline-fixture','protocol':'openai-responses','auth_mode':'optional'} for role in ('agent','operators')},
        'runtime':{'batch_rows':1,'reduce_fan_in':2,'operator_concurrency':concurrency},
        'budget':{'ledger_path':str(tmp_path/'offline-ledger.json'),'max_requests':40,'deadline_seconds':60.0}})
    owner=Supervisor()
    owner.audit=AuditLog(tmp_path/'session')
    owner.store=Store(owner.audit)
    owner.config=cfg
    owner.kernel=SimpleNamespace(cancelled=cancelled or threading.Event(),active_execution='fixture-cell')
    owner.budget=BudgetLedger(cfg.budget,owner.audit)
    owner.provider=Provider(cfg.models.operators,owner.budget,owner.audit,transport=httpx.MockTransport(handler),concurrency=concurrency)
    owner.engine=Engine(cfg,owner.provider,owner.store,owner.audit)
    return owner


def response(body):
    return httpx.Response(200,json={'model':'offline-fixture','status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(body)}]}],
        'usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2}})


def request_for(op='tag',rows=4):
    texts=pd.Series([f'text {i}' for i in range(rows)],index=[f'r{i}' for i in range(rows)])
    return KernelRequest(op=op,snapshot=snapshot(texts).model_dump(mode='json'),query='fixture semantic query')


def body_for(prompt,joint=False):
    if 'intent-classification' in prompt:
        return {'mode':'joint','domain_query':'discover themes'} if joint else {'mode':'independent','domain_query':None}
    if 'Discover corpus-level' in prompt or 'Consolidate the supplied' in prompt:
        return {'categories':[{'name':'Topic'}]}
    rows=json.loads(prompt.split('\nRows: ',1)[1].split('\nReturn only one JSON',1)[0])
    return {'items':[{'rid':row['rid'],'value':'Topic'} for row in rows]}


@pytest.mark.parametrize('concurrency',[1,2])
def test_cancellation_after_intent_stops_assignment_and_commit(tmp_path,concurrency):
    cancelled=threading.Event()
    dispatch_flags=[]
    def handler(request):
        dispatch_flags.append(cancelled.is_set())
        prompt=json.loads(request.content)['input']
        if 'intent-classification' in prompt:cancelled.set()
        return response(body_for(prompt))
    owner=make_owner(tmp_path,concurrency,handler,cancelled)
    try:
        error=None
        try:owner.kernel_call(request_for())
        except RuntimeError as exc:error=exc
        assert dispatch_flags==[False], f'HTTP dispatch after cancel: {dispatch_flags}'
        assert error is not None and 'cancel' in str(error).lower()
        assert owner.store.list()==[]
        assert len(owner.budget.snapshot().reservations)==1
    finally:owner.audit.close()


def phase_of(prompt):
    if 'intent-classification' in prompt:return 'intent'
    if 'Discover corpus-level' in prompt:return 'map'
    if 'Consolidate the supplied' in prompt:return 'reduce'
    return 'assignment'


@pytest.mark.parametrize('concurrency',[1,2])
@pytest.mark.parametrize('phase',['intent','map','reduce','assignment'])
def test_supervised_phase_cancel_drains_dispatched_and_stops_queued(tmp_path,concurrency,phase):
    from tada_runtime.cancellation import ExecutionCancellation,ExecutionCancelled
    token=ExecutionCancellation()
    barrier=threading.Barrier(1 if phase=='intent' else concurrency)
    calls=[]
    lock=threading.Lock()
    def handler(request):
        prompt=json.loads(request.content)['input']
        current=phase_of(prompt)
        with lock:calls.append((current,token.is_set()))
        if current==phase:
            barrier.wait(timeout=5)
            token.set()
        return response(body_for(prompt,joint=True))
    owner=make_owner(tmp_path,concurrency,handler,token)
    try:
        with pytest.raises(ExecutionCancelled):owner.kernel_call(request_for(rows=8))
        assert calls and all(not after_cancel for _,after_cancel in calls),calls
        target_calls=[current for current,_ in calls if current==phase]
        assert len(target_calls)==(1 if phase=='intent' else concurrency),calls
        reservations=owner.budget.snapshot().reservations
        assert len(reservations)==len(calls)
        assert all(item.status=='success' for item in reservations)
        states=owner.store.list()
        if phase=='assignment':
            assert len(states)==1 and states[0].op=='categorize' # committed before cancellation
        else:assert states==[]
        atoms=[e for e in owner.audit.events if e.type=='atom/end']
        assert atoms[-1].payload.status=='cancelled'
        assert not any(s.op=='tag' for s in states)
    finally:owner.audit.close()


@pytest.mark.parametrize('concurrency',[1,2])
def test_cancel_queued_provider_slots_never_reserves_or_dispatches(tmp_path,concurrency):
    from concurrent.futures import ThreadPoolExecutor
    from tada_runtime.cancellation import ExecutionCancellation,ExecutionCancelled
    from tada_runtime.contracts import TagEnvelope
    token=ExecutionCancellation()
    started=threading.Barrier(concurrency+1)
    release=threading.Event()
    calls=[]
    def handler(request):
        calls.append(token.is_set())
        started.wait(timeout=5)
        assert release.wait(timeout=5)
        return response({'items':[]})
    owner=make_owner(tmp_path,concurrency,handler,token)
    try:
        with ThreadPoolExecutor(max_workers=concurrency+1) as pool:
            active=[pool.submit(owner.provider.call,'fixture',TagEnvelope,role='tag-batch',cancellation=token) for _ in range(concurrency)]
            started.wait(timeout=5)
            queued=pool.submit(owner.provider.call,'fixture',TagEnvelope,role='tag-batch',cancellation=token)
            token.set()
            try:
                with pytest.raises(ExecutionCancelled):queued.result(timeout=2)
            finally:release.set()
            assert all(future.result(timeout=5).items==[] for future in active)
        assert calls==[False]*concurrency
        assert len(owner.budget.snapshot().reservations)==concurrency
    finally:release.set();owner.audit.close()


@pytest.mark.parametrize('boundary',['before_reserve','after_reserve','transport_entry','commit'])
def test_cancellation_at_last_boundaries_prevents_dispatch_or_commit(tmp_path,monkeypatch,boundary):
    from tada_runtime.cancellation import ExecutionCancellation,ExecutionCancelled
    token=ExecutionCancellation()
    calls=[]
    def handler(request):
        calls.append(token.is_set())
        return response(body_for(json.loads(request.content)['input']))
    owner=make_owner(tmp_path,1,handler,token)
    try:
        if boundary in ('before_reserve','after_reserve'):
            original=owner.budget.reserve
            def reserve(*args,**kwargs):
                if boundary=='before_reserve':
                    token.set();token.check()
                reservation=original(*args,**kwargs)
                token.set()
                return reservation
            monkeypatch.setattr(owner.budget,'reserve',reserve)
        elif boundary=='transport_entry':
            original=token.admit_dispatch
            def admit():
                token.set();original()
            monkeypatch.setattr(token,'admit_dispatch',admit)
        else:
            original=owner.engine._commit
            def commit(*args,**kwargs):
                token.set();return original(*args,**kwargs)
            monkeypatch.setattr(owner.engine,'_commit',commit)
        with pytest.raises(ExecutionCancelled):owner.kernel_call(request_for(rows=1))
        assert owner.store.list()==[]
        reservations=owner.budget.snapshot().reservations
        if boundary=='commit':
            assert calls==[False,False]
            assert all(item.status=='success' for item in reservations)
        elif boundary=='before_reserve':
            assert calls==[] and reservations==[]
        else:
            assert calls==[] and len(reservations)==1
            assert reservations[0].status=='cancelled'
    finally:owner.audit.close()


def test_cancelled_execution_cannot_be_revived_by_later_token(tmp_path):
    from tada_runtime.cancellation import ExecutionCancellation,ExecutionCancelled
    old_token=ExecutionCancellation();old_token.set()
    owner=make_owner(tmp_path,1,lambda request:pytest.fail('unexpected provider request'),old_token)
    try:
        # A request waiting on the atom lock captured the old token; a fresh
        # execution token never clears that old cancellation state.
        new_token=ExecutionCancellation()
        owner.kernel.cancelled=new_token
        assert not new_token.is_set() and old_token.is_set()
        with pytest.raises(ExecutionCancelled):
            owner.engine.tag(pd.Series(['x'],index=['r1']),'fixture',cancellation=old_token)
        assert owner.budget.snapshot().reservations==[]
    finally:owner.audit.close()


@pytest.mark.parametrize('concurrency',[1,2])
def test_actual_kernel_cancel_propagates_to_supervised_operator(tmp_path,concurrency):
    from tada_runtime.rpc_contracts import Initialize
    cfg=tmp_path/'config.toml'
    cfg.write_text('''[models.agent]
base_url="http://127.0.0.1:1/v1"
model="offline-fixture"
protocol="anthropic-messages"
auth_mode="optional"
[models.operators]
base_url="http://127.0.0.1:1/v1"
model="offline-fixture"
protocol="openai-responses"
auth_mode="optional"
[runtime]
operator_concurrency='''+str(concurrency)+'''
batch_rows=1
[budget]
ledger_path="'''+str(tmp_path/'offline-ledger.json')+'''"
max_requests=10
deadline_seconds=60.0
''')
    owner=Supervisor()
    entered=threading.Event();release=threading.Event()
    calls=[];results=[];errors=[]
    try:
        owner.initialize(Initialize(config_path=str(cfg),session_dir=str(tmp_path/'session')))
        kernel_id=owner.kernel.kernel_id
        def handler(request):
            calls.append(owner.kernel.cancelled.is_set())
            entered.set()
            assert release.wait(timeout=10)
            return response(body_for(json.loads(request.content)['input']))
        owner.provider.transport=httpx.MockTransport(handler)
        def execute():
            try:results.append(owner.kernel.execute("saved = 42\ntags = tag(pd.Series(['a','b','c'],index=['r1','r2','r3']), 'sentiment')"))
            except BaseException as exc:errors.append(exc)
        worker=threading.Thread(target=execute)
        worker.start()
        assert entered.wait(timeout=10)
        old_token=owner.kernel.cancelled
        owner.kernel.cancel()
        assert old_token.is_set()
        release.set()
        worker.join(timeout=15)
        assert not worker.is_alive() and not errors
        # The HTTP bridge retains ownership until its already-sent response is
        # logged and its cancelled atom terminates, even after kernel interrupt.
        with owner.atom_lock:pass
        assert calls==[False] and owner.store.list()==[]
        assert results[0].status=='cancelled'
        assert owner.kernel.execute('print(saved)').stdout.strip()=='42'
        assert owner.kernel.kernel_id==kernel_id
        assert owner.kernel.cancelled is not old_token and not owner.kernel.cancelled.is_set()
        assert old_token.is_set()
        assert len(owner.budget.snapshot().reservations)==1
    finally:
        release.set()
        owner.close()
