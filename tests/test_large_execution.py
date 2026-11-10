"""Large open-semantic pipeline through real provider contracts and offline HTTP."""
import json
import threading
from types import SimpleNamespace
import httpx
import pandas as pd
import pytest
from pydantic import ValidationError
from tada_runtime.audit import AuditLog,read_events,verify_blob
from tada_runtime.budget import BudgetLedger
from tada_runtime.config import Config
from tada_runtime.cancellation import ExecutionCancellation,ExecutionCancelled
from tada_runtime.providers import Provider
from tada_runtime.state import Store,snapshot,restore
from tada_runtime.supervisor import Supervisor
from tada_runtime.rpc_contracts import KernelRequest,Initialize
from tada_runtime.large_contracts import LargeExecutionEvidence,PilotEvidence,CardinalitySpec
from tada_operators.engine import Engine
from tada_operators.large import CapacityExceeded,CardinalityUnsupported


def config(tmp_path,**large):
    defaults={'threshold_rows':4,'target_chunk_rows':3,'pilot_rows':6,'pilot_max_categories':4,
        'max_name_chars':64,'max_description_chars':64,'output_reserve_bytes':1000,
        'context_budget_bytes':100000,'reducer_budget_bytes':100000,
        'planner':{'outer_draws':4,'inner_draws':4,'r_max':4,'no_tail_budget':4,'seed':123}}
    defaults.update(large)
    return Config.model_validate({'models':{role:{'base_url':'http://127.0.0.1:1/v1','model':'offline','protocol':'openai-responses','auth_mode':'optional'} for role in ['agent','operators']},
        'runtime':{'batch_rows':3,'large_scale':defaults},
        'budget':{'ledger_path':str(tmp_path/'offline-ledger.json'),'max_requests':60,'deadline_seconds':120.0}})


def decode_field(prompt,marker):
    return json.loads(prompt.split(marker,1)[1].split('\nReturn only one JSON',1)[0])


class FixtureHTTP:
    def __init__(self,*,joint=False,overfill=False,underfill=False,final_count=2,no_tail=False):
        self.calls=[];self.global_inputs=[];self.joint=joint;self.overfill=overfill;self.underfill=underfill;self.final_count=final_count;self.no_tail=no_tail
        self.on_phase=lambda phase:None
    def __call__(self,request):
        prompt=json.loads(request.content)['input']
        if 'RETENTION_STAGE: pilot-discovery' in prompt:
            phase='pilot-discovery';data={'categories':[{'name':'T'+str(i)} for i in range(2 if self.no_tail else 3)]}
        elif 'RETENTION_STAGE: pilot-assignment' in prompt:
            phase='pilot-assignment';rows=decode_field(prompt,'\nRows: ')
            data={'items':[{'rid':r['rid'],'value':f'p{int(r["rid"][1:])%(2 if self.no_tail else 3):04d}'} for r in reversed(rows)]}
        elif 'RETENTION_STAGE: mapper' in prompt:
            phase='map';r=int(prompt.split('\nLOCAL_BUDGET: ',1)[1].split('\n',1)[0]);size=r+1 if self.overfill else (1 if self.underfill else min(r,3))
            data={'categories':[{'name':'T'+str(i)} for i in range(size)]}
        elif 'RETENTION_STAGE: global-reducer' in prompt:
            phase='reduce';self.global_inputs.append(decode_field(prompt,'\nGLOBAL_INPUT_JSON: '));data={'categories':[{'name':'T'+str(i)} for i in range(self.final_count)]}
        elif 'Determine the explicitly requested' in prompt:
            phase='cardinality';data={'status':'top_k','cardinality':{'k':2,'mode':'exact'},'reason':'fixture numeric request'}
        elif 'intent-classification' in prompt:
            phase='intent';data={'mode':'joint','domain_query':'Discover exactly 2 topics'} if self.joint else {'mode':'independent','domain_query':None}
        else:
            phase='assignment';rows=decode_field(prompt,'\nRows: ');data={'items':[{'rid':row['rid'],'value':'T0'} for row in rows]}
        self.calls.append(phase);self.on_phase(phase)
        return httpx.Response(200,json={'model':'offline','status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(data)}]}],
            'usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2}})


def owner_for(tmp_path,http,**kwargs):
    cfg=config(tmp_path,**kwargs);owner=Supervisor();owner.config=cfg;owner.audit=AuditLog(tmp_path/'session');owner.store=Store(owner.audit)
    owner.kernel=SimpleNamespace(cancelled=ExecutionCancellation(),active_execution='fixture-execution')
    owner.budget=BudgetLedger(cfg.budget,owner.audit)
    owner.provider=Provider(cfg.models.operators,owner.budget,owner.audit,transport=httpx.MockTransport(http))
    owner.engine=Engine(cfg,owner.provider,owner.store,owner.audit)
    return owner


def request(op='categorize',n=7,cardinality={'k':2,'mode':'exact'}):
    return KernelRequest(op=op,snapshot=snapshot(pd.Series(['text']*n,index=[f'r{i}' for i in range(n)])).model_dump(mode='json'),
        query='Discover exactly 2 topics',cardinality=cardinality)


def evidence(owner):
    event=next(e for e in reversed(owner.audit.events) if e.type=='retention/stage' and e.payload.phase=='complete')
    return LargeExecutionEvidence.model_validate_json(verify_blob(owner.audit.root,event.payload.evidence_ref))


def test_pilot_balanced_maps_single_global_and_typed_evidence(tmp_path):
    http=FixtureHTTP();owner=owner_for(tmp_path,http)
    try:
        result=restore(owner.kernel_call(request()))
        assert result['name'].tolist()==['T0','T1']
        assert http.calls==['pilot-discovery','pilot-assignment','map','map','map','reduce']
        observed=evidence(owner)
        sizes=[len(batch) for batch in observed.partition.batches]
        assert sizes==[3,2,2] and observed.partition.geometry.n_min==2
        pilot=PilotEvidence.model_validate_json(verify_blob(owner.audit.root,observed.pilot_ref))
        assert sum(pilot.counts.counts)==6 and len(pilot.assignments)==6
        assert len(http.global_inputs)==1 and len(http.global_inputs[0]['local_candidates'])==3
        assert http.global_inputs[0]['pilot']['counts']==pilot.counts.model_dump()
        assert observed.actual_candidate_volume==sum(x.actual_candidates for x in observed.mapper_evidence)
        final=owner.store.list()[-1]
        assert final.mode=='paper-guided-open-retention' and len(final.child_state_ids)==3 and final.evidence_refs
        assert owner.store.get(final.state_id).index.tolist()==['c0000','c0001']
    finally:owner.audit.close()


def test_large_missing_cardinality_is_counted_and_small_no_pilot(tmp_path):
    http=FixtureHTTP();owner=owner_for(tmp_path,http)
    try:
        owner.kernel_call(request(cardinality=None))
        assert http.calls[0]=='cardinality' and http.calls.count('cardinality')==1
        assert len(owner.budget.snapshot().reservations)==len(http.calls)
    finally:owner.audit.close()


def test_independent_tag_no_retention_joint_uses_frozen_domain(tmp_path):
    for joint in (False,True):
        path=tmp_path/str(joint);path.mkdir();http=FixtureHTTP(joint=joint);owner=owner_for(path,http)
        try:
            result=restore(owner.kernel_call(request(op='tag')))
            assert result.index.tolist()==[f'r{i}' for i in range(7)] and result.tolist()==['T0']*7
            assert http.calls.count('pilot-discovery')==int(joint)
            assert http.calls.count('pilot-assignment')==int(joint)
            assert http.calls.count('reduce')==int(joint)
            if joint:assert http.calls[-3:]==['assignment']*3
        finally:owner.audit.close()


@pytest.mark.parametrize('mode,raises',[('exact',True),('at_most',False)])
def test_final_underfill_and_mapper_underfill_are_distinct(tmp_path,mode,raises):
    http=FixtureHTTP(underfill=True,final_count=1);owner=owner_for(tmp_path,http)
    try:
        if raises:
            with pytest.raises(ValidationError):owner.kernel_call(request(cardinality={'k':2,'mode':mode}))
            assert not any(s.mode=='paper-guided-open-retention' for s in owner.store.list())
        else:
            result=restore(owner.kernel_call(request(cardinality={'k':2,'mode':mode})))
            assert len(result)==1 and all(m.underfilled for m in evidence(owner).mapper_evidence)
    finally:owner.audit.close()


def test_mapper_overfill_fails_before_global_reduce(tmp_path):
    http=FixtureHTTP(overfill=True);owner=owner_for(tmp_path,http)
    try:
        with pytest.raises(ValidationError):owner.kernel_call(request())
        assert 'reduce' not in http.calls
        assert not any(s.mode=='paper-guided-open-retention' for s in owner.store.list())
    finally:owner.audit.close()


@pytest.mark.parametrize('boundary',['preflight','reducer'])
def test_capacity_refusal_never_dispatches_over_capacity(tmp_path,boundary):
    http=FixtureHTTP();limits={'context_budget_bytes':1000} if boundary=='preflight' else {'reducer_budget_bytes':1000}
    owner=owner_for(tmp_path,http,**limits)
    try:
        with pytest.raises(CapacityExceeded):owner.kernel_call(request())
        assert 'reduce' not in http.calls
        if boundary=='preflight':assert http.calls==[] and owner.store.list()==[]
        else:assert http.calls[:2]==['pilot-discovery','pilot-assignment'] and len(owner.store.list())==3
        assert any(e.type=='retention/stage' and e.payload.status=='capacity_refused' for e in owner.audit.events)
    finally:owner.audit.close()


def test_no_tail_budget_not_shrunk_to_pilot_domain(tmp_path):
    http=FixtureHTTP(no_tail=True);owner=owner_for(tmp_path,http)
    try:
        owner.kernel_call(request())
        report=evidence(owner)
        assert report.decision.selected_r==4 and report.decision.status=='fallback_no_tail'
        assert len(report.decision.diagnostics)==0
        assert all(m.requested_r==4 for m in report.mapper_evidence)
    finally:owner.audit.close()


def test_identity_reuse_all_policy_seed_and_fresh_changes(tmp_path):
    http=FixtureHTTP();owner=owner_for(tmp_path,http)
    try:
        owner.kernel_call(request());before=len(http.calls);first=evidence(owner)
        owner.kernel_call(request());assert len(http.calls)==before
        owner.config.runtime.large_scale.planner.seed+=1
        owner.kernel_call(request());assert len(http.calls)>before
        changed=evidence(owner);assert first.effective_seed!=changed.effective_seed
        before=len(http.calls);fresh=request();fresh.fresh=True
        owner.kernel_call(fresh);assert len(http.calls)>before and evidence(owner).effective_seed!=changed.effective_seed
        before=len(http.calls);owner.config.runtime.large_scale.pilot_max_categories+=1
        owner.kernel_call(request());assert len(http.calls)>before
    finally:owner.audit.close()


@pytest.mark.parametrize('phase',['pilot-discovery','pilot-assignment','map','reduce'])
def test_cancelled_pipeline_preserves_paid_evidence_never_final_commits(tmp_path,phase):
    http=FixtureHTTP();owner=owner_for(tmp_path,http)
    http.on_phase=lambda current:owner.kernel.cancelled.set() if current==phase else None
    try:
        with pytest.raises(ExecutionCancelled):owner.kernel_call(request())
        assert http.calls[-1]==phase
        assert not any(s.mode=='paper-guided-open-retention' for s in owner.store.list())
        assert all(r.status=='success' for r in owner.budget.snapshot().reservations)
    finally:owner.audit.close()


def test_general_large_discovery_explicitly_refuses_without_invented_k(tmp_path):
    class Unbounded(FixtureHTTP):
        def __call__(self,request):
            prompt=json.loads(request.content)['input']
            if 'Determine the explicitly requested' in prompt:
                self.calls.append('cardinality')
                data={'status':'unbounded','cardinality':None,'reason':'No numeric request'}
                return httpx.Response(200,json={'model':'offline','status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(data)}]}]})
            return super().__call__(request)
    http=Unbounded();owner=owner_for(tmp_path,http)
    try:
        with pytest.raises(CardinalityUnsupported):owner.kernel_call(request(cardinality=None))
        assert http.calls==['cardinality'] and owner.store.list()==[]
    finally:owner.audit.close()


def test_empty_final_at_most_rejected(tmp_path):
    http=FixtureHTTP(final_count=0);owner=owner_for(tmp_path,http)
    try:
        with pytest.raises(ValidationError):owner.kernel_call(request(cardinality={'k':2,'mode':'at_most'}))
    finally:owner.audit.close()


def test_cpu_cancellation_blocks_all_maps_and_final_commit(tmp_path,monkeypatch):
    import tada_operators.large as large
    original=large.plan_retention
    http=FixtureHTTP();owner=owner_for(tmp_path,http)
    def cancelled(*args,**kwargs):
        owner.kernel.cancelled.set()
        return original(*args,**kwargs)
    monkeypatch.setattr(large,'plan_retention',cancelled)
    try:
        with pytest.raises(ExecutionCancelled):owner.kernel_call(request())
        assert http.calls==['pilot-discovery','pilot-assignment']
        assert len(owner.store.list())==3
    finally:owner.audit.close()


def test_pilot_invalid_reference_rejected_before_planning(tmp_path):
    class InvalidPilot(FixtureHTTP):
        def __call__(self,req):
            prompt=json.loads(req.content)['input']
            if 'RETENTION_STAGE: pilot-assignment' in prompt:
                self.calls.append('pilot-assignment')
                rows=decode_field(prompt,'\nRows: ')
                data={'items':[{'rid':r['rid'],'value':'unknown-cid'} for r in rows]}
                return httpx.Response(200,json={'model':'offline','status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(data)}]}]})
            return super().__call__(req)
    http=InvalidPilot();owner=owner_for(tmp_path,http)
    try:
        with pytest.raises(ValidationError):owner.kernel_call(request())
        assert http.calls==['pilot-discovery','pilot-assignment']
        assert len(owner.store.list())==1
        assert not any(e.type=='retention/stage' and e.payload.phase=='plan' for e in owner.audit.events)
    finally:owner.audit.close()


@pytest.mark.parametrize('joint',[False,True])
def test_real_jupyter_bridge_runs_large_categorize_and_joint(tmp_path,monkeypatch,joint):
    cfg=config(tmp_path)
    monkeypatch.setattr('tada_runtime.supervisor.load_config',lambda *a,**k:cfg)
    owner=Supervisor();http=FixtureHTTP(joint=joint)
    try:
        owner.initialize(Initialize(config_path='unused-fixture',session_dir=str(tmp_path/'session')))
        owner.provider.transport=httpx.MockTransport(http)
        op='tag' if joint else 'categorize'
        result=owner.kernel.execute("texts=pd.Series(['text']*7,index=[f'r{i}' for i in range(7)])\n"+
            f"result={op}(texts,'Discover exactly 2 topics',name='large_result',cardinality={{'k':2,'mode':'exact'}})\nprint(result.shape)")
        assert result.status=='ok',result.model_dump()
        assert http.calls.count('pilot-discovery')==1 and http.calls.count('pilot-assignment')==1
        assert http.calls.count('reduce')==1
        saved=owner.store.get('large_result')
        assert len(saved)==(7 if joint else 2)
        followup=owner.kernel.execute("print(state.get('large_result').equals(result))")
        assert followup.status=='ok' and followup.stdout.strip()=='True'
        assert len(owner.budget.snapshot().reservations)==len(http.calls)
    finally:owner.close()


def test_typed_cardinality_and_small_explicit_contract():
    from tada_runtime.contracts import CategoryEnvelope
    with pytest.raises(ValidationError):CardinalitySpec(k=True,mode='exact')
    with pytest.raises(ValidationError):CardinalitySpec(k=2,mode='approximately')
    with pytest.raises(ValidationError):CategoryEnvelope.model_validate({'categories':[{'name':'one'}]},context={'exact_categories':2})
    assert len(CategoryEnvelope.model_validate({'categories':[{'name':'one'}]},context={'max_categories':2}).categories)==1


def test_joint_parent_identity_tracks_child_planner_policy(tmp_path):
    http=FixtureHTTP(joint=True);owner=owner_for(tmp_path,http)
    try:
        owner.kernel_call(request(op='tag'));before=len(http.calls)
        owner.kernel_call(request(op='tag'));assert len(http.calls)==before
        owner.config.runtime.large_scale.planner.seed+=7
        owner.kernel_call(request(op='tag'))
        assert len(http.calls)>before and http.calls.count('pilot-discovery')==2
    finally:owner.audit.close()


def test_small_categorize_stays_single_call_without_cardinality_planning(tmp_path):
    class Small(FixtureHTTP):
        def __call__(self,req):
            prompt=json.loads(req.content)['input']
            assert 'Discover corpus-level categories' in prompt and 'RETENTION_STAGE' not in prompt
            self.calls.append('generic-small')
            return httpx.Response(200,json={'model':'offline','status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'{"categories":[{"name":"T0"},{"name":"T1"}]}'}]}]})
    http=Small();owner=owner_for(tmp_path,http)
    try:
        result=restore(owner.kernel_call(request(n=3,cardinality=None)))
        assert len(result)==2 and http.calls==['generic-small']
        assert not any(e.type=='retention/stage' for e in owner.audit.events)
    finally:owner.audit.close()


def test_pilot_evidence_rejects_forged_symbolic_count(tmp_path):
    http=FixtureHTTP();owner=owner_for(tmp_path,http)
    try:
        owner.kernel_call(request())
        report=evidence(owner)
        pilot=PilotEvidence.model_validate_json(verify_blob(owner.audit.root,report.pilot_ref))
        forged=pilot.model_dump();forged['counts']['counts'][0]+=1
        with pytest.raises(ValidationError):PilotEvidence.model_validate(forged)
    finally:owner.audit.close()


def test_correctly_hashed_relational_evidence_rejects_forgery(tmp_path):
    from tada_operators.large import GlobalReducerInput
    http=FixtureHTTP();owner=owner_for(tmp_path,http)
    try:
        owner.kernel_call(request())
        report=evidence(owner)
        reducer=http.global_inputs[0]
        forged=json.loads(json.dumps(reducer));forged['pilot_support'][next(iter(forged['pilot_support']))]=.99
        with pytest.raises(ValidationError):GlobalReducerInput.model_validate(forged)
        forged=json.loads(json.dumps(reducer));forged['local_candidates'][1]['batch_index']=0
        with pytest.raises(ValidationError):GlobalReducerInput.model_validate(forged)
        for change in ('seed','budget','rid','count'):
            forged=report.model_dump()
            if change=='seed':forged['effective_seed']+=1
            elif change=='budget':forged['mapper_evidence'][0]['requested_r']+=1
            elif change=='rid':forged['mapper_evidence'][0]['rids'][0]='unknown-rid'
            else:forged['mapper_evidence'].pop()
            with pytest.raises(ValidationError):LargeExecutionEvidence.model_validate(forged)
    finally:owner.audit.close()
