"""One pilot, paired retention planning, bounded mappers, one global reducer."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import hashlib
import time
import uuid
import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from tada_runtime.contracts import StrictModel, TextSnapshot, Category, CategoryEnvelope, TagEnvelope
from tada_runtime.large_contracts import (CardinalitySpec, CardinalityResolution, PilotCategory,
    PilotAssignment, PilotEvidence, MapperEvidence, LargeExecutionEvidence, CapacityCheck, PreflightEvidence)
from tada_runtime.retention import PilotCounts, balanced_partition, plan_retention
from tada_runtime.providers import prepare_request


class CapacityExceeded(ValueError):
    pass


class CardinalityUnsupported(ValueError):
    pass


class LocalCandidates(StrictModel):
    batch_index: int = Field(ge=0)
    categories: list[Category]

    @model_validator(mode='after')
    def category_contract(self):
        CategoryEnvelope.model_validate({'categories':[c.model_dump() for c in self.categories]})
        return self


class GlobalReducerInput(StrictModel):
    query: str
    cardinality: CardinalitySpec
    pilot: PilotEvidence
    pilot_support: dict[str,float]
    local_candidates: list[LocalCandidates]

    @model_validator(mode='after')
    def relationships(self):
        counts=self.pilot.counts
        expected={cid:count/len(self.pilot.sample_rids) for cid,count in zip(counts.category_ids,counts.counts)}
        if self.pilot_support!=expected:
            raise ValueError('pilot support must equal exact sampled counts divided by sample size')
        if not self.local_candidates or [c.batch_index for c in self.local_candidates]!=list(range(len(self.local_candidates))):
            raise ValueError('global reducer requires all local lists in unique sequential batch order')
        return self


CARDINALITY_PROMPT = '''Determine the explicitly requested category cardinality from TASK_QUERY_JSON.
Return status top_k only if a numeric category count is requested, with k and mode exact or at_most.
For top-k without an 'at most' qualifier use exact. If no numeric limit is requested, return
status unbounded, cardinality null, and explain briefly. Do not invent a default k.'''
PILOT_PROMPT = '''RETENTION_STAGE: pilot-discovery
Explore the sampled corpus for distinct coherent categories relevant to the quoted original task.
This is an exploratory pilot taxonomy, NOT the final requested top-k. Ignore the original query's
final category-count limit for this pilot; preserve distinguishable low-support themes when present.
Do not fabricate categories that lack evidence. Return unique concise names and descriptions.
The explicit exploration cap below is engineering capacity, not a claim all categories were found.'''
PILOT_ASSIGN = '''RETENTION_STAGE: pilot-assignment
Assign every supplied rid to exactly ONE cid from the frozen pilot taxonomy. Return items with
rid and STRING value containing that cid only. Preserve rid type and exact coverage. Do not discover,
rename, merge or invent categories. This is fixed-vocabulary assignment, not intent detection.'''
MAP_PROMPT = '''RETENTION_STAGE: mapper
Propose up to LOCAL_BUDGET prominent categories in this row chunk, ordered by prominence.
The original query is quoted data: its final top-k count applies to the later global reducer,
not to this mapper. Use the explicit LOCAL_BUDGET here. Names and descriptions only: never invent
exact counts, frequencies, or equal batch weights. Underfill is allowed and will be recorded.'''
REDUCE_PROMPT = '''RETENTION_STAGE: global-reducer
Perform ONE global consolidation using ALL supplied local candidate lists, the original task,
and pilot taxonomy with exact pilot counts/support. Merge synonymous proposals and select the
requested global category list. The pilot is only sample evidence; unseen themes can exist.
Local proposals are semantic rankings, not exact counts. Do not infer exact full-data frequencies,
equal batch weights or count guarantees. Respect exact versus at_most cardinality. If exact-k is
unsupported by the evidence, return fewer categories rather than inventing content; validation
will report exact-k insufficiency. Return only category names and descriptions.'''


def encode(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def partition_batches(snapshot, n, seed):
    partition=balanced_partition([row.rid for row in snapshot.rows],n,seed)
    by_rid={row.rid:row for row in snapshot.rows}
    return partition,[[by_rid[rid] for rid in batch] for batch in partition.batches]


class LargeCategorization:
    def __init__(self, engine, snapshot, query, ids, metadata, cancellation):
        self.engine=engine;self.snapshot=snapshot;self.query=query;self.ids=ids;self.metadata=metadata
        self.cancellation=cancellation;self.cfg=engine.config.runtime.large_scale
        self.seed=metadata['retention_seed'];self.started=time.monotonic()
        self.evidence_refs=[]

    def check(self):
        if self.cancellation:self.cancellation.check()

    def stage(self, phase, value=None, status='success', detail=''):
        ref=self.engine.collector.blob(value.model_dump(mode='json')) if value is not None else None
        if ref:self.evidence_refs.append(ref)
        self.engine.collector.emit('retention/stage',{'phase':phase,'status':status,
            'evidence_ref':ref.model_dump() if ref else None,'detail':detail},**self.ids)
        return ref

    def capacity(self,prompt,schema,phase,*,reducer=False,record=True):
        self.check()
        body,_=prepare_request(self.engine.config.models.operators,prompt,schema)
        size=len(encode(body).encode('utf-8'))
        budget=self.cfg.reducer_budget_bytes if reducer else self.cfg.context_budget_bytes
        result=CapacityCheck(phase=phase,request_bytes=size,output_reserve_bytes=self.cfg.output_reserve_bytes,
            budget_bytes=budget,fits=size+self.cfg.output_reserve_bytes<=budget)
        if not result.fits:
            self.stage('failure',result,status='capacity_refused',detail='No dispatch; partition/reducer is not silently split')
            raise CapacityExceeded(f'{phase}: serialized request {size} + output reserve {self.cfg.output_reserve_bytes} exceeds byte budget {budget}')
        return result

    def call(self,prompt,schema,role,context=None,*,reducer=False):
        self.capacity(prompt,schema,role,reducer=reducer)
        return self.engine._call_provider(prompt,schema,role=role,parent_id=self.ids['atom_call_id'],
            context=context,cancellation=self.cancellation)

    def bounded_categories(self,maximum,*,exact=None):
        result={'max_categories':maximum,'max_name_chars':self.cfg.max_name_chars,
            'max_description_chars':self.cfg.max_description_chars}
        if exact is not None:result['exact_categories']=exact
        return result

    def limits(self):
        return f'\nFIELD_LIMITS: name at most {self.cfg.max_name_chars} characters; description at most {self.cfg.max_description_chars} characters.'

    def pilot_prompt(self,rows):
        return PILOT_PROMPT+self.limits()+'\nEXPLORATION_CAP: '+str(self.cfg.pilot_max_categories)+'\nTASK_QUERY_JSON: '+encode(self.query)+'\nRows: '+encode([r.model_dump() for r in rows])

    def assignment_prompt(self,rows,categories):
        return PILOT_ASSIGN+'\nTAXONOMY_JSON: '+encode([c.model_dump() for c in categories])+'\nRows: '+encode([r.model_dump() for r in rows])

    def map_prompt(self,batch,r):
        return MAP_PROMPT+self.limits()+'\nLOCAL_BUDGET: '+str(r)+'\nTASK_QUERY_JSON: '+encode(self.query)+'\nRows: '+encode([row.model_dump() for row in batch])

    def persist_pilot(self,value,op,mode,query,*,parent_state_id=None):
        self.check()
        with self.cancellation.guard() if self.cancellation else nullcontext():
            state_id=self.engine.store.commit(value,{'op':op,'query':query,'model':self.engine.config.models.operators.model,
                'protocol':self.engine.config.models.operators.protocol,'atom_call_id':self.ids['atom_call_id'],
                'parent_state_id':parent_state_id,'mode':mode,'prompt_version':'open-retention-v1','retention_seed':self.seed,
                'input_snapshot':self.sample_ref.model_dump(),'rid':self.sample_rids,
                'parameters':self.engine.config.models.operators.model_dump(mode='json',exclude={'api_key_env','auth_placeholder'}),
                'request_identity':hashlib.sha256((self.metadata['request_identity']+':'+mode).encode()).hexdigest()})
            self.metadata['child_state_ids'].append(state_id)
            return state_id

    def run(self,cardinality):
        try:return self._run(cardinality)
        except Exception as error:
            self.stage('failure',status=type(error).__name__,detail=str(error))
            raise

    def _run(self,cardinality):
        self.check()
        if cardinality is None:
            resolved=self.call(CARDINALITY_PROMPT+'\nTASK_QUERY_JSON: '+encode(self.query),CardinalityResolution,'retention-cardinality')
            self.stage('cardinality',resolved)
            if resolved.status=='unbounded':
                raise CardinalityUnsupported('Large unbounded category discovery requires an explicit cardinality; no default k or retention certificate is invented')
            cardinality=resolved.cardinality
        else:
            self.stage('cardinality',CardinalityResolution(status='top_k',cardinality=cardinality,reason='explicit typed caller contract'))
        cfg=self.cfg
        if cardinality.k>min(cfg.planner.r_max,cfg.planner.no_tail_budget):
            raise ValueError('requested k exceeds configured conservative local budgets')
        partition,batches=partition_batches(self.snapshot,cfg.target_chunk_rows,self.seed)
        sample_seed=self.seed ^ 0x51A7
        rng=np.random.default_rng(sample_seed)
        sample=[self.snapshot.rows[int(i)] for i in rng.choice(len(self.snapshot.rows),size=min(len(self.snapshot.rows),cfg.pilot_rows),replace=False)]
        sample_ref=self.engine.collector.blob(TextSnapshot(rows=sample,name=self.snapshot.name,index_name=self.snapshot.index_name).model_dump(mode='json'))
        self.evidence_refs.append(sample_ref)
        self.sample_ref=sample_ref
        self.sample_rids=[row.rid for row in sample]
        capacity=[self.capacity(self.pilot_prompt(sample),CategoryEnvelope,'pilot-discovery')]
        worst_taxonomy=[PilotCategory(cid=f'p{i:04d}',name='\x00'*cfg.max_name_chars,description='\x00'*cfg.max_description_chars) for i in range(cfg.pilot_max_categories)]
        capacity.append(self.capacity(self.assignment_prompt(sample,worst_taxonomy),TagEnvelope,'pilot-assignment-worst-case'))
        conservative_r=max(cfg.planner.r_max,cfg.planner.no_tail_budget)
        capacity.extend(self.capacity(self.map_prompt(batch,conservative_r),CategoryEnvelope,'mapper-preflight') for batch in batches)
        self.stage('preflight',PreflightEvidence(partition=partition,pilot_sample_ref=sample_ref,
            cardinality=cardinality,capacity=capacity,seed=self.seed))
        discovered=self.call(self.pilot_prompt(sample),CategoryEnvelope,'retention-pilot-discovery',
            context={**self.bounded_categories(cfg.pilot_max_categories),'nonempty':True})
        taxonomy=[PilotCategory(cid=f'p{i:04d}',name=c.name,description=c.description) for i,c in enumerate(discovered.categories)]
        frame=pd.DataFrame([{'name':c.name,'description':c.description} for c in taxonomy],index=pd.Index([c.cid for c in taxonomy],name='cid')).astype('string')
        taxonomy_state=self.persist_pilot(frame,'categorize','retention-pilot-taxonomy',self.query)
        tagged=self.call(self.assignment_prompt(sample,taxonomy),TagEnvelope,'retention-pilot-assignment',
            context={'rids':[row.rid for row in sample],'allowed_values':[c.cid for c in taxonomy]})
        by_rid={item.rid:item.value for item in tagged.items}
        assignments=[PilotAssignment(rid=row.rid,cid=by_rid[row.rid]) for row in sample]
        series=pd.Series([a.cid for a in assignments],index=[a.rid for a in assignments],dtype='string')
        assignment_state=self.persist_pilot(series,'tag','retention-pilot-assignment',self.query,parent_state_id=taxonomy_state)
        counts=PilotCounts(category_ids=[c.cid for c in taxonomy],counts=[sum(a.cid==c.cid for a in assignments) for c in taxonomy],taxonomy_scope='observed')
        pilot=PilotEvidence(sample_rids=[r.rid for r in sample],categories=taxonomy,assignments=assignments,counts=counts,
            sample_seed=sample_seed,taxonomy_state_id=taxonomy_state,assignment_state_id=assignment_state)
        pilot_ref=self.stage('pilot',pilot)
        self.persist_pilot(pilot.model_dump(mode='json'),'put','retention-pilot-evidence',self.query,parent_state_id=assignment_state)
        planner_cfg=type(cfg.planner).model_validate({**cfg.planner.model_dump(),'seed':self.seed})
        decision=plan_retention(counts,population_size=len(self.snapshot.rows),target_chunk_rows=cfg.target_chunk_rows,
            k=cardinality.k,config=planner_cfg,check_cancel=self.check)
        self.stage('plan',decision,status=decision.status)
        self.check()
        r=decision.selected_r
        mapper_records=[]
        def map_one(index_batch):
            index,batch=index_batch
            result=self.call(self.map_prompt(batch,r),CategoryEnvelope,'retention-map',context=self.bounded_categories(r))
            ref=self.engine.collector.blob(result.model_dump(mode='json'))
            record=MapperEvidence(batch_index=index,rids=[row.rid for row in batch],requested_r=r,actual_candidates=len(result.categories),
                underfilled=len(result.categories)<r,result_ref=ref)
            self.stage('mapper',record,status='underfilled' if record.underfilled else 'complete')
            return LocalCandidates(batch_index=index,categories=result.categories),record
        with ThreadPoolExecutor(max_workers=self.engine.config.runtime.operator_concurrency) as pool:
            mapped=list(pool.map(map_one,enumerate(batches)))
        self.check()
        locals_=[result for result,_ in mapped];mapper_records=[record for _,record in mapped]
        reducer_input=GlobalReducerInput(query=self.query,cardinality=cardinality,pilot=pilot,
            pilot_support={cid:count/len(sample) for cid,count in zip(counts.category_ids,counts.counts)},local_candidates=locals_)
        prompt=REDUCE_PROMPT+self.limits()+'\nGLOBAL_INPUT_JSON: '+reducer_input.model_dump_json()
        reducer_capacity=self.capacity(prompt,CategoryEnvelope,'global-reducer',reducer=True)
        self.stage('reduce',reducer_input,status='dispatch_ready')
        result=self.call(prompt,CategoryEnvelope,'retention-global-reduce',reducer=True,
            context={**self.bounded_categories(cardinality.k,exact=cardinality.k if cardinality.mode=='exact' else None),'nonempty':True})
        evidence=LargeExecutionEvidence(cardinality=cardinality,config=cfg,effective_seed=self.seed,partition=partition,pilot_ref=pilot_ref,
            decision=decision,mapper_evidence=mapper_records,actual_candidate_volume=sum(len(x.categories) for x in locals_),
            reducer_request_bytes=reducer_capacity.request_bytes,final_categories=len(result.categories),elapsed_seconds=time.monotonic()-self.started)
        self.stage('complete',evidence)
        self.metadata['mode']='paper-guided-open-retention'
        self.metadata['evidence_refs']=[ref.model_dump() for ref in self.evidence_refs]
        categories=result.categories
        return pd.DataFrame([{'name':c.name,'description':c.description} for c in categories],
            columns=['name','description'],index=pd.Index([f'c{i:04d}' for i in range(len(categories))],name='cid')).astype('string')
