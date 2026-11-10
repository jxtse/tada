import json
import re
import pandas as pd
import pytest
from pydantic import ValidationError
from tada_operators import Engine, snapshot_series
from tada_runtime.audit import AuditLog, read_events
from tada_runtime.state import Store
from tada_runtime.config import Config
from tada_runtime.contracts import TagEnvelope, CategoryEnvelope, IntentEnvelope

class FixtureProvider:
    def __init__(self, joint=False, malformed=False):
        self.calls=[]
        self.joint=joint
        self.malformed=malformed
    def call(self,prompt,schema,*,role,parent_id,context=None):
        self.calls.append((role,parent_id))
        if schema is IntentEnvelope:
            data={'mode':'joint','domain_query':'discover topics'} if self.joint else {'mode':'independent','domain_query':None}
        elif schema is CategoryEnvelope:
            data={'categories':[{'name':'Shipping'},{'name':'Service'}]}
        else:
            data={'items':[{'rid':rid,'value': {} if self.malformed else ('Shipping' if self.joint else 'positive')} for rid in reversed(context['rids'])]}
        return schema.model_validate(data,context=context)

@pytest.fixture
def setup(tmp_path):
    cfg=Config.model_validate({'models':{role:{'base_url':'http://localhost:9999/v1','model':'exact','protocol':'openai-responses','auth_mode':'optional'} for role in ['agent','operators']},'runtime':{'batch_rows':2,'reduce_fan_in':2}})
    audit=AuditLog(tmp_path/'session')
    store=Store(audit)
    yield cfg,audit,store
    audit.close()


def test_empty_invalid_and_alignment(setup):
    cfg,audit,store=setup
    provider=FixtureProvider()
    engine=Engine(cfg,provider,store,audit)
    assert engine.tag(pd.Series([],dtype='string'),'x').empty
    assert engine.categorize(pd.Series([],dtype='string'),'x').empty
    assert provider.calls==[]
    for series in [pd.Series(['a','b'],index=['x','x']),pd.Series(['a',None],index=['x','y'])]:
        with pytest.raises(ValidationError):engine.tag(series,'x')
    assert provider.calls==[]
    result=engine.tag(pd.Series(['a','b','c'],index=['r3','r1','r2']),'sentiment')
    assert result.index.tolist()==['r3','r1','r2'] and result.tolist()==['positive']*3
    assert [x[0] for x in provider.calls]==['intent','tag-batch','tag-batch']


def test_reuse_changes_fresh_alias_and_immutability(setup):
    cfg,audit,store=setup
    provider=FixtureProvider()
    engine=Engine(cfg,provider,store,audit)
    source=pd.Series(['a','b'],index=['r1','r2'])
    result=engine.tag(source,'sentiment',name='sentiment')
    state_id=result.attrs['state_id']
    result.iloc[0]='mutated'
    assert store.get(state_id).iloc[0]=='positive'
    before=len(provider.calls)
    hit=engine.tag(source,'sentiment',name='renamed')
    assert len(provider.calls)==before and hit.attrs['state_id']==state_id
    assert store.get('renamed').iloc[0]=='positive'
    source.iloc[0]='changed'
    engine.tag(source,'sentiment')
    assert len(provider.calls)>before
    before=len(provider.calls)
    cfg.runtime.batch_rows=1
    engine.tag(source,'sentiment')
    assert len(provider.calls)>before
    before=len(provider.calls)
    cfg.models.operators.model='new-model'
    engine.tag(source,'sentiment')
    assert len(provider.calls)>before
    before=len(provider.calls)
    engine.tag(source,'sentiment',fresh=True)
    assert len(provider.calls)>before
    before=len(provider.calls)
    cfg.runtime.reuse='off'
    engine.tag(source,'sentiment')
    assert len(provider.calls)>before


def test_joint_child_state_and_frozen_vocabulary(setup):
    cfg,audit,store=setup
    provider=FixtureProvider(joint=True)
    engine=Engine(cfg,provider,store,audit)
    tags=engine.tag(pd.Series(['a','b','c'],index=[1,2,3]),'discover themes')
    assert tags.tolist()==['Shipping']*3
    roles=[x[0] for x in provider.calls]
    assert roles==['intent','categorize-map','categorize-map','categorize-reduce','tag-batch','tag-batch']
    events=read_events(audit.root)[0]
    start=next(e for e in events if e.type=='atom/start' and e.payload.op=='tag')
    child=next(e for e in events if e.type=='atom/start' and e.payload.op=='categorize')
    assert child.parent_id==start.atom_call_id
    end=next(e for e in events if e.type=='atom/end' and e.payload.op=='tag')
    cats=store.get(end.payload.child_state_ids[0])
    assert cats.index.name=='cid' and cats.index.is_unique and cats['name'].tolist()==['Shipping','Service']


def test_batch_failure_never_commits_partial(setup):
    cfg,audit,store=setup
    engine=Engine(cfg,FixtureProvider(malformed=True),store,audit)
    with pytest.raises(ValidationError):engine.tag(pd.Series(['a','b','c'],index=[1,2,3]),'sentiment')
    assert not any(e.type=='state/commit' for e in read_events(audit.root)[0])


def test_categorization_hierarchical_reduce_and_restore(setup):
    cfg,audit,store=setup
    provider=FixtureProvider()
    engine=Engine(cfg,provider,store,audit)
    result=engine.categorize(pd.Series(['x']*10,index=list(range(10))),'themes',name='themes')
    assert sum(role=='categorize-reduce' for role,_ in provider.calls)==4
    restored=store.get(result.attrs['state_id'])
    pd.testing.assert_frame_equal(result,restored)


class PromptRecordingProvider(FixtureProvider):
    """Overfills unless the prompt carries the typed cardinality, like a real model would."""
    def __init__(self):
        super().__init__();self.prompts=[]
    def call(self,prompt,schema,*,role,parent_id,context=None):
        import json,re
        self.prompts.append((role,prompt));self.calls.append((role,parent_id))
        match=re.search(r'CARDINALITY_JSON: (\{[^\n]*\})',prompt)
        n=json.loads(match.group(1))['k'] if match else 9
        return schema.model_validate({'categories':[{'name':f'C{i}'} for i in range(n)]},context=context)


@pytest.mark.parametrize('mode',['exact','at_most'])
def test_generic_categorize_states_cardinality_in_every_prompt(setup,mode):
    """Typed cardinality must reach map and reduce prompts, not only post-hoc validation."""
    cfg,audit,store=setup
    provider=PromptRecordingProvider()
    result=Engine(cfg,provider,store,audit).categorize(pd.Series(['x']*6,index=list(range(6))),'themes',cardinality={'k':3,'mode':mode})
    assert len(result)==3
    assert {role for role,_ in provider.prompts}=={'categorize-map','categorize-reduce'}
    for role,prompt in provider.prompts:
        field=json.loads(re.search(r'CARDINALITY_JSON: (\{[^\n]*\})',prompt).group(1))
        assert field=={'k':3,'mode':mode}, role


def test_generic_categorize_without_cardinality_has_no_cardinality_field(setup):
    cfg,audit,store=setup
    provider=PromptRecordingProvider()
    Engine(cfg,provider,store,audit).categorize(pd.Series(['x']*4,index=list(range(4))),'themes')
    assert all('CARDINALITY_JSON: ' not in prompt for _,prompt in provider.prompts)
