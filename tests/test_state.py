import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pydantic import ValidationError

from tada_runtime.audit import AuditLog, atomic_write, canonical
from tada_runtime.state import Store,snapshot,restore
from tada_runtime.storage_contracts import SNAPSHOT_ADAPTER, CommitMetadata


@pytest.mark.parametrize('value',[
    pd.Series(['x','y'],index=pd.Index(['r2','r1'],name='rid'),name='tag',dtype='string'),
    pd.Series([1,None],dtype='Int64',name='count'),
    pd.Series([True,False],index=[3,1],name='valid'),
    pd.Series(pd.Categorical(['low','high'],categories=['low','high'],ordered=True)),
    pd.DataFrame({'text':['a','b'],'count':[1,2],'value':[1.2,float('nan')],'flag':[True,False]},index=['r2','r1']),
    {'value':['safe',3,True,None]}, None,
])
def test_safe_snapshot_roundtrip(value):
    encoded=snapshot(value)
    decoded=restore(SNAPSHOT_ADAPTER.validate_json(encoded.model_dump_json()))
    if isinstance(value,pd.Series):
        assert_series_equal(value,decoded)
    elif isinstance(value,pd.DataFrame):
        assert_frame_equal(value,decoded)
    else:
        assert value==decoded


def test_commit_immutability_alias_recovery_and_orphan(tmp_path):
    frame=pd.DataFrame({'name':['Shipping','Quality']},index=pd.Index(['c1','c2'],name='cid'))
    with AuditLog(tmp_path) as log:
        store=Store(log)
        state_id=store.commit(frame,{'op':'categorize','request_identity':'hash','rid':['r1','r2'],'query':'Discover topics'})
        store.alias('topics',state_id)
        frame.loc['c1','name']='changed'
        assert store.get('topics').loc['c1','name']=='Shipping'
        assert store.find('hash')==state_id
        second=store.put('topics',{'changed':True})
        assert store.get('topics')=={'changed':True}
        assert store.get(state_id).loc['c1','name']=='Shipping'
    atomic_write(tmp_path/'states'/'orphan.data.json',canonical(snapshot('not committed')))
    with AuditLog(tmp_path,resume=True) as log:
        store=Store(log)
        assert store.get('topics')=={'changed':True}
        assert store.find('hash')==state_id
        assert list(store.get(state_id).index)==['c1','c2']
        assert store.orphans==['states/orphan.data.json']
        assert any(e.type=='recovery/interrupted' and e.payload.status=='orphan' for e in log.events)
        assert len(store.list())==2


def test_snapshot_and_metadata_reject_unsafe_values():
    with pytest.raises(ValidationError):
        snapshot({'bad':float('inf')})
    with pytest.raises(ValidationError):
        snapshot(lambda:None)
    with pytest.raises(ValueError):
        snapshot(pd.DataFrame({'x':[1,2]},index=pd.MultiIndex.from_tuples([('a',1),('b',2)])))
    for rid in [['x','x'],['x',1],[True],['']]:
        with pytest.raises(ValidationError):
            CommitMetadata(rid=rid)
    with pytest.raises(ValidationError):
        SNAPSHOT_ADAPTER.validate_json('{"kind":"invalid"}')


def test_offline_validation_checks_committed_data(tmp_path):
    from tada_runtime.audit import validate_session
    with AuditLog(tmp_path) as log:
        store=Store(log)
        metadata=store.put('result',pd.Series(['a'],index=['r1']))
    validate_session(tmp_path)
    (tmp_path/metadata.data_ref.path).write_text('corrupt')
    with pytest.raises(ValueError,match='integrity'):
        validate_session(tmp_path)


def test_state_metadata_references_and_secret_keys(tmp_path):
    secret='fixture-credential-key'
    with AuditLog(tmp_path,secrets=[secret]) as log:
        store=Store(log)
        with pytest.raises(ValueError,match='uncommitted'):
            store.commit('value',{'child_state_ids':['missing']})
        meta=store.put(secret,{secret:'value'})
        assert secret not in (tmp_path/'states'/'manifest.json').read_text()
        assert secret not in (tmp_path/meta.data_ref.path).read_text()
        assert store.get(meta.state_id)=={'[REDACTED]':'value'}
