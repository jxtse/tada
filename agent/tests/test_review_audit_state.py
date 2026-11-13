"""Regression evidence for independent review findings R3, R5, and R6."""
from pathlib import Path
import pandas as pd
import pytest
from tada_runtime.audit import AuditLog, _pending, export_html, export_jsonl, load_checkpoint, read_events, validate_session
from tada_runtime.rpc_contracts import Initialize, RpcRequest
from tada_runtime.state import Store, snapshot, restore
from tada_runtime.supervisor import Supervisor


def test_r3_stale_checkpoint_remains_forensically_readable(tmp_path):
    session=tmp_path/'session'
    with AuditLog(session) as audit:
        audit.append('session/start',{'status':'open'})
        audit.checkpoint({'messages':[],'pi_entries':[]})
        audit.append('user/message',{'text':'durable interrupted follow-up'},turn_id='turn-2')
    original=(session/'events.jsonl').read_bytes()
    report=validate_session(session)
    assert report.valid and not report.resumable
    assert report.checkpoint_status=='stale' and report.resume_warnings
    export_html(session,tmp_path/'review.html')
    export_jsonl(session,tmp_path/'review.jsonl')
    assert 'durable interrupted follow-up' in (tmp_path/'review.html').read_text()
    assert (tmp_path/'review.jsonl').read_bytes()==original
    assert (session/'events.jsonl').read_bytes()==original
    with pytest.raises(ValueError,match='later conversation'):
        AuditLog(session,resume=True)
    assert (session/'events.jsonl').read_bytes()==original


def test_r5_successful_compaction_is_not_recovered_as_interrupted(tmp_path):
    session=tmp_path/'session'
    with AuditLog(session) as audit:
        owner=Supervisor();owner.audit=audit
        owner.dispatch(RpcRequest(id=1,method='compaction',params={'phase':'start','messages':[]}))
        owner.dispatch(RpcRequest(id=2,method='compaction',params={'phase':'end','messages':[],'summary':'complete','status':'success'}))
        audit.checkpoint({'messages':[],'pi_entries':[]})
        assert not _pending(audit.events)
        events=[event for event in audit.events if event.type.startswith('compaction/')]
        assert len({event.compaction_id for event in events})==1 and events[0].compaction_id
    for _ in range(2):
        with AuditLog(session,resume=True) as audit:
            assert not any(event.type=='recovery/interrupted' for event in audit.events)


def test_r6_categorical_index_store_roundtrip_preserves_unused_categories(tmp_path):
    value=pd.Series([1,2],index=pd.CategoricalIndex(['b','a'],categories=['c','b','a'],ordered=True,name='rid'))
    with AuditLog(tmp_path/'session') as audit:
        store=Store(audit)
        meta=store.put('categories',value)
        pd.testing.assert_series_equal(store.get(meta.state_id),value)
    with AuditLog(tmp_path/'session',resume=True) as audit:
        pd.testing.assert_series_equal(Store(audit).get('categories'),value)


def test_r6_categorical_index_actual_kernel_bridge_roundtrip(tmp_path):
    config=tmp_path/'config.toml'
    config.write_text('''[models.agent]
base_url="http://127.0.0.1:1/v1"
model="offline-fixture"
protocol="anthropic-messages"
auth_mode="optional"
[models.operators]
base_url="http://127.0.0.1:1/v1"
model="offline-fixture"
protocol="openai-responses"
auth_mode="optional"
[budget]
ledger_path="'''+str(tmp_path/'offline-ledger.json')+'''"
max_requests=1
deadline_seconds=60.0
''')
    owner=Supervisor()
    try:
        owner.initialize(Initialize(config_path=str(config),session_dir=str(tmp_path/'session')))
        observation=owner.kernel.execute("value = pd.Series([1, 2], index=pd.CategoricalIndex(['b', 'a'], categories=['c', 'b', 'a'], ordered=True, name='rid'))\nstate.put('category_index', value)\nrestored = state.get('category_index')\npd.testing.assert_series_equal(value, restored)\nprint('bridge roundtrip verified')")
        assert observation.status=='ok',observation.exception
        assert observation.stdout.strip()=='bridge roundtrip verified'
        assert not owner.budget.snapshot().reservations
    finally:
        owner.close()


def test_r3_first_turn_without_checkpoint_and_torn_tail_export_readonly(tmp_path,monkeypatch):
    session=tmp_path/'session'
    with AuditLog(session) as audit:
        audit.append('user/message',{'text':'first durable turn'},turn_id='first')
        audit.append('model/request',{'role':'agent','requested_model':'fixture','protocol':'anthropic-messages','request_ref':audit.blob({'messages':[]})},request_id='pending')
        audit.append('model/stream',{'channel':'text','text':'partial retained'},request_id='pending')
    with (session/'events.jsonl').open('ab') as stream:
        stream.write(b'{"unfinished":')
    before={str(path.relative_to(session)):path.read_bytes() for path in session.rglob('*') if path.is_file()}
    monkeypatch.setattr('socket.socket',lambda *_a,**_k: (_ for _ in ()).throw(AssertionError('offline reader attempted network')))
    report=validate_session(session)
    assert report.valid and report.torn_tail and not report.resumable
    assert report.checkpoint_status=='missing' and report.unknown_requests==['pending']
    export_html(session,tmp_path/'partial.html')
    export_jsonl(session,tmp_path/'partial.jsonl')
    assert 'partial retained' in (tmp_path/'partial.html').read_text()
    assert b'partial retained' in (tmp_path/'partial.jsonl').read_bytes()
    assert b'unfinished' not in (tmp_path/'partial.jsonl').read_bytes()
    assert before=={str(path.relative_to(session)):path.read_bytes() for path in session.rglob('*') if path.is_file()}


def test_r3_invalid_checkpoint_is_separate_from_event_integrity(tmp_path):
    session=tmp_path/'session'
    with AuditLog(session) as audit:
        audit.append('user/message',{'text':'valid events survive invalid projection'})
        audit.checkpoint({'messages':[],'pi_entries':[]})
    path=session/'pi-checkpoints'/'latest.json'
    path.write_text('{"damaged":true}')
    before=path.read_bytes()
    report=validate_session(session)
    assert report.valid and not report.resumable and report.checkpoint_status=='invalid'
    export_html(session,tmp_path/'invalid-projection.html')
    assert 'valid events survive invalid projection' in (tmp_path/'invalid-projection.html').read_text()
    with pytest.raises(ValueError):
        AuditLog(session,resume=True)
    assert path.read_bytes()==before


def test_r5_interrupted_compaction_recovers_exactly_once(tmp_path):
    session=tmp_path/'session'
    with AuditLog(session) as audit:
        audit.checkpoint({'messages':[],'pi_entries':[]})
        owner=Supervisor();owner.audit=audit
        start=owner.dispatch(RpcRequest(id=1,method='compaction',params={'phase':'start','messages':[]}))
        identity=start['compaction_id']
        assert identity
    before=(session/'events.jsonl').read_bytes()
    report=validate_session(session)
    export_html(session,tmp_path/'interrupted-compaction.html')
    assert report.valid and (session/'events.jsonl').read_bytes()==before
    for _ in range(2):
        with AuditLog(session,resume=True) as audit:
            recovered=[event for event in audit.events if event.type=='recovery/interrupted' and event.payload.interrupted_type=='compaction/start']
            assert len(recovered)==1
            assert recovered[0].payload.interrupted_id==identity
            assert recovered[0].compaction_id==identity
            assert not _pending(audit.events)


def test_r5_compaction_rejects_overlapping_or_unmatched_lifecycle(tmp_path):
    with AuditLog(tmp_path/'session') as audit:
        owner=Supervisor();owner.audit=audit
        with pytest.raises(ValueError,match='active compaction'):
            owner.dispatch(RpcRequest(id=1,method='compaction',params={'phase':'end','messages':[]}))
        owner.dispatch(RpcRequest(id=2,method='compaction',params={'phase':'start','messages':[]}))
        with pytest.raises(ValueError,match='already in progress'):
            owner.dispatch(RpcRequest(id=3,method='compaction',params={'phase':'start','messages':[]}))


@pytest.mark.parametrize('categories',[
    pd.Index([3,2,1],dtype='int32'),
    pd.Index(['unused','b','a'],dtype='string'),
    pd.DatetimeIndex(['2026-01-03','2026-01-02','2026-01-01']),
])
def test_r6_categorical_index_category_dtype_is_preserved(categories):
    index=pd.CategoricalIndex([categories[1],categories[2]],categories=categories,ordered=True,name='rid')
    value=pd.DataFrame({'value':[1,2]},index=index)
    pd.testing.assert_frame_equal(value,restore(snapshot(value)))


@pytest.mark.parametrize('as_frame', [False, True])
@pytest.mark.parametrize('unused_timestamp', [False, True])
def test_r6_lossy_object_timestamp_categories_rejected_before_commit(tmp_path, as_frame, unused_timestamp):
    timestamp = pd.Timestamp('2020-01-01')
    categories = pd.Index(['used', timestamp], dtype=object) if unused_timestamp else pd.Index(
        [pd.Timestamp('2020-01-02'), timestamp], dtype=object)
    label = 'used' if unused_timestamp else timestamp
    index = pd.CategoricalIndex([label], categories=categories, ordered=True, name='rid')
    value = pd.DataFrame({'value': [1]}, index=index) if as_frame else pd.Series([1], index=index)
    with AuditLog(tmp_path/'session') as audit:
        store = Store(audit)
        with pytest.raises(ValueError, match='CategoricalIndex categories cannot be restored losslessly'):
            store.put('lossy_categories', value)
        assert not store.list() and not store.aliases
        assert not any(event.type in {'state/commit', 'state/alias'} for event in audit.events)
        assert not list((audit.root/'states').glob('*.data.json'))


def test_r6_json_scalar_object_categories_still_roundtrip():
    categories = pd.Index(['unused', 1, 'used'], dtype=object)
    value = pd.Series([2, 3], index=pd.CategoricalIndex(
        [1, 'used'], categories=categories, ordered=True, name='rid'))
    pd.testing.assert_series_equal(value, restore(snapshot(value)))


def test_r6_legacy_categorical_index_fails_explicitly_instead_of_guessing():
    encoded={'kind':'series','index':{'name':'rid','dtype':'category','values':['b','a'],'range':None},
             'column':{'name':None,'dtype':'int64','values':[1,2],'categories':None,'ordered':False}}
    with pytest.raises(ValueError,match='lacks categories/order metadata'):
        restore(encoded)
