import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tada_runtime.audit import AuditLog, export_html, read_events, validate_session, canonical
from tada_runtime.audit_contracts import EVENT_ADAPTER


def request(log,identity='r1'):
    ref=log.blob({'model':'fixture','messages':[]})
    return log.append('model/request',{'role':'agent','requested_model':'fixture','protocol':'anthropic-messages','request_ref':ref},request_id=identity)


def test_typed_fragments_partial_recovery_and_read_only(tmp_path):
    with AuditLog(tmp_path) as log:
        request(log)
        for i,channel in enumerate(['reasoning','text','tool_call']):
            log.append('model/stream',{'channel':channel,'text':'fragment','provider_sequence':i},request_id='r1')
        log.append('tool/call',{'arguments':'{"code":invalid'},tool_call_id='t1')
    before=(tmp_path/'events.jsonl').read_bytes()
    with (tmp_path/'events.jsonl').open('ab') as file:
        file.write(b'{"torn":')
    damaged=(tmp_path/'events.jsonl').read_bytes()
    report=validate_session(tmp_path)
    assert report.torn_tail and report.unknown_requests==['r1']
    assert (tmp_path/'events.jsonl').read_bytes()==damaged
    with AuditLog(tmp_path,resume=True) as log:
        events=log.events
        assert [e.payload.channel for e in events if e.type=='model/stream']==['reasoning','text','tool_call']
        assert any(e.type=='recovery/interrupted' and e.payload.status=='unknown' for e in events)
        assert (tmp_path/'events.jsonl').read_bytes().startswith(before)
    assert not validate_session(tmp_path).torn_tail


def test_lock_dedup_validation_and_flush_fail_closed(tmp_path,monkeypatch):
    with AuditLog(tmp_path) as log:
        with pytest.raises(RuntimeError,match='write owner'):
            AuditLog(tmp_path,resume=True)
        a=log.append('user/message',{'text':'hello'},source_event_id='same')
        b=log.append('user/message',{'text':'hello'},source_event_id='same')
        assert a.seq==b.seq==1
        with pytest.raises(ValueError,match='Conflicting'):
            log.append('user/message',{'text':'changed'},source_event_id='same')
        with pytest.raises(ValidationError):
            log.append('user/message',{'text':'hello','extra':True})
        monkeypatch.setattr('tada_runtime.audit.os.fsync',lambda *_: (_ for _ in ()).throw(OSError('disk full')))
        with pytest.raises(OSError):
            log.append('user/message',{'text':'failure'})
        with pytest.raises(RuntimeError,match='dispatch prohibited'):
            log.assert_healthy()


def test_redaction_html_escape_and_no_network(tmp_path,monkeypatch):
    secret='fixture-secret-sensitive'
    with AuditLog(tmp_path,secrets=[secret]) as log:
        ref=log.blob({'Authorization':'Bearer secret','usage':{'input_tokens':100},'text':secret})
        log.append('model/stream',{'channel':'reasoning','text':f'{secret}<script>fetch("https://invalid.example")</script>','raw_ref':ref})
        log.append('user/message',{'text':secret})
    monkeypatch.setattr('socket.socket',lambda *_a,**_k: (_ for _ in ()).throw(AssertionError('network forbidden')))
    output=tmp_path/'views'/'view.html'
    export_html(tmp_path,output)
    data=output.read_text()
    assert secret not in data and 'Bearer secret' not in data
    assert '<script>fetch' not in data and '&lt;script&gt;' in data
    assert 'input_tokens' in data and '100' in data
    assert any(e.redacted for e in read_events(tmp_path)[0])


def test_checkpoint_sequence_projection_and_no_mutation(tmp_path):
    with AuditLog(tmp_path) as log:
        log.append('user/message',{'text':'one'})
        checkpoint=log.checkpoint([{'role':'user','content':'one'}])
        assert checkpoint.event_seq==1
        assert log.load_checkpoint()==checkpoint
        log.append('user/message',{'text':'two'})
        with pytest.raises(ValueError,match='later conversation'):
            log.load_checkpoint()
        log.checkpoint([{'role':'user','content':'one'},{'role':'user','content':'two'}])
    path=tmp_path/'pi-checkpoints'/'latest.json'
    original=path.read_bytes()
    data=json.loads(original)
    data['event_seq']=999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='sequence'):
        AuditLog(tmp_path,resume=True)
    path.write_bytes(original)
    with AuditLog(tmp_path,resume=True):
        pass


def test_strict_event_negative_and_missing_blob(tmp_path):
    with AuditLog(tmp_path) as log:
        event=log.append('user/message',{'text':'valid'})
        body=event.model_dump()
        for field,value in [('schema_version','2.0'),('seq',True),('type','invalid')]:
            with pytest.raises(ValidationError):
                EVENT_ADAPTER.validate_python({**body,field:value})
        with pytest.raises(ValidationError):
            log.append('model/request',{'role':'agent','requested_model':'x','protocol':'x','request_ref':log.blob({}),'parameters':{'bad':float('nan')}})
        ref=log.blob({'ok':True})
        log.append('artifact/register',{'artifact':ref})
    (tmp_path/ref.path).write_bytes(b'bad')
    with pytest.raises(ValueError,match='integrity'):
        validate_session(tmp_path)


def test_complete_invalid_tail_is_not_silently_discarded(tmp_path):
    with AuditLog(tmp_path) as log:
        log.append('user/message',{'text':'valid'})
    with (tmp_path/'events.jsonl').open('ab') as file:
        file.write(b'{"type":"unsupported"}')
    with pytest.raises(ValidationError):
        read_events(tmp_path)


def test_completed_record_without_newline_is_preserved(tmp_path):
    with AuditLog(tmp_path) as log:
        log.append('user/message',{'text':'valid'})
    path=tmp_path/'events.jsonl'
    path.write_bytes(path.read_bytes().rstrip(b'\n'))
    with AuditLog(tmp_path,resume=True) as log:
        assert len(log.events)==1
        log.append('user/message',{'text':'next'})
    assert len(read_events(tmp_path)[0])==2


def test_concurrent_collectors_preserve_one_total_order(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    with AuditLog(tmp_path) as log:
        def append(i):
            return log.append('model/stream',{'channel':'text','text':str(i),'provider_sequence':i},
                              source_event_id=f'producer:{i}',request_id='request').seq
        with ThreadPoolExecutor(max_workers=2) as executor:
            seqs=list(executor.map(append,range(20)))
        assert sorted(seqs)==list(range(1,21))
    events,_=read_events(tmp_path)
    assert [event.seq for event in events]==list(range(1,21))
    assert {event.payload.text for event in events}=={str(i) for i in range(20)}


def test_jsonl_export_is_readonly_and_preserves_full_attempt(tmp_path):
    from tada_runtime.audit import export_jsonl
    with AuditLog(tmp_path) as log:
        request(log)
        log.append('model/stream',{'channel':'text','text':'partial'},request_id='r1')
        log.append('model/attempt_end',{'status':'unknown'},request_id='r1')
    original=(tmp_path/'events.jsonl').read_bytes()
    output=tmp_path/'views'/'review.jsonl'
    export_jsonl(tmp_path,output)
    assert output.read_bytes()==original
    assert (tmp_path/'events.jsonl').read_bytes()==original


@pytest.mark.parametrize('relative',['events.jsonl','session.json','states/snapshot.json','blobs/data','inputs/reviews.json','pi-checkpoints/latest.json','.writer.lock'])
def test_exports_never_overwrite_source_session(tmp_path,relative):
    from tada_runtime.audit import export_jsonl
    with AuditLog(tmp_path) as log:
        log.append('user/message',{'text':'preserve'})
    original=(tmp_path/'events.jsonl').read_bytes()
    for exporter in (export_html,export_jsonl):
        with pytest.raises(ValueError,match='authoritative session data'):
            exporter(tmp_path,tmp_path/relative)
    assert (tmp_path/'events.jsonl').read_bytes()==original


def test_export_symlink_cannot_escape_derived_views(tmp_path):
    from tada_runtime.audit import export_jsonl
    with AuditLog(tmp_path) as log:
        log.append('user/message',{'text':'preserve'})
    (tmp_path/'states').mkdir()
    (tmp_path/'views').symlink_to(tmp_path/'states',target_is_directory=True)
    with pytest.raises(ValueError,match='authoritative session data'):
        export_jsonl(tmp_path,tmp_path/'views'/'overwrite.jsonl')



def test_html_view_is_bounded_but_complete_record_stays_authoritative(tmp_path):
    """Thousands of raw transport chunks must not balloon the readable view; nothing is lost on disk."""
    with AuditLog(tmp_path) as log:
        log.append('model/request',{'role':'agent','requested_model':'m','protocol':'anthropic-messages','request_ref':log.blob({'messages':[]})},request_id='q1')
        for i in range(3000):
            log.append('model/stream',{'channel':'provider_raw','text':'','provider_sequence':i,'availability':'opaque','raw_ref':log.blob_bytes(b'x'*4000,media_type='application/octet-stream')},request_id='q1')
        for i,piece in enumerate(['Hello',' world']):
            log.append('model/stream',{'channel':'text','text':piece,'provider_sequence':3000+i},request_id='q1')
    small=tmp_path/'views'/'small.html';full=tmp_path/'views'/'full.html'
    export_html(tmp_path,small);export_html(tmp_path,full,include_transport=True)
    text=small.read_text()
    assert small.stat().st_size<200_000<full.stat().st_size
    assert '3000 raw chunks' in text and 'Hello world' in text
    assert sum(1 for _ in open(tmp_path/'events.jsonl'))>=3003
