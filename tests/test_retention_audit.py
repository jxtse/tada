"""Offline evidence validation never treats a content hash as a schema proof."""
import hashlib

import pytest
from pydantic import ValidationError

from tada_runtime.audit import AuditLog, export_html, export_jsonl, validate_session
from tada_runtime.contracts import TextSnapshot
from tada_runtime.large_contracts import CardinalitySpec, MapperEvidence, PreflightEvidence
from tada_runtime.retention import PilotCounts, RetentionConfig, balanced_partition, plan_retention
from tada_runtime.state import Store


def contents(root):
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file()}


def deny_network(monkeypatch):
    def denied(*args,**kwargs):
        raise AssertionError('offline validation attempted a socket')
    monkeypatch.setattr('socket.socket',denied)
    monkeypatch.setattr('socket.create_connection',denied)


@pytest.mark.parametrize('phase',['cardinality','preflight','pilot','plan','mapper','reduce','complete','failure'])
def test_correct_hash_does_not_validate_malformed_typed_stage(tmp_path,monkeypatch,phase):
    with AuditLog(tmp_path) as log:
        ref = log.blob({'arbitrary':'JSON with a correct content hash'})
        log.append('retention/stage',{'phase':phase,'status':'success','evidence_ref':ref})
    before = contents(tmp_path)
    deny_network(monkeypatch)
    with pytest.raises(ValidationError): validate_session(tmp_path)
    assert contents(tmp_path) == before


def test_plan_revalidates_jeffreys_and_mc_receipt(tmp_path):
    decision = plan_retention(PilotCounts(category_ids=['a','b'],counts=[6,4],taxonomy_scope='fixed'),
        population_size=10,target_chunk_rows=5,k=1,config=RetentionConfig(outer_draws=2,inner_draws=2,r_max=2))
    raw = decision.model_dump(mode='json')
    raw['posterior_parameters'] = [7.0,5.0]
    with AuditLog(tmp_path) as log:
        log.append('retention/stage',{'phase':'plan','status':'selected','evidence_ref':log.blob(raw)})
    with pytest.raises(ValidationError,match='Jeffreys'): validate_session(tmp_path)


def test_nested_mapper_result_requires_category_schema_and_actual_volume(tmp_path):
    with AuditLog(tmp_path) as log:
        ref = log.blob({'categories':[]})
        record = MapperEvidence(batch_index=0,rids=['r1'],requested_r=2,actual_candidates=1,
            underfilled=True,result_ref=ref)
        log.append('retention/stage',{'phase':'mapper','status':'underfilled','evidence_ref':log.blob(record.model_dump())})
    with pytest.raises(ValueError,match='Mapper result count'): validate_session(tmp_path)


def test_nested_mapper_result_does_not_accept_other_valid_json(tmp_path):
    with AuditLog(tmp_path) as log:
        ref = log.blob({'items':[{'rid':'r1','value':'a'}]})
        record = MapperEvidence(batch_index=0,rids=['r1'],requested_r=2,actual_candidates=1,
            underfilled=True,result_ref=ref)
        log.append('retention/stage',{'phase':'mapper','status':'underfilled','evidence_ref':log.blob(record.model_dump())})
    with pytest.raises(ValidationError): validate_session(tmp_path)


def test_nested_preflight_snapshot_hash_and_rid_coverage(tmp_path):
    with AuditLog(tmp_path) as log:
        sample = log.blob(TextSnapshot.model_validate({'rows':[{'rid':'outside','text':'x'}]}).model_dump())
        evidence = PreflightEvidence(partition=balanced_partition(['r1','r2'],1,0),
            pilot_sample_ref=sample,cardinality=CardinalitySpec(k=1,mode='exact'),capacity=[],seed=0)
        log.append('retention/stage',{'phase':'preflight','status':'success','evidence_ref':log.blob(evidence.model_dump())})
    with pytest.raises(ValueError,match='outside the balanced partition'): validate_session(tmp_path)


def test_state_metadata_evidence_refs_are_verified_even_without_stage(tmp_path):
    with AuditLog(tmp_path) as log:
        ref = log.blob({'untyped':'evidence'})
        Store(log).commit({'result':1},{'evidence_refs':[ref]})
    with pytest.raises(ValueError,match='supported typed retention artifact'): validate_session(tmp_path)


def test_state_evidence_nested_blob_hash_not_hidden_by_metadata_hash(tmp_path):
    with AuditLog(tmp_path) as log:
        ref = log.blob(TextSnapshot.model_validate({'rows':[{'rid':'r1','text':'x'}]}).model_dump())
        Store(log).commit({'result':1},{'evidence_refs':[ref]})
    (tmp_path/ref.path).write_text('changed')
    with pytest.raises(ValueError,match='Artifact integrity mismatch'): validate_session(tmp_path)


def test_valid_typed_evidence_socket_disabled_exports_preserve_source(tmp_path,monkeypatch):
    root = tmp_path/'source'
    with AuditLog(root) as log:
        ref = log.blob(TextSnapshot.model_validate({'rows':[{'rid':'r1','text':'x'}]}).model_dump())
        preflight = PreflightEvidence(partition=balanced_partition(['r1','r2'],1,0),
            pilot_sample_ref=ref,cardinality=CardinalitySpec(k=1,mode='exact'),capacity=[],seed=0)
        typed_ref = log.blob(preflight.model_dump())
        log.append('retention/stage',{'phase':'preflight','status':'success','evidence_ref':typed_ref})
        log.append('retention/stage',{'phase':'failure','status':'ExecutionCancelled','detail':'cancelled'})
        Store(log).commit({'result':1},{'evidence_refs':[ref,typed_ref]})
    before = contents(root)
    deny_network(monkeypatch)
    assert validate_session(root).committed_states == 1
    export_html(root,tmp_path/'review.html')
    export_jsonl(root,tmp_path/'review.jsonl')
    assert 'retention/stage' in (tmp_path/'review.html').read_text()
    assert contents(root) == before


def test_successful_retention_stage_requires_evidence(tmp_path):
    with AuditLog(tmp_path) as log:
        log.append('retention/stage',{'phase':'plan','status':'selected'})
    with pytest.raises(ValueError,match='missing typed evidence'): validate_session(tmp_path)
