"""Reproducible Pydantic JSON Schema generation; --check rejects drift."""
import argparse
from pathlib import Path
from typing import Annotated, Literal
from pydantic import Field
from .audit import canonical, digest
from .config import Config
from .contracts import StrictModel, ExecuteArgs, TextSnapshot, TagEnvelope, CategoryEnvelope, IntentEnvelope, NormalizedResponse, OperatorIdentity
from .audit_contracts import EVENT_ADAPTER, Checkpoint, ValidationReport
from .storage_contracts import SNAPSHOT_ADAPTER, StateMetadata, StateManifest
from .rpc_contracts import RpcRequest, RpcSuccess, RpcFailure, Observation, KernelRequest, PARAMS, RESULTS
from .budget import LedgerState
from .receipts import AcceptanceReceipt
from .large_scale_receipts import LargeScaleReceipt
from .retention import (RetentionConfig, PilotCounts, BalancedGeometry, BalancedPartitionPlan,
    RetentionRequest, TheoremDiagnostic, CandidateRetentionDiagnostic, RetentionDecision, CountTable)
from .large_contracts import (CardinalitySpec, CardinalityResolution, LargeScaleConfig, PilotCategory,
    PilotAssignment, PilotEvidence, MapperEvidence, LargeExecutionEvidence, CapacityCheck, PreflightEvidence)
from tada_operators.large import GlobalReducerInput, LocalCandidates
from tada_operators.retention_evaluation import (FixedPopulation, PartitionRequest, EvaluationConfig,
    TrialRecord, RecoveryMetric, PairedComparison, EvaluationCase, EvaluationReport)
from .cli import DoctorReceipt

class SchemaManifest(StrictModel):
    schema_version: Literal['1.0'] = '1.0'
    hashes: dict[str, Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]]

def documents():
    models={'schema_manifest':SchemaManifest,'doctor_receipt':DoctorReceipt,'config':Config,'execute_python':ExecuteArgs,'text_snapshot':TextSnapshot,'tag':TagEnvelope,
            'category':CategoryEnvelope,'intent':IntentEnvelope,'normalized_response':NormalizedResponse,
            'acceptance_receipt':AcceptanceReceipt,'log_validation':ValidationReport,'operator_identity':OperatorIdentity,'checkpoint':Checkpoint,'state':StateMetadata,
            'state_manifest':StateManifest,'rpc_request':RpcRequest,'rpc_success':RpcSuccess,
            'rpc_failure':RpcFailure,'observation':Observation,'kernel_request':KernelRequest,'ledger':LedgerState}
    models.update({'rpc_'+name:model for name,model in PARAMS.items()})
    models.update({name: model for name, model in [
        ('large_scale_receipt', LargeScaleReceipt), ('retention_config', RetentionConfig),
        ('pilot_counts', PilotCounts), ('balanced_geometry', BalancedGeometry),
        ('balanced_partition', BalancedPartitionPlan), ('retention_request', RetentionRequest),
        ('theorem_diagnostic', TheoremDiagnostic), ('candidate_retention_diagnostic', CandidateRetentionDiagnostic),
        ('retention_decision', RetentionDecision), ('count_table', CountTable),
        ('cardinality_spec', CardinalitySpec), ('cardinality_resolution', CardinalityResolution),
        ('large_scale_config', LargeScaleConfig), ('pilot_category', PilotCategory),
        ('pilot_assignment', PilotAssignment), ('pilot_evidence', PilotEvidence),
        ('mapper_evidence', MapperEvidence), ('large_execution_evidence', LargeExecutionEvidence),
        ('capacity_check', CapacityCheck), ('preflight_evidence', PreflightEvidence),
        ('global_reducer_input', GlobalReducerInput), ('local_candidates', LocalCandidates),
        ('fixed_population', FixedPopulation), ('partition_request', PartitionRequest),
        ('retention_evaluation_config', EvaluationConfig), ('retention_trial', TrialRecord),
        ('recovery_metric', RecoveryMetric), ('paired_comparison', PairedComparison),
        ('retention_evaluation_case', EvaluationCase), ('retention_evaluation_report', EvaluationReport)]})
    values={name:model.model_json_schema() for name,model in models.items()}
    values.update({'rpc_'+name+'_result':adapter.json_schema() for name,adapter in RESULTS.items()})
    values.update(event=EVENT_ADAPTER.json_schema(),snapshot=SNAPSHOT_ADAPTER.json_schema())
    return values

def generate(root=Path('schemas'),check=False):
    values=documents()
    values['manifest']=SchemaManifest(hashes={name:digest(canonical(schema)) for name,schema in sorted(values.items())}).model_dump(mode='json')
    for name,schema in sorted(values.items()):
        path=root/f'{name}.json';body=canonical(schema)+b'\n'
        if check:
            if not path.exists() or path.read_bytes()!=body:raise RuntimeError(f'schema drift: {path}')
        else:
            root.mkdir(parents=True,exist_ok=True);path.write_bytes(body)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    generate(check=parser.parse_args().check)
