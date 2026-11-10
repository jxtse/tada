"""Revision-bound acceptance self-report, never a substitute for evidence."""
from typing import Literal
from pydantic import Field, model_validator
from .contracts import StrictModel

class CommandResult(StrictModel):
    command: str
    exit_code: int
    evidence_path: str
    assertions: list[str] = Field(default_factory=list)

class Gate(StrictModel):
    status: Literal['passed','partial','blocked','pending','failed']
    evidence_paths: list[str]
    assertions: list[str]
    blockers: list[str] = Field(default_factory=list)

class RouteEvidence(StrictModel):
    role: str
    requested_model: str
    reported_models: list[str]
    protocol: str
    endpoint: str
    evidence_paths: list[str]
    scope: str

class AcceptanceReceipt(StrictModel):
    schema_version: Literal['1.0'] = '1.0'
    report_kind: Literal['self-report-pending-independent-review'] = 'self-report-pending-independent-review'
    baseline_revision: str
    dirty_tree_sha256: str
    changed_files: list[str]
    dependencies: dict[str,str]
    input_hashes: dict[str,str]
    commands: list[CommandResult]
    gates: dict[Literal['G0','G1','G2','G3','G4'],Gate]
    provider_requests: int
    failed_attempts: int
    unknown_attempts: int
    model_protocol_evidence: list[RouteEvidence]
    failed_atoms: int = Field(default=0, ge=0)
    provider_request_roles: dict[str, int] = Field(default_factory=dict)
    remaining_blockers: list[str]

    @model_validator(mode='after')
    def complete_gates(self):
        if set(self.gates) != {'G0','G1','G2','G3','G4'}:
            raise ValueError('acceptance requires all five gates')
        if not 0 <= self.provider_requests <= 60:
            raise ValueError('acceptance request count outside authorized ceiling')
        return self
