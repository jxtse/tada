"""Typed, revision-bound evidence for the paper-guided retention implementation."""
from typing import Literal
from pydantic import Field, model_validator
from .contracts import StrictModel
from .receipts import CommandResult, Gate, RouteEvidence


class EvidenceArtifact(StrictModel):
    path: str
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    description: str


class LargeScaleReceipt(StrictModel):
    schema_version: Literal['1.0'] = '1.0'
    report_kind: Literal['self-report-pending-independent-review'] = 'self-report-pending-independent-review'
    baseline_revision: str
    accepted_core_tree_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    final_tree_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    changed_since_accepted_core: list[str]
    dependencies: dict[str, str]
    commands: list[CommandResult]
    gates: dict[Literal['numeric_contract', 'runtime_integration', 'controlled_scale', 'live_functional', 'core_regression'], Gate]
    artifacts: list[EvidenceArtifact]
    provider_requests_before: Literal[52] = 52
    provider_requests_after: int = Field(ge=52, le=60)
    provider_request_cap: Literal[60] = 60
    original_reservations_unchanged: Literal[True] = True
    model_protocol_evidence: list[RouteEvidence] = Field(default_factory=list)
    limitations: list[str]

    @model_validator(mode='after')
    def gate_coverage(self):
        expected = {'numeric_contract', 'runtime_integration', 'controlled_scale', 'live_functional', 'core_regression'}
        if set(self.gates) != expected:
            raise ValueError('large-scale receipt must cover all verification gates')
        return self
