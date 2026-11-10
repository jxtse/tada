"""Explicit engineering policy and validated evidence for open semantic retention."""
from __future__ import annotations
from typing import Literal
from pydantic import Field, StrictInt, StrictStr, model_validator
from .audit_contracts import AuditModel, BlobRef
from .retention import RetentionConfig, PilotCounts, BalancedPartitionPlan, RetentionDecision


class CardinalitySpec(AuditModel):
    k: int = Field(ge=1,le=1000)
    mode: Literal['exact','at_most']


class CardinalityResolution(AuditModel):
    status: Literal['top_k','unbounded']
    cardinality: CardinalitySpec | None
    reason: str

    @model_validator(mode='after')
    def coherent(self):
        if (self.status=='top_k') != (self.cardinality is not None):
            raise ValueError('top_k requires cardinality; unbounded must not invent k')
        return self


class LargeScaleConfig(AuditModel):
    enabled: bool = True
    threshold_rows: int = Field(default=1000,ge=1)
    target_chunk_rows: int = Field(default=1000,ge=1,le=10000)
    pilot_rows: int = Field(default=128,ge=1,le=1000)
    pilot_max_categories: int = Field(default=32,ge=1,le=1000)
    context_budget_bytes: int = Field(default=200000,ge=1000)
    reducer_budget_bytes: int = Field(default=200000,ge=1000)
    output_reserve_bytes: int = Field(default=32768,ge=0)
    max_name_chars: int = Field(default=128,ge=1,le=10000)
    max_description_chars: int = Field(default=512,ge=1,le=10000)
    planner: RetentionConfig = Field(default_factory=RetentionConfig)
    capacity_policy: Literal['serialized-utf8-plus-output-reserve-v1'] = 'serialized-utf8-plus-output-reserve-v1'
    pilot_policy: Literal['explore-without-final-k-v1'] = 'explore-without-final-k-v1'
    reducer_policy: Literal['one-global-call-or-refuse-v1'] = 'one-global-call-or-refuse-v1'
    general_discovery_policy: Literal['refuse-unbounded-large-v1'] = 'refuse-unbounded-large-v1'
    engine_version: Literal['open-retention-v1'] = 'open-retention-v1'


class PilotCategory(AuditModel):
    cid: str
    name: str
    description: str = ''


class PilotAssignment(AuditModel):
    rid: StrictStr | StrictInt
    cid: str


class PilotEvidence(AuditModel):
    sample_rids: list[StrictStr | StrictInt]
    categories: list[PilotCategory]
    assignments: list[PilotAssignment]
    counts: PilotCounts
    sample_seed: int = Field(ge=0,le=2**63-1)
    taxonomy_state_id: str
    assignment_state_id: str
    @model_validator(mode='after')
    def consistent(self):
        cids=[c.cid for c in self.categories]
        if not self.sample_rids or not cids or any(not c.cid or not c.name.strip() for c in self.categories):
            raise ValueError('pilot sample and taxonomy must be nonempty')
        if len({type(rid) for rid in self.sample_rids})>1 or '' in self.sample_rids:
            raise ValueError('pilot rid domain must be nonempty and type stable')
        if len(cids)!=len(set(cids)) or cids!=self.counts.category_ids:
            raise ValueError('pilot taxonomy and counts must share a unique canonical order')
        rids=[a.rid for a in self.assignments]
        if len(self.sample_rids)!=len(set(self.sample_rids)) or rids!=self.sample_rids:
            raise ValueError('pilot assignments must cover sample exactly in sample order')
        observed={cid:0 for cid in cids}
        for assignment in self.assignments:
            if assignment.cid not in observed:raise ValueError('invalid pilot category reference')
            observed[assignment.cid]+=1
        if list(observed.values())!=self.counts.counts:
            raise ValueError('pilot counts must equal symbolic assignment counts')
        return self


class MapperEvidence(AuditModel):
    batch_index: int = Field(ge=0)
    rids: list[StrictStr | StrictInt]
    requested_r: int = Field(ge=1)
    actual_candidates: int = Field(ge=0)
    underfilled: bool
    result_ref: BlobRef
    @model_validator(mode='after')
    def bounded(self):
        if not self.rids or len(set(self.rids))!=len(self.rids) or '' in self.rids or len({type(rid) for rid in self.rids})>1:
            raise ValueError('mapper rid domain must be unique, nonempty and type stable')
        if self.actual_candidates>self.requested_r or self.underfilled!=(self.actual_candidates<self.requested_r):
            raise ValueError('mapper budget/underfill mismatch')
        return self


class LargeExecutionEvidence(AuditModel):
    cardinality: CardinalitySpec
    config: LargeScaleConfig
    effective_seed: int
    partition: BalancedPartitionPlan
    pilot_ref: BlobRef
    decision: RetentionDecision
    mapper_evidence: list[MapperEvidence]
    actual_candidate_volume: int
    reducer_request_bytes: int
    final_categories: int
    elapsed_seconds: float = Field(ge=0)
    semantic_guarantee: Literal['conditional-retention-guidance-only'] = 'conditional-retention-guidance-only'
    @model_validator(mode='after')
    def volume(self):
        request=self.decision.request
        if self.partition.geometry!=request.geometry or self.partition.seed!=self.effective_seed or request.config.seed!=self.effective_seed:
            raise ValueError('partition, planner and effective seed/geometry must agree')
        if request.config.model_dump()!={**self.config.planner.model_dump(),'seed':self.effective_seed}:
            raise ValueError('effective planner configuration mismatch')
        if request.k!=self.cardinality.k or self.partition.geometry.target_chunk_rows!=self.config.target_chunk_rows:
            raise ValueError('cardinality or frozen chunk rows mismatch')
        if [m.batch_index for m in self.mapper_evidence]!=list(range(self.partition.geometry.B)):
            raise ValueError('mapper records must cover every batch once in plan order')
        for mapper,batch in zip(self.mapper_evidence,self.partition.batches):
            if mapper.rids!=batch or mapper.requested_r!=self.decision.selected_r:
                raise ValueError('mapper coverage or selected budget mismatch')
        if sum(m.actual_candidates for m in self.mapper_evidence)!=self.actual_candidate_volume:
            raise ValueError('actual candidate volume mismatch')
        if self.final_categories>self.cardinality.k or (self.cardinality.mode=='exact' and self.final_categories!=self.cardinality.k):
            raise ValueError('final cardinality mismatch')
        return self


class CapacityCheck(AuditModel):
    phase: str
    request_bytes: int = Field(ge=0)
    output_reserve_bytes: int = Field(ge=0)
    budget_bytes: int = Field(ge=1)
    fits: bool
    @model_validator(mode='after')
    def correct(self):
        if self.fits!=(self.request_bytes+self.output_reserve_bytes<=self.budget_bytes):
            raise ValueError('capacity outcome inconsistent')
        return self


class PreflightEvidence(AuditModel):
    partition: BalancedPartitionPlan
    pilot_sample_ref: BlobRef
    cardinality: CardinalitySpec
    capacity: list[CapacityCheck]
    seed: int
