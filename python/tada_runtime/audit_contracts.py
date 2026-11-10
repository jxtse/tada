"""Authoritative typed event vocabulary. Raw vendor data is explicitly opaque."""
from __future__ import annotations

import math
from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, StrictStr, TypeAdapter, create_model, model_validator


class AuditModel(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', allow_inf_nan=False)

    @model_validator(mode='after')
    def finite_json(self):
        def check(value):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('non-finite JSON value')
            if isinstance(value, dict):
                for child in value.values():
                    check(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    check(child)
        check(self.model_dump())
        return self


class BlobRef(AuditModel):
    path: str
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    size: int = Field(ge=0)
    media_type: str = 'application/json'
    redacted: bool = False


class ErrorInfo(AuditModel):
    kind: str
    message: str
    details: list[str] = Field(default_factory=list)
    raw_ref: BlobRef | None = None


class Usage(AuditModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)


class LifecyclePayload(AuditModel):
    status: str = ''
    reason: str = ''
    revision: str = ''
    config_ref: BlobRef | None = None
    versions: dict[str, str] = Field(default_factory=dict)
    mode: Literal['trusted-local'] = 'trusted-local'
    boundary_seq: int | None = None
    transient_variables_lost: list[str] = Field(default_factory=list)
    max_unflushed_seconds: float = 0.0
    producer_redaction_tail_chars: int = Field(default=0, ge=0)
    producer_timeout_seconds: float = Field(default=0.0, ge=0)


class UserPayload(AuditModel):
    text: str
    origin: Literal['human', 'system'] = 'human'
    attachments: list[BlobRef] = Field(default_factory=list)


class RequestPayload(AuditModel):
    role: str
    requested_model: str
    protocol: str
    request_ref: BlobRef
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    context_refs: list[BlobRef] = Field(default_factory=list)
    attempt: int = Field(default=1, ge=1)


class StreamPayload(AuditModel):
    channel: Literal['reasoning', 'text', 'tool_call', 'observation', 'provider_raw']
    text: str = ''
    provider_sequence: int = Field(default=0, ge=0)
    availability: Literal['exposed', 'summary_only', 'opaque', 'not_provided', 'redacted'] = 'not_provided'
    raw_ref: BlobRef | None = None
    block_index: int | None = None
    fragment_offset: int = Field(default=0, ge=0)
    fragment_length: int | None = Field(default=None, ge=0)
    redacted: bool = False


class TextBlock(AuditModel):
    type: Literal['text'] = 'text'
    text: str


class ReasoningBlock(AuditModel):
    type: Literal['reasoning'] = 'reasoning'
    text: str = ''
    availability: Literal['exposed', 'summary_only', 'opaque', 'not_provided', 'redacted']


class ToolBlock(AuditModel):
    type: Literal['tool_call'] = 'tool_call'
    call_id: str
    name: str
    arguments: str


ContentBlock = Annotated[TextBlock | ReasoningBlock | ToolBlock, Field(discriminator='type')]


class ResponsePayload(AuditModel):
    requested_model: str
    reported_model: str | None = None
    protocol: str
    content: list[ContentBlock] = Field(default_factory=list)
    response_ref: BlobRef | None = None
    usage: Usage | None = None
    finish_reason: str = ''
    reasoning_availability: Literal['exposed', 'summary_only', 'opaque', 'not_provided', 'redacted'] = 'not_provided'
    provider_raw: JsonValue = None


class AttemptPayload(AuditModel):
    status: Literal['success', 'failed', 'cancelled', 'unknown', 'empty', 'schema_error']
    error: ErrorInfo | None = None
    response_ref: BlobRef | None = None
    elapsed_seconds: float = Field(default=0.0, ge=0)
    retry_of: str | None = None


class ToolCallPayload(AuditModel):
    name: str = 'execute_python'
    arguments: str
    code: str | None = None
    error: ErrorInfo | None = None


class ObservationPayload(AuditModel):
    status: str
    observation_ref: BlobRef
    full_output_ref: BlobRef | None = None
    truncated: bool = False


class CellStartPayload(AuditModel):
    code: str
    kernel_id: str = ''
    parent_message_id: str = ''


class CellStreamPayload(AuditModel):
    channel: Literal['stdout', 'stderr', 'display', 'exception', 'status']
    text: str = ''
    raw_ref: BlobRef | None = None
    parent_message_id: str = ''
    late: bool = False


class CellEndPayload(AuditModel):
    status: str
    stdout: str = ''
    stderr: str = ''
    error: ErrorInfo | None = None
    namespace_summary: list[str] = Field(default_factory=list)
    committed_state_refs: list[str] = Field(default_factory=list)
    output_ref: BlobRef | None = None


class AtomPayload(AuditModel):
    op: Literal['tag', 'categorize']
    query: str = ''
    input_snapshot: BlobRef | None = None
    request_identity: str = ''
    mode: str = ''
    batches: list[list[StrictStr | StrictInt]] = Field(default_factory=list)
    row_limit: int | None = None
    token_limit: int | None = None
    char_limit: int | None = None
    concurrency: int = Field(default=1, ge=1, le=2)
    status: str = ''
    state_id: str | None = None
    error: ErrorInfo | None = None
    child_state_ids: list[str] = Field(default_factory=list)
    replicate: str = ''
    retry_of: str | None = None


class StatePayload(AuditModel):
    state_id: str
    metadata_ref: BlobRef | None = None
    request_identity: str = ''
    alias: str | None = None
    previous_state_id: str | None = None


class CompactionPayload(AuditModel):
    summary: str = ''
    before_ref: BlobRef | None = None
    after_ref: BlobRef | None = None
    first_seq: int | None = None
    last_seq: int | None = None
    status: str = ''


class BudgetPayload(AuditModel):
    request_id: str
    role: str = ''
    reserved_count: int = Field(default=0, ge=0)
    limit: int = Field(default=0, ge=0)
    output_cap: int = Field(default=0, ge=0)
    status: str = ''
    usage: Usage | None = None


class RecoveryPayload(AuditModel):
    interrupted_type: str
    interrupted_id: str
    status: Literal['unknown', 'interrupted', 'orphan', 'torn_tail'] = 'interrupted'
    reason: str = ''
    original_seq: int | None = None


class ArtifactPayload(AuditModel):
    artifact: BlobRef
    name: str = ''
    source: str = ''


class RetentionPayload(AuditModel):
    phase: Literal['cardinality','preflight','pilot','plan','mapper','reduce','complete','failure']
    status: str
    evidence_ref: BlobRef | None = None
    detail: str = ''


class EventBase(AuditModel):
    schema_version: Literal['1.0'] = '1.0'
    session_id: str
    event_id: str
    seq: int = Field(ge=1)
    timestamp: str
    elapsed_seconds: float = Field(ge=0)
    source: Literal['supervisor', 'agent', 'kernel', 'operator'] = 'supervisor'
    turn_id: str | None = None
    step_id: str | None = None
    request_id: str | None = None
    attempt_id: str | None = None
    tool_call_id: str | None = None
    execution_id: str | None = None
    atom_call_id: str | None = None
    compaction_id: str | None = None
    parent_id: str | None = None
    state_id: str | None = None
    source_event_id: str | None = None
    redacted: bool = False


PAYLOAD_TYPES = {
    **{k: LifecyclePayload for k in ['session/start','session/end','session/resume','turn/start','turn/end','step/start','step/end']},
    'user/message': UserPayload, 'model/request': RequestPayload, 'model/stream': StreamPayload,
    'model/response': ResponsePayload, 'model/attempt_end': AttemptPayload,
    'tool/call': ToolCallPayload, 'tool/result': ObservationPayload,
    'cell/start': CellStartPayload, 'cell/stream': CellStreamPayload, 'cell/end': CellEndPayload,
    **{k: AtomPayload for k in ['atom/start','atom/plan','atom/end']},
    **{k: StatePayload for k in ['state/commit','state/reuse','state/alias']},
    **{k: CompactionPayload for k in ['compaction/start','compaction/summary','compaction/end']},
    'budget/reserve': BudgetPayload, 'budget/settle': BudgetPayload,
    'recovery/interrupted': RecoveryPayload, 'artifact/register': ArtifactPayload,
    'retention/stage': RetentionPayload,
}
EVENT_TYPES = {
    name: create_model(''.join(part.title() for part in name.split('/'))+'Event', __base__=EventBase,
                       type=(Literal[name], name), payload=(payload, ...))
    for name, payload in PAYLOAD_TYPES.items()
}
Event = Annotated[Union[tuple(EVENT_TYPES.values())], Field(discriminator='type')]
EVENT_ADAPTER = TypeAdapter(Event)


class Checkpoint(AuditModel):
    schema_version: Literal['1.0'] = '1.0'
    session_id: str
    event_seq: int = Field(ge=0)
    event_prefix_sha256: str
    projection_sha256: str
    provider_raw: JsonValue


class ValidationReport(AuditModel):
    schema_version: Literal['1.0'] = '1.0'
    session_id: str
    events: int
    torn_tail: bool
    committed_states: int
    unknown_requests: list[str]
    checkpoint_seq: int | None = None
    checkpoint_status: Literal['missing','current','stale','invalid'] = 'missing'
    resumable: bool = False
    resume_warnings: list[str] = Field(default_factory=list)
    valid: bool = True
