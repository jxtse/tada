"""Validated process and kernel boundaries; vendor projections remain explicitly raw."""
from typing import Annotated, Literal
from pydantic import Field, JsonValue, TypeAdapter, model_validator
from .contracts import StrictModel, ExecuteArgs, Category, LongRowsOverride
from .audit_contracts import BlobRef, ErrorInfo, Usage

class InputSpec(StrictModel):
    name: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_]*$')
    path: str

class Initialize(StrictModel):
    config_path: str | None = None
    session_dir: str | None = None
    inputs: list[InputSpec] = Field(default_factory=list)
    resume: bool = False
    overrides: list[str] = Field(default_factory=list)

class Execute(ExecuteArgs):
    tool_call_id: str | None = None

class Empty(StrictModel):
    pass

class EventIds(StrictModel):
    source: Literal['agent'] = 'agent'
    turn_id: str | None = None
    step_id: str | None = None
    request_id: str | None = None
    attempt_id: str | None = None
    tool_call_id: str | None = None
    execution_id: str | None = None
    atom_call_id: str | None = None
    parent_id: str | None = None
    state_id: str | None = None
    source_event_id: str | None = None

class EventParams(StrictModel):
    type: str
    payload: dict[str, JsonValue]
    ids: EventIds = Field(default_factory=EventIds)

class Reserve(StrictModel):
    role: Literal['agent', 'compaction']
    provider_raw: JsonValue
    turn_id: str | None = None
    step_id: str | None = None

class Stream(StrictModel):
    request_id: str
    channel: Literal['reasoning','text','tool_call','provider_raw']
    text: str = ''
    provider_sequence: int = 0
    availability: Literal['exposed','not_provided','opaque','summary_only','redacted'] = 'not_provided'
    block_index: int | None = None
    provider_raw: JsonValue = None

class Compaction(StrictModel):
    phase: Literal['start','end']
    messages: list[JsonValue]
    summary: str = ''
    status: str = ''

class Settle(StrictModel):
    request_id: str
    status: Literal['success', 'failed', 'cancelled', 'unknown', 'empty', 'schema_error']
    provider_raw: JsonValue = None
    reported_model: str | None = None
    error: str | None = None

class CheckpointParams(StrictModel):
    messages: JsonValue
    pi_entries: list[JsonValue] = Field(default_factory=list)

class RpcRequest(StrictModel):
    schema_version: Literal['1.0'] = '1.0'
    id: str | int
    method: Literal['initialize','execute','inventory','cancel','event','model_reserve','model_settle','model_stream','compaction','checkpoint','close','receipt','prepare_tool','execute_tool','tool_observation']
    params: dict[str, JsonValue] = Field(default_factory=dict)

class RpcSuccess(StrictModel):
    id: str | int
    result: JsonValue

class RpcFailure(StrictModel):
    id: str | int
    error: ErrorInfo

class Observation(StrictModel):
    execution_id: str
    status: Literal['ok','error','cancelled','timeout']
    stdout: str
    stderr: str
    exception: ErrorInfo | None = None
    displays: list[BlobRef] = Field(default_factory=list)
    artifacts: list[BlobRef] = Field(default_factory=list)
    namespace_summary: list[str] = Field(default_factory=list)
    committed_state_refs: list[str] = Field(default_factory=list)
    truncated: bool = False
    full_output_ref: BlobRef
    # Remaining model-request budget after this cell, so the agent can plan semantic work and deliver in time.
    budget: dict[str, JsonValue] | None = None

from .large_contracts import CardinalitySpec

class KernelRequest(StrictModel):
    op: Literal['tag','categorize','state.list','state.get','state.put','inventory']
    snapshot: JsonValue = None
    query: str | None = None
    name: str | None = None
    fresh: bool = False
    cardinality: CardinalitySpec | None = None
    labels: list[Category] | None = None
    multi: bool = False
    long_rows: LongRowsOverride | None = None
    key: str | None = None
    namespace: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def tag_only_options(self):
        if self.op != 'tag' and (self.labels is not None or self.multi or self.long_rows is not None):
            raise ValueError('labels, multi and long_rows apply to tag only')
        return self

class KernelResult(StrictModel):
    value: JsonValue

PARAMS = {'initialize':Initialize, 'execute':Execute, 'inventory':Empty, 'cancel':Empty,
          'event':EventParams,'model_reserve':Reserve,'model_settle':Settle,
          'model_stream':Stream,'compaction':Compaction,'checkpoint':CheckpointParams,'close':Empty}

# Responses are validated by the selected method before the transport envelope.
from .config import Config
from .storage_contracts import StateMetadata
from .audit_contracts import Checkpoint, EVENT_ADAPTER

class InitializeResult(StrictModel):
    config: Config
    session_dir: str
    session_id: str
    agent_api_key: str | None
    tool_schema: dict[str, JsonValue]
    restored_messages: list[JsonValue]
    restored_entries: list[JsonValue]
    namespace_summary: str

class InventoryResult(StrictModel):
    namespace_summary: list[str]
    states: list[StateMetadata]
    outputs: list[str]

class StatusResult(StrictModel):
    status: Literal['cancelling','closed','success','failed','cancelled','unknown','empty','schema_error']

class ReserveResult(StrictModel):
    request_id: str

class StreamResult(StrictModel):
    buffered: bool
    emitted_parts: int = Field(ge=0)

RESULTS = {'initialize':TypeAdapter(InitializeResult),'execute':TypeAdapter(Observation),
           'inventory':TypeAdapter(InventoryResult),'cancel':TypeAdapter(StatusResult),
           'close':TypeAdapter(StatusResult),'model_reserve':TypeAdapter(ReserveResult),
           'model_settle':TypeAdapter(StatusResult),'checkpoint':TypeAdapter(Checkpoint),
           'event':EVENT_ADAPTER,'model_stream':TypeAdapter(StreamResult),'compaction':EVENT_ADAPTER}

class CliReceipt(StrictModel):
    session_id: str | None
    status: Literal['turn_completed','completed','cancelled','failed']
    message: JsonValue = None  # Pi-owned projection; typed domain events live in the audit.
    inventory: InventoryResult | None = None
    error: str | None = None

    @model_validator(mode='after')
    def completion_fields(self):
        if self.status == 'turn_completed' and (self.session_id is None or self.message is None):
            raise ValueError('completed turn requires session and Pi message projection')
        if self.status == 'completed' and (self.session_id is None or self.inventory is None):
            raise ValueError('completed session requires inventory')
        if self.status == 'failed' and not self.error:
            raise ValueError('failed receipt requires error')
        return self

PARAMS['receipt'] = CliReceipt
RESULTS['receipt'] = TypeAdapter(CliReceipt)

class PrepareTool(StrictModel):
    tool_call_id: str = Field(min_length=1)
    name: str
    arguments: str
    sdk_arguments_error: str | None = None
    ids: EventIds = Field(default_factory=EventIds)

class ExecuteTool(StrictModel):
    tool_call_id: str = Field(min_length=1)

class ToolAdmission(StrictModel):
    tool_call_id: str
    status: Literal['accepted','rejected']
    code: str | None = None
    error: ErrorInfo | None = None

    @model_validator(mode='after')
    def consistent(self):
        if self.status == 'accepted' and (self.code is None or self.error is not None):
            raise ValueError('accepted tool requires code and no error')
        if self.status == 'rejected' and (self.code is not None or self.error is None):
            raise ValueError('rejected tool requires error and no code')
        return self

class ToolObserved(StrictModel):
    tool_call_id: str
    name: str
    is_error: bool
    provider_raw: JsonValue

class ToolObservationRecord(StrictModel):
    tool_call_id: str
    is_error: bool
    stage: Literal['pre_execution','execution']
    error: ErrorInfo | None = None
    provider_raw: JsonValue

class Recorded(StrictModel):
    status: Literal['recorded'] = 'recorded'

PARAMS.update(prepare_tool=PrepareTool,execute_tool=ExecuteTool,tool_observation=ToolObserved)
RESULTS.update(prepare_tool=TypeAdapter(ToolAdmission),execute_tool=TypeAdapter(Observation),tool_observation=TypeAdapter(Recorded))
