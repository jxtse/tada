"""Strict shared contracts. Pydantic is the authority at every domain boundary."""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, StrictStr, TypeAdapter, ValidationInfo, model_validator

SCHEMA_VERSION = '1.0'

from .audit_contracts import AuditModel
from .large_contracts import LargeScaleConfig, CardinalitySpec

class StrictModel(AuditModel):
    pass

RID = StrictStr | StrictInt
JSON = TypeAdapter(JsonValue, config=ConfigDict(strict=True, allow_inf_nan=False))

from .audit_contracts import ErrorInfo, Usage, ContentBlock, TextBlock, ReasoningBlock, ToolBlock as ToolCallBlock

class ExecuteArgs(StrictModel):
    code: StrictStr

class TextRow(StrictModel):
    rid: RID
    text: StrictStr

class TextSnapshot(StrictModel):
    rows: list[TextRow]
    name: StrictStr | None = None
    index_name: StrictStr | None = None

    @model_validator(mode='after')
    def identities(self):
        ids = [row.rid for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate rid')
        if any(rid == '' for rid in ids):
            raise ValueError('empty rid')
        if len({type(rid) for rid in ids}) > 1:
            raise ValueError('rid types must be stable')
        return self

class TagItem(StrictModel):
    rid: RID
    value: StrictStr

class TagEnvelope(StrictModel):
    items: list[TagItem]

    @model_validator(mode='after')
    def coverage(self, info: ValidationInfo):
        ids = [item.rid for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate output rid')
        context = info.context or {}
        if 'rids' in context and {(type(x), x) for x in ids} != {(type(x), x) for x in context['rids']}:
            raise ValueError('output rid coverage mismatch')
        if 'allowed_values' in context and any(x.value not in context['allowed_values'] for x in self.items):
            raise ValueError('invalid category reference')
        return self

class Category(StrictModel):
    name: Annotated[StrictStr, Field(min_length=1)]
    description: StrictStr = ''

class CategoryEnvelope(StrictModel):
    categories: list[Category]

    @model_validator(mode='after')
    def unique_names(self, info: ValidationInfo):
        names = [x.name for x in self.categories]
        if len(set(names)) != len(names):
            raise ValueError('duplicate category name')
        if any(not x.strip() for x in names):
            raise ValueError('blank category name')
        context = info.context or {}
        if context.get('max_categories') is not None and len(names) > context['max_categories']:
            raise ValueError('category output exceeds requested upper bound')
        if context.get('exact_categories') is not None and len(names) != context['exact_categories']:
            raise ValueError('exact-k category insufficiency or overfill')
        if any(len(c.name)>context.get('max_name_chars',10**9) or len(c.description)>context.get('max_description_chars',10**9) for c in self.categories):
            raise ValueError('category field exceeds configured capacity bound')
        if info.context and info.context.get('nonempty') and not names:
            raise ValueError('nonempty input requires categories')
        return self

class IntentEnvelope(StrictModel):
    mode: Literal['independent', 'joint', 'invalid']
    domain_query: StrictStr | None = None

    @model_validator(mode='after')
    def joint_query(self):
        if self.mode == 'joint' and not self.domain_query:
            raise ValueError('joint intent requires domain_query')
        if self.mode != 'joint' and self.domain_query is not None:
            raise ValueError('only joint intent has domain_query')
        return self

class NormalizedResponse(StrictModel):
    schema_version: Literal['1.0'] = SCHEMA_VERSION
    requested_model: str
    reported_model: str | None
    protocol: Literal['openai-responses', 'openai-chat', 'anthropic-messages']
    content: list[ContentBlock]
    usage: Usage
    finish_reason: str
    reasoning_availability: Literal['exposed', 'summary_only', 'opaque', 'not_provided', 'redacted']
    provider_raw: JsonValue

class ProviderIdentity(StrictModel):
    base_url: str
    model: str
    protocol: Literal['openai-responses', 'openai-chat', 'anthropic-messages']
    auth_mode: Literal['optional', 'required']
    max_output_tokens: int
    timeout_seconds: float
    temperature: float | None = None
    reasoning_effort: str | None = None

class OperatorIdentity(StrictModel):
    op: Literal['tag', 'categorize']
    query: str
    snapshot: TextSnapshot
    mode: Literal['query-driven', 'categorize']
    provider: ProviderIdentity
    prompt_version: str = 'tada-strict-v2'
    engine_version: str = '1.0'
    batch_plan: list[list[RID]]
    batch_rows: int
    batch_chars: int
    reduce_fan_in: int
    replicate: str
    large_scale: LargeScaleConfig | None = None
    cardinality: CardinalitySpec | None = None
    effective_seed: int | None = None
    operator_concurrency: int = 1
