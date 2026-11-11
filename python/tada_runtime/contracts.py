"""Strict shared contracts. Pydantic is the authority at every domain boundary."""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, StrictStr, TypeAdapter, ValidationInfo, field_validator, model_validator

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
        if 'label_sets' in context:
            for item in self.items:
                parse_label_set(item.value, context['label_sets'])
        return self


def parse_label_set(value: str, allowed: list[str]) -> list[str]:
    """Strictly decode one multi-label value: a JSON array of unique allowed label names."""
    import json
    try:
        decoded = json.loads(value)
    except ValueError:
        raise ValueError('multi-label value must be a JSON-encoded array string') from None
    if not isinstance(decoded, list) or any(not isinstance(name, str) for name in decoded):
        raise ValueError('multi-label value must be a JSON array of label-name strings')
    if len(set(decoded)) != len(decoded):
        raise ValueError('multi-label value repeats a label')
    if any(name not in allowed for name in decoded):
        raise ValueError('multi-label value references a label outside the supplied labels')
    return decoded


def canonical_label_set(names, order: list[str]) -> str:
    """Host serialization of a validated label set in the caller's label order.

    This reorders already-validated names only; it never adds, drops or renames a label.
    """
    import json
    chosen = set(names)
    return json.dumps([name for name in order if name in chosen], ensure_ascii=False)

class Category(StrictModel):
    name: Annotated[StrictStr, Field(min_length=1)]
    description: StrictStr = ''

    @field_validator('name', 'description')
    @classmethod
    def no_control_or_surrogate(cls, value):
        for char in value:
            code = ord(char)
            if code < 0x20 and char not in '\t\n\r':
                raise ValueError('category text contains a control character')
            if 0xD800 <= code <= 0xDFFF:
                raise ValueError('category text contains a lone surrogate')
        return value

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

class LongRowsConfig(StrictModel):
    """Scale-transparent handling of rows longer than one request can carry.

    Rows are segmented inside tag; the row-level result is aggregated by a fixed
    rule (labels+multi -> union, otherwise reduce) unless a caller override asks
    for another supported aggregate. The default trigger only segments rows that
    previously failed with "one row exceeds batch_chars", so short-text behavior
    and requests are unchanged.
    """
    enabled: bool = True
    trigger: Literal['over_batch_chars', 'over_segment_chars'] = 'over_batch_chars'
    segment_chars: Annotated[int, Field(ge=200, le=500000)] = 8000
    overlap_chars: Annotated[int, Field(ge=0, le=100000)] = 600
    head_chars: Annotated[int, Field(ge=0, le=100000)] = 1500
    boundary: Literal['paragraph', 'line', 'chars'] = 'paragraph'
    max_segments_per_row: Annotated[int, Field(ge=1, le=10000)] = 64
    policy: Literal['segment-aggregate-v1'] = 'segment-aggregate-v1'

    @model_validator(mode='after')
    def geometry(self):
        if self.overlap_chars * 2 >= self.segment_chars:
            raise ValueError('overlap_chars must be less than half of segment_chars')
        if self.head_chars >= self.segment_chars:
            raise ValueError('head_chars must be smaller than segment_chars')
        return self

class LongRowsOverride(StrictModel):
    """Optional per-call override; unset fields keep the runtime configuration."""
    aggregate: Literal['union', 'reduce', 'collect'] | None = None
    trigger: Literal['over_batch_chars', 'over_segment_chars'] | None = None
    segment_chars: Annotated[int, Field(ge=200, le=500000)] | None = None
    overlap_chars: Annotated[int, Field(ge=0, le=100000)] | None = None
    head_chars: Annotated[int, Field(ge=0, le=100000)] | None = None
    boundary: Literal['paragraph', 'line', 'chars'] | None = None

class LabelSpec(StrictModel):
    """Caller-supplied closed label set for tag(labels=...)."""
    labels: list[Category] = Field(min_length=1)
    multi: bool = False

    @model_validator(mode='after')
    def unique(self):
        CategoryEnvelope.model_validate({'categories': [c.model_dump() for c in self.labels]}, context={'nonempty': True})
        return self

    def names(self) -> list[str]:
        return [c.name for c in self.labels]

class OperatorIdentity(StrictModel):
    op: Literal['tag', 'categorize']
    query: str
    snapshot: TextSnapshot
    mode: Literal['query-driven', 'categorize']
    provider: ProviderIdentity
    prompt_version: str = 'tada-strict-v2'
    engine_version: str = '1.1'
    labels: list[Category] | None = None
    multi: bool = False
    long_rows: LongRowsConfig | None = None
    long_rows_aggregate: Literal['union', 'reduce', 'collect'] | None = None
    batch_retries: int = 0
    batch_plan: list[list[RID]]
    batch_rows: int
    batch_chars: int
    reduce_fan_in: int
    replicate: str
    large_scale: LargeScaleConfig | None = None
    cardinality: CardinalitySpec | None = None
    effective_seed: int | None = None
    operator_concurrency: int = 1
