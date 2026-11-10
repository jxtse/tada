"""Portable typed state snapshots; no executable serialization formats."""
from __future__ import annotations

from typing import Annotated, Literal
from pydantic import Field, JsonValue, StrictInt, StrictStr, TypeAdapter, model_validator
from .audit_contracts import AuditModel, BlobRef

RID = StrictStr | StrictInt


class JsonSnapshot(AuditModel):
    kind: Literal['json'] = 'json'
    value: JsonValue


class ColumnData(AuditModel):
    name: StrictStr | StrictInt | None
    dtype: str
    values: list[JsonValue]
    categories: list[JsonValue] | None = None
    ordered: bool = False


class IndexData(AuditModel):
    name: StrictStr | StrictInt | None = None
    dtype: str
    values: list[JsonValue]
    range: list[StrictInt] | None = Field(default=None, min_length=3, max_length=3)
    categories: list[JsonValue] | None = None
    categories_dtype: str | None = None
    ordered: bool = False

    @model_validator(mode='after')
    def categorical_metadata(self):
        if self.categories is not None:
            if self.dtype != 'category' or self.categories_dtype is None or self.range is not None:
                raise ValueError('Categorical index metadata is inconsistent')
        elif self.categories_dtype is not None or self.ordered:
            raise ValueError('Categorical index metadata requires categories')
        return self


class SeriesSnapshot(AuditModel):
    kind: Literal['series'] = 'series'
    index: IndexData
    column: ColumnData

    @model_validator(mode='after')
    def aligned(self):
        if len(self.column.values) != len(self.index.values):
            raise ValueError('Series/index length mismatch')
        return self


class FrameSnapshot(AuditModel):
    kind: Literal['dataframe'] = 'dataframe'
    index: IndexData
    columns: list[ColumnData]
    columns_name: StrictStr | StrictInt | None = None

    @model_validator(mode='after')
    def aligned(self):
        names = [column.name for column in self.columns]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate columns unsupported in registered state')
        if any(len(column.values) != len(self.index.values) for column in self.columns):
            raise ValueError('DataFrame/index length mismatch')
        return self


Snapshot = Annotated[JsonSnapshot | SeriesSnapshot | FrameSnapshot, Field(discriminator='kind')]
SNAPSHOT_ADAPTER = TypeAdapter(Snapshot)


class CommitMetadata(AuditModel):
    op: Literal['tag', 'categorize', 'put', 'input'] = 'put'
    request_identity: str = ''
    input_snapshot: BlobRef | None = None
    rid: list[RID] = Field(default_factory=list)
    query: str = ''
    model: str = ''
    protocol: str = ''
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    prompt_version: str = ''
    engine_version: str = '1.0'
    parent_state_id: str | None = None
    child_state_ids: list[str] = Field(default_factory=list)
    atom_call_id: str | None = None
    mode: str = ''
    replicate: str = ''
    retention_seed: int | None = None
    evidence_refs: list[BlobRef] = Field(default_factory=list)

    @model_validator(mode='after')
    def valid_rids(self):
        if len(set(self.rid)) != len(self.rid):
            raise ValueError('Duplicate rid in metadata')
        if self.rid and (any(type(rid) is not type(self.rid[0]) for rid in self.rid) or '' in self.rid):
            raise ValueError('rid must have a stable non-empty type')
        return self


class StateMetadata(CommitMetadata):
    schema_version: Literal['1.0'] = '1.0'
    state_id: str
    created_seq: int = Field(ge=1)
    data_ref: BlobRef
    data_hash: str
    schema_hash: str
    kind: Literal['json', 'series', 'dataframe']


class StateManifest(AuditModel):
    schema_version: Literal['1.0'] = '1.0'
    states: list[StateMetadata]
    aliases: dict[str, str]
    through_seq: int = Field(ge=0)


# Version 1.0 snapshots before categorical-index metadata was added. Their declared
# hash remains inspectable; restore rejects an old categorical index whose original
# categories/order cannot be recovered from its data rather than guessing.
LEGACY_SNAPSHOT_SCHEMA_HASHES = frozenset({'2f33ff308e28b7167b8cda4092d546f77e508231f6655f30e9bdebf7b60626ae'})
