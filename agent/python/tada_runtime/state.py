"""Immutable registered objects and event-derived aliases/exact identities."""
from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from pydantic import JsonValue, TypeAdapter

from .audit import AuditLog, atomic_write, canonical, digest, verify_blob
from .audit_contracts import BlobRef
from .storage_contracts import (ColumnData, CommitMetadata, FrameSnapshot, IndexData,
    JsonSnapshot, SeriesSnapshot, Snapshot, SNAPSHOT_ADAPTER, StateManifest, StateMetadata, LEGACY_SNAPSHOT_SCHEMA_HASHES)

_JSON = TypeAdapter(JsonValue)


def _scalar(value):
    if value is pd.NA or value is pd.NaT or value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return _JSON.validate_python(value, strict=True)


def _column(series: pd.Series, name) -> ColumnData:
    dtype = str(series.dtype)
    categories = [_scalar(value) for value in series.cat.categories] if isinstance(series.dtype, pd.CategoricalDtype) else None
    return ColumnData(name=name, dtype=dtype, values=[_scalar(value) for value in series],
                      categories=categories, ordered=series.cat.ordered if categories is not None else False)


def snapshot(value) -> Snapshot:
    """Convert supported Python objects at an explicit serialization boundary."""
    if isinstance(value, (pd.Series, pd.DataFrame)):
        if isinstance(value.index, pd.MultiIndex):
            raise ValueError('Register a flat index explicitly; MultiIndex restore is unsupported')
        index = IndexData(name=_scalar(value.index.name), dtype=str(value.index.dtype),
            values=[_scalar(item) for item in value.index],
            range=[value.index.start,value.index.stop,value.index.step] if isinstance(value.index,pd.RangeIndex) else None,
            categories=[_scalar(item) for item in value.index.categories] if isinstance(value.index,pd.CategoricalIndex) else None,
            categories_dtype=str(value.index.categories.dtype) if isinstance(value.index,pd.CategoricalIndex) else None,
            ordered=value.index.ordered if isinstance(value.index,pd.CategoricalIndex) else False)
        if isinstance(value.index, pd.CategoricalIndex):
            # Object dtype cannot recover scalar types from their JSON spelling.
            # Reject loss before any bridge payload, artifact or commit is emitted.
            try:
                pd.testing.assert_index_equal(value.index.categories,
                    pd.Index(index.categories, dtype=index.categories_dtype), exact=True)
            except AssertionError as exc:
                raise ValueError('CategoricalIndex categories cannot be restored losslessly; '
                                 'normalize the category dtype before registration') from exc
        if isinstance(value, pd.Series):
            result = SeriesSnapshot(index=index, column=_column(value,value.name))
        else:
            if isinstance(value.columns,pd.MultiIndex):
                raise ValueError('MultiIndex columns must be flattened before registration')
            result = FrameSnapshot(index=index, columns=[_column(value.iloc[:,i],_scalar(name)) for i,name in enumerate(value.columns)],
                                   columns_name=_scalar(value.columns.name))
    else:
        result = JsonSnapshot(value=_JSON.validate_python(value,strict=True))
    # Verify JSON serialization before persistence and before returning the bridge payload.
    return SNAPSHOT_ADAPTER.validate_json(result.model_dump_json())


def _restore_column(column: ColumnData):
    if column.categories is not None:
        return pd.Series(pd.Categorical(column.values,categories=column.categories,ordered=column.ordered), name=column.name)
    return pd.Series(column.values,dtype=column.dtype,name=column.name)


def restore(value: Snapshot | dict):
    value = SNAPSHOT_ADAPTER.validate_python(value)
    if isinstance(value, JsonSnapshot):
        return _JSON.validate_json(canonical(value.value))
    if value.index.dtype == 'category':
        if value.index.categories is None or value.index.categories_dtype is None:
            raise ValueError('Legacy categorical-index snapshot lacks categories/order metadata; restore cannot infer its original semantics')
        categories = pd.Index(value.index.categories,dtype=value.index.categories_dtype)
        index = pd.CategoricalIndex(value.index.values,categories=categories,ordered=value.index.ordered,name=value.index.name)
    else:
        index = pd.RangeIndex(*value.index.range,name=value.index.name) if value.index.range is not None else pd.Index(
            value.index.values,dtype=value.index.dtype,name=value.index.name)
    if len(index) != len(value.index.values) or [_scalar(item) for item in index] != value.index.values:
        raise ValueError('Index encoding mismatch')
    if isinstance(value,SeriesSnapshot):
        result = _restore_column(value.column)
        result.index = index
        return result
    columns = [_restore_column(column) for column in value.columns]
    result = pd.concat(columns,axis=1) if columns else pd.DataFrame(index=range(len(index)))
    result.index = index
    result.columns.name = value.columns_name
    return result


class Store:
    def __init__(self, audit: AuditLog):
        self.audit = audit
        self.root = audit.root
        self._states: dict[str,StateMetadata] = {}
        self._aliases: dict[str,str] = {}
        self._identities: dict[str,str] = {}
        for event in audit.events:
            if event.type == 'state/commit':
                if event.payload.metadata_ref is None:
                    raise ValueError('Committed state missing metadata reference')
                metadata = StateMetadata.model_validate_json(verify_blob(self.root,event.payload.metadata_ref))
                if metadata.created_seq != event.seq or metadata.state_id != event.payload.state_id:
                    raise ValueError('State commit/metadata mismatch')
                verify_blob(self.root,metadata.data_ref)
                self._states[metadata.state_id] = metadata
                if metadata.request_identity:
                    self._identities[metadata.request_identity] = metadata.state_id
            elif event.type == 'state/alias':
                if event.payload.state_id not in self._states or event.payload.alias is None:
                    raise ValueError('Alias references an uncommitted state')
                self._aliases[event.payload.alias] = event.payload.state_id
        # Existing files without commits remain diagnostic orphans; never promote them.
        committed_paths = {s.data_ref.path for s in self._states.values()}
        self.orphans = [str(path.relative_to(self.root)) for path in (self.root/'states').glob('*.data.json')
                        if str(path.relative_to(self.root)) not in committed_paths]
        if self.orphans:
            for orphan in self.orphans:
                audit.append('recovery/interrupted', {'interrupted_type':'state/artifact',
                    'interrupted_id':orphan,'status':'orphan','reason':'No authoritative state commit; ignored'})

    def commit(self,value,metadata: CommitMetadata | dict | None=None) -> str:
        metadata = CommitMetadata.model_validate(metadata or {})
        referenced = metadata.child_state_ids + ([metadata.parent_state_id] if metadata.parent_state_id else [])
        if any(state_id not in self._states for state_id in referenced):
            raise ValueError('State metadata references an uncommitted parent or child')
        encoded = snapshot(value)
        with self.audit._mutex:
            self.audit.assert_healthy()
            state_id = 'state-'+uuid.uuid4().hex
            clean, redacted = self.audit._redact(encoded.model_dump(mode='json'))
            encoded = SNAPSHOT_ADAPTER.validate_python(clean)
            data = canonical(encoded)
            relative = f'states/{state_id}.data.json'
            ref = BlobRef(path=relative,sha256=digest(data),size=len(data),redacted=redacted)
            clean_metadata,_ = self.audit._redact(metadata.model_dump(mode='json'))
            meta = StateMetadata(**clean_metadata,state_id=state_id,created_seq=self.audit.seq+1,
                data_ref=ref,data_hash=ref.sha256,schema_hash=digest(canonical(SNAPSHOT_ADAPTER.json_schema())),kind=encoded.kind)
            metadata_bytes = canonical(meta)
            meta_relative = f'states/{state_id}.metadata.json'
            meta_ref = BlobRef(path=meta_relative,sha256=digest(metadata_bytes),size=len(metadata_bytes))
            atomic_write(self.root/relative,data)
            atomic_write(self.root/meta_relative,metadata_bytes)
            self.audit.append('state/commit',{'state_id':state_id,'metadata_ref':meta_ref,'request_identity':meta.request_identity},
                              state_id=state_id,atom_call_id=meta.atom_call_id)
            self._states[state_id] = meta
            if meta.request_identity:
                self._identities[meta.request_identity] = state_id
            self._manifest()
            return state_id

    def _manifest(self):
        manifest=StateManifest(states=list(self._states.values()),aliases=self._aliases,through_seq=self.audit.seq)
        clean,_ = self.audit._redact(manifest.model_dump(mode='json'))
        atomic_write(self.root/'states'/'manifest.json',canonical(StateManifest.model_validate(clean)))

    def alias(self,name:str,state_id:str):
        if not isinstance(name,str) or not name:
            raise ValueError('Alias must be a non-empty string')
        if state_id not in self._states:
            raise KeyError(state_id)
        with self.audit._mutex:
            self.audit.append('state/alias',{'state_id':state_id,'alias':name,'previous_state_id':self._aliases.get(name)},state_id=state_id)
            self._aliases[name]=state_id
            self._manifest()

    def put(self,alias,value,*,op='put',request_identity='',metadata=None):
        payload=CommitMetadata.model_validate(metadata or {'op':op,'request_identity':request_identity})
        state_id=self.commit(value,payload)
        self.alias(alias,state_id)
        return self._states[state_id]

    def find(self,request_identity:str):
        return self._identities.get(request_identity)

    def lookup(self,request_identity:str):
        state_id=self.find(request_identity)
        return self._states.get(state_id) if state_id else None

    def get(self,id_or_alias:str):
        state_id=self._aliases.get(id_or_alias,id_or_alias)
        meta=self._states[state_id]
        encoded=SNAPSHOT_ADAPTER.validate_json(verify_blob(self.root,meta.data_ref))
        if meta.schema_hash not in {digest(canonical(SNAPSHOT_ADAPTER.json_schema())), *LEGACY_SNAPSHOT_SCHEMA_HASHES}:
            raise ValueError('State snapshot schema hash mismatch')
        result=restore(encoded)
        if isinstance(result,(pd.Series,pd.DataFrame)):
            result.attrs['state_id']=state_id
        return result

    def list(self):
        return list(self._states.values())

    @property
    def aliases(self):
        return dict(self._aliases)


StateStore = Store
