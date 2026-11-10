"""Single-writer, durable audit storage and side-effect-free offline readers."""
from __future__ import annotations

import fcntl
import hashlib
import html
import json
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from .audit_contracts import (BlobRef, Checkpoint, EVENT_ADAPTER, EVENT_TYPES,
                              ValidationReport)

JSON = TypeAdapter(JsonValue)


def canonical(value) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode='json')
    value = JSON.validate_python(value, strict=True)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.staging-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        dirfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_events(session_dir: Path | str):
    """Read and validate without locking, recovering, executing, or networking."""
    path = Path(session_dir) / 'events.jsonl'
    data = path.read_bytes() if path.exists() else b''
    events = []
    offset = 0
    torn = False
    lines = data.splitlines(keepends=True)
    session_id = None
    event_ids = set()
    source_ids = set()
    for i, line in enumerate(lines):
        try:
            event = EVENT_ADAPTER.validate_json(line)
        except ValidationError as error:
            syntax_failure = any(item['type'] == 'json_invalid' for item in error.errors())
            if syntax_failure and i == len(lines)-1 and not line.endswith(b'\n'):
                torn = True
                break
            raise
        if event.seq != len(events)+1:
            raise ValueError(f'Non-contiguous event sequence: {event.seq}')
        if session_id is not None and event.session_id != session_id:
            raise ValueError('Mixed session IDs')
        session_id = event.session_id
        if event.event_id in event_ids:
            raise ValueError('Duplicate event ID')
        if event.source_event_id is not None and event.source_event_id in source_ids:
            raise ValueError('Duplicate source event ID')
        event_ids.add(event.event_id)
        if event.source_event_id:
            source_ids.add(event.source_event_id)
        events.append(event)
        offset += len(line)
    return events, torn


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact escapes session directory')
    return path


def verify_blob(root: Path, ref: BlobRef) -> bytes:
    data = _safe_path(root, ref.path).read_bytes()
    if len(data) != ref.size or digest(data) != ref.sha256:
        raise ValueError(f'Artifact integrity mismatch: {ref.path}')
    return data


def _refs(value):
    if isinstance(value, BlobRef):
        yield value
    elif isinstance(value, BaseModel):
        for name in value.__class__.model_fields:
            yield from _refs(getattr(value, name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _refs(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _refs(item)


def _retention_evidence_validator(root: Path):
    """Build a read-only typed evidence decoder without importing operators at startup."""
    from .large_contracts import (CardinalityResolution, PreflightEvidence, PilotEvidence,
        MapperEvidence, LargeExecutionEvidence, CapacityCheck)
    from .retention import RetentionDecision
    from .contracts import TextSnapshot, CategoryEnvelope
    from tada_operators.large import GlobalReducerInput
    phases = {'cardinality':CardinalityResolution, 'preflight':PreflightEvidence,
        'pilot':PilotEvidence, 'plan':RetentionDecision, 'mapper':MapperEvidence,
        'reduce':GlobalReducerInput, 'complete':LargeExecutionEvidence, 'failure':CapacityCheck}
    cache = {}

    def decode(ref, model=None):
        key = (ref.path,ref.sha256,model)
        if key in cache:
            return cache[key]
        raw = verify_blob(root,ref)
        if model is None:
            # State evidence stores the same typed artifacts as stage records;
            # the sample snapshot is the only additional evidence variant.
            from pydantic import ValidationError
            for candidate in (*phases.values(),TextSnapshot):
                try:
                    value = candidate.model_validate_json(raw)
                except ValidationError:
                    continue
                break
            else:
                raise ValueError('State evidence does not match a supported typed retention artifact')
        else:
            value = model.model_validate_json(raw)
        cache[key] = value
        for nested in _refs(value):
            verify_blob(root,nested)
        if isinstance(value,PreflightEvidence):
            sample = decode(value.pilot_sample_ref,TextSnapshot)
            population = {rid for batch in value.partition.batches for rid in batch}
            if not {row.rid for row in sample.rows}.issubset(population):
                raise ValueError('Pilot snapshot contains rows outside the balanced partition')
        elif isinstance(value,MapperEvidence):
            mapped = decode(value.result_ref,CategoryEnvelope)
            if len(mapped.categories) != value.actual_candidates or len(mapped.categories) > value.requested_r:
                raise ValueError('Mapper result count does not match typed retention evidence')
        elif isinstance(value,LargeExecutionEvidence):
            pilot = decode(value.pilot_ref,PilotEvidence)
            if pilot.counts != value.decision.request.pilot:
                raise ValueError('Planner pilot counts do not match immutable pilot evidence')
            for mapper in value.mapper_evidence:
                mapped = decode(mapper.result_ref,CategoryEnvelope)
                if len(mapped.categories) != mapper.actual_candidates or len(mapped.categories) > mapper.requested_r:
                    raise ValueError('Mapper result count does not match completed retention evidence')
        return value

    def stage(event):
        if event.payload.evidence_ref is None:
            if event.payload.phase != 'failure':
                raise ValueError('Retention stage is missing typed evidence')
            return
        decode(event.payload.evidence_ref,phases[event.payload.phase])
    return stage,decode


def _pending(events):
    pending = {}
    pairs = {'model/request': ('model/attempt_end', 'request_id'),
             'turn/start': ('turn/end', 'turn_id'), 'cell/start': ('cell/end', 'execution_id'),
             'atom/start': ('atom/end', 'atom_call_id'),
             'compaction/start': ('compaction/end', 'compaction_id')}
    for event in events:
        if event.type in pairs:
            end, key = pairs[event.type]
            identity = getattr(event, key) or event.event_id
            pending[(end, identity)] = event
        elif event.type == 'recovery/interrupted':
            for key in list(pending):
                if key[1] == event.payload.interrupted_id:
                    del pending[key]
        else:
            for start, (end, key) in pairs.items():
                if event.type == end:
                    pending.pop((end, getattr(event, key)), None)
    return pending


class AuditLog:
    def __init__(self, session_dir: Path | str, session_id: str | None = None,
                 secrets=(), resume: bool = False):
        self.root = self.session_dir = Path(session_dir)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self._lockfile = (self.root / '.writer.lock').open('a+b')
        try:
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lockfile.close()
            raise RuntimeError('Session already has a write owner') from None
        self._mutex = threading.RLock()
        self._secrets = tuple(secret for secret in secrets if secret)
        self._started = time.monotonic()
        self._failed = False
        self._closed = False
        self.path = self.root / 'events.jsonl'
        try:
            self.events, torn = read_events(self.root)
            if self.events and not resume:
                raise ValueError('Existing session requires explicit resume')
            self.session_id = session_id or (self.events[0].session_id if self.events else self.root.name)
            if self.events and self.events[0].session_id != self.session_id:
                raise ValueError('Session ID mismatch')
            self._source_ids = {e.source_event_id: e for e in self.events if e.source_event_id}
            if resume and (self.root/'pi-checkpoints'/'latest.json').exists():
                self.load_checkpoint()
            if torn:
                data = self.path.read_bytes()
                offset = data.rfind(b'\n')+1
                atomic_write(self.root/'blobs'/f'torn-tail-{digest(data[offset:])}.bin', data[offset:])
                with self.path.open('r+b') as file:
                    file.truncate(offset)
                    file.flush()
                    os.fsync(file.fileno())
            if self.path.exists() and self.path.stat().st_size and not self.path.read_bytes().endswith(b'\n'):
                with self.path.open('ab') as file:
                    file.write(b'\n')
                    file.flush()
                    os.fsync(file.fileno())
            self._file = self.path.open('ab', buffering=0)
            os.chmod(self.path, 0o600)
            if torn:
                self.append('recovery/interrupted', {'interrupted_type':'physical_record',
                    'interrupted_id':'torn_tail','status':'torn_tail','reason':'Preserved torn tail in blobs; truncated incomplete final physical record'})
            if resume:
                for (_, identity), event in list(_pending(self.events).items()):
                    self.append('recovery/interrupted', {'interrupted_type':event.type,
                        'interrupted_id':identity, 'status':'unknown' if event.type=='model/request' else 'interrupted',
                        'reason':'Previous writer ended without a terminal event; never automatically resent', 'original_seq':event.seq},
                        request_id=event.request_id, compaction_id=event.compaction_id, parent_id=event.event_id)
        except BaseException:
            if hasattr(self, '_file'):
                self._file.close()
            self._lockfile.close()
            raise

    @property
    def seq(self):
        return len(self.events)

    def _redact(self, value):
        changed = False
        sensitive = {'authorization','x-api-key','api_key','apikey','cookie','password','access_token','refresh_token'}
        def visit(item):
            nonlocal changed
            if isinstance(item, str):
                for secret in self._secrets:
                    if secret in item:
                        item = item.replace(secret, '[REDACTED]')
                        changed = True
                return item
            if isinstance(item, list):
                return [visit(child) for child in item]
            if isinstance(item, dict):
                result = {}
                for key, child in item.items():
                    safe_key = visit(key)
                    if safe_key in result:
                        raise ValueError('Secret redaction produced duplicate JSON object keys')
                    if key.lower().replace('-', '_') in {s.replace('-', '_') for s in sensitive}:
                        result[safe_key] = '[REDACTED]'
                        changed = True
                    else:
                        result[safe_key] = visit(child)
                return result
            return item
        return visit(value), changed

    def assert_healthy(self):
        if self._failed or self._closed:
            raise RuntimeError('Audit writer unavailable; provider dispatch prohibited')

    def append(self, event_type: str, payload, **ids):
        with self._mutex:
            self.assert_healthy()
            source_id = ids.get('source_event_id')
            if source_id and source_id in self._source_ids:
                previous = self._source_ids[source_id]
                candidate_payload = EVENT_TYPES[event_type].model_fields['payload'].annotation.model_validate(payload)
                cleaned, _ = self._redact(candidate_payload.model_dump(mode='json'))
                if previous.type != event_type or previous.payload.model_dump(mode='json') != cleaned:
                    raise ValueError('Conflicting source_event_id replay')
                return previous
            if isinstance(payload, BaseModel):
                payload = payload.model_dump(mode='json')
            event = EVENT_ADAPTER.validate_python(dict(schema_version='1.0', session_id=self.session_id,
                event_id=str(uuid.uuid4()), seq=self.seq+1, timestamp=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.monotonic()-self._started, type=event_type, payload=payload, **ids))
            clean, redacted = self._redact(event.model_dump(mode='json'))
            clean['redacted'] = redacted
            event = EVENT_ADAPTER.validate_python(clean)
            line = canonical(event)+b'\n'
            try:
                written = self._file.write(line)
                if written != len(line):
                    raise OSError('Short audit append; writer disabled')
                self.flush()
            except BaseException:
                self._failed = True
                raise
            self.events.append(event)
            if source_id:
                self._source_ids[source_id] = event
            return event

    emit = append

    def blob(self, value, media_type='application/json') -> BlobRef:
        with self._mutex:
            self.assert_healthy()
            if isinstance(value, BaseModel):
                value = value.model_dump(mode='json')
            validated = JSON.validate_python(value, strict=True)
            clean, redacted = self._redact(validated)
            data = canonical(clean) if media_type == 'application/json' else str(clean).encode('utf-8')
            sha = digest(data)
            relative = f'blobs/{sha}'
            path = self.root / relative
            try:
                if not path.exists():
                    atomic_write(path, data)
            except BaseException:
                self._failed = True
                raise
            return BlobRef(path=relative,sha256=sha,size=len(data),media_type=media_type,redacted=redacted)

    def blob_bytes(self, data: bytes, media_type='application/octet-stream') -> BlobRef:
        """Register a file artifact; bytes are not misrepresented as JSON."""
        with self._mutex:
            self.assert_healthy()
            redacted = False
            for secret in self._secrets:
                encoded = secret.encode('utf-8')
                if encoded in data:
                    data = data.replace(encoded, b'[REDACTED]')
                    redacted = True
            sha = digest(data)
            relative = f'blobs/{sha}'
            try:
                if not (self.root / relative).exists():
                    atomic_write(self.root / relative, data)
            except BaseException:
                self._failed = True
                raise
            return BlobRef(path=relative, sha256=sha, size=len(data), media_type=media_type, redacted=redacted)

    def flush(self):
        self.assert_healthy()
        try:
            self._file.flush()
            os.fsync(self._file.fileno())
        except BaseException:
            self._failed = True
            raise

    def checkpoint(self, messages) -> Checkpoint:
        with self._mutex:
            self.flush()
            clean, _ = self._redact(JSON.validate_python(messages, strict=True))
            value = Checkpoint(session_id=self.session_id, event_seq=self.seq,
                event_prefix_sha256=digest(self.path.read_bytes()), projection_sha256=digest(canonical(clean)), provider_raw=clean)
            atomic_write(self.root/'pi-checkpoints'/f'{self.seq:09d}.json', canonical(value))
            atomic_write(self.root/'pi-checkpoints'/'latest.json', canonical(value))
            return value

    def load_checkpoint(self) -> Checkpoint:
        return load_checkpoint(self.root)

    def close(self):
        with self._mutex:
            if self._closed:
                return
            try:
                if not self._failed:
                    self.flush()
            finally:
                self._closed = True
                self._file.close()
                fcntl.flock(self._lockfile, fcntl.LOCK_UN)
                self._lockfile.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def checkpoint_integrity(root: Path | str) -> Checkpoint:
    root = Path(root)
    checkpoint = Checkpoint.model_validate_json((root/'pi-checkpoints'/'latest.json').read_bytes())
    events, _ = read_events(root)
    lines = (root/'events.jsonl').read_bytes().splitlines(keepends=True)
    if checkpoint.event_seq > len(events) or (events and checkpoint.session_id != events[0].session_id):
        raise ValueError('Checkpoint sequence/session mismatch')
    if digest(b''.join(lines[:checkpoint.event_seq])) != checkpoint.event_prefix_sha256:
        raise ValueError('Checkpoint event prefix hash mismatch')
    if digest(canonical(checkpoint.provider_raw)) != checkpoint.projection_sha256:
        raise ValueError('Checkpoint projection hash mismatch')
    return checkpoint


def load_checkpoint(root: Path | str) -> Checkpoint:
    """Load only a checkpoint safe for continuing the conversation."""
    checkpoint = checkpoint_integrity(root)
    events, _ = read_events(root)
    # Trailing lifecycle closers are safe; new context-bearing events require a new checkpoint.
    unsafe = {'user/message','model/request','model/response','tool/result','compaction/summary'}
    if any(e.type in unsafe for e in events[checkpoint.event_seq:]):
        raise ValueError('Checkpoint does not cover later conversation events; explicit recovery required')
    return checkpoint


def validate_session(session_dir: Path | str) -> ValidationReport:
    root = Path(session_dir)
    events, torn = read_events(root)
    for event in events:
        for ref in _refs(event):
            verify_blob(root, ref)
    from .storage_contracts import StateMetadata, SNAPSHOT_ADAPTER, LEGACY_SNAPSHOT_SCHEMA_HASHES
    committed = set()
    settled = set()
    retention_validator = None
    for event in events:
        if event.type == 'retention/stage':
            if retention_validator is None:
                retention_validator = _retention_evidence_validator(root)
            retention_validator[0](event)
        if event.type == 'state/commit':
            if event.payload.metadata_ref is None:
                raise ValueError('Committed state missing metadata')
            metadata = StateMetadata.model_validate_json(verify_blob(root, event.payload.metadata_ref))
            for ref in _refs(metadata):
                verify_blob(root,ref)
            if metadata.evidence_refs:
                if retention_validator is None:
                    retention_validator = _retention_evidence_validator(root)
                for ref in metadata.evidence_refs:
                    retention_validator[1](ref)
            if metadata.created_seq != event.seq or metadata.state_id != event.payload.state_id:
                raise ValueError('State metadata/event mismatch')
            if metadata.schema_hash not in {digest(canonical(SNAPSHOT_ADAPTER.json_schema())), *LEGACY_SNAPSHOT_SCHEMA_HASHES}:
                raise ValueError('State schema hash mismatch')
            raw = verify_blob(root, metadata.data_ref)
            if digest(raw) != metadata.data_hash:
                raise ValueError('State data hash mismatch')
            SNAPSHOT_ADAPTER.validate_json(raw)
            committed.add(metadata.state_id)
        elif event.type in ('state/alias', 'state/reuse') and event.payload.state_id not in committed:
            raise ValueError('State reference precedes commit')
        elif event.type == 'budget/settle':
            if event.payload.request_id in settled:
                raise ValueError('Request budget settled more than once')
            settled.add(event.payload.request_id)
    checkpoint = None
    checkpoint_status = 'missing'
    warnings = []
    if (root/'pi-checkpoints'/'latest.json').exists():
        try:
            checkpoint = checkpoint_integrity(root)
        except (ValueError, OSError) as error:
            checkpoint_status = 'invalid'
            warnings.append('Checkpoint integrity or schema validation failed; task resume is prohibited: '+type(error).__name__)
        else:
            try:
                load_checkpoint(root)
            except ValueError as error:
                checkpoint_status = 'stale'
                warnings.append(str(error))
            else:
                checkpoint_status = 'current'
    else:
        warnings.append('No Pi checkpoint is available; durable events remain inspectable but task resume is unavailable.')
    if torn:
        warnings.append('Incomplete physical tail retained unchanged; write-owner reconciliation is required before task resume.')
    unknown = {event.request_id for event in _pending(events).values() if event.type=='model/request'}
    unknown.update(e.request_id for e in events if e.type=='model/attempt_end' and e.payload.status=='unknown')
    unknown.update(e.payload.interrupted_id for e in events if e.type=='recovery/interrupted' and e.payload.status=='unknown')
    return ValidationReport(session_id=events[0].session_id if events else root.name, events=len(events),
        torn_tail=torn, committed_states=sum(e.type=='state/commit' for e in events),
        unknown_requests=sorted(x for x in unknown if x), checkpoint_seq=checkpoint.event_seq if checkpoint else None,
        checkpoint_status=checkpoint_status, resumable=checkpoint_status=='current' and not torn, resume_warnings=warnings)


def _export_destination(session_dir: Path | str, output: Path | str) -> Path:
    root = Path(session_dir).resolve()
    destination = Path(output).resolve()
    views = root / 'views'
    if destination.is_relative_to(root) and not destination.is_relative_to(views):
        raise ValueError('Offline export cannot overwrite authoritative session data; use views/ or an external path')
    return destination


def export_html(session_dir: Path | str, output: Path | str) -> None:
    output = _export_destination(session_dir, output)
    root = Path(session_dir)
    report = validate_session(root)
    events, _ = read_events(root)
    request_roles = {event.request_id:event.payload.role for event in events if event.type=='model/request'}
    cards = []
    options = {key:set() for key in ('session','turn','step','role','state','type')}
    groups = {}
    for event in events:
        body = html.escape(event.model_dump_json(indent=2))
        attachments = []
        for ref in _refs(event):
            raw = verify_blob(root, ref).decode('utf-8', errors='replace')
            attachments.append('<details><summary>'+html.escape(ref.path)+'</summary><pre>'+html.escape(raw)+'</pre></details>')
        role = request_roles.get(event.request_id, 'operator' if event.source=='operator' else event.source)
        state_id = event.state_id or getattr(event.payload,'state_id',None) or ''
        metadata = {'session':event.session_id,'turn':event.turn_id or '', 'step':event.step_id or '',
                    'role':role,'state':state_id,'type':event.type}
        for key,value in metadata.items():
            if value:
                options[key].add(value)
        failed = getattr(event.payload,'status','') in ('error','failed','unknown','schema_error','cancelled','timeout') or getattr(event.payload,'error',None) is not None
        attrs = ' '.join(f'data-{key}="{html.escape(value,quote=True)}"' for key,value in metadata.items())
        attrs += ' data-error="'+('yes' if failed else 'no')+'"'
        refs = ' '.join(f'{key}={getattr(event,key) or ""}' for key in ['turn_id','step_id','request_id','atom_call_id','compaction_id','parent_id','state_id'])
        label = f'{event.seq} · {event.type} · {refs}'
        reasoning = event.type=='model/stream' and event.payload.channel=='reasoning'
        cards.append(f'<details id="event-{event.seq}" class="event" {attrs}'+('' if reasoning else ' open')+'><summary>'+html.escape(label)+'</summary><pre>'+body+'</pre>'+''.join(attachments)+'</details>')
        group_id = event.request_id or event.atom_call_id or event.compaction_id or event.execution_id
        if group_id:
            group = groups.setdefault(group_id, {'parent':event.parent_id,'role':role,'events':[]})
            if group['parent'] is None and event.parent_id:
                group['parent'] = event.parent_id
            group['events'].append(event)
    def trace(group_id, ancestry=()):
        if group_id in ancestry:
            return '<li>Cycle rejected in trace outline: '+html.escape(group_id)+'</li>'
        group = groups[group_id]
        links = ''.join(f'<li><a href="#event-{event.seq}">{event.seq} '+html.escape(event.type)+'</a></li>' for event in group['events'])
        children = ''.join(trace(child,ancestry+(group_id,)) for child,item in groups.items() if item['parent']==group_id)
        return '<li><details><summary>'+html.escape(group['role']+' · '+group_id)+'</summary><ul>'+links+children+'</ul></details></li>'
    outline = ''.join(trace(group_id) for group_id,group in groups.items() if group['parent'] not in groups)
    controls = []
    for key,values in options.items():
        controls.append('<label>'+html.escape(key.title())+' <select class="selector" data-field="'+key+'"><option value="">All</option>'+''.join('<option>'+html.escape(value)+'</option>' for value in sorted(values))+'</select></label>')
    schema_hash = digest(canonical(EVENT_ADAPTER.json_schema()))
    page = """<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>TADA offline session</title><style>body{font:14px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #bbb;padding:.6rem;margin:.5rem 0}input[type=search]{width:95%;padding:.7rem}summary{cursor:pointer}small{color:#555}label{display:inline-block;margin:.4rem}select{max-width:18rem}</style>
<h1>TADA offline session</h1><p>Read-only audit view. Raw provider records are separate from normalized content. Stream and assembled response are distinct views; usage is settled once. Reasoning is collapsed by default. The outline links nested requests to the canonical chronological events below.</p><input type="search" id="filter" placeholder="Search event text or correlation IDs"><div>"""+''.join(controls)+"""<label><input id="errors" type="checkbox">Errors and uncertainty only</label></div><small>No network or Python execution. Export is redacted and is not a byte-exact provider transcript. Validated by Pydantic event schema SHA-256: """+schema_hash+'</small><pre>'+html.escape(report.model_dump_json(indent=2))+'</pre><details><summary>Nested request and execution outline</summary><ul>'+outline+'</ul></details>'+''.join(cards)+"""<script>
const selectors=[...document.querySelectorAll('.selector')];
function filter(){const q=document.getElementById('filter').value.toLowerCase();const errors=document.getElementById('errors').checked;document.querySelectorAll('.event').forEach(card=>{card.hidden=!card.textContent.toLowerCase().includes(q)||(errors&&card.dataset.error!=='yes')||selectors.some(control=>control.value&&card.dataset[control.dataset.field]!==control.value);});}
document.getElementById('filter').addEventListener('input',filter);document.getElementById('errors').addEventListener('change',filter);selectors.forEach(control=>control.addEventListener('change',filter));
document.querySelectorAll('a[href^="#event-"]').forEach(link=>link.addEventListener('click',()=>{const target=document.getElementById(link.getAttribute('href').slice(1));if(target){target.hidden=false;target.open=true;}}));
</script>"""
    atomic_write(Path(output), page.encode('utf-8'))


def export_jsonl(session_dir: Path | str, output: Path | str) -> None:
    output = _export_destination(session_dir, output)
    validate_session(session_dir)
    events, _ = read_events(session_dir)
    atomic_write(output, b''.join(canonical(event)+b'\n' for event in events))
