"""Single-owner persistent Jupyter kernel with correlated shell/IOPub completion."""
from __future__ import annotations
import base64
from datetime import datetime
import os
from pathlib import Path
from queue import Empty
import sys
import threading
import time
import uuid
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from .rpc_contracts import Observation
from .audit_contracts import ErrorInfo
from .streaming import StreamingRedactor
from .cancellation import ExecutionCancellation


class JupyterWire(BaseModel):
    """Vendor projection; extension fields remain in the separately retained raw envelope."""
    model_config = ConfigDict(strict=True, extra='ignore', allow_inf_nan=False)


class JupyterHeader(JupyterWire):
    msg_id: str = ''
    msg_type: str = ''


class JupyterEnvelope(JupyterWire):
    header: JupyterHeader
    parent_header: JupyterHeader = Field(default_factory=JupyterHeader)
    content: dict[str, JsonValue]


class StreamContent(JupyterWire):
    name: Literal['stdout', 'stderr']
    text: str


class ErrorContent(JupyterWire):
    ename: str
    evalue: str
    traceback: list[str]


class StatusContent(JupyterWire):
    execution_state: str


def wire_json(value):
    """Encode Jupyter's decoded dates/buffers explicitly, then validate JSON."""
    def encode(item):
        if isinstance(item, datetime):
            return item.isoformat()
        if isinstance(item, (bytes, memoryview)):
            return {'encoding':'base64','data':base64.b64encode(bytes(item)).decode('ascii')}
        if isinstance(item, dict):
            return {key:encode(child) for key,child in item.items()}
        if isinstance(item, (list,tuple)):
            return [encode(child) for child in item]
        return item
    return TypeAdapter(JsonValue).validate_python(encode(value),strict=True)


class Kernel:
    def __init__(self, audit, store, config, bridge_url, bridge_token):
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpec
        self.audit, self.store, self.config = audit, store, config
        self.lock = threading.Lock()
        self.cancellation = ExecutionCancellation()
        self.cancelled = self.cancellation
        self.namespace = []
        self.active_execution = None
        self.closed = False
        self.closing = False
        self.cancel_time = None
        self.poisoned = False
        self.client = None
        self.km = KernelManager(connection_file=str(audit.root/'kernel.json'))
        self.km._kernel_spec = KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'], display_name='TADA Python', language='python')
        env = {key:os.environ[key] for key in ('PATH','HOME','TMPDIR','LANG') if key in os.environ}
        env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]), JUPYTER_CONFIG_DIR=str(audit.root/'jupyter'), IPYTHONDIR=str(audit.root/'ipython'))
        try:
            self.km.start_kernel(env=env,cwd=str(audit.root/'outputs'))
            self.client = self.km.client()
            self.client.start_channels()
            self.client.wait_for_ready(timeout=30)
            self.kernel_id = str(self.km.kernel_id)
            bootstrap = ('from tada_runtime.kernel_client import install as _tada_install\n'
                         f'_tada_install(globals(), {bridge_url!r}, {bridge_token!r}, {list(store.aliases)!r})\n'
                         '_tada_inventory()')
            _, _, _, error, _, _ = self._cell(bootstrap, 'bootstrap', visible=False)
            if error:
                raise RuntimeError(f'kernel bootstrap failed: {error.message}')
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    def _persist_stream(self, text, parent, channel, execution_id, late=False, raw_ref=None):
        # A parent can emit late output after idle. Keep its privacy tail until close.
        if not hasattr(self, '_output_redactors'):
            self._output_redactors = {}
        key = (parent, channel)
        redactor = self._output_redactors.setdefault(key, StreamingRedactor(self.audit._secrets))
        metadata = {'execution_id': execution_id, 'parent': parent, 'channel': channel, 'late': late, 'raw_ref': raw_ref}
        self._emit_output_parts(redactor.feed(text, metadata))

    def _emit_output_parts(self, parts):
        for part in parts:
            meta = part.metadata
            self.audit.append('cell/stream', {'channel': meta['channel'], 'text': part.text,
                'parent_message_id': meta['parent'], 'late': meta['late'], 'raw_ref':meta['raw_ref']},
                execution_id=meta['execution_id'], source='kernel', redacted=part.redacted)

    def _flush_output_redactors(self):
        for redactor in getattr(self, '_output_redactors', {}).values():
            self._emit_output_parts(redactor.finish())
        self._output_redactors = {}

    def _safe_raw(self, message):
        if not self.audit._secrets:
            return message
        # Escaping and envelope boundaries make raw chunks unsafe to scrub separately.
        # Sanitized typed streams are retained; never retain reconstructable raw halves.
        return {'redacted': True, 'reason': 'credentialed Jupyter raw envelope; see typed streams',
                'header': message.get('header', {}), 'parent_header': message.get('parent_header', {})}

    def _cell(self, code, execution_id, visible=True):
        if self.closed or self.poisoned:
            raise RuntimeError('kernel unavailable; explicit session recovery required')
        if visible:
            # Durable intent precedes the Jupyter send; correlated message IDs follow in streams.
            self.audit.append('cell/start',{'code':code,'kernel_id':self.kernel_id},execution_id=execution_id,source='kernel')
        msg_id = self.client.execute(code, store_history=visible, stop_on_error=True)
        stdout, stderr, displays, raw = [], [], [], []
        error = None
        shell_done = idle = False
        deadline = time.monotonic()+self.config.cell_timeout_seconds
        interrupted = False
        while not (shell_done and idle):
            if self.cancel_time is not None and time.monotonic() > self.cancel_time + 10:
                self.poisoned = True
                raise TimeoutError('kernel did not settle after cancellation; explicit recovery required')
            if time.monotonic()>deadline:
                if not interrupted:
                    self.cancelled.set()
                    self.km.interrupt_kernel()
                    interrupted=True
                    deadline=time.monotonic()+10
                else:
                    self.poisoned = True
                    raise TimeoutError('kernel did not settle after interrupt; owner must close before further execution')
            try:
                raw_message=wire_json(self.client.get_iopub_msg(timeout=.05))
                message=JupyterEnvelope.model_validate(raw_message)
                kind=message.header.msg_type
                parent=message.parent_header.msg_id
                content=message.content
                if parent!=msg_id:
                    if kind=='stream':
                        stream=StreamContent.model_validate(content)
                        self._persist_stream(stream.text,parent,stream.name,execution_id,late=True,raw_ref=self.audit.blob(self._safe_raw(raw_message)).model_dump())
                    if kind in ('display_data','execute_result','error'):
                        self.audit.append('cell/stream',{'channel':'status','raw_ref':self.audit.blob(self._safe_raw(raw_message)).model_dump(),'parent_message_id':parent,'late':True},execution_id=execution_id,source='kernel')
                    continue
                raw.append(self._safe_raw(raw_message))
                if kind=='status' and StatusContent.model_validate(content).execution_state=='idle':
                    idle=True
                elif kind=='stream':
                    stream=StreamContent.model_validate(content)
                    (stdout if stream.name=='stdout' else stderr).append(stream.text)
                    if visible:
                        self._persist_stream(stream.text,parent,stream.name,execution_id)
                elif kind in ('display_data','execute_result'):
                    ref=self.audit.blob(self._safe_raw(raw_message) if self.audit._secrets else content)
                    displays.append(ref)
                    if visible:
                        self.audit.append('cell/stream',{'channel':'display','raw_ref':ref.model_dump(),'parent_message_id':parent},execution_id=execution_id,source='kernel')
                elif kind=='error':
                    exception=ErrorContent.model_validate(content)
                    error=ErrorInfo(kind=exception.ename,message=exception.evalue,details=exception.traceback)
                    if visible:
                        self.audit.append('cell/stream',{'channel':'exception','text':error.message,'raw_ref':self.audit.blob(self._safe_raw(raw_message) if self.audit._secrets else content).model_dump(),'parent_message_id':parent},execution_id=execution_id,source='kernel')
            except Empty:
                pass
            try:
                raw_reply=wire_json(self.client.get_shell_msg(timeout=.01))
                reply=JupyterEnvelope.model_validate(raw_reply)
                if reply.parent_header.msg_id==msg_id:
                    raw.append(self._safe_raw(raw_reply))
                    shell_done=True
            except Empty:
                pass
        return ''.join(stdout), ''.join(stderr), displays, error, raw, interrupted

    def execute(self, code):
        if self.closed or self.closing or self.poisoned:
            raise RuntimeError('kernel unavailable; explicit session recovery required')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('only one cell may be in flight')
        execution_id=uuid.uuid4().hex
        self.active_execution=execution_id
        self.cancellation = ExecutionCancellation()
        self.cancelled = self.cancellation
        self.cancel_time = None
        before={s.state_id for s in self.store.list()}
        try:
            out,err,displays,error,raw,timedout=self._cell(code,execution_id)
            self._cell('_tada_inventory()',execution_id,visible=False)
            full=self.audit.blob({'stdout':out,'stderr':err,'exception':error.model_dump() if error else None,'provider_raw':raw})
            status='timeout' if timedout else ('cancelled' if self.cancelled.is_set() else ('error' if error else 'ok'))
            limit=self.config.output_view_chars
            out=self.audit._redact(out)[0]
            err=self.audit._redact(err)[0]
            if error:
                error=ErrorInfo.model_validate(self.audit._redact(error.model_dump(mode='json'))[0])
            observation=Observation(execution_id=execution_id,status=status,stdout=out[:limit],stderr=err[:limit],exception=error,displays=displays,namespace_summary=self.namespace,committed_state_refs=[s.state_id for s in self.store.list() if s.state_id not in before],truncated=len(out)>limit or len(err)>limit,full_output_ref=full)
            self.audit.append('cell/end',{'status':status,'stdout':observation.stdout,'stderr':observation.stderr,'error':error.model_dump() if error else None,'namespace_summary':self.namespace,'committed_state_refs':observation.committed_state_refs,'output_ref':full.model_dump()},execution_id=execution_id,source='kernel')
            return observation
        except BaseException:
            # I/O/protocol failures can leave work in flight. Never dispatch another cell blindly.
            self.poisoned=True
            raise
        finally:
            self.active_execution=None
            self.lock.release()

    def cancel(self):
        if self.active_execution and not self.closed:
            self.cancelled.set()
            self.cancel_time = time.monotonic()
            self.km.interrupt_kernel()

    def close(self):
        if self.closed:
            return
        self.closing = True
        try:
            self.cancel()
            with self.lock:
                if self.km.has_kernel:
                    self.km.shutdown_kernel(now=True)
        finally:
            try:
                self._flush_output_redactors()
            finally:
                self.closed=True
                if self.client:
                    self.client.stop_channels()
