"""R4: the real Jupyter consumer must sanitize before any persistence."""
from queue import Empty
from types import SimpleNamespace
import pytest
from tada_runtime.audit import AuditLog, export_html
from tada_runtime.kernel import Kernel

SECRET='TEST_SECRET_VALUE_123'

class Client:
    def __init__(self,parts,parent,channel):
        self.messages=[{'header':{'msg_type':'stream'},'parent_header':{'msg_id':parent},'content':{'name':channel,'text':part}} for part in parts]
        self.messages.append({'header':{'msg_type':'status'},'parent_header':{'msg_id':'current'},'content':{'execution_state':'idle'}})
        self.shell=[{'header':{'msg_type':'execute_reply'},'parent_header':{'msg_id':'current'},'content':{'status':'ok'}}]
    def execute(self,*a,**k):return 'current'
    def get_iopub_msg(self,**k):
        if self.messages:return self.messages.pop(0)
        raise Empty
    def get_shell_msg(self,**k):
        if self.shell:return self.shell.pop(0)
        raise Empty

@pytest.mark.parametrize('boundary',range(1,len(SECRET)))
@pytest.mark.parametrize('parent',['current','previous'])
@pytest.mark.parametrize('channel',['stdout','stderr'])
def test_all_split_boundaries_are_safe_in_events_blobs_html(tmp_path,boundary,parent,channel):
    with AuditLog(tmp_path/'session',secrets=[SECRET]) as audit:
        kernel=Kernel.__new__(Kernel);kernel.audit=audit;kernel.closed=False;kernel.poisoned=False
        kernel.cancel_time=None;kernel.kernel_id='fixture';kernel.config=SimpleNamespace(cell_timeout_seconds=1.)
        kernel.client=Client([SECRET[:boundary],SECRET[boundary:]],parent,channel)
        out,err,_,_,raw,_=kernel._cell('pass','exec')
        audit.blob({'stdout':out,'stderr':err,'provider_raw':raw})
        if hasattr(kernel,'_flush_output_redactors'):kernel._flush_output_redactors()
        fragments=[e.payload.text for e in audit.events if e.type=='cell/stream' and e.payload.channel==channel]
        # Include raw stored envelope text too; it may not retain separate secret halves.
        alltext=''.join(p.read_text(errors='replace') for p in (tmp_path/'session').rglob('*') if p.is_file() and p.name!='.writer.lock')
        assert SECRET not in ''.join(fragments)
        assert SECRET[:boundary] not in alltext or SECRET[boundary:] not in alltext
        assert '[REDACTED]' in ''.join(fragments)
    export_html(tmp_path/'session',tmp_path/'review.html')
    assert SECRET not in (tmp_path/'review.html').read_text()


def test_secret_spanning_late_cells_keeps_parent_tail(tmp_path):
    with AuditLog(tmp_path/'session',secrets=[SECRET]) as audit:
        kernel=Kernel.__new__(Kernel);kernel.audit=audit;kernel.closed=False;kernel.poisoned=False
        kernel.cancel_time=None;kernel.kernel_id='fixture';kernel.config=SimpleNamespace(cell_timeout_seconds=1.)
        kernel.client=Client([SECRET[:10]],'previous','stderr');kernel._cell('pass','exec1')
        kernel.client=Client([SECRET[10:]],'previous','stderr');kernel._cell('pass','exec2')
        kernel._flush_output_redactors()
        pieces=''.join(e.payload.text for e in audit.events if e.type=='cell/stream')
        assert SECRET not in pieces and '[REDACTED]' in pieces
        for blob in (tmp_path/'session'/'blobs').iterdir():
            text=blob.read_text(errors='replace')
            assert SECRET[:10] not in text and SECRET[10:] not in text
