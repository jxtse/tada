"""Cumulative, process-safe request accounting. Unknown attempts consume budget."""
from __future__ import annotations
import fcntl
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Literal
from pydantic import Field, model_validator
from .contracts import StrictModel, Usage
from .config import BudgetConfig

class Reservation(StrictModel):
    request_id: str
    role: str
    requested_model: str
    reserved_at: float
    status: Literal['reserved', 'success', 'failed', 'unknown', 'cancelled'] = 'reserved'
    usage: Usage = Field(default_factory=Usage)

class LedgerState(StrictModel):
    schema_version: Literal['1.0'] = '1.0'
    max_requests: int = Field(ge=1, le=2000)
    deadline_at: float
    reservations: list[Reservation] = Field(default_factory=list)

    @model_validator(mode='after')
    def unique_ids(self):
        ids = [x.request_id for x in self.reservations]
        if len(ids) != len(set(ids)) or len(ids) > self.max_requests:
            raise ValueError('invalid budget ledger reservations')
        return self

class BudgetExceeded(RuntimeError):
    pass

class BudgetLedger:
    def __init__(self, config: BudgetConfig, collector=None):
        self.config = config
        self.path = Path(config.ledger_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_path = self.path.with_suffix(self.path.suffix + '.lock')
        self.collector = collector
        self._thread_lock = threading.RLock()
        self._transaction(lambda state: None)

    def _transaction(self, operation):
        with self._thread_lock, self.lock_path.open('a+b') as lock:
            os.chmod(self.lock_path, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = (LedgerState.model_validate_json(self.path.read_bytes()) if self.path.exists() else
                     LedgerState(max_requests=self.config.max_requests, deadline_at=time.time() + self.config.deadline_seconds))
            state.max_requests = min(state.max_requests, self.config.max_requests)
            result = operation(state)
            state = LedgerState.model_validate(state.model_dump())
            fd, tmp = tempfile.mkstemp(prefix='.ledger-', dir=self.path.parent)
            try:
                with os.fdopen(fd, 'wb') as handle:
                    handle.write(state.model_dump_json().encode())
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, self.path)
                directory = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            return result

    def snapshot(self) -> LedgerState:
        return self._transaction(lambda state: state.model_copy(deep=True))

    def reserve(self, role: str, requested_model: str, request_id: str, *, output_cap: int = 0) -> Reservation:
        def reserve(state):
            if any(x.request_id == request_id for x in state.reservations):
                raise ValueError('request already reserved; never dispatch twice')
            if time.time() >= state.deadline_at:
                raise BudgetExceeded('request deadline exhausted')
            if len(state.reservations) >= state.max_requests:
                raise BudgetExceeded('cumulative request limit exhausted')
            item = Reservation(request_id=request_id, role=role, requested_model=requested_model, reserved_at=time.time())
            state.reservations.append(item)
            return item, len(state.reservations), state.max_requests
        item, used, limit = self._transaction(reserve)
        if self.collector:
            self.collector.emit('budget/reserve', {'role': role, 'request_id': request_id, 'reserved_count': used, 'limit': limit, 'output_cap': output_cap}, request_id=request_id)
        return item

    def settle(self, request_id: str, status: Literal['success', 'failed', 'unknown', 'cancelled'], usage: Usage | None = None):
        usage = usage or Usage()
        def settle(state):
            entries = [x for x in state.reservations if x.request_id == request_id]
            if not entries:
                raise ValueError('cannot settle unreserved request')
            entry = entries[0]
            if entry.status != 'reserved':
                if entry.status == status and entry.usage == usage:
                    return False
                raise ValueError('request already settled differently')
            entry.status, entry.usage = status, usage
            return True
        changed = self._transaction(settle)
        if changed and self.collector:
            self.collector.emit('budget/settle', {'request_id': request_id, 'status': status, 'usage': usage.model_dump()}, request_id=request_id)
