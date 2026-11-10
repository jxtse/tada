"""Per-execution cancellation; tokens are never reset or shared with later cells."""
from __future__ import annotations
from contextlib import contextmanager
import threading


class ExecutionCancelled(RuntimeError):
    pass


class ExecutionCancellation:
    def __init__(self, event: threading.Event | None = None):
        self._event = event if event is not None else threading.Event()
        self._gate = threading.RLock()

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        # A commit already inside its guard may finish; cancellation wins every
        # later commit/dispatch admission. Never hold this gate during network I/O.
        with self._gate:
            self._event.set()

    cancel = set

    def check(self) -> None:
        if self.is_set():
            raise ExecutionCancelled('execution cancelled; no new semantic request or state commit')

    @contextmanager
    def guard(self):
        with self._gate:
            self.check()
            yield

    def admit_dispatch(self) -> None:
        # This is the dispatch linearization point. An admitted request remains
        # owned and charged even if cancellation arrives while its I/O drains.
        with self.guard():
            pass


def cancellation_token(value) -> ExecutionCancellation | None:
    if value is None or isinstance(value, ExecutionCancellation):
        return value
    # Compatibility for explicit fixture/event callers; production kernels own
    # immutable ExecutionCancellation instances instead of clearing an Event.
    return ExecutionCancellation(value)
