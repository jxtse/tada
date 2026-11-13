"""Bounded secret removal across stream fragments, retaining source boundaries."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Literal
import base64
from pydantic import field_validator
from .contracts import StrictModel


@dataclass(frozen=True)
class RedactedPart:
    text: str
    fragment_index: int
    fragment_offset: int
    fragment_length: int
    redacted: bool
    metadata: object = None


@dataclass
class _Fragment:
    index: int
    length: int
    remaining: int
    offset: int
    metadata: object


class StreamingRedactor:
    """Retain at most longest-secret minus one characters between calls.

    Parts identify their original fragment and offset even when safe emission
    splits that fragment. Redaction changes emitted length, never source offsets.
    A failed/cancelled stream must call finish() to preserve its partial content.
    """
    def __init__(self, secrets=()):
        self.secrets = tuple(set(secret for secret in secrets if secret))
        self.window = max((len(secret) for secret in self.secrets), default=1) - 1
        self._text = ''
        self._masked: list[bool] = []
        self._fragments: deque[_Fragment] = deque()
        self._next = 0

    def feed(self, text: str, metadata=None) -> list[RedactedPart]:
        index = self._next
        self._next += 1
        if not text:
            return [RedactedPart('', index, 0, 0, False, metadata)]
        self._text += text
        self._masked.extend([False] * len(text))
        self._fragments.append(_Fragment(index, len(text), len(text), 0, metadata))
        self._mark()
        return self._emit(max(0, len(self._text) - self.window))

    def finish(self) -> list[RedactedPart]:
        self._mark()
        return self._emit(len(self._text))

    def _mark(self):
        for secret in self.secrets:
            start = 0
            while (found := self._text.find(secret, start)) >= 0:
                self._masked[found:found + len(secret)] = [True] * len(secret)
                start = found + 1

    def _emit(self, count):
        parts = []
        consumed = 0
        while count:
            fragment = self._fragments[0]
            take = min(count, fragment.remaining)
            chars = self._text[consumed:consumed + take]
            mask = self._masked[consumed:consumed + take]
            output = []
            hidden = False
            for char, redact in zip(chars, mask):
                if redact:
                    if not hidden:
                        output.append('[REDACTED]')
                    hidden = True
                else:
                    output.append(char)
                    hidden = False
            parts.append(RedactedPart(''.join(output), fragment.index, fragment.offset,
                fragment.length, any(mask), fragment.metadata))
            fragment.remaining -= take
            fragment.offset += take
            if not fragment.remaining:
                self._fragments.popleft()
            consumed += take
            count -= take
        self._text = self._text[consumed:]
        self._masked = self._masked[consumed:]
        return parts


def opaque_raw_fragment(raw: bytes, secrets) -> tuple[bytes, bool]:
    """Hide raw wire fragments when credentials could span JSON string fields.

    Structured text/reasoning/tool deltas are retained through StreamingRedactor.
    Raw bytes cannot safely be replayed verbatim with configured secrets: JSON
    escaping and SSE framing can separate the pieces of one secret. Explicitly
    hiding this raw-only copy avoids retaining a reconstructable credential.
    No-secret streams retain byte-exact wire data.
    """
    if any(secrets):
        return b'[REDACTED RAW STREAM: configured secret; see sanitized typed fragments]', True
    return raw, False


class RawTransportChunk(StrictModel):
    encoding: Literal['base64']
    data: str
    transport_chunk: Literal[True]

    @field_validator('data')
    @classmethod
    def valid_base64(cls, value):
        base64.b64decode(value, validate=True)
        return value

    def decode(self) -> bytes:
        return base64.b64decode(self.data, validate=True)
