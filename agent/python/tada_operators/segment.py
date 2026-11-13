"""Deterministic long-row segmentation for scale-transparent Tagging.

Pure functions: no provider calls, no randomness. A long row is cut into
segments of at most ``segment_chars`` body characters, preferring paragraph or
line boundaries, with ``overlap_chars`` of shared text between neighbours. Every
segment after the first is prefixed with the document's first ``head_chars``
characters so parties and definitions stay visible. The rendered segment length
is bounded by ``head_chars + segment_chars + SEGMENT_OVERHEAD``.
"""
from __future__ import annotations

SEGMENT_MARKER = '::seg'
SEGMENT_OVERHEAD = 96
HEAD_OPEN = '[DOCUMENT START EXCERPT]\n'
HEAD_CLOSE = '\n[END EXCERPT]\n'


def segment_number(index: int) -> int:
    """1-based segment number shared by the segment id and the rendered header."""
    return index + 1


def segment_id(rid, index: int) -> str:
    """Host-assigned segment identifier; always a string, never model supplied.

    The suffix uses the same 1-based number as the ``[SEGMENT k/n]`` header inside the
    segment text. A 0-based suffix next to a 1-based header made models answer with the
    header number (``::seg4`` for the row ``::seg3``), which failed rid coverage.
    """
    return f'{rid}{SEGMENT_MARKER}{segment_number(index)}'


def _cut(text: str, start: int, limit: int, boundary: str) -> int:
    """End offset (exclusive) of the body that begins at ``start``."""
    hard = min(start + limit, len(text))
    if hard == len(text) or boundary == 'chars':
        return hard
    floor = start + max(1, limit // 2)
    separators = ('\n\n', '\n') if boundary == 'paragraph' else ('\n',)
    for separator in separators:
        position = text.rfind(separator, floor, hard)
        if position != -1:
            return position + len(separator)
    position = text.rfind(' ', floor, hard)
    return position + 1 if position != -1 else hard


def split_bodies(text: str, *, segment_chars: int, overlap_chars: int, boundary: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets that cover ``text`` completely, in order."""
    if segment_chars < 2 or overlap_chars < 0 or overlap_chars * 2 >= segment_chars:
        raise ValueError('segmentation requires segment_chars >= 2 and overlap_chars < segment_chars / 2')
    if boundary not in ('paragraph', 'line', 'chars'):
        raise ValueError('unknown segmentation boundary')
    spans, start = [], 0
    while True:
        end = _cut(text, start, segment_chars, boundary)
        spans.append((start, end))
        if end >= len(text):
            return spans
        # end >= start + segment_chars/2 > start + overlap_chars, so the next start always advances.
        start = end - overlap_chars


def render_segments(text: str, *, segment_chars: int, overlap_chars: int, head_chars: int, boundary: str) -> list[str]:
    """Rendered segment texts; each is at most head_chars + segment_chars + SEGMENT_OVERHEAD long."""
    spans = split_bodies(text, segment_chars=segment_chars, overlap_chars=overlap_chars, boundary=boundary)
    head = text[:head_chars]
    total = len(spans)
    rendered = []
    for index, (start, end) in enumerate(spans):
        prefix = HEAD_OPEN + head + HEAD_CLOSE if head and start > 0 else ''
        rendered.append(f'{prefix}[SEGMENT {segment_number(index)}/{total}]\n{text[start:end]}')
    return rendered
