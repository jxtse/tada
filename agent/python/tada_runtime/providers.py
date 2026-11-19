"""Explicit operator adapters with durable, typed request/stream/attempt accounting."""
from __future__ import annotations
import time
import codecs
import threading
import uuid
from typing import Annotated, Literal, TypeVar
import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, TypeAdapter, ValidationError
from .contracts import StrictModel, JSON, NormalizedResponse, TextBlock, ReasoningBlock, ToolCallBlock, Usage, ErrorInfo
from .config import ProviderConfig
from .budget import BudgetLedger
from .cancellation import ExecutionCancellation, ExecutionCancelled, cancellation_token
from .audit_contracts import BlobRef
from .streaming import StreamingRedactor, opaque_raw_fragment

T = TypeVar('T', bound=BaseModel)

class WireModel(BaseModel):
    """Vendor envelope projection; unknown fields remain in separately retained raw data."""
    model_config = ConfigDict(strict=True, extra='ignore', allow_inf_nan=False)

class WireUsage(WireModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def normalized(self):
        return Usage(input_tokens=self.input_tokens if self.input_tokens is not None else self.prompt_tokens,
                     output_tokens=self.output_tokens if self.output_tokens is not None else self.completion_tokens,
                     total_tokens=self.total_tokens)

class WireContent(WireModel):
    type: str
    text: str | None = None
    refusal: str | None = None

class WireOutput(WireModel):
    type: str
    phase: str | None = None
    content: list[WireContent] = Field(default_factory=list)
    summary: list[WireContent] = Field(default_factory=list)
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None

class WireResponse(WireModel):
    model: str | None = None
    status: str = ''
    output: list[WireOutput] = Field(default_factory=list)
    usage: WireUsage | None = None
    error: JsonValue = None
    incomplete_details: JsonValue = None

class ResponseEvent(WireModel):
    type: str
    delta: str | None = None
    response: WireResponse | None = None
    sequence_number: int | None = None

class WireFunction(WireModel):
    name: str | None = None
    arguments: str | None = None

class WireTool(WireModel):
    index: int = 0
    id: str | None = None
    function: WireFunction | None = None

class ChatMessage(WireModel):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[WireTool] = Field(default_factory=list)

class ChatChoice(WireModel):
    index: int = 0
    delta: ChatMessage | None = None
    message: ChatMessage | None = None
    finish_reason: str | None = None

class ChatChunk(WireModel):
    model: str | None = None
    choices: list[ChatChoice] = Field(default_factory=list)
    usage: WireUsage | None = None

class ReasoningParameters(StrictModel):
    effort: str

class InputText(StrictModel):
    type: Literal['input_text'] = 'input_text'
    text: str

class InputImage(StrictModel):
    type: Literal['input_image'] = 'input_image'
    image_url: str

class InputMessage(StrictModel):
    role: Literal['user'] = 'user'
    content: list[InputText | InputImage]

class ResponsesRequest(StrictModel):
    model: str
    input: str | list[InputMessage]
    max_output_tokens: int
    stream: Literal[True] = True
    temperature: float | None = None
    reasoning: ReasoningParameters | None = None

class ChatImageUrl(StrictModel):
    url: str

class ChatText(StrictModel):
    type: Literal['text'] = 'text'
    text: str

class ChatImage(StrictModel):
    type: Literal['image_url'] = 'image_url'
    image_url: ChatImageUrl

class RequestMessage(StrictModel):
    role: Literal['user'] = 'user'
    content: str | list[ChatText | ChatImage]

class StreamOptions(StrictModel):
    include_usage: Literal[True] = True

class ChatRequest(StrictModel):
    model: str
    messages: list[RequestMessage]
    max_completion_tokens: int
    stream: Literal[True] = True
    stream_options: StreamOptions = Field(default_factory=StreamOptions)
    temperature: float | None = None
    reasoning_effort: str | None = None

class RequestImage(StrictModel):
    """One image attached to an operator request: a text label followed by the image itself."""
    label: str
    media_type: Literal['image/png', 'image/jpeg', 'image/gif', 'image/webp']
    data_base64: str

    def data_url(self) -> str:
        return f'data:{self.media_type};base64,{self.data_base64}'


def prepare_request(cfg: ProviderConfig, prompt: str, schema: type[BaseModel], images: list[RequestImage] | None = None):
    import json
    prompt += '\nReturn only one JSON object matching this schema, without markdown fences:\n' + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    if cfg.protocol == 'openai-responses':
        if images:
            parts: list = [InputText(text=prompt)]
            for image in images:
                parts += [InputText(text=image.label), InputImage(image_url=image.data_url())]
            payload = [InputMessage(content=parts)]
        else:
            payload = prompt
        request = ResponsesRequest(model=cfg.model, input=payload, max_output_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature, reasoning=ReasoningParameters(effort=cfg.reasoning_effort) if cfg.reasoning_effort else None)
        path = '/responses'
    elif cfg.protocol == 'openai-chat':
        if images:
            content: list = [ChatText(text=prompt)]
            for image in images:
                content += [ChatText(text=image.label), ChatImage(image_url=ChatImageUrl(url=image.data_url()))]
            message = RequestMessage(content=content)
        else:
            message = RequestMessage(content=prompt)
        request = ChatRequest(model=cfg.model, messages=[message], max_completion_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature, reasoning_effort=cfg.reasoning_effort)
        path = '/chat/completions'
    else:
        raise ValueError('operator protocol must be explicitly Responses or Chat')
    body = request.model_dump(mode='json', exclude_none=True)
    return body, path


class ProviderFailure(RuntimeError):
    pass


class TransientProviderFailure(ProviderFailure):
    """A failure that a later identical request may not repeat: HTTP 408/429/5xx, a broken
    connection, or a stream that ended without a terminal response. Retried by the engine."""


class OutputTruncated(ProviderFailure):
    """The model stopped because it hit max_output_tokens (often all spent on reasoning).
    The engine splits a multi-row batch in halves instead of failing the whole operator."""


TRANSIENT_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

def structured_answer_text(content) -> str:
    """Return the one answer text that strict schema validation receives.

    The Responses API may return several assistant messages labelled with a phase:
    'commentary' is narration and is never part of the answer. When any phase is
    reported, the answer comes from the remaining (non-commentary) messages; parts
    inside one message are joined. Several answer messages are accepted only when
    they decode to the same JSON value (or are textually identical); otherwise the
    response is refused instead of silently choosing one. Without phase metadata the
    text blocks are joined exactly as before.
    """
    import json
    blocks = [block for block in content if block.type == 'text']
    if not any(block.phase for block in blocks):
        return ''.join(block.text for block in blocks)
    messages: dict = {}
    for block in blocks:
        if block.phase != 'commentary':
            messages.setdefault(block.message_index, []).append(block.text)
    answers = [''.join(parts) for parts in messages.values()]
    if not answers:
        raise ProviderFailure('response contains only commentary messages and no final answer')
    if len(answers) > 1:
        try:
            decoded = [json.loads(answer) for answer in answers]
            same = all(value == decoded[0] for value in decoded[1:])
        except ValueError:
            same = len(set(answers)) == 1
        if not same:
            raise ProviderFailure(f'response contains {len(answers)} differing answer messages; refusing to choose one')
    return answers[0]

class _CancellationTransport(httpx.BaseTransport):
    """Check again at transport entry, after httpx has prepared the request."""
    def __init__(self, inner, cancellation):
        self.inner, self.cancellation = inner, cancellation
        self.dispatched = False

    def handle_request(self, request):
        self.cancellation.admit_dispatch()
        self.dispatched = True
        return self.inner.handle_request(request)

    def close(self):
        self.inner.close()


class Provider:
    def __init__(self, config: ProviderConfig, budget: BudgetLedger, collector, *, transport=None, concurrency: int = 1):
        self.config, self.budget, self.collector = config, budget, collector
        self.transport = transport
        concurrency = TypeAdapter(Annotated[StrictInt, Field(ge=1, le=8)]).validate_python(concurrency)
        self.concurrency = concurrency
        self._slots = threading.BoundedSemaphore(concurrency)
        self.last_response: NormalizedResponse | None = None
        self._local = threading.local()

    @property
    def last_request_id(self) -> str | None:
        """Request id of the most recent call made from the current thread (for retry linkage)."""
        local = getattr(self, '_local', None)
        return getattr(local, 'request_id', None) if local is not None else None

    def call(self, prompt: str, schema: type[T], *, role: str, parent_id: str | None = None,
             context: dict | None = None, cancellation: ExecutionCancellation | None = None,
             retry_of: str | None = None, images: list[RequestImage] | None = None) -> T:
        cancellation = cancellation_token(cancellation)
        # Poll only while queued; an already-dispatched request is drained and
        # settled once rather than discarded or silently retried.
        while True:
            if cancellation:
                cancellation.check()
            if self._slots.acquire(timeout=.05):
                break
        try:
            if cancellation:
                cancellation.check()
            return self._call_impl(prompt, schema, role=role, parent_id=parent_id,
                context=context, cancellation=cancellation, retry_of=retry_of, images=images)
        finally:
            self._slots.release()

    @staticmethod
    def _incomplete_reason(normalized) -> str | None:
        """The provider's stated reason for an incomplete response, when it reported one."""
        raw = normalized.provider_raw
        candidates = []
        if isinstance(raw, dict):
            candidates.append(raw)
        elif isinstance(raw, list):
            candidates.extend(item.get('response') for item in reversed(raw) if isinstance(item, dict) and isinstance(item.get('response'), dict))
        for response in candidates:
            details = response.get('incomplete_details') if isinstance(response, dict) else None
            if isinstance(details, dict) and isinstance(details.get('reason'), str):
                return details['reason']
        return None

    def _call_impl(self, prompt: str, schema: type[T], *, role: str, parent_id: str | None = None,
                   context: dict | None = None, cancellation: ExecutionCancellation | None = None,
                   retry_of: str | None = None, images: list[RequestImage] | None = None) -> T:
        cfg = self.config
        body, path = prepare_request(cfg, prompt, schema, images)
        key = cfg.api_key()
        headers = {'Accept': 'text/event-stream', 'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer ' + key
        rid = uuid.uuid4().hex
        if getattr(self, '_local', None) is not None:
            self._local.request_id = rid
        ids = {'request_id': rid, 'attempt_id': rid + ':1', 'parent_id': parent_id, 'source': 'operator'}
        if cancellation:
            cancellation.check()
        request_ref = self.collector.blob(body)
        self.collector.emit('model/request', {'role': role, 'requested_model': cfg.model, 'protocol': cfg.protocol,
                                             'request_ref': request_ref.model_dump(), 'parameters': cfg.model_dump(exclude={'api_key_env', 'auth_placeholder'})}, **ids)
        started = time.monotonic()
        status = 'failed'
        reserved = False
        raw_ref = None
        usage = Usage()
        try:
            if cancellation:
                cancellation.check()
            self.budget.reserve(role, cfg.model, rid, output_cap=cfg.max_output_tokens)
            reserved = True
            status = 'unknown'
            if cancellation:
                cancellation.check()
            transport = self.transport
            if cancellation:
                transport = _CancellationTransport(transport if transport is not None else httpx.HTTPTransport(), cancellation)
            try:
                with httpx.Client(timeout=cfg.timeout_seconds, transport=transport, trust_env=False) as client:
                    with client.stream('POST', cfg.base_url.rstrip('/') + path, json=body, headers=headers) as response:
                        if response.is_error:
                            raw_ref = self.collector.blob(response.read().decode('utf-8', errors='replace'))
                            status = 'failed'
                            failure = TransientProviderFailure if response.status_code in TRANSIENT_HTTP_STATUS else ProviderFailure
                            raise failure(f'provider HTTP {response.status_code}; see raw response artifact')
                        if 'text/event-stream' in response.headers.get('content-type', ''):
                            normalized, raw_ref = self._stream(response, ids)
                        else:
                            raw_bytes = response.read()
                            raw_ref = self.collector.blob_bytes(raw_bytes, media_type='application/json')
                            raw = raw_bytes.decode('utf-8')
                            parsed = JSON.validate_json(raw)
                            normalized = self._normalize_response(parsed) if cfg.protocol == 'openai-responses' else self._normalize_chat(parsed)
            except httpx.TransportError as exc:
                # The request may or may not have reached the provider; the attempt stays 'unknown'.
                raise TransientProviderFailure(f'transport error: {type(exc).__name__}: {exc}') from exc
            usage = normalized.usage
            self.last_response = normalized
            response_payload = normalized.model_dump(exclude={'schema_version', 'provider_raw'})
            # The immutable response blob retains vendor extensions and complete stream records.
            response_payload['provider_raw'] = None
            response_payload['response_ref'] = raw_ref.model_dump()
            self.collector.emit('model/response', response_payload, **ids)
            if normalized.finish_reason not in ('completed', 'stop'):
                status = 'failed'
                reason = self._incomplete_reason(normalized)
                if normalized.finish_reason == 'length' or reason == 'max_output_tokens':
                    raise OutputTruncated(f'provider did not complete: {normalized.finish_reason} '
                        f'(reason: max_output_tokens={cfg.max_output_tokens}; reasoning may have used the whole allowance)')
                raise ProviderFailure('provider did not complete: ' + normalized.finish_reason + (f' (reason: {reason})' if reason else ''))
            status = 'failed'
            text = structured_answer_text(normalized.content)
            result = schema.model_validate_json(text, context=context)
            status = 'success'
            self.collector.emit('model/attempt_end', {'status': status, 'elapsed_seconds': time.monotonic() - started, 'retry_of': retry_of}, **ids)
            return result
        except Exception as exc:
            if isinstance(exc, ExecutionCancelled):
                status = 'cancelled'
            details = [f'{e["loc"]}: {e["type"]}' for e in exc.errors(include_input=False, include_url=False)] if isinstance(exc, ValidationError) else []
            error = ErrorInfo(kind=type(exc).__name__, message='provider or response contract failure; inspect retained payload' if isinstance(exc, ValidationError) else str(exc), details=details, raw_ref=raw_ref)
            self.collector.emit('model/attempt_end', {'status': 'schema_error' if isinstance(exc, ValidationError) else status,
                'error': error.model_dump(), 'elapsed_seconds': time.monotonic() - started, 'retry_of': retry_of}, **ids)
            raise
        finally:
            if reserved:
                self.budget.settle(rid, status, usage)

    def _stream(self, response, ids):
        redactors = {}
        try:
            return self._read_stream(response, ids, redactors)
        finally:
            for channel, redactor in redactors.items():
                self._emit_parts(channel, redactor.finish(), ids)

    def _emit_parts(self, channel, parts, ids):
        for part in parts:
            metadata = part.metadata
            self.collector.emit('model/stream', {'channel': channel, 'text': part.text,
                'provider_sequence': metadata['provider_sequence'], 'availability': metadata['availability'],
                'block_index': part.fragment_index, 'fragment_offset': part.fragment_offset,
                'fragment_length': part.fragment_length, 'redacted': part.redacted}, **ids)

    def _stream_delta(self, redactors, channel, text, sequence, availability, ids):
        redactor = redactors.setdefault(channel, StreamingRedactor(self.collector._secrets))
        self._emit_parts(channel, redactor.feed(text, {'provider_sequence': sequence, 'availability': availability}), ids)

    def _stream_raw(self, value):
        if self.collector._secrets:
            ref = self.collector.blob({'redacted': True, 'reason': 'Raw stream can contain secrets split across encoded fields; see sanitized typed content'})
            return BlobRef.model_validate({**ref.model_dump(), 'redacted': True})
        return self.collector.blob(value)

    def _read_stream(self, response, ids, redactors):
        raws: list[JsonValue] = []
        final = None
        chat_text, chat_reasoning = '', ''
        reported_model = None
        usage = Usage()
        finish = ''
        sequence = 0
        data_lines: list[str] = []
        def lines():
            decoder = codecs.getincrementaldecoder('utf-8')()
            buffer = ''
            for chunk_index, chunk in enumerate(response.iter_bytes()):
                # Preserve exact decoded HTTP-body chunks before text/SSE parsing can fail.
                safe_chunk, hidden = opaque_raw_fragment(chunk, self.collector._secrets)
                chunk_ref = self.collector.blob_bytes(safe_chunk, media_type='text/event-stream')
                if hidden:
                    chunk_ref = BlobRef.model_validate({**chunk_ref.model_dump(), 'redacted': True})
                self.collector.emit('model/stream', {'channel': 'provider_raw', 'provider_sequence': chunk_index,
                    'block_index': chunk_index, 'fragment_length': len(chunk), 'redacted': hidden, 'raw_ref': chunk_ref.model_dump()}, **ids)
                buffer += decoder.decode(chunk)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    yield line.rstrip('\r')
            buffer += decoder.decode(b'', final=True)
            if buffer:
                yield buffer.rstrip('\r')

        def events():
            for line in lines():
                if line.startswith('data:'):
                    # SSE permits one optional leading space, not arbitrary stripping.
                    value = line[5:]
                    data_lines.append(value[1:] if value.startswith(' ') else value)
                elif not line and data_lines:
                    yield '\n'.join(data_lines)
                    data_lines.clear()
            if data_lines:
                yield '\n'.join(data_lines)
        for raw in events():
            sequence += 1
            raw_ref = self._stream_raw(raw)
            self.collector.emit('model/stream', {'channel': 'provider_raw', 'provider_sequence': sequence, 'raw_ref': raw_ref.model_dump()}, **ids)
            if raw == '[DONE]':
                continue
            parsed = JSON.validate_json(raw)
            raws.append(parsed)
            if self.config.protocol == 'openai-responses':
                event = ResponseEvent.model_validate(parsed)
                channel = {'response.output_text.delta': 'text', 'response.reasoning_summary_text.delta': 'reasoning',
                           'response.reasoning_text.delta': 'reasoning', 'response.function_call_arguments.delta': 'tool_call'}.get(event.type)
                if channel and event.delta is not None:
                    self._stream_delta(redactors, channel, event.delta, sequence, 'summary_only' if 'summary' in event.type else 'exposed', ids)
                if event.type in ('response.completed', 'response.incomplete', 'response.failed'):
                    if event.response is None:
                        raise ProviderFailure('terminal response event has no response')
                    final = event.response
            else:
                chunk = ChatChunk.model_validate(parsed)
                reported_model = chunk.model or reported_model
                if chunk.usage:
                    usage = chunk.usage.normalized()
                for choice in chunk.choices:
                    if choice.index != 0:
                        raise ProviderFailure('multiple chat choices are not supported')
                    finish = choice.finish_reason or finish
                    delta = choice.delta
                    if delta:
                        for channel, text in [('text', delta.content), ('reasoning', delta.reasoning_content)]:
                            if text:
                                self._stream_delta(redactors, channel, text, sequence, 'exposed', ids)
                                if channel == 'text': chat_text += text
                                else: chat_reasoning += text
                        for tool in delta.tool_calls:
                            self._stream_delta(redactors, 'tool_call', tool.function.arguments if tool.function and tool.function.arguments else '', sequence, 'not_provided', ids)
        raw_ref = self._stream_raw(raws)
        if self.config.protocol == 'openai-responses':
            if final is None:
                raise TransientProviderFailure('stream disconnected without terminal response; result unknown')
            normalized = self._normalize_response(final.model_dump())
            normalized.provider_raw = raws
            normalized = NormalizedResponse.model_validate(normalized.model_dump())
            return normalized, raw_ref
        if not finish:
            raise TransientProviderFailure('stream disconnected without finish reason; result unknown')
        blocks = [TextBlock(text=chat_text)]
        if chat_reasoning:
            blocks.insert(0, ReasoningBlock(text=chat_reasoning, availability='exposed'))
        return NormalizedResponse(requested_model=self.config.model, reported_model=reported_model, protocol=self.config.protocol,
            content=blocks, usage=usage, finish_reason=finish, reasoning_availability='exposed' if chat_reasoning else 'not_provided', provider_raw=raws), raw_ref

    def _normalize_response(self, raw):
        wire = WireResponse.model_validate(raw)
        content = []
        availability = 'not_provided'
        for index, item in enumerate(wire.output):
            if item.type == 'message':
                content.extend(TextBlock(text=part.text, phase=item.phase, message_index=index)
                               for part in item.content if part.type == 'output_text' and part.text is not None)
            elif item.type == 'reasoning':
                availability = 'summary_only' if item.summary else 'opaque'
                content.extend(ReasoningBlock(text=part.text or '', availability=availability) for part in item.summary)
            elif item.type == 'function_call':
                content.append(ToolCallBlock(call_id=item.call_id or '', name=item.name or '', arguments=item.arguments or ''))
        return NormalizedResponse(requested_model=self.config.model, reported_model=wire.model, protocol=self.config.protocol,
            content=content, usage=wire.usage.normalized() if wire.usage else Usage(), finish_reason=wire.status,
            reasoning_availability=availability, provider_raw=raw)

    def _normalize_chat(self, raw):
        wire = ChatChunk.model_validate(raw)
        if len(wire.choices) != 1 or wire.choices[0].message is None:
            raise ProviderFailure('expected exactly one chat message')
        message = wire.choices[0].message
        content = [TextBlock(text=message.content or '')]
        if message.reasoning_content:
            content.insert(0, ReasoningBlock(text=message.reasoning_content, availability='exposed'))
        return NormalizedResponse(requested_model=self.config.model, reported_model=wire.model, protocol=self.config.protocol,
            content=content, usage=wire.usage.normalized() if wire.usage else Usage(), finish_reason=wire.choices[0].finish_reason or '',
            reasoning_availability='exposed' if message.reasoning_content else 'not_provided', provider_raw=raw)
