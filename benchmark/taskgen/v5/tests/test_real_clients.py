"""Real API implementations are tested exclusively with injected transports."""
import copy
import json
from pathlib import Path
import shutil
import socket

import pytest

from tada_taskgen import clients
from tada_taskgen.clients import OpenAICompatibleClient, ModelRequestError
from tada_taskgen.compiler import compile_task
from tada_taskgen.docgen import generate_document
from screened_support import integration_adapters, screened_spec


@pytest.fixture(autouse=True)
def no_sockets(monkeypatch):
    def prohibited(*args, **kwargs):
        pytest.fail('No real HTTP or socket calls permitted in client tests')
    monkeypatch.setattr(clients, 'http_transport', prohibited)
    monkeypatch.setattr(socket, 'create_connection', prohibited)
    monkeypatch.setattr(socket.socket, 'connect', prohibited)
    monkeypatch.setenv('R3_TEST_CREDENTIAL', 'test-only-not-a-real-key')


def config(api='chat_completions', **extra):
    return {'model': 'test-writer' if api == 'chat_completions' else 'test-extractor',
            'base_url': 'http://127.0.0.1:1/v1', 'api': api,
            'key_env': 'R3_TEST_CREDENTIAL', 'supported_parameters': [], **extra}


def chat(text):
    return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}


def responses(text):
    return {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]}


def writer_payload():
    return {'id': 'notice', 'kind': 'email', 'facts': {'cutoff': '2026-10-01'}, 'attempt': 1}


def test_writer_protocol_natural_prose_prompt_and_header_free_logs(tmp_path):
    calls = []
    def transport(url, body, headers, timeout):
        calls.append((url, body, headers, timeout))
        return 200, chat('Dear team, please complete the review by October 1, 2026.')
    client = OpenAICompatibleClient(config(temperature=.4, seed=7, supported_parameters=['temperature', 'seed']),
                                    tmp_path, transport=transport)
    result = client.complete('write_document', writer_payload())
    url, body, headers, _ = calls[0]
    assert url.endswith('/v1/chat/completions')
    assert body['model'] == 'test-writer' and body['temperature'] == .4 and body['seed'] == 7
    prompt = body['messages'][0]['content']
    assert 'natural, varied prose' in prompt and 'Do not list the facts as a table' in prompt and 'JSON' in prompt
    assert headers['Authorization'] == 'Bearer test-only-not-a-real-key'
    assert result['text'].startswith('Dear team')
    log = (tmp_path/'build/model_requests.jsonl').read_text()
    assert 'Authorization' not in log and 'headers' not in log and 'test-only-not-a-real-key' not in log
    assert json.loads(log)['request']['temperature'] == .4


def test_extractor_receives_only_document_and_field_names(tmp_path):
    calls = []
    def transport(url, body, headers, timeout):
        calls.append((url, body))
        return 200, responses('{"facts":{"cutoff":"2026-10-01"}}')
    client = OpenAICompatibleClient(config('responses', temperature=.9, seed=99), tmp_path, transport=transport)
    result = client.complete('extract_facts', {'text': 'Cutoff: October 1, 2026', 'fact_names': ['cutoff'],
        'facts': {'cutoff': 'DO-NOT-TRANSMIT'}, 'history': 'DO-NOT-TRANSMIT', 'instruction': 'DO-NOT-TRANSMIT'})
    assert result == {'facts': {'cutoff': '2026-10-01'}}
    url, body = calls[0]
    assert url.endswith('/v1/responses') and body['store'] is False
    assert 'temperature' not in body and 'seed' not in body
    assert 'DO-NOT-TRANSMIT' not in json.dumps(body)
    assert json.loads(body['input'][1]['content']) == {'document': 'Cutoff: October 1, 2026', 'field_names': ['cutoff']}


def test_transient_status_backoff_has_at_most_four_retries(tmp_path):
    statuses = iter([429, 500, 503, 429, 200]); delays = []; calls = []
    def transport(*args):
        calls.append(args)
        return next(statuses), chat('Final notice text')
    client = OpenAICompatibleClient(config(), tmp_path, transport=transport, sleep=delays.append)
    assert client.complete('write_document', writer_payload())['text'] == 'Final notice text'
    assert len(calls) == 5 and delays == [1, 2, 4, 8]
    assert len((tmp_path/'build/model_requests.jsonl').read_text().splitlines()) == 5


@pytest.mark.parametrize('status, expected_calls', [(429, 5), (500, 5), (503, 5), (400, 1), (401, 1)])
def test_retry_exhaustion_and_permanent_failures_do_not_expose_credentials(tmp_path, status, expected_calls):
    calls = []
    def transport(*args):
        calls.append(args)
        return status, {'error': {'message': 'test-only-not-a-real-key rejected'}}
    client = OpenAICompatibleClient(config(), tmp_path, transport=transport, sleep=lambda _: None)
    with pytest.raises(ModelRequestError, match=f'HTTP {status}') as error:
        client.complete('write_document', writer_payload())
    assert len(calls) == expected_calls
    assert 'test-only-not-a-real-key' not in str(error.value)
    assert 'test-only-not-a-real-key' not in (tmp_path/'build/model_requests.jsonl').read_text()


def test_transport_exception_message_never_logged(tmp_path):
    def transport(*args):
        raise OSError('Authorization: Bearer test-only-not-a-real-key')
    client = OpenAICompatibleClient(config(), tmp_path, transport=transport)
    with pytest.raises(ModelRequestError) as error:
        client.complete('write_document', writer_payload())
    assert 'test-only-not-a-real-key' not in str(error.value)
    assert 'test-only-not-a-real-key' not in (tmp_path/'build/model_requests.jsonl').read_text()


def test_cached_response_survives_copy_and_missing_key(tmp_path, monkeypatch):
    calls = []
    def transport(*args):
        calls.append(args)
        return 200, chat('A persisted document')
    original = tmp_path/'original'
    first = OpenAICompatibleClient(config(), original, transport=transport)
    assert first.complete('write_document', writer_payload()) == {'text': 'A persisted document'}
    copied = tmp_path/'copied'; shutil.copytree(original, copied)
    monkeypatch.delenv('R3_TEST_CREDENTIAL')
    second = OpenAICompatibleClient(config(), copied)
    assert second.complete('write_document', writer_payload()) == {'text': 'A persisted document'}
    assert len(calls) == 1
    assert json.loads((copied/'build/model_requests.jsonl').read_text().splitlines()[-1])['event'] == 'cache_hit'


def test_successful_response_with_secret_echo_redacted_before_cache(tmp_path):
    client = OpenAICompatibleClient(config(), tmp_path, transport=lambda *args: (200, chat('Echo test-only-not-a-real-key')))
    assert client.complete('write_document', writer_payload())['text'] == 'Echo [REDACTED]'
    assert all('test-only-not-a-real-key' not in p.read_text() for p in (tmp_path/'build').rglob('*') if p.is_file())


def test_invalid_success_output_is_cached_and_not_recalled(tmp_path, monkeypatch):
    calls = []
    def transport(*args):
        calls.append(args)
        return 200, responses('not valid JSON')
    client = OpenAICompatibleClient(config('responses'), tmp_path, transport=transport)
    payload = {'text': 'Notice', 'fact_names': ['deadline']}
    with pytest.raises(ModelRequestError, match='invalid'):
        client.complete('extract_facts', payload)
    monkeypatch.delenv('R3_TEST_CREDENTIAL')
    with pytest.raises(ModelRequestError, match='invalid'):
        OpenAICompatibleClient(config('responses'), tmp_path).complete('extract_facts', payload)
    assert len(calls) == 1


@pytest.mark.parametrize('change', ['model', 'prompt', 'temperature', 'seed', 'api', 'operation'])
def test_cache_identity_changes_with_every_semantic_input(tmp_path, change):
    calls = []
    def transport(url, body, *args):
        calls.append((url, body))
        return 200, responses('{"facts":{}}') if url.endswith('/responses') else chat('{"facts":{}}')
    conf = config(); payload = writer_payload(); operation = 'write_document'
    OpenAICompatibleClient(conf, tmp_path, transport=transport).complete(operation, payload)
    altered = dict(conf)
    if change == 'model': altered['model'] = 'other-model'
    if change == 'prompt': payload['facts']['cutoff'] = '2026-11-01'
    if change in ('temperature', 'seed'):
        altered.update(supported_parameters=[change], **{change: .2 if change == 'temperature' else 42})
    if change == 'api': altered['api'] = 'responses'
    if change == 'operation': operation, payload = 'extract_facts', {'text': 'x', 'fact_names': []}
    OpenAICompatibleClient(altered, tmp_path, transport=transport).complete(operation, payload)
    assert len(calls) == 2


def test_credential_is_not_part_of_cache_key(tmp_path, monkeypatch):
    transport = lambda *args: (200, chat('Persist me'))
    OpenAICompatibleClient(config(), tmp_path, transport=transport).complete('write_document', writer_payload())
    monkeypatch.setenv('R3_TEST_CREDENTIAL', 'different-test-value')
    assert OpenAICompatibleClient(config(), tmp_path).complete('write_document', writer_payload())['text'] == 'Persist me'


def test_missing_credentials_fail_before_transport(tmp_path, monkeypatch):
    monkeypatch.delenv('R3_TEST_CREDENTIAL')
    with pytest.raises(ModelRequestError, match='R3_TEST_CREDENTIAL'):
        OpenAICompatibleClient(config(), tmp_path).complete('write_document', writer_payload())


@pytest.mark.parametrize('extra', [{'max_retries': 5}, {'api': 'unknown'}, {'base_url': 'https://' + 'secret' + '@host/v1'},
                                  {'supported_parameters': ['unknown']}])
def test_invalid_transport_configuration_refused(tmp_path, extra):
    with pytest.raises(ValueError):
        OpenAICompatibleClient(config(**extra), tmp_path)


def test_default_model_routes_use_shared_local_configuration(tmp_path, monkeypatch):
    settings = tmp_path / 'routes.toml'
    settings.write_text('writer_endpoint="http://localhost:8001/v1"\nextractor_endpoint="http://localhost:8002/v1"\n')
    monkeypatch.setenv('TADA_LOCAL_PATHS_FILE', str(settings))
    monkeypatch.delenv('TADA_WRITER_ENDPOINT', raising=False)
    monkeypatch.delenv('TADA_EXTRACTOR_ENDPOINT', raising=False)
    writer = OpenAICompatibleClient.from_config('writer', tmp_path)
    assert writer.url == 'http://localhost:8001/v1/chat/completions'
    extractor = OpenAICompatibleClient.from_config('extractor', tmp_path)
    assert extractor.url == 'http://localhost:8002/v1/responses'
    monkeypatch.setenv('TADA_WRITER_ENDPOINT', 'http://localhost:8003/v1')
    assert OpenAICompatibleClient.from_config('writer', tmp_path).url == 'http://localhost:8003/v1/chat/completions'


def test_missing_default_model_route_names_the_setting(tmp_path, monkeypatch):
    monkeypatch.setenv('TADA_LOCAL_PATHS_FILE', str(tmp_path / 'absent.toml'))
    monkeypatch.delenv('TADA_WRITER_ENDPOINT', raising=False)
    with pytest.raises(ValueError, match='writer_endpoint'):
        OpenAICompatibleClient.from_config('writer', tmp_path)


def test_corrupt_cache_is_not_silently_regenerated(tmp_path):
    client = OpenAICompatibleClient(config(), tmp_path, transport=lambda *args: (200, chat('A notice')))
    client.complete('write_document', writer_payload())
    next((tmp_path/'build/model_cache').glob('*.json')).write_text('{}')
    with pytest.raises(ModelRequestError, match='cache'):
        OpenAICompatibleClient(config(), tmp_path).complete('write_document', writer_payload())


def test_real_compile_and_copied_rebuild_never_recall_models(tmp_path, monkeypatch):
    monkeypatch.setenv('TADA_LOCAL_ROUTE_KEY', 'test-writer-credential')
    monkeypatch.setenv('TADA_ASTRA_API_KEY', 'test-extractor-credential')
    original_factory = OpenAICompatibleClient.from_config.__func__
    def factory(cls, role, task_dir, **kwargs):
        return original_factory(cls, role, task_dir, cache_dir=tmp_path/'shared-cache', **kwargs)
    monkeypatch.setattr(OpenAICompatibleClient, 'from_config', classmethod(factory))
    calls = []; notices = {}
    def transport(url, body, headers, timeout):
        calls.append((url, body))
        if url.endswith('/chat/completions'):
            content = json.loads(body['messages'][1]['content'])
            text = 'Professional notice: ' + content['id'] + '\n' + json.dumps(content['facts'], sort_keys=True)
            notices[text] = content['facts']
            return 200, chat(text)
        material = json.loads(body['input'][1]['content'])
        assert set(material) == {'document', 'field_names'}
        return 200, responses(json.dumps({'facts': notices[material['document']]}))
    with integration_adapters():
        spec, report = screened_spec('MAUD')
        task = compile_task(spec, 7, tmp_path/'task', client='real', model_transport=transport, screening_report=report)
        archive = json.loads((task/'tests/roundtrip_report.json').read_text())
        assert archive['pass'] and not archive['fake']
        assert len(calls) == 2 * len(archive['documents'])
        before = len(calls)
        visible = {str(p.relative_to(task/'environment/public')): p.read_bytes()
                   for p in (task/'environment/public').rglob('*') if p.is_file()}
        monkeypatch.delenv('TADA_LOCAL_ROUTE_KEY'); monkeypatch.delenv('TADA_ASTRA_API_KEY')
        compile_task(spec, 7, task, client='real', screening_report=report)
        copied = tmp_path/'copy'; shutil.copytree(task, copied)
        compile_task(spec, 7, copied, client='real', screening_report=report)
        compile_task(spec, 7, tmp_path/'fresh', client='real', screening_report=report)
        assert len(calls) == before
        assert visible == {str(p.relative_to(copied/'environment/public')): p.read_bytes()
                           for p in (copied/'environment/public').rglob('*') if p.is_file()}
        build = (copied/'build/build.py').read_text()
        assert "CLIENT = 'real'" in build and 'no live model calls permitted' in build


def test_real_compile_cannot_reuse_fake_archive(tmp_path):
    with integration_adapters():
        spec, report = screened_spec('MAUD')
        task = compile_task(spec, 7, tmp_path/'task', screening_report=report)
        with pytest.raises(ValueError, match='fake document artifacts'):
            compile_task(spec, 7, task, client='real', screening_report=report)


@pytest.mark.parametrize('api, body', [('responses', {'output': [None]}),
                                      ('chat_completions', {'choices': [{'message': {'content': [None]}}]})])
def test_malformed_shapes_use_safe_invalid_output_error(tmp_path, api, body):
    client = OpenAICompatibleClient(config(api), tmp_path, transport=lambda *args: (200, body))
    with pytest.raises(clients.InvalidModelOutputError):
        client.complete('write_document', writer_payload())


def test_malformed_writer_and_extractor_outputs_use_bounded_regeneration(tmp_path):
    calls = []; count = {'writer': 0, 'extractor': 0}
    def writer_transport(*args):
        count['writer'] += 1
        return 200, {} if count['writer'] == 1 else chat('Document attempt ' + str(count['writer']))
    def extractor_transport(*args):
        count['extractor'] += 1
        return 200, responses('not JSON' if count['extractor'] == 1 else '{"facts":{"approved":true}}')
    writer = OpenAICompatibleClient(config(), tmp_path/'writer', transport=writer_transport)
    extractor = OpenAICompatibleClient(config('responses'), tmp_path/'extractor', transport=extractor_transport)
    result = generate_document({'id': 'a', 'kind': 'notice', 'facts': {'approved': True}}, writer, extractor)
    assert result['pass'] and len(result['attempts']) == 3
    assert count == {'writer': 3, 'extractor': 2}
    repeated = generate_document({'id': 'a', 'kind': 'notice', 'facts': {'approved': True}}, writer, extractor)
    assert repeated == result and count == {'writer': 3, 'extractor': 2}


def test_archive_boolean_integer_mismatch_refused_without_clearing_public(tmp_path):
    with integration_adapters():
        spec, report = screened_spec('MAUD')
        spec['documents'].append({'id': 'typed', 'path': 'typed.txt', 'kind': 'notice', 'facts': {'approved': True}})
        task = compile_task(spec, 7, tmp_path/'task', screening_report=report)
        archive = json.loads((task/'tests/roundtrip_report.json').read_text())
        next(d for d in archive['documents'] if d['id'] == 'typed')['facts']['approved'] = 1
        existing = (task/'environment/public/typed.txt').read_bytes()
        with pytest.raises(ValueError, match='exact scenario'):
            compile_task(spec, 7, task, document_archive=archive, screening_report=report)
        assert (task/'environment/public/typed.txt').read_bytes() == existing


@pytest.mark.parametrize('borrowed', [False, True])
def test_real_requests_honor_shared_opus_slot_and_cache_hits_need_none(tmp_path, monkeypatch, borrowed):
    from contextlib import contextmanager
    from tada_taskgen import concurrency
    events = []
    @contextmanager
    def slot(model):
        events.append(('enter', model))
        yield
        events.append(('exit', model))
    monkeypatch.setattr(concurrency, 'model_slot', slot)
    if borrowed:
        monkeypatch.setenv('TADA_MODEL_SLOT_HELD', '1')
        monkeypatch.setenv('TADA_MODEL_SLOT_FAMILY', 'opus')
    else:
        monkeypatch.delenv('TADA_MODEL_SLOT_HELD', raising=False)
    client = OpenAICompatibleClient(config(model='claude-opus-test'), tmp_path,
                                    transport=lambda *args: (200, chat('A notice')))
    client.complete('write_document', writer_payload())
    client.complete('write_document', writer_payload())
    assert events == ([] if borrowed else [('enter', 'claude-opus-test'), ('exit', 'claude-opus-test')])


def test_borrowed_opus_slot_does_not_skip_astra_slot(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from tada_taskgen import concurrency
    events = []
    @contextmanager
    def slot(model):
        events.append(model)
        yield
    monkeypatch.setattr(concurrency, 'model_slot', slot)
    monkeypatch.setenv('TADA_MODEL_SLOT_HELD', '1')
    monkeypatch.setenv('TADA_MODEL_SLOT_FAMILY', 'opus')
    client = OpenAICompatibleClient(config(model='gpt-6-astra'), tmp_path,
                                    transport=lambda *args: (200, chat('A notice')))
    client.complete('write_document', writer_payload())
    assert events == ['gpt-6-astra']
