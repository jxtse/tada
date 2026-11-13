"""Responses API message phases must not be concatenated into the strict JSON answer.

Regression for the Harbor merger-review run on 6dcc73a: a 'commentary' message followed by a
'final_answer' message (or two final answers) was joined into one string and failed with
json_invalid although the final JSON alone was valid.
"""
import json
import httpx
import pytest
from tada_runtime.contracts import TagEnvelope, TextBlock
from tada_runtime.config import ProviderConfig, BudgetConfig
from tada_runtime.budget import BudgetLedger
from tada_runtime.providers import Provider, ProviderFailure, structured_answer_text
from tada_runtime.audit import AuditLog, read_events

FINAL = json.dumps({'items': [{'rid': 'r1', 'value': 'Yes'}]}, separators=(',', ':'))
CONTEXT = {'rids': ['r1'], 'allowed_values': ['Yes', 'No']}


def message(text, phase=None):
    item = {'type': 'message', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': text}]}
    if phase is not None:
        item['phase'] = phase
    return item


def run(tmp_path, output, *, stream=True):
    response = {'model': 'reported', 'status': 'completed', 'output': output,
                'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}}
    def handler(request):
        if stream:
            events = [{'type': 'response.completed', 'response': response}]
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                  text=''.join('data: ' + json.dumps(e) + '\n\n' for e in events))
        return httpx.Response(200, json=response)
    audit = AuditLog(tmp_path / 'session')
    budget = BudgetLedger(BudgetConfig(ledger_path=str(tmp_path / 'ledger.json'), max_requests=5, deadline_seconds=60.0), audit)
    provider = Provider(ProviderConfig(base_url='http://localhost:9999/v1', model='exact', protocol='openai-responses', auth_mode='optional'),
                        budget, audit, transport=httpx.MockTransport(handler))
    try:
        return provider.call('test', TagEnvelope, role='tag-batch', context=CONTEXT), provider, audit
    finally:
        audit.close()


@pytest.mark.parametrize('stream', [True, False])
def test_commentary_is_excluded_from_the_answer(tmp_path, stream):
    result, provider, _ = run(tmp_path, [message('I will return the label next.', 'commentary'), message(FINAL, 'final_answer')], stream=stream)
    assert result.items[0].value == 'Yes'
    phases = [(b.phase, b.message_index) for b in provider.last_response.content if b.type == 'text']
    assert phases == [('commentary', 0), ('final_answer', 1)]


def test_identical_duplicate_final_answers_are_accepted(tmp_path):
    spaced = json.dumps({'items': [{'rid': 'r1', 'value': 'Yes'}]})
    result, _, _ = run(tmp_path, [message(FINAL, 'final_answer'), message(spaced, 'final_answer')])
    assert result.items[0].value == 'Yes'


def test_differing_final_answers_are_refused_not_chosen(tmp_path):
    other = json.dumps({'items': [{'rid': 'r1', 'value': 'No'}]})
    with pytest.raises(ProviderFailure, match='differing answer messages'):
        run(tmp_path, [message(FINAL, 'final_answer'), message(other, 'final_answer')])
    events, _ = read_events(tmp_path / 'session')
    end = [e for e in events if e.type == 'model/attempt_end'][-1]
    assert end.payload.status == 'failed'


def test_commentary_only_is_refused(tmp_path):
    with pytest.raises(ProviderFailure, match='only commentary'):
        run(tmp_path, [message('thinking out loud', 'commentary')])


def test_without_phase_metadata_parts_are_joined_as_before():
    blocks = [TextBlock(text='{"items":'), TextBlock(text='[]}')]
    assert structured_answer_text(blocks) == '{"items":[]}'
    split = [TextBlock(text='{"items":', phase='final_answer', message_index=2), TextBlock(text='[]}', phase='final_answer', message_index=2)]
    assert structured_answer_text(split) == '{"items":[]}'
