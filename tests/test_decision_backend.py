"""Native decision backend (Jev / TypeSafe SystemOne) for closed-label tag paths, fully offline."""
import json
import httpx
import pandas as pd
import pytest
from pydantic import ValidationError
from tada_operators import Engine
from tada_operators.engine import SEGMENT_NONE
from tada_runtime.audit import AuditLog, read_events
from tada_runtime.state import Store
from tada_runtime.config import Config, BudgetConfig, DecisionConfig
from tada_runtime.budget import BudgetLedger
from tada_runtime.contracts import IntentEnvelope, CategoryEnvelope
from tada_runtime.decision import DecisionClient
from tada_runtime.providers import ProviderFailure

LABELS = ['Exclusivity', 'Termination for convenience', 'Change of control']


class LLM:
    """Generative operator fixture: records roles; intent can be joint."""
    def __init__(self, joint=False):
        self.roles, self.joint = [], joint
    def call(self, prompt, schema, *, role, parent_id, context=None, cancellation=None):
        self.roles.append(role)
        if schema is IntentEnvelope:
            return schema.model_validate({'mode': 'joint', 'domain_query': 'discover'} if self.joint else {'mode': 'independent', 'domain_query': None})
        if schema is CategoryEnvelope:
            return schema.model_validate({'categories': [{'name': 'Shipping', 'description': 'delivery'}, {'name': 'Service'}]}, context=context)
        return schema.model_validate({'items': [{'rid': r, 'value': 'llm'} for r in context['rids']]}, context=context)


def jev_answer(question, text, *, yes_if):
    if question['type'] == 'noul':
        return {'type': 'noul', 'noul': 0.9 if yes_if(text, question) else 0.1}
    options = list(question['criteria'])
    if options == ['Yes', 'No']:
        choice = 'Yes' if yes_if(text, question) else 'No'
        return {'type': 'choice', 'choice': choice, 'probabilities': {'Yes': 0.9 if choice == 'Yes' else 0.1, 'No': 0.1 if choice == 'Yes' else 0.9}}
    hit = next((o for o in options if o != SEGMENT_NONE and o.split()[0].upper() in text.upper()), None)
    choice = hit or (SEGMENT_NONE if SEGMENT_NONE in options else options[0])
    return {'type': 'choice', 'choice': choice, 'probabilities': {o: (0.8 if o == choice else 0.2 / (len(options) - 1)) for o in options}}


class FakeJev:
    def __init__(self, model='jev-1.13.0', break_first=0, status_first=None):
        self.requests, self.model, self.break_first, self.status_first = [], model, break_first, status_first
    def __call__(self, request):
        assert request.url.path == '/v1/systemone'
        body = json.loads(request.content)
        self.requests.append((body, request.headers.get('authorization')))
        if self.status_first and len(self.requests) == 1:
            return httpx.Response(self.status_first, json={'error': 'x'})
        items = body['state']['items']
        def yes_if(text, question):
            label = question['instructions']['question'].split('"')[1]
            return label.split()[0].upper() in text.upper()
        answers = {key: jev_answer(q, items[q['instructions']['item'].split('.', 1)[1]], yes_if=yes_if) for key, q in body['questions'].items()}
        if self.break_first and len(self.requests) <= self.break_first:
            answers.pop(next(iter(answers)))
        return httpx.Response(200, json={'model': self.model, 'answers': answers, 'usage': {'input_tokens': 10, 'output_tokens': 2}})


def setup(tmp_path, fake, *, decision=None, runtime=None, llm=None, monkeypatch=None):
    if monkeypatch:
        monkeypatch.setenv('TYPESAFE_API_KEY', 'test-jev-key')
    cfg = Config.model_validate({'models': {**{role: {'base_url': 'http://localhost:9999/v1', 'model': 'exact',
        'protocol': 'openai-responses', 'auth_mode': 'optional'} for role in ['agent', 'operators']},
        'decision': {'base_url': 'https://api.typesafe.ai/v1', **(decision or {})}},
        'runtime': {'batch_retries': 1, **(runtime or {})}})
    audit = AuditLog(tmp_path / 'session')
    budget = BudgetLedger(BudgetConfig(ledger_path=str(tmp_path / 'ledger.json'), max_requests=50, deadline_seconds=60.0), audit)
    client = DecisionClient(cfg.models.decision, budget, audit, transport=httpx.MockTransport(fake))
    llm = llm or LLM()
    return Engine(cfg, llm, Store(audit), audit, decision=client), llm, audit, budget


TEXTS = pd.Series(['This EXCLUSIVITY clause binds.', 'Either party may terminate; CHANGE of control requires consent.'], index=['c1', 'c2'])


def test_single_label_routes_to_decision_and_records_probabilities(tmp_path, monkeypatch):
    fake = FakeJev()
    engine, llm, audit, budget = setup(tmp_path, fake, monkeypatch=monkeypatch)
    result = engine.tag(TEXTS, 'main provision', labels=LABELS)
    assert result.tolist() == ['Exclusivity', 'Change of control']
    assert llm.roles == [] and len(fake.requests) == 1
    body, auth = fake.requests[0]
    assert auth == 'Bearer test-jev-key' and body['model'] == 'jev-1.13.0'
    assert set(body['questions']['i0']['criteria']) == set(LABELS)
    events, _ = read_events(audit.root)
    request = next(e for e in events if e.type == 'model/request')
    assert request.payload.protocol == 'jev-systemone' and request.payload.role == 'decision-tag-batch'
    blob = (audit.root / request.payload.request_ref.path).read_text()
    assert 'test-jev-key' not in blob
    end = next(e for e in events if e.type == 'atom/end' and e.payload.op == 'tag')
    probs = engine.store.get(end.payload.child_state_ids[-1])
    assert set(probs) == {'c1', 'c2'} and probs['c1']['Exclusivity'] == 0.8
    assert len(budget.snapshot().reservations) == 1
    audit.close()


def test_multi_label_binary_choice_and_noul(tmp_path, monkeypatch):
    fake = FakeJev()
    engine, _, audit, _ = setup(tmp_path, fake, monkeypatch=monkeypatch)
    result = engine.tag(TEXTS, 'which provisions', labels=LABELS, multi=True)
    assert result.tolist() == ['["Exclusivity"]', '["Change of control"]']
    assert len(fake.requests[0][0]['questions']) == 6
    audit.close()
    fake = FakeJev()
    engine, _, audit, _ = setup(tmp_path / 'n', fake, decision={'multi_mode': 'noul', 'noul_threshold': 0.5}, monkeypatch=monkeypatch)
    assert engine.tag(TEXTS, 'which provisions', labels=LABELS, multi=True).tolist() == ['["Exclusivity"]', '["Change of control"]']
    assert {q['type'] for q in fake.requests[0][0]['questions'].values()} == {'noul'}
    audit.close()


def test_question_limit_splits_requests(tmp_path, monkeypatch):
    fake = FakeJev()
    engine, _, audit, _ = setup(tmp_path, fake, decision={'max_questions': 3}, monkeypatch=monkeypatch)
    engine.tag(TEXTS, 'which provisions', labels=LABELS, multi=True)
    assert len(fake.requests) == 2 and all(len(b['questions']) == 3 for b, _ in fake.requests)
    audit.close()


def test_contract_violation_is_retried_and_linked(tmp_path, monkeypatch):
    fake = FakeJev(break_first=1)
    engine, _, audit, budget = setup(tmp_path, fake, monkeypatch=monkeypatch)
    assert engine.tag(TEXTS, 'main provision', labels=LABELS).tolist() == ['Exclusivity', 'Change of control']
    events, _ = read_events(audit.root)
    attempts = [e for e in events if e.type == 'model/attempt_end']
    assert [a.payload.status for a in attempts] == ['schema_error', 'success']
    assert attempts[1].payload.retry_of == attempts[0].request_id
    assert len(budget.snapshot().reservations) == 2
    audit.close()


def test_identity_mismatch_and_client_errors_are_not_retried(tmp_path, monkeypatch):
    fake = FakeJev(model='jev-9.9.9')
    engine, _, audit, _ = setup(tmp_path, fake, monkeypatch=monkeypatch)
    with pytest.raises(ProviderFailure, match='unexpected model identity'):
        engine.tag(TEXTS, 'main provision', labels=LABELS)
    assert len(fake.requests) == 1
    audit.close()
    fake = FakeJev(status_first=400)
    engine, _, audit, _ = setup(tmp_path / 'b', fake, monkeypatch=monkeypatch)
    with pytest.raises(ProviderFailure, match='decision HTTP 400'):
        engine.tag(TEXTS, 'main provision', labels=LABELS)
    assert len(fake.requests) == 1
    audit.close()
    fake = FakeJev(status_first=429)
    engine, _, audit, _ = setup(tmp_path / 'c', fake, monkeypatch=monkeypatch)
    assert engine.tag(TEXTS, 'main provision', labels=LABELS).tolist() == ['Exclusivity', 'Change of control']
    assert len(fake.requests) == 2
    audit.close()


LONG = 'Preamble. ' + ('ordinary text. ' * 900) + 'EXCLUSIVITY obligations here. ' + ('ordinary text. ' * 900)


def test_long_rows_single_label_use_max_probability_without_reduce(tmp_path, monkeypatch):
    fake = FakeJev()
    engine, llm, audit, _ = setup(tmp_path, fake, decision={'batch_chars': 9000},
        runtime={'long_rows': {'segment_chars': 4000, 'overlap_chars': 200, 'head_chars': 300}}, monkeypatch=monkeypatch)
    result = engine.tag(pd.Series([LONG], index=['doc']), 'main provision', labels=LABELS)
    assert result.tolist() == ['Exclusivity'] and llm.roles == []
    criteria = [set(q['criteria']) for b, _ in fake.requests for q in b['questions'].values()]
    assert all(SEGMENT_NONE in c for c in criteria)
    events, _ = read_events(audit.root)
    plan = next(e for e in events if e.type == 'atom/plan')
    assert plan.payload.long_rows['aggregate'] == 'decision-max-probability'
    audit.close()


def test_long_rows_multi_label_union(tmp_path, monkeypatch):
    fake = FakeJev()
    engine, llm, audit, _ = setup(tmp_path, fake, decision={'batch_chars': 9000},
        runtime={'long_rows': {'segment_chars': 4000, 'overlap_chars': 200, 'head_chars': 300}}, monkeypatch=monkeypatch)
    result = engine.tag(pd.Series([LONG, 'CHANGE of control only'], index=['doc', 'short']), 'which', labels=LABELS, multi=True)
    assert result.tolist() == ['["Exclusivity"]', '["Change of control"]'] and llm.roles == []
    audit.close()


def test_joint_assignment_route(tmp_path, monkeypatch):
    fake = FakeJev()
    llm = LLM(joint=True)
    engine, llm, audit, _ = setup(tmp_path, fake, llm=llm, monkeypatch=monkeypatch)
    result = engine.tag(pd.Series(['SHIPPING was late', 'SERVICE was rude'], index=[1, 2]), 'discover complaint themes')
    assert result.tolist() == ['Shipping', 'Service']
    assert llm.roles == ['intent', 'categorize-map'] and len(fake.requests) == 1
    assert fake.requests[0][0]['state']['labels'] == {'Shipping': 'delivery', 'Service': ''}
    audit.close()


def test_disabled_route_uses_the_generative_provider(tmp_path, monkeypatch):
    class Labeller(LLM):
        def call(self, prompt, schema, *, role, parent_id, context=None, cancellation=None):
            self.roles.append(role)
            return schema.model_validate({'items': [{'rid': r, 'value': LABELS[0]} for r in context['rids']]}, context=context)
    fake = FakeJev()
    engine, llm, audit, _ = setup(tmp_path, fake, decision={'routes': ['labels-multi']}, llm=Labeller(), monkeypatch=monkeypatch)
    assert engine.tag(TEXTS, 'main provision', labels=LABELS).tolist() == [LABELS[0]] * 2
    assert fake.requests == [] and llm.roles == ['tag-batch']
    audit.close()


def test_decision_config_validation(monkeypatch):
    with pytest.raises(ValidationError):
        DecisionConfig(multi_mode='noul')
    with pytest.raises(ValidationError):
        DecisionConfig(base_url='https://api.typesafe.ai/v1', auth_mode='optional')
    with pytest.raises(ValidationError):
        DecisionConfig(model='jev-latest')
    monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    with pytest.raises(ValueError, match='TYPESAFE_API_KEY'):
        DecisionConfig().api_key()
