"""Closed/multi-label Tagging and scale-transparent long-row segmentation (offline fixtures)."""
import json
import pandas as pd
import pytest
from pydantic import ValidationError
from tada_operators import Engine
from tada_operators.engine import labels_frame
from tada_operators.segment import split_bodies, render_segments, SEGMENT_OVERHEAD
from tada_runtime.audit import AuditLog, read_events
from tada_runtime.state import Store
from tada_runtime.config import Config
from tada_runtime.contracts import IntentEnvelope, CategoryEnvelope, TagEnvelope
from tada_runtime.rpc_contracts import KernelRequest


class ScriptedProvider:
    """Answers each request through a role-specific function of (prompt, rids)."""
    def __init__(self, answer=None, intent='independent'):
        self.calls, self.prompts = [], []
        self.answer = answer or (lambda role, prompt, rid: 'positive')
        self.intent = intent

    def call(self, prompt, schema, *, role, parent_id, context=None):
        self.calls.append(role)
        self.prompts.append(prompt)
        if schema is IntentEnvelope:
            data = {'mode': 'joint', 'domain_query': 'discover'} if self.intent == 'joint' else {'mode': self.intent, 'domain_query': None}
        elif schema is CategoryEnvelope:
            data = {'categories': [{'name': 'Shipping'}, {'name': 'Service'}]}
        else:
            data = {'items': [{'rid': rid, 'value': self.answer(role, prompt, rid)} for rid in context['rids']]}
        return schema.model_validate(data, context=context)


@pytest.fixture
def setup(tmp_path):
    def make(runtime=None):
        cfg = Config.model_validate({'models': {role: {'base_url': 'http://localhost:9999/v1', 'model': 'exact',
            'protocol': 'openai-responses', 'auth_mode': 'optional'} for role in ['agent', 'operators']},
            'runtime': {'batch_rows': 4, **(runtime or {})}})
        audit = AuditLog(tmp_path / f'session{len(made)}')
        made.append(audit)
        return cfg, audit, Store(audit)
    made = []
    yield make
    for audit in made:
        audit.close()


LABELS = ['Exclusivity', 'Termination for convenience', 'Change of control']


# ---------- closed and multi-label tagging ----------

def test_single_label_skips_intent_and_validates_domain(setup):
    cfg, audit, store = setup()
    provider = ScriptedProvider(lambda role, prompt, rid: 'Exclusivity')
    result = Engine(cfg, provider, store, audit).tag(pd.Series(['a', 'b'], index=['r1', 'r2']), 'clause type', labels=LABELS)
    assert result.tolist() == ['Exclusivity'] * 2
    assert provider.calls == ['tag-batch']
    assert 'LABELS_JSON' in provider.prompts[0]
    bad = ScriptedProvider(lambda role, prompt, rid: 'Not a label')
    with pytest.raises(ValidationError):
        Engine(cfg, bad, store, audit).tag(pd.Series(['a'], index=['r1']), 'clause type', labels=LABELS)


def test_multi_label_values_are_validated_and_canonical(setup):
    cfg, audit, store = setup()
    answers = {'r1': '["Change of control", "Exclusivity"]', 'r2': '[]'}
    provider = ScriptedProvider(lambda role, prompt, rid: answers[rid])
    result = Engine(cfg, provider, store, audit).tag(pd.Series(['a', 'b'], index=['r1', 'r2']), 'which clauses', labels=LABELS, multi=True)
    assert result.tolist() == ['["Exclusivity", "Change of control"]', '[]']
    flags = labels_frame(result, LABELS)
    assert flags.loc['r1'].tolist() == [True, False, True] and not flags.loc['r2'].any()
    for broken in ['Exclusivity', '["Exclusivity", "Exclusivity"]', '["Other"]', '[1]', '{"a": 1}']:
        provider = ScriptedProvider(lambda role, prompt, rid, broken=broken: broken)
        with pytest.raises(ValidationError):
            Engine(cfg, provider, store, audit).tag(pd.Series(['x'], index=['r9']), 'which clauses', labels=LABELS, multi=True, fresh=True)


def test_labels_accept_category_table_and_change_reuse_identity(setup):
    cfg, audit, store = setup()
    provider = ScriptedProvider(lambda role, prompt, rid: 'Shipping')
    engine = Engine(cfg, provider, store, audit)
    table = pd.DataFrame({'name': ['Shipping', 'Service'], 'description': ['delivery', 'staff']},
                         index=pd.Index(['c0000', 'c0001'], name='cid')).astype('string')
    texts = pd.Series(['a', 'b'], index=['r1', 'r2'])
    first = engine.tag(texts, 'assign', labels=table)
    assert '"description": "delivery"' in provider.prompts[0]
    before = len(provider.calls)
    assert engine.tag(texts, 'assign', labels=table).attrs['state_id'] == first.attrs['state_id']
    assert len(provider.calls) == before
    engine.tag(texts, 'assign', labels=['Shipping', 'Service'])
    assert len(provider.calls) > before


def test_label_argument_errors(setup):
    cfg, audit, store = setup()
    provider = ScriptedProvider()
    engine = Engine(cfg, provider, store, audit)
    texts = pd.Series(['a'], index=['r1'])
    with pytest.raises(ValueError):
        engine.tag(texts, 'q', labels=LABELS, cardinality={'k': 2, 'mode': 'exact'})
    with pytest.raises(ValidationError):
        engine.tag(texts, 'q', labels=['A', 'A'])
    with pytest.raises(TypeError):
        engine.tag(texts, 'q', labels='Exclusivity')
    assert provider.calls == []
    with pytest.raises(ValueError, match='multi=True requires'):
        engine.tag(texts, 'q', multi=True)
    with pytest.raises(ValidationError):
        KernelRequest.model_validate({'op': 'categorize', 'query': 'q', 'labels': [{'name': 'A'}]})


def test_joint_multi_assigns_frozen_vocabulary_sets(setup):
    cfg, audit, store = setup()
    provider = ScriptedProvider(lambda role, prompt, rid: '["Service", "Shipping"]', intent='joint')
    result = Engine(cfg, provider, store, audit).tag(pd.Series(['a', 'b'], index=[1, 2]), 'topics', multi=True)
    assert result.tolist() == ['["Shipping", "Service"]'] * 2


# ---------- segmentation primitives ----------

def test_split_bodies_cover_overlap_and_prefer_paragraphs():
    text = ('alpha beta gamma. ' * 30 + '\n\n') * 20
    spans = split_bodies(text, segment_chars=1000, overlap_chars=100, boundary='paragraph')
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert s2 == e1 - 100 and s1 < s2 < e1
        assert text[e1 - 2:e1] == '\n\n'
    assert all(e - s <= 1000 for s, e in spans)
    hard = split_bodies('x' * 2500, segment_chars=1000, overlap_chars=0, boundary='paragraph')
    assert hard == [(0, 1000), (1000, 2000), (2000, 2500)]
    with pytest.raises(ValueError):
        split_bodies('abc', segment_chars=10, overlap_chars=5, boundary='paragraph')


def test_rendered_segments_are_bounded_and_carry_head():
    text = 'HEADER parties and definitions. ' + 'clause text. ' * 2000
    rendered = render_segments(text, segment_chars=2000, overlap_chars=200, head_chars=300, boundary='paragraph')
    assert len(rendered) > 1
    assert rendered[0].startswith('[SEGMENT 1/')
    assert all(r.startswith('[DOCUMENT START EXCERPT]\nHEADER') for r in rendered[1:])
    assert all(len(r) <= 300 + 2000 + SEGMENT_OVERHEAD for r in rendered)


# ---------- long rows inside tag ----------

LONG = 'Preamble. ' + ('ordinary text. ' * 900) + 'EXCLUSIVE dealing here. ' + ('ordinary text. ' * 900)


def long_runtime(**extra):
    return {'batch_chars': 6000, 'long_rows': {'segment_chars': 4000, 'overlap_chars': 200, 'head_chars': 500, **extra}}


def test_short_rows_are_unchanged(setup):
    cfg, audit, store = setup(long_runtime())
    provider = ScriptedProvider()
    Engine(cfg, provider, store, audit).tag(pd.Series(['short', 'rows'], index=['r1', 'r2']), 'sentiment')
    assert provider.calls == ['intent', 'tag-batch']
    plan = next(e for e in read_events(audit.root)[0] if e.type == 'atom/plan')
    assert plan.payload.long_rows is None


def test_long_row_union_with_multi_labels(setup):
    cfg, audit, store = setup(long_runtime())
    def answer(role, prompt, rid):
        if '::seg' in str(rid):
            return '["Exclusivity"]' if 'EXCLUSIVE' in prompt.split(str(rid))[1][:5000] else '[]'
        return '["Change of control"]'
    provider = ScriptedProvider(answer)
    texts = pd.Series([LONG, 'short contract'], index=['long', 'short'])
    result = Engine(cfg, provider, store, audit).tag(texts, 'which clauses', labels=LABELS, multi=True)
    assert result.loc['short'] == '["Change of control"]'
    assert result.loc['long'] == '["Exclusivity"]'
    assert 'intent' not in provider.calls and 'tag-segment-reduce' not in provider.calls
    assert provider.calls.count('tag-segment-batch') >= 1
    events = read_events(audit.root)[0]
    plan = next(e for e in events if e.type == 'atom/plan')
    assert plan.payload.long_rows['aggregate'] == 'union'
    end = next(e for e in events if e.type == 'atom/end' and e.payload.op == 'tag')
    segments = store.get(end.payload.child_state_ids[0])
    assert all(rid.startswith('long::seg') for rid in segments.index)


def test_long_row_default_reduce_and_collect_override(setup):
    cfg, audit, store = setup(long_runtime())
    def answer(role, prompt, rid):
        if role == 'tag-segment-reduce':
            return 'document-level answer'
        return 'found' if role == 'tag-segment-batch' else 'whole'
    provider = ScriptedProvider(answer)
    engine = Engine(cfg, provider, store, audit)
    texts = pd.Series([LONG], index=[7])
    assert engine.tag(texts, 'summarize obligations').tolist() == ['document-level answer']
    assert provider.calls[0] == 'intent' and provider.calls[-1] == 'tag-segment-reduce'
    reduce_prompt = provider.prompts[-1]
    rows = json.loads(reduce_prompt.split('\nRows: ', 1)[1])
    assert 'SEGMENT_ANSWERS_JSON' in reduce_prompt and rows[0]['rid'] == 7
    assert {entry['answer'] for entry in json.loads(rows[0]['text'])} == {'found'}
    collected = engine.tag(texts, 'extract amounts', long_rows={'aggregate': 'collect'})
    assert json.loads(collected.iloc[0]) == ['found']


def test_long_row_single_label_reduce_allows_empty_segments(setup):
    cfg, audit, store = setup(long_runtime())
    def answer(role, prompt, rid):
        return '' if role == 'tag-segment-batch' else 'Exclusivity'
    provider = ScriptedProvider(answer)
    result = Engine(cfg, provider, store, audit).tag(pd.Series([LONG], index=['d']), 'main clause', labels=LABELS)
    assert result.tolist() == ['Exclusivity']
    assert provider.calls[-1] == 'tag-segment-reduce'


def test_long_row_refusals_send_no_requests(setup):
    texts = pd.Series([LONG], index=['d'])
    cfg, audit, store = setup({'batch_chars': 6000, 'long_rows': {'enabled': False}})
    provider = ScriptedProvider()
    with pytest.raises(ValueError, match='one row exceeds batch_chars'):
        Engine(cfg, provider, store, audit).tag(texts, 'q')
    cfg, audit, store = setup(long_runtime(max_segments_per_row=2))
    with pytest.raises(ValueError, match='max_segments_per_row'):
        Engine(cfg, provider, store, audit).tag(texts, 'q')
    cfg, audit, store = setup({'batch_chars': 3000})
    with pytest.raises(ValueError, match='exceeds batch_chars'):
        Engine(cfg, provider, store, audit).tag(texts, 'q')
    cfg, audit, store = setup(long_runtime())
    with pytest.raises(ValueError, match='collect aggregation'):
        Engine(cfg, provider, store, audit).tag(texts, 'q', labels=LABELS, long_rows={'aggregate': 'collect'})
    with pytest.raises(ValueError, match='union aggregation'):
        Engine(cfg, provider, store, audit).tag(texts, 'q', long_rows={'aggregate': 'union'})
    assert provider.calls == []


def test_joint_over_long_rows_is_refused(setup):
    cfg, audit, store = setup(long_runtime())
    provider = ScriptedProvider(intent='joint')
    with pytest.raises(ValueError, match='joint tagging over long rows'):
        Engine(cfg, provider, store, audit).tag(pd.Series([LONG], index=['d']), 'discover topics')
    assert provider.calls == ['intent']


def test_segment_trigger_override_segments_mid_length_rows(setup):
    cfg, audit, store = setup(long_runtime())
    provider = ScriptedProvider(lambda role, prompt, rid: '[]' if '::seg' in str(rid) else '["Exclusivity"]')
    mid = 'word ' * 1000
    Engine(cfg, provider, store, audit).tag(pd.Series([mid], index=['m']), 'q', labels=LABELS, multi=True)
    assert provider.calls == ['tag-batch']
    Engine(cfg, provider, store, audit).tag(pd.Series([mid], index=['m']), 'q', labels=LABELS, multi=True,
        long_rows={'trigger': 'over_segment_chars', 'segment_chars': 2000, 'overlap_chars': 100, 'head_chars': 0})
    assert provider.calls[-1] == 'tag-segment-batch'
