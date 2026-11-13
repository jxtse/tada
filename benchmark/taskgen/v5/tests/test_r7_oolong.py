"""Official deterministic Oolong/SQE scoring and input allowlist regressions."""
import json
import pytest
from tada_taskgen.adapters import benchmark_oolong as source
from tada_taskgen.benchmark_oolong_score import score_oolong


@pytest.mark.parametrize('expected,predicted,kind,extra,score', [
    ('[10]', 'Answer: 11', 'oolong_synth', {'answer_type': 'ANSWER_TYPE.NUMERIC'}, .75),
    ('[10]', [10], 'oolong_synth', {'answer_type': 'ANSWER_TYPE.NUMERIC'}, 1.),
    ("['incorrect']", 'Label: incorrect', 'oolong_synth', {'answer_type': 'ANSWER_TYPE.LABEL'}, 1.),
    ('[datetime.date(2020, 1, 2)]', '2020-01-02', 'oolong_synth', {'answer_type': 'ANSWER_TYPE.DATE'}, 1.),
    ('12', r'\boxed{13}', 'oolong_real', {}, .75),
    ('Alice, Bob', ['Alice'], 'oolong_real', {}, .5),
    ('Alice', ' ALICE ', 'oolong_real', {}, 1.),
    (['1','two'], [1,'bad'], 'sqe_derivation', {}, .5),
    ([{'id':1},{'id':2}], [{'id':1}], 'sqe_set_f1', {}, 2/3),
])
def test_original_rules(expected, predicted, kind, extra, score):
    spec = {'kind':kind, **extra}
    assert score_oolong(predicted, expected, spec) == pytest.approx(score)
    assert score_oolong(expected, expected, spec) == 1


def test_oolong_context_stays_lazy_and_hidden_alternate_never_exported(monkeypatch, tmp_path):
    row = {'question':'Count categories', 'context_window_text':'Public statements',
           'context_window_text_with_labels':'PRIVATE_CODED_CONTEXT', 'answer':'[2]',
           'answer_type':'ANSWER_TYPE.NUMERIC'}
    calls=[]
    def fake_units(*args, **kwargs):
        calls.append(args)
        yield {'unit_id':'original-item', 'raw_row':row}
    source._oolong_index.cache_clear()
    data=tmp_path/'original.json';data.write_text('{}')
    monkeypatch.setattr(source,'REPO',tmp_path)
    monkeypatch.setattr(source,'plan',lambda _: [{'table_id':'table','path':str(data)}])
    monkeypatch.setattr(source,'units',fake_units)
    item=next(source.iter_items('oolong-synth'))
    assert item['text']=='' and item['text_bytes']>0 and len(calls)==1
    text=item['text_loader']()
    assert text=='Public statements\n\nQuestion: Count categories'
    assert 'PRIVATE' not in text and '[2]' not in text
    assert item['answer']=='[2]' and len(calls)==2
    assert list(source.selected_texts('oolong-synth',['original-item'])) == [('original-item',text)]
    assert len(calls)==3
    source._oolong_index.cache_clear()


def test_sqe_public_projection_is_allowlisted():
    assert source._public({'summary':'Facts', 'is_winner':True, 'answer':'secret'}, ['summary']) == {'summary':'Facts'}


def test_sqe_selection_and_join_preserve_official_merge_multiplicity():
    positive={'id':1};negative={'id':2};unknown={'id':3}
    assert score_oolong([positive,positive,negative],[positive],
                        {'kind':'sqe_set_f1','mode':'selection','negative_rows':[negative]})==pytest.approx(4/5)
    assert score_oolong([positive,unknown],[positive],
                        {'kind':'sqe_set_f1','mode':'selection','negative_rows':[negative]})==1
    assert score_oolong([positive,unknown],[positive],
                        {'kind':'sqe_set_f1','mode':'join'})==pytest.approx(2/3)
