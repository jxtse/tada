"""Original benchmark checkers, visibility, transport and bounded SQL execution."""
import json
from pathlib import Path
import sqlite3
import pytest
from tada_taskgen.adapters.benchmark import BenchmarkItemAdapter, LazyRecord
from tada_taskgen.benchmark_checkers import score_answer, sql_rows

@pytest.mark.parametrize('kind,expected,bad,spec',[
 ('exact',{'a':1},{'a':2},{}),
 ('dabstep','1, 2, 5','8, 7, 6',{}),
 ('blade_mcq',{'cvar_str':'DV: outcome','rationale':'private','is_llm_generated':False},3,{'correct_index':1}),
 ('palimpzest_fields',{'sender':'a@example.org','subject':'hello'},{'sender':'b@example.org','subject':'wrong'},{}),
 ('infiagent',[['number','1.5'],['label','Yes']],[['number','4'],['label','No']],{}),
 ('tablebench','10.6','77.8',{'metric':'em'}),
 ('tablebench','A sequence of different words','unrelated response',{'metric':'rouge_l'}),
 ('tablebench','100','150',{'metric':'tolerance_10'}),
 ('krama_success',1.234,9.8,{}),
 ('krama_rae',5,15,{}),
 ('krama_f1',[1,'hello'],[9,'wrong'],{}),
 ('set_f1',[['first'],['second']],[['wrong']],{}),
 ('lro_order',['a','b','c'],['c','b','a'],{}),
 ('lro_nmi',[1,1,2,2],[1,2,1,2],{}),
 ('loft_sql',[[1,'x']],[[9,'z']],{}),
 ('loft_sql_multi',[[[1]],[[2]]],[[[9]],[[9]]],{}),
 ('loft_em',['the cat'],['dog'],{}),
 ('loft_set_em',['cat','dog'],['horse'],{}),
 ('loft_retrieval',[['doc1',1],['doc2',1]],['doc3'],{}),
 ('spider2_table',{'tables':[[['1','x']]]},[['7','z']],{'ignore_order':False}),
])
def test_original_checker_gold_and_perturbation(kind,expected,bad,spec):
    spec={'kind':kind,**spec}
    assert score_answer(json.dumps(expected),expected,spec)==1
    assert score_answer(json.dumps(bad),expected,spec)<1
    assert score_answer('INVALID_RESPONSE',expected,spec)<1


def test_sql_result_sets_readonly_and_bounded(tmp_path):
    db=tmp_path/'data.sqlite';con=sqlite3.connect(db)
    con.executescript('CREATE TABLE t(x INTEGER); INSERT INTO t VALUES(1),(2);');con.close()
    spec={'kind':'sqlite_result_set','database':'data.sqlite'}
    assert score_answer('select x from t order by x desc','select x from t',spec,tmp_path)==1
    assert score_answer('select 7','select x from t',spec,tmp_path)==0
    assert score_answer('delete from t','select x from t',spec,tmp_path)==0
    assert score_answer('select 1','select 1',{'kind':'sqlite_result_set','database':'../outside'},tmp_path)==0
    assert score_answer('select 1.0','select 1',spec,tmp_path)==1
    assert score_answer("select '1'",'select 1',spec,tmp_path)==0
    assert sql_rows('select count(*) from t',db)==[[2]]


@pytest.mark.parametrize('source',['tag-bench','blade','dabstep','palimpzest','infiagent-dabench','tablebench','kramabench','lrobench','semtpch'])
def test_real_original_items_and_visibility(source):
    adapter=BenchmarkItemAdapter(source)
    assert adapter.status=='available' and adapter.label_pool()
    for (record,q),(expected,_) in list(adapter.label_pool().items())[:3]:
        visible=adapter.records()[record]
        assert adapter.decode(q,adapter.encode(q,expected))==expected
        assert score_answer(adapter.encode(q,expected),expected,adapter.benchmark_checker(record,q))==1
        assert score_answer('THIS_IS_AN_INCORRECT_RESPONSE',expected,adapter.benchmark_checker(record,q))<1
        assert set(visible)<= {'text','assets','text_bytes','text_sha256','_loader'}
        assert 'rationale' not in visible and 'answer' not in visible
        assert visible['text']
        for a in visible['assets']:
            assert Path(a['source_path']).is_file()
            assert not Path(a['path']).is_absolute() and '..' not in Path(a['path']).parts
    assert adapter.metadata['checker_source']


def test_dabstep_all_ten_dev_answers_bijective():
    adapter=BenchmarkItemAdapter('dabstep')
    assert len(adapter.label_pool())==10
    assert all(len(adapter.encode(q,a))<=2000 and adapter.decode(q,adapter.encode(q,a))==a for (_,q),(a,_) in adapter.label_pool().items())


def test_lazy_input_does_not_materialize_for_sizing():
    called=[];record=LazyRecord(text='',text_bytes=999,assets=[],_loader=lambda:called.append(1) or 'visible')
    assert record['assets']==[] and record.get('text_bytes')==999 and not called
    assert record['text']=='visible' and called==[1]


def test_infiagent_duplicate_original_field_names_follow_official_last_value():
    answer=[['a','0.38'],['a','0.39']]
    assert score_answer(json.dumps(answer),answer,{'kind':'infiagent'})==1
    assert score_answer('@a[0.39]',answer,{'kind':'infiagent'})==1


@pytest.mark.parametrize('source',['discoverybench','longds-bench','dsbench'])
def test_judge_only_sources_report_specific_original_verifier(source):
    adapter=BenchmarkItemAdapter(source)
    assert adapter.status=='requires_llm_judge'
    assert not adapter.label_pool() and adapter.reason
    assert 'judge' in adapter.reason.lower() or 'GPT' in adapter.reason


def test_krama_integer_types_and_percent_parity():
    assert score_answer(10001,10000,{'kind':'krama_success'})==0
    assert score_answer(10000,10000,{'kind':'krama_success'})==1
    assert score_answer('25%',.25,{'kind':'krama_rae'})==1
    assert score_answer('50%',.25,{'kind':'krama_rae'})==.5


def test_loft_official_uncapped_recall_five():
    expected=[[str(i),1] for i in range(6)]
    assert score_answer(expected,expected,{'kind':'loft_retrieval'})==5/6


def test_lro_dataframe_cell_casting_and_loft_multi_retrieval():
    assert score_answer([[1]], [['1']], {'kind':'lro_set_f1'})==1
    assert score_answer([[2]], [['1']], {'kind':'lro_set_f1'})==0
    spec={'kind':'loft_retrieval','multi_turn':True}
    assert score_answer([['doc1'],['doc2']],[['doc1',1],['doc2',1]],spec)==1
    assert score_answer([['wrong'],['doc2']],[['doc1',1],['doc2',1]],spec)==.5


def test_repaired_assets_require_explicit_visible_role(tmp_path):
    from tada_taskgen.adapters.benchmark import linked_assets
    p=tmp_path/'data.csv';p.write_text('visible')
    raw={'path':str(p),'status':'available'}
    assert not linked_assets({'input_assets':[raw]},'spider2')
    assert not linked_assets({'input_assets':[{**raw,'source_role':'hidden_test'}]},'loft')
    assert linked_assets({'input_assets':[{**raw,'source_role':'task_external_knowledge'}]},'spider2')


@pytest.mark.parametrize('source',['tag-bench','blade','dabstep','palimpzest','infiagent-dabench','tablebench','kramabench','lrobench','semtpch'])
def test_benchmark_quarantine_preserves_inputs(source,tmp_path):
    source_adapter=BenchmarkItemAdapter(source)
    (record,q),_=next(iter(source_adapter.label_pool().items()))
    quarantine=tmp_path/'quarantine.json'
    quarantine.write_text(json.dumps({source+'_labels':[{'record':record,'question':q}]}))
    adapter=BenchmarkItemAdapter(source,quarantine=quarantine)
    assert (record,q) not in adapter.label_pool()
    assert adapter.records()[record]==source_adapter.records()[record]
