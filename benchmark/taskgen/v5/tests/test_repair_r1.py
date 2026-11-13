"""R1 admission failures cannot trigger seed searches or erase measurement rows."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tada_taskgen.compiler import guard_instance, InstanceCompilationError
from tada_taskgen.gates import classify, aggregate, run_multiseed, POLICY


def scenario():
    return {'source':'MAUD','questions':[f'q{j}' for j in range(5)], 'work_items':[{'id':'coding'}],
            'thresholds':{'include_all':.90,'ignore_revisions':.90}}


def fake_adapter(monkeypatch, answers):
    pool={(f'r{i}',f'q{j}'):(a,[]) for i,a in enumerate(answers) for j in range(5)}
    adapter=SimpleNamespace(label_pool=lambda:pool)
    adapter.reference=lambda:adapter
    monkeypatch.setattr('tada_taskgen.adapters.get_adapter',lambda _:adapter)
    return {'seed':7,'spec':scenario(),'records':[{'record_id':f'r{i}','questions':[f'q{j}' for j in range(5)]} for i in range(len(answers))]}


@pytest.mark.parametrize('answers',[['Yes']*16,['Yes']*15+['No'],['Yes']*7+['No']*8+['Other']])
def test_instance_guard_rejects_single_or_singleton_class(monkeypatch,answers):
    sp=fake_adapter(monkeypatch,answers)
    with pytest.raises(InstanceCompilationError) as error:guard_instance(sp)
    assert error.value.details['invalid_questions']['q0']


def test_instance_guard_accepts_two_of_each(monkeypatch):
    sp=fake_adapter(monkeypatch,['Yes','No','Yes','No'])
    assert guard_instance(sp)['class_counts']=={f'q{j}':{'No':2,'Yes':2} for j in range(5)}


def test_failed_compilation_counts_all_five_seeds_without_retry(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr('tada_taskgen.schema.validate_spec',lambda spec:spec)
    def compile_fail(spec,seed,out,**kwargs):
        seen.append(seed)
        raise InstanceCompilationError('minority has one cell',{'seed':seed})
    monkeypatch.setattr('tada_taskgen.compiler.compile_task',compile_fail)
    monkeypatch.setattr('tada_taskgen.gates.run_controller',lambda *a,**k:pytest.fail('must not run controller'))
    result=run_multiseed(scenario(),tmp_path/'five')
    assert seen==POLICY['seeds']
    assert result['compilation_failures']==5 and result['eligible_instances']==[]
    assert not result['pass'] and result['keyword_gate']['passes']==0
    for row in result['instances']:
        assert row['failure']['details']['seed']==row['seed']
        assert all(metric['value'] is None and not metric['pass'] for metric in row['metrics'].values())
        assert Path(row['report']).exists()


def test_build_failed_controller_still_aggregates_missing_measurements():
    spec=scenario()
    rows=[dict(seed=s,**classify({'verdict':'build_failed','steps':{}},spec)) for s in POLICY['seeds']]
    result=aggregate(rows,'frozen')
    assert not result['pass']
    assert result['worst_values']['keyword'] is None
    assert result['metric_pass_counts']['keyword']==0


def test_schema_keeps_historical_screening_report_only():
    from tada_taskgen.schema import validate_spec
    legacy=json.loads((Path(__file__).parents[1]/'specs/maud_r7.json').read_text())
    assert validate_spec(legacy) is legacy
