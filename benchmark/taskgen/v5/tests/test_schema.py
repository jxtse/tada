import copy
import json
from pathlib import Path
import pytest
from tada_taskgen.schema import validate_spec_structure as validate_spec, validate_graph, safe_path


def spec():
    return {'version': 5, 'id': 'x', 'source': 'MAUD', 'category': 'parallel_coding', 'questions': ['q'], 'sample_size': 2,
            'work_items': [{'id': 'scope', 'title': 'Scope', 'order': 1, 'depends_on': []},
                           {'id': 'coding', 'title': 'Code', 'order': 2, 'depends_on': ['scope']}],
            'documents': [], 'dependency_graph': {'nodes': [], 'edges': []}, 'instruction': 'Review.', 'design': 'Work.',
            'thresholds': {'include_all': .90, 'ignore_revisions': .90}}


def test_valid_spec_and_machine_schemas():
    assert validate_spec(spec())
    for p in (Path(__file__).parents[1] / 'schemas').glob('*.json'):
        assert json.loads(p.read_text())['$schema']


@pytest.mark.parametrize('mutate', [lambda s:s.update(sample_size=True), lambda s:s.update(questions=[]),
    lambda s:s['work_items'][0].update(depends_on=['coding']), lambda s:s['thresholds'].update(include_all=1.1),
    lambda s:s.update(work_items=s['work_items'][:1])])
def test_invalid_specs(mutate):
    s=spec(); mutate(s)
    with pytest.raises(ValueError): validate_spec(s)


@pytest.mark.parametrize('path', ['/etc/passwd','../x','a/../x','a\\b',''])
def test_path_escape(path):
    with pytest.raises(ValueError): safe_path(path)


def test_cycle_and_derived_gold():
    n={'id':'a','type':'derived_quantity','provision':'solver-derived','scoring':'consistency-only'}
    with pytest.raises(ValueError): validate_graph({'nodes':[n],'edges':[{'from':'a','to':'a'}]})
    with pytest.raises(ValueError): validate_graph({'nodes':[{**n,'gold':3}],'edges':[]})
    with pytest.raises(ValueError): validate_graph({'nodes':[{**n,'type':'construction_fact'}],'edges':[]})


def test_four_designer_files_bind_and_disagreements_fail(tmp_path, monkeypatch):
    monkeypatch.setattr("tada_taskgen.screening.require_eligible", lambda *args, **kwargs: {})
    from tada_taskgen.schema import load_designer_output
    s=json.loads((Path(__file__).resolve().parents[2]/'v5/specs/maud_screened.json').read_text())
    (tmp_path/'scenario_spec.json').write_text(json.dumps(s))
    (tmp_path/'dependency_graph.json').write_text(json.dumps(s['dependency_graph']))
    (tmp_path/'instruction.md').write_text(s['instruction'])
    (tmp_path/'design.md').write_text(s['design'])
    assert load_designer_output(tmp_path) == s
    # Standalone files are authoritative: a stale embedded copy can no longer cause a rejection.
    (tmp_path/'instruction.md').write_text('A revised request.')
    assert load_designer_output(tmp_path)['instruction'] == 'A revised request.'
    (tmp_path/'instruction.md').write_text(s['instruction'])
    (tmp_path/'build.py').write_text('pass')
    with pytest.raises(ValueError,match='four'):load_designer_output(tmp_path)
