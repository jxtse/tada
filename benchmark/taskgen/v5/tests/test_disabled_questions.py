"""Owner-approved question-level disablement (2026-11-13).

Original labels and codebook stay intact so frozen/admitted packages still decode and rebuild;
a disabled question is only withheld from new design (intents, designer catalogue, format check).
"""
import json

from tada_taskgen.adapters import get_adapter
from tada_taskgen.adapters.base import QUARANTINE, disabled_questions


def _scratch(tmp_path, questions):
    data = json.loads(QUARANTINE.read_text())
    data['disabled_questions'] = [{'source': 'MAUD', 'question': q} for q in questions]
    path = tmp_path / 'quarantine.json'
    path.write_text(json.dumps(data))
    return path


def test_disabled_question_keeps_labels_and_codebook(tmp_path):
    path = _scratch(tmp_path, ['DP041'])
    adapter = get_adapter('MAUD', quarantine=path)
    assert disabled_questions(path, 'MAUD') == {'DP041'}
    assert 'DP041' in adapter.questions()
    assert any(q == 'DP041' for _, q in adapter.label_pool())


def test_disablement_is_source_scoped(tmp_path):
    path = _scratch(tmp_path, ['DP041'])
    assert disabled_questions(path, 'CUAD') == frozenset()


def test_live_quarantine_disables_dp040_for_new_design():
    assert 'DP040' in disabled_questions(QUARANTINE, 'MAUD')
    from tada_taskgen.designer_sources import catalogues
    catalogue = catalogues(["MAUD"])
    assert 'DP040' not in catalogue['MAUD']['questions']
    assert 'DP041' in catalogue['MAUD']['questions']


def test_designer_check_rejects_disabled_question(tmp_path, monkeypatch):
    from tada_taskgen import designer_check, schema
    monkeypatch.setattr(schema, 'load_designer_output',
                        lambda _d: {'questions': ['DP040', 'DP041'], 'source': 'MAUD'})
    monkeypatch.setattr('tada_taskgen.designer_sources.spec_sources', lambda spec: ['MAUD'])
    result = designer_check.check(tmp_path)
    assert not result['ok'] and 'MAUD:DP040' in result['error']
