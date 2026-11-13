import importlib.util
import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, REPO / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_v5_designer_has_schema_and_no_code_contract():
    designer = module('run_designer')
    prompt = designer.assemble('MAUD', v5=True)
    assert '{{' not in prompt
    assert len(designer.V5_REQUIRED) == 4
    assert 'web_search' not in designer.V5_TOOLS
    assert 'filenames' in prompt and 'consistency-only' in prompt
    assert not designer.audit_prompt(prompt)


def test_old_designer_still_assembles_unchanged():
    designer = module('run_designer')
    assert 'build/build.py' in designer.REQUIRED
    assert 'web_search' in designer.TOOLS
    if not (REPO / 'designer_pi.md').is_file():
        with pytest.raises(ValueError, match='Legacy v4 designer inputs'):
            designer.assemble('MAUD')
    else:
        assert '{{' not in designer.assemble('MAUD')


def test_v5_reviewer_contract():
    reviewer = module('review_package')
    assert 'may list 2–4 ordered work items' in reviewer.INSTRUCTIONS
    assert 'round-trip report' in reviewer.INSTRUCTIONS
    assert 'scenario_spec.json' in reviewer.V5_FILES
