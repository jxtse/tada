import json
import importlib.util
from pathlib import Path

import pytest

from tada_taskgen import design_loop

REPO = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, REPO / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_reviewer_failure_stops_without_spending_revisions(tmp_path):
    calls = []

    def designer(intent, output, previous, feedback):
        calls.append(1)
        return json.loads((REPO / 'v5/specs/maud_screened.json').read_text())

    def gates(spec, out):
        return {'task': str(tmp_path / 'task'), 'hard_gates': {k: {'pass': True} for k in design_loop.HARD_GATES}}

    def reviewer(*args):
        raise RuntimeError('prompt too long')

    result = design_loop.prepare_candidate({}, tmp_path / 'c', designer=designer, compile_and_gate=gates,
                                           reviewer=reviewer)
    assert result['verdict'] == 'infra_failed' and len(calls) == 1 and result['revisions'] == 0


def test_review_materials_bound_agreement_text(tmp_path):
    task = tmp_path / 'task'
    agreements = task / 'environment/public/agreements'
    agreements.mkdir(parents=True)
    (task / 'instruction.md').write_text('Prepare the workbook.')
    (agreements / 'A.txt').write_text('x' * 500_000)
    (task / 'environment/public/notice.txt').write_text('y' * 60_000)
    material = module('review_package').v5_materials(task, {})
    limit = module('review_package').REVIEW_AGREEMENT_CHARS
    assert len(material['environment/public/agreements/A.txt']) < limit + 300
    assert 'full unchanged agreement' in material['environment/public/agreements/A.txt']
    assert len(material['environment/public/notice.txt']) == 60_000
