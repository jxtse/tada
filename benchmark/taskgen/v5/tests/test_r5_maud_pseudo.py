"""V5 name packaging keeps legal forms, source answers, and provision context."""
from pathlib import Path
import re

import pytest

from tada_taskgen.adapters.maud import MAUDAdapter
from tada_taskgen.adapters import maud_pseudo as pseudo
from tada_taskgen.adapters.base import exclusions, normalize


class NameAPI:
    FIRST = ['Alder', 'Birchwood', 'Cobalt']
    SECOND = ['Systems', 'Networks']
    SUFFIX = re.compile(r',?\s+(?:Inc\.?|LLC|Holdings?)$', re.I)

    def __init__(self, target, subs=()):
        self.parties = {'target': target, 'acquirer': 'Buyer Industries LLC', 'subs': subs}

    def _parties(self, record):
        return self.parties


@pytest.mark.parametrize('suffix', [', Inc.', ', INC.', ' Incorporated', ' Corporation', ' CORP.',
    ' LLC', ' L.L.C.', ' L.P.', ' LP', ' LLP', ' L.L.P.', ' plc', ' N.V.', ' B.V.', ' GmbH',
    ' Ltd.', ' LIMITED', ' Trust', ' Company', ' Co.', ' S.A.', ' AG', ' Limited Liability Company',
    ' Public Limited Company', ' Limited Partnership'])
def test_legal_form_and_original_case_are_preserved(suffix):
    name = 'Original Name' + suffix
    api = NameAPI(name)
    original = name + ', a Delaware limited liability company.'
    changed = pseudo.pseudonymize(api, 'synthetic', original)
    assert 'Original Name' not in changed
    assert changed.endswith(suffix + ', a Delaware limited liability company.')
    assert pseudo.core_name(name) == 'Original Name'
    upper = pseudo.pseudonymize(api, 'synthetic', name.upper())
    assert upper.endswith(suffix.upper()) and 'ORIGINAL NAME' not in upper


@pytest.mark.parametrize('name', [
    'Charter of the Interim Surviving Entity. At the Effective Time',
    'Subsidiary of Standard Industries Holdings Inc., and Gibraltar Merger Sub Inc.',
    'A heading with no legal entity suffix',
    'Not an entity. Sub may merge',
    'Long ' * 20 + 'Merger Sub Inc.',
])
def test_malformed_subsidiary_parses_are_skipped(name):
    api = NameAPI('Target Inc.', [name])
    assert not pseudo.subsidiary_name(name)
    text = '1.9 ' + name + ', the certificate of incorporation remains.'
    assert pseudo.pseudonymize(api, 'synthetic', text) == text


@pytest.mark.parametrize('name', ['Bridge Merger Corp.', 'Acme Bidco', 'Acme MergeCo', 'Stone Sub LLC'])
def test_short_entity_subsidiary_parses_are_used(name):
    assert pseudo.subsidiary_name(name)
    api = NameAPI('Target Inc.', [name])
    result = pseudo.pseudonymize(api, 'synthetic', name)
    assert result != name
    suffix = pseudo.LEGAL_FORM.search(name)
    if suffix:
        assert result.endswith(suffix.group())


@pytest.fixture(scope='module')
def real():
    return MAUDAdapter(permutation_seed='')


def test_contract_109_heading_and_principal_private_audit(real):
    heading = 'Charter of the Interim Surviving Entity. At the Effective Time'
    assert heading in re.sub(r'\s+', ' ', real.records()['contract_109']['text'])
    assert not pseudo.residual_original_names(real.api)


def test_labels_are_unchanged_and_evidence_uses_the_visible_text(real):
    old = real.api._labels()
    pool = real.label_pool()
    shared = set(old) & set(pool)
    assert shared and all(old[key][0] == pool[key][0] for key in shared)
    normalized = {record: normalize(row['text']) for record, row in real.records().items()}
    assert all(all(normalize(segment) in normalized[record] for segment in evidence)
               for (record, _), (_, evidence) in pool.items())
    assert {q for _, q in old} == {q for _, q in pool}
    assert all(real.scorable(*key) for key in list(pool)[:5])


def test_question_provision_context_without_option_changes(real):
    old = real.api.deal_points()
    current = real.questions()
    assert set(old) == set(current)
    for question, metadata in current.items():
        assert metadata['question'] == metadata['provision'] + ': ' + old[question]['question']
        assert metadata['options'] == old[question]['answer_options']
        assert all(real.encode(question, value) == value for value in metadata['options'])
    assert current['DP068']['provision'] == current['DP069']['provision'] == 'Intervening Event Definition'
    assert current['DP071']['provision'] == 'Superior Offer Definition'


def test_screening_version_binds_pseudonymiser(real):
    from tada_taskgen.screening import _snapshot
    version = _snapshot(real)[-1]
    assert version['pseudonymiser_version'] == pseudo.VERSION
    assert version['rule_sha256']


def test_maud_permutation_retains_public_text_and_scorable_pairs(real):
    shuffled = MAUDAdapter(permutation_seed='991')
    assert shuffled.records() == real.records()
    original, changed = real.label_pool(), shuffled.label_pool()
    assert set(original) == set(changed)
    assert any(original[key][0] != changed[key][0] for key in original)
