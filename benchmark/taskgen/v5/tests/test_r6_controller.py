"""Controller provenance preserves original integer label types, independently of values."""
import json

import pytest

from tada_taskgen import multisource
from weekend_checks import strict_provenance


def test_prec_public_catalogue_caveats_do_not_advertise_private_scan_tokens():
    from controller import PRIVATE_STRINGS
    from tada_taskgen.designer_sources import catalogues
    public = json.dumps(catalogues(['PREC']))
    assert not [token for token in PRIVATE_STRINGS if token in public]
    assert 'precedent-alteration variable' in public


@pytest.mark.parametrize('original,impostor', [(0, False), (1, True), (0, 0.0), (1, 1.0)])
def test_provenance_rejects_values_equal_to_b_labels_with_different_type(tmp_path, monkeypatch, original, impostor):
    text = 'The public study reports the specified outcome in the full article.'
    pair = ('comparison-1', 'direction')

    class Adapter:
        def reference(self):
            return self

        def label_pool(self):
            return {pair: (original, ['specified outcome'])}

        def records(self):
            return {pair[0]: {'text': text}}

    monkeypatch.setattr(multisource, 'adapter_for', lambda spec: Adapter())
    tests = tmp_path / 'tests'
    public = tmp_path / 'environment/public'
    tests.mkdir()
    public.mkdir(parents=True)
    (public / 'article.txt').write_text(text)
    (tests / 'scenario_spec.json').write_text(json.dumps({'version': 5, 'source': 'B'}))
    origin = {'source': 'B', 'record': pair[0], 'question': pair[1]}
    key = '|'.join(pair)
    gold = {'schema_version': 5, 'cells': [dict(origin, origin=origin)],
            'sheets': {'Coding': {'rows': {key: {}}}}}
    (tests / 'gold.json').write_text(json.dumps(gold))
    dependency = {'source_id': ':'.join(pair), 'label': original, 'visible_file': 'article.txt'}
    provenance = {'rows': {'Coding|' + key: {'dependencies': [dependency]}}}
    path = tests / 'provenance.json'
    path.write_text(json.dumps(provenance))
    baseline = strict_provenance(tmp_path, 'B', {})
    assert baseline['pass'], baseline

    # Python equality alone accepts all these substitutions; source identity must not.
    assert impostor == original and type(impostor) is not type(original)
    dependency['label'] = impostor
    path.write_text(json.dumps(provenance))
    result = strict_provenance(tmp_path, 'B', {})
    assert not result['pass']
    assert result['failures'] == [{'key': 'Coding|' + key, 'source_id': ':'.join(pair),
                                  'error': 'unknown source or wrong label'}]
