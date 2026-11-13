"""Synthetic integration pools independent of real screening/seed measurements.

Use six real question schemas with controlled balanced labels and identical text.
Adapter property tests separately exercise every real source encoding and pool.
"""
import copy
import json
from pathlib import Path
from unittest.mock import patch

from tada_taskgen.adapters import get_adapter as real_adapter
from tada_taskgen.adapters.base import SourceAdapter
from tada_taskgen.screening import screen_adapter


class IntegrationAdapter(SourceAdapter):
    def __init__(self, source):
        self.original = real_adapter(source, permutation_seed='')
        self.source = source
        self.quarantine = self.original.quarantine
        self.permutation_seed = ''
        self._questions = {q: row for q, row in sorted(self.original.questions().items())[:6]}

    def reference(self):
        return self

    def questions(self):
        return self._questions

    def records(self):
        return {f'fixture-{i:02d}': {'text': 'This agreement sets out the contractual obligations.'}
                for i in range(32)}

    def label_pool(self):
        return {(record, q): (row['options'][i % 2], [])
                for i, record in enumerate(self.records()) for q, row in self.questions().items()}

    def keyword_predict(self, question, text):
        return self.original.keyword_predict(question, text)


_ADAPTERS = {}
_REPORTS = {}


def integration_adapters():
    """Patch adapter lookup only for compiler/gate unit integration modules."""
    for source in ('MAUD', 'CUAD'):
        if source not in _ADAPTERS:
            _ADAPTERS[source] = IntegrationAdapter(source)
    return patch('tada_taskgen.adapters.get_adapter', side_effect=lambda source, **kw: _ADAPTERS[source])


def screened_spec(source):
    adapter = _ADAPTERS[source]
    if source not in _REPORTS:
        _REPORTS[source] = screen_adapter(adapter)
    report = copy.deepcopy(_REPORTS[source])
    spec = json.loads((Path(__file__).resolve().parents[1] / 'specs' /
                       ('maud_r7.json' if source == 'MAUD' else 'cuad_c2.json')).read_text())
    spec['questions'] = sorted(adapter.questions())
    spec['sample_size'] = 32  # Entire synthetic population; never search real seeds.
    return spec, report
