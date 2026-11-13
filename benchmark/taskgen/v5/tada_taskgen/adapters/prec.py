"""Original SCDB precedentAlteration choices, using public CAP majority opinions.

LoC pairs are not converted to negatives or current-law labels. Their positive-only
inventory cannot supply label-independent candidate pairs in this adapter.
"""
from functools import cached_property
import re
from .base import SourceAdapter, exclusions, load_source
from .public_records import PublicRecords


class PRECAdapter(SourceAdapter):
    source = 'PREC'
    pseudonymiser_version = 'public-case-citations-identity-v1'
    limitations = ('Uses the original SCDB precedent-alteration variable only (CC BY-NC 3.0 US); not a current-law determination.',
                  'LoC overruling pairs are not scored; absence from their incomplete table is never a negative label.',
                  'Public case names and citations are retained to preserve legal reference meaning.')

    @cached_property
    def api(self):
        return load_source('prec_sources', self.permutation_seed)

    def questions(self):
        return {'precedent_alteration': {'question': 'Does the majority opinion say it formally alters precedent?',
                'options': ['0', '1'], 'answer_options': ['0', '1'],
                'coding_conventions': ['1 means the majority says it formally alters precedent; 0 means it does not.',
                    'This is the original SCDB variable, not whether any cited case remains good law.',
                    'Do not treat missing LoC table entries as evidence of no overruling.']}}

    @cached_property
    def _metadata(self):
        original = load_source('prec_sources', '')
        return {row['cite']: row for row in original.iter_cases()
                if '#' not in row['cite']
                and original.scdb_precedent_alteration(row['cite']) in ('0', '1')
                and 'majority' in row['opinion_types']}

    def _public(self, cite):
        case = self._metadata[cite]
        text = '\n\n'.join(o['text'] for o in self.api.opinions(cite) if o['type'] == 'majority')
        return {'text': f"{case['name']}\n{cite}\n{case['date']}\n\nMajority opinion\n\n{text}"}

    @cached_property
    def _records(self):
        return PublicRecords(self._metadata, self._public)

    def records(self):
        return self._records

    @cached_property
    def _labels(self):
        return {(cite, 'precedent_alteration'): (self.api.scdb_precedent_alteration(cite), [])
                for cite in self._metadata}

    def label_pool(self):
        pairs, records = exclusions(self.quarantine, self.source)
        return {key: value for key, value in self._labels.items() if key not in pairs and key[0] not in records}

    def scorable(self, record, question):
        pairs, records = exclusions(self.quarantine, self.source)
        return (record, question) in self._labels and (record, question) not in pairs and record not in records

    def keyword_predict(self, question, text):
        self.questions()[question]
        return '1' if re.search(r'\b(?:we (?:now )?overrule|is hereby overruled)\b', text, re.I) else '0'
