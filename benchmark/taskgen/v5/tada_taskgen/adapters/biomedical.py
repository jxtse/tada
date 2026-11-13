"""Evidence Inference comparison labels with public paper and registry protocol.

Only unanimous valid expert comparisons from the legacy API are eligible. Registry
posted results and expert annotations are never rendered. Original direction
integers remain gold; their public encoding is an exact decimal string.
"""
from functools import cached_property, lru_cache
import json
from .base import SourceAdapter, exclusions, load_source, normalize
from .public_records import PublicRecords


class BAdapter(SourceAdapter):
    source = 'B'
    pseudonymiser_version = 'public-scientific-text-identity-v1'
    limitations = ('Original interventions, comparators, outcomes and public study identifiers are retained for scientific meaning.',
                  'Registry records contain protocol only; no posted results or expert annotation fields are public.')

    @cached_property
    def api(self):
        return load_source('b_sources', self.permutation_seed)

    def questions(self):
        return {'direction': {'question': 'What is the statistically significant outcome direction for the specified intervention versus comparator?',
                'options': [-1, 0, 1], 'answer_options': [-1, 0, 1],
                'coding_conventions': ['-1: significantly decreased; 0: no significant difference; 1: significantly increased.',
                    'Evaluate the record-specific intervention, comparator and outcome using the full paper.',
                    'Registry protocol supplies study context, not outcome results.']}}

    @cached_property
    def _comparisons(self):
        return {row['prompt_id']: row for row in load_source('b_sources', '').iter_comparisons()}

    @lru_cache(maxsize=64)
    def _paper(self, pmc):
        return self.api.paper_text(pmc)

    def _public(self, record):
        row = self._comparisons[record]
        context = {key: row[key] for key in ('intervention', 'comparator', 'outcome')}
        return {'text': 'Comparison\n' + json.dumps(context, ensure_ascii=False, indent=2)
                + '\n\nStudy protocol\n' + json.dumps(self.api.registry(row['nct']), ensure_ascii=False, indent=2)
                + '\n\nFull article\n' + self._paper(row['pmc'])}

    @cached_property
    def _records(self):
        return PublicRecords(self._comparisons, self._public)

    def records(self):
        return self._records

    @cached_property
    def _eligible(self):
        out = set()
        for record, row in self._comparisons.items():
            text = normalize(self._paper(row['pmc']))
            if row['evidence_spans'] and all(normalize(span) in text for span in row['evidence_spans']):
                out.add((record, 'direction'))
        return out

    @cached_property
    def _labels(self):
        return {(row['prompt_id'], 'direction'): (row['direction'], list(row['evidence_spans']))
                for row in self.api.iter_comparisons()}

    def label_pool(self):
        pairs, records = exclusions(self.quarantine, self.source)
        q = json.loads(self.quarantine.read_text()) if self.quarantine.exists() else {}
        papers = {str(row['pmc']) for row in q.get('B_papers', [])}
        prompts = {str(row.get('prompt_id', row.get('PromptID'))) for row in q.get('B_prompts', [])}
        return {key: self._labels[key] for key in sorted(self._eligible)
                if key not in pairs and key[0] not in records | prompts
                and self._comparisons[key[0]]['pmc'] not in papers}

    def scorable(self, record, question):
        if (record, question) not in self._eligible:
            return False
        pairs, records = exclusions(self.quarantine, self.source)
        q = json.loads(self.quarantine.read_text()) if self.quarantine.exists() else {}
        papers = {str(row['pmc']) for row in q.get('B_papers', [])}
        prompts = {str(row.get('prompt_id', row.get('PromptID'))) for row in q.get('B_prompts', [])}
        return ((record, question) not in pairs and record not in records | prompts
                and self._comparisons[record]['pmc'] not in papers)

    def encode(self, question, answer):
        self.questions()[question]
        if type(answer) is not int or answer not in (-1, 0, 1):
            raise ValueError('Invalid original Evidence Inference direction')
        return str(answer)

    def decode(self, question, visible):
        self.questions()[question]
        if type(visible) is not str or visible not in ('-1', '0', '1'):
            raise ValueError('Invalid encoded Evidence Inference direction')
        return int(visible)

    def keyword_predict(self, question, text):
        self.questions()[question]
        return 0

    def scoring_rule(self, question):
        self.questions()[question]
        return {'name': 'categorical_exact', 'decode': 'exact decimal string to original integer',
                'scope': 'original unanimous Evidence Inference direction'}
