"""ContractNLI original expert choices; public NDA text, no annotation dumps."""
from functools import cached_property
from .base import SourceAdapter, exclusions, load_source, normalize


class NDAAdapter(SourceAdapter):
    source = 'NDA'
    pseudonymiser_version = 'public-nda-text-identity-v1'
    limitations = ('Public template party names are retained exactly; only record identifiers are pseudonymised.',)

    @cached_property
    def api(self):
        return load_source('nda_sources', self.permutation_seed)

    @cached_property
    def _questions(self):
        return {qid: {'question': row['text'], 'options': ['Entailment', 'Contradiction', 'NotMentioned'],
                      'answer_options': ['Entailment', 'Contradiction', 'NotMentioned'],
                      'coding_conventions': ['Assess the supplied hypothesis against the whole NDA.',
                        'Use Entailment, Contradiction or NotMentioned exactly; absence of a statement is not contradiction.']}
                for qid, row in self.api.hypotheses().items()}

    def questions(self):
        return self._questions

    @cached_property
    def _records(self):
        # Deliberately omit original file names, URLs, splits and all annotations.
        return {row['doc_id']: {'text': row['text']} for row in self.api.iter_documents()}

    def records(self):
        return self._records

    @cached_property
    def _eligible(self):
        original = load_source('nda_sources', '')._labels()
        texts = {record: normalize(row['text']) for record, row in self.records().items()}
        return {key for key, (answer, spans) in original.items()
                if answer in self.questions()[key[1]]['options']
                and (answer == 'NotMentioned' or spans)
                and all(normalize(span) in texts[key[0]] for span in spans)}

    def label_pool(self):
        pairs, records = exclusions(self.quarantine, self.source)
        labels = self.api._labels()
        return {key: (labels[key][0], list(labels[key][1])) for key in sorted(self._eligible)
                if key not in pairs and key[0] not in records}

    def scorable(self, record, question):
        pairs, records = exclusions(self.quarantine, self.source)
        return (record, question) in self._eligible and (record, question) not in pairs and record not in records
