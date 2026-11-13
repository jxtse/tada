"""CUAD C2 categorical adapter; cache text already has principal-party pseudonyms."""
from functools import cached_property
import string
import json
from .base import SourceAdapter, load_source, load_unpermuted_source, exclusions, normalize


def jaccard_words(prediction, reference):
    """CUAD span overlap hook: lower-case, delete punctuation, compare word sets.

    C2 is a categorical Yes/No projection. Its cells use exact categorical equality;
    span overlap is available separately and never substitutes for label equality.
    """
    def words(value):
        return set(str(value).lower().translate(str.maketrans('', '', string.punctuation)).split())
    a, b = words(prediction), words(reference)
    return len(a & b) / len(a | b) if a | b else 1.0


class CUADAdapter(SourceAdapter):
    source = 'CUAD'
    visible_prefix = 'C2'

    @cached_property
    def api(self):
        return load_source('c2_sources', self.permutation_seed)

    def questions(self):
        return {qid: {'question': text, 'options': ['No', 'Yes'], 'answer_options': ['No', 'Yes'],
                      'coding_conventions': ['Use the original category question, including all alternatives.',
                         'Yes denotes an annotated qualifying clause; No denotes its absence in the reviewed agreement.',
                         'Do not narrow a category to one trigger, remedy, or party.']}
                for qid, text in self.api.CATEGORY_QUESTIONS.items()}

    @cached_property
    def _raw(self):
        return {r['contract_id']: r for r in self.api.iter_contracts()}

    @cached_property
    def _records(self):
        return {rid: {k: v for k, v in row.items() if k not in ('labels', 'evidence_spans')}
                for rid, row in self._raw.items()}

    def records(self):
        return self._records

    @cached_property
    def _eligible_keys(self):
        # Eligibility is bound to the unpermuted cache, including in leak-test builds.
        result = set()
        for record in json.loads(self.api.CACHE.read_text())['contracts']:
            text = normalize(record['text'])
            for qid in self.questions():
                answer = record['labels'].get(qid)
                evidence = record['evidence_spans'].get(qid, [])
                if answer not in ('Yes', 'No') or (answer == 'Yes' and not evidence):
                    continue
                if all(normalize(span) in text for span in evidence):
                    result.add((record['contract_id'], qid))
        return result

    def label_pool(self):
        pairs, records = exclusions(self.quarantine, self.source)
        return {(rid, qid): (self._raw[rid]['labels'][qid], list(self._raw[rid]['evidence_spans'][qid]))
                for rid, qid in sorted(self._eligible_keys) if (rid, qid) not in pairs and rid not in records}

    def scorable(self, record, question):
        pairs, records = exclusions(self.quarantine, self.source)
        return record not in records and (record, question) not in pairs and (record, question) in self._eligible_keys

    def unpermuted(self):
        return self.reference()

    def scoring_rule(self, question):
        result = super().scoring_rule(question)
        result['span_hook'] = 'jaccard_words (not used for categorical Yes/No scoring)'
        return result

    def keyword_predict(self, question, text):
        import re
        patterns = {
            'change_of_control': r'change\s+(?:in|of)\s+control|merger|substantially all',
            'anti_assignment': r'(?:may|shall) not assign|prior written consent.{0,50}assign|assign.{0,70}consent',
            'non_compete': r'not.{0,30}compet|non.?compet',
            'exclusivity': r'exclusive|exclusivity|sole supplier',
            'no_solicit_customers': r'not.{0,40}solicit.{0,40}(?:customer|client)',
            'non_transferable_license': r'non.?transferable|license.{0,50}not.{0,30}transfer',
            'termination_for_convenience': r'without cause|for convenience|any reason',
            'rofr_rofo_rofn': r'right of first (?:refusal|offer|negotiation)',
        }
        return 'Yes' if re.search(patterns[question], text, re.I | re.S) else 'No'
