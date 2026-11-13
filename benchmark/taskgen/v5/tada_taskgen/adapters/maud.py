"""MAUD adapter wrapping the existing audited source loader without relabelling."""
from functools import cached_property
import json
from pathlib import Path
from .base import SourceAdapter, load_source, exclusions
from . import maud_pseudo


class MAUDAdapter(SourceAdapter):
    source = 'MAUD'
    pseudonymiser_version = maud_pseudo.VERSION

    @cached_property
    def api(self):
        return load_source('maud_sources', self.permutation_seed)

    @cached_property
    def _questions(self):
        conventions = json.loads(Path(__file__).with_name('maud_conventions.json').read_text())
        out = {}
        provisions = {(question, subquestion): sorted(set(map(str, group.text_type.dropna())))
                      for (question, subquestion), group in self.api._labels_raw().groupby(['question', 'subquestion'])}
        for qid, q in self.api.deal_points().items():
            conv = [text for _id, scope, text in conventions if scope == 'All deal points' or qid in scope.split(', ')]
            provision = provisions[q['question'], q['subquestion']]
            if len(provision) != 1:
                raise ValueError(f'Expected one provision context for {qid}')
            out[qid] = dict(q, provision=provision[0], question=provision[0] + ': ' + q['question'],
                           options=list(q['answer_options']), coding_conventions=conv)
        return out

    def questions(self):
        return self._questions

    @cached_property
    def _records(self):
        return maud_pseudo.records(self.api)

    def records(self):
        return self._records

    def label_pool(self):
        pairs, records = exclusions(self.quarantine, self.source)
        return {k: (v[0], list(v[1])) for k, v in maud_pseudo.grounded_labels(self.api).items()
                if k not in pairs and k[0] not in records}

    def scorable(self, record, question):
        pairs, records = exclusions(self.quarantine, self.source)
        return record not in records and (record, question) not in pairs and (record, question) in maud_pseudo.grounded_labels(self.api)

    def pseudonymize(self, record, text):
        return maud_pseudo.pseudonymize(self.api, record, text)

    def unpermuted(self):
        return self.reference()

    def keyword_predict(self, question, text):
        from .maud_keywords import DEDICATED_QUESTIONS, predict
        if question in DEDICATED_QUESTIONS:
            return predict(question, text, self.questions())
        return super().keyword_predict(question, text)
