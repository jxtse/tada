"""Read-only source adapter contract and reversible public encodings."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from functools import lru_cache
import re
from abc import ABC, abstractmethod
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
from local_config import path as local_path, require
QUARANTINE = local_path('quarantine')


def load_source(name, permutation_seed=None):
    require('quarantine')
    seed = os.environ.get('TADA_LABEL_PERMUTE', '') if permutation_seed is None else str(permutation_seed)
    quarantine_digest = hashlib.sha256(QUARANTINE.read_bytes()).hexdigest() if QUARANTINE.exists() else 'missing'
    return _load_source(name, seed, quarantine_digest)


def load_unpermuted_source(name):
    return load_source(name, '')


class _SourceOS:
    """Freeze only the loader's permutation flag; never mutate process environment."""
    def __init__(self, permutation_seed):
        self.environ = dict(os.environ)
        self.environ.pop('TADA_LABEL_PERMUTE', None)
        if permutation_seed:
            self.environ['TADA_LABEL_PERMUTE'] = str(permutation_seed)

    def __getattr__(self, name):
        return getattr(os, name)


@lru_cache(maxsize=16)
def _load_source(name, permutation_seed, quarantine_digest):
    spec = importlib.util.spec_from_file_location('tada_v5_' + name, REPO / 'sources' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.os = _SourceOS(permutation_seed)
    return module


def normalize(text):
    return re.sub(r'[^a-z0-9]+', ' ', str(text).lower()).strip()


def exclusions(path, source):
    data = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    pairs, records = set(), set()
    keys = [source + '_labels'] + (['C2_labels', 'C_labels'] if source == 'CUAD' else [])
    for key in keys:
        for item in data.get(key, []):
            record = item.get('contract_id', item.get('record', item.get('doc_id', item.get('prompt_id', item.get('cite')))))
            question = item.get('dp', item.get('question', item.get('category', item.get('hypothesis'))))
            if record and question:
                pairs.add((str(record), str(question)))
    aliases = {'CUAD': ['C_contracts', 'CUAD_contracts', 'C2_contracts'],
               'CUADX': ['C_contracts', 'CUAD_contracts', 'C2_contracts', 'CUADX_contracts'],
               'NDA': ['NDA_documents'], 'PREC': ['PREC_cases'], 'B': ['B_comparisons']}
    for key in aliases.get(source, [source + '_contracts']):
        for item in data.get(key, []):
            record = item if isinstance(item, str) else item.get('contract_id', item.get('record', item.get('doc_id', item.get('prompt_id', item.get('cite')))))
            if record:
                records.add(str(record))
    return pairs, records


def disabled_questions(path, source):
    """Owner-approved question-level disablement (``disabled_questions`` in the quarantine file).

    A disabled question keeps its original labels and codebook untouched (frozen and admitted
    packages still decode and rebuild). It is removed only from NEW task design: intents, the
    designer catalogue and the designer format check. Used when a whole question rests on an
    annotation convention that per-record exclusion cannot contain.
    """
    path = Path(path)
    stamp = path.stat().st_mtime_ns if path.exists() else None
    return _disabled_questions(str(path), str(source).lower(), stamp)


@lru_cache(maxsize=64)
def _disabled_questions(path, source, stamp):
    data = json.loads(Path(path).read_text()) if stamp is not None else {}
    return frozenset(str(row['question']) for row in data.get('disabled_questions', [])
                     if str(row.get('source', '')).lower() == source)


class SourceAdapter(ABC):
    source = ''

    def __init__(self, quarantine=QUARANTINE, permutation_seed=None):
        self.quarantine = Path(quarantine)
        self.permutation_seed = os.environ.get('TADA_LABEL_PERMUTE', '') if permutation_seed is None else str(permutation_seed)

    def reference(self):
        return type(self)(quarantine=self.quarantine, permutation_seed='')

    @abstractmethod
    def questions(self):
        """Question metadata keyed by stable source question ID."""

    @abstractmethod
    def records(self):
        """Label-free, pseudonymised records keyed by original record ID."""

    @abstractmethod
    def label_pool(self):
        """Eligible (record, question) -> (original answer, evidence segments)."""

    def scorable(self, record, question):
        return (record, question) in self.label_pool()

    def encode(self, question, answer):
        options = self.questions()[question]['options']
        if answer not in options:
            raise ValueError(f'Invalid {self.source} answer for {question}: {answer!r}')
        return answer

    def decode(self, question, visible):
        # Deliberately exact: synonyms and fuzzy matching would break the bijection.
        return self.encode(question, visible)

    def record_map(self):
        result = {r: getattr(self, 'visible_prefix', self.source) + '-' + hashlib.sha256((self.source + ':' + r).encode()).hexdigest()[:16]
                  for r in sorted(self.records())}
        if len(set(result.values())) != len(result):
            raise ValueError('Record ID collision')
        return result

    def visible_id(self, record):
        return self.record_map()[record]

    def record_id(self, visible):
        return {v: k for k, v in self.record_map().items()}[visible]

    def pseudonymize(self, record, text):
        if record not in self.records():
            raise KeyError(record)
        return text

    def scoring_rule(self, question):
        self.questions()[question]
        return {'name': 'categorical_exact', 'decode': 'identity', 'scope': 'original categorical answer'}

    def official_score(self, question, predicted, expected):
        try:
            return float(self.decode(question, predicted) == expected)
        except ValueError:
            return 0.0

    def keyword_predict(self, question, text):
        return generic_term_predict(self.questions()[question]['options'], text)


@lru_cache(maxsize=256)
def _text_tokens(text):
    return frozenset(re.findall(r'[a-z0-9]+', str(text).lower()))


@lru_cache(maxsize=512)
def _distinctive_tokens(options):
    tokens = [_text_tokens(option) for option in options]
    return tuple(terms - frozenset().union(*(other for j, other in enumerate(tokens) if j != i))
                 for i, terms in enumerate(tokens))


def distinctive_tokens(options):
    """Tokens unique to one option; exact lowercase ASCII word/number tokens.

    No stopwords, stemming, frequency weights, or learned thresholds are used.
    """
    return [set(terms) for terms in _distinctive_tokens(tuple(options))]


def generic_term_predict(options, text):
    """Choose the first option whose nonempty distinctive-token set is present.

    Every distinctive token must occur as a complete token in the agreement.
    Multiple matches use public option order; no match returns its first option.
    This deterministic definition is frozen before R1 pool measurement.
    """
    if not options:
        raise ValueError('At least one option is required')
    present = _text_tokens(text)
    for option, terms in zip(options, _distinctive_tokens(tuple(options))):
        if terms and terms <= present:
            return option
    return options[0]
