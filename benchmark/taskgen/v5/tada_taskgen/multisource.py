"""Source-qualified adapter views; each cell remains one original expert label."""
from collections.abc import Mapping
from contextvars import ContextVar
from contextlib import contextmanager
import copy

def get_adapter(source, **kwargs):
    # Resolve at call time so the same registry is used by controller and tests.
    from .adapters import get_adapter as lookup
    return lookup(source, **kwargs)

_permutation = ContextVar('v5_label_permutation', default=None)


@contextmanager
def label_permutation(seed):
    token = _permutation.set(str(seed))
    try:
        yield
    finally:
        _permutation.reset(token)


def _adapter(source):
    seed = _permutation.get()
    return get_adapter(source, **({'permutation_seed': seed} if seed is not None else {}))


def scenario_sources(spec):
    from .adapters import canonical_source, AVAILABLE_SOURCES
    values = spec.get('sources', [spec.get('source')])
    if not isinstance(values, list) or not 1 <= len(values) <= 3:
        raise ValueError('sources must contain 1-3 dataset IDs')
    sources = [canonical_source(value) for value in values]
    from .batch1 import IDS
    if not set(sources) <= set(AVAILABLE_SOURCES) | {canonical_source(source) for source in IDS}:
        raise ValueError('unsupported source')
    if len(set(sources)) != len(sources):
        raise ValueError('duplicate source')
    return sorted(sources)


def qualify(source, key):
    return source + '::' + key


def split(key):
    return key.split('::', 1)


class _Records(Mapping):
    def __init__(self, adapters):
        self.adapters = adapters

    def __iter__(self):
        return (qualify(source, record) for source, adapter in self.adapters.items() for record in adapter.records())

    def __len__(self):
        return sum(len(adapter.records()) for adapter in self.adapters.values())

    def __getitem__(self, key):
        source, record = split(key)
        return self.adapters[source].records()[record]


class CompositeAdapter:
    def __init__(self, sources, *, adapters=None):
        self.adapters = adapters or {source: _adapter(source) for source in sorted(sources)}
        self.source = '+'.join(self.adapters)
        self._questions = {qualify(source, q): dict(row, source=source, source_question=q, kind=getattr(adapter, 'kind', 'record_label'))
                           for source, adapter in self.adapters.items() for q, row in adapter.questions().items()}
        self._pool = None

    def questions(self):
        return self._questions

    def records(self):
        return _Records(self.adapters)

    def iter_record_texts(self, records):
        for source, adapter in self.adapters.items():
            selected = [split(record)[1] for record in records if split(record)[0] == source]
            iterator = adapter.iter_record_texts(selected) if hasattr(adapter, 'iter_record_texts') else (
                (record, adapter.records()[record]['text']) for record in selected)
            for record, text in iterator:
                yield qualify(source, record), text

    def label_pool(self):
        if self._pool is None:
            self._pool = {(qualify(source, record), qualify(source, q)): value
                          for source, adapter in self.adapters.items() for (record, q), value in adapter.label_pool().items()}
        return self._pool

    def scorable(self, record, question):
        return (record, question) in self.label_pool()

    def benchmark_checker(self, record, question):
        source, original_record = split(record)
        question_source, original_question = split(question)
        if source != question_source:
            raise ValueError('cross-source benchmark answer')
        return self.adapters[source].benchmark_checker(original_record, original_question)

    def reference(self):
        return CompositeAdapter(list(self.adapters), adapters={s: a.reference() for s, a in self.adapters.items()})

    def visible_id(self, record):
        source, original = split(record)
        return source + '-' + self.adapters[source].visible_id(original)

    def encode(self, question, answer):
        source, original = split(question)
        return self.adapters[source].encode(original, answer)

    def decode(self, question, answer):
        source, original = split(question)
        return self.adapters[source].decode(original, answer)


def adapter_for(spec):
    if isinstance(spec, str):
        return CompositeAdapter(spec.split('+')) if '+' in spec else _adapter(spec)
    return CompositeAdapter(scenario_sources(spec)) if 'sources' in spec else _adapter(spec['source'])


def normalized_spec(spec):
    result = copy.deepcopy(spec)
    if 'sources' in result and isinstance(result['questions'], dict):
        sources = scenario_sources(result)
        result['source'] = '+'.join(sources)
        result['sources'] = sources
        result['questions'] = [qualify(source, question) for source in sources for question in result['questions'][source]]
    return result


def source_of(spec, question=None, record=None):
    from .adapters import canonical_source
    return split(question or record)[0] if 'sources' in spec else canonical_source(spec['source'])


def origin(spec, record, question):
    source = source_of(spec, question=question)
    if 'sources' in spec:
        record_source, record = split(record)
        question_source, question = split(question)
        if record_source != question_source:
            raise ValueError('cross-source label pair')
    return {'source': source, 'record': record, 'question': question}


def example_spec(sources):
    """Deterministic fake-client fixture; no balance-directed sampling or retries."""
    sources = sorted(sources)
    preferred = {'MAUD': ['DP041', 'DP068', 'DP110', 'DP114', 'DP086'],
                 'CUAD': ['change_of_control', 'exclusivity', 'termination_for_convenience']}
    questions = {source: preferred.get(source, list(get_adapter(source).questions())) for source in sources}
    # A large single draw supports sparse/rare classes without selecting by labels.
    sizes = {source: min(80 if source in ('B', 'PREC') else 32, len(get_adapter(source).records())) for source in sources}
    nodes, edges = [], []
    for source in sources:
        for template, kind, scoring, work in [('record_scope', 'construction_fact', 'prerequisite', 'scope'),
                ('record_question', 'original_label', 'label', 'coding'),
                ('question_count', 'derived_quantity', 'consistency-only', 'summary')]:
            nodes.append({'id': source + '_' + template, 'template': template, 'source': source,
                          'type': kind, 'provision': 'solver-derived' if template == 'question_count' else 'in-document',
                          'scoring': scoring, 'carrier': 'assigned_records', 'work_item': work})
        edges.extend([{'from': source + '_record_scope', 'to': source + '_record_question'},
                      {'from': source + '_record_question', 'to': source + '_question_count'}])
    return {'version': 5, 'id': 'combination-' + '-'.join(sources).lower(), 'sources': sources,
            'questions': questions, 'sample_size': 80, 'sample_sizes': sizes, 'category': 'sequential_dependency',
            'work_items': [{'id': 'scope', 'title': 'Review scope', 'order': 1, 'depends_on': []},
                           {'id': 'coding', 'title': 'Code records', 'order': 2, 'depends_on': ['scope']},
                           {'id': 'summary', 'title': 'Summarize coding', 'order': 3, 'depends_on': ['coding']}], 'documents': [],
            'dependency_graph': {'nodes': nodes, 'edges': edges}, 'thresholds': {'include_all': 1, 'ignore_revisions': 1},
            'instruction': 'Review all assigned source records. Reconcile the current notice, code each assigned question, '
                           'and summarize the first published option per question in /output/review.xlsx.',
            'design': 'A combined review with source-qualified identifiers and original categorical labels.'}


def is_benchmark(adapter, question):
    kind = getattr(adapter, 'kind', 'record_label')
    # The original categorical guard also accepts label-pool-only adapters.
    # Composite adapters declare the kind per question instead.
    if hasattr(adapter, 'questions'):
        kind = adapter.questions()[question].get('kind', kind)
    return kind == 'benchmark_item'
