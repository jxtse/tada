"""Oolong and Semantic Query Engines items through source_adapters.iter_units.

Oolong public inputs are context_window_text and question ONLY: the alternate
context_window_text_with_labels, answer and task annotations stay private.
The original serialized answer is retained bijectively; the official scorer
parses it (github.com/abertsch72/oolong/src/eval/eval_helpers.py).
SQE public inputs are query.nlq and query.cols / cols_left / cols_right. Hidden
evaluation columns and gold result tables are read only to bind original query
answers. Its deterministic exact_match_accuracy and selection/join f1_score
come from experiment_runner/evaluation.py; semantic LLM scoring stays unscored.
Query YAML is specification metadata; all table records use the unified API.
"""
from functools import lru_cache
import hashlib
import fcntl
import json
from pathlib import Path
import yaml
from .benchmark_reader import DATA, UNIFIED, plan, units
from .base import REPO

SQE = 'semantic-query-engines-experiments'
OOLONG_SOURCE = 'https://github.com/abertsch72/oolong/blob/main/src/eval/eval_helpers.py'


def metadata(dataset):
    return {'status': 'available', 'reason': '',
            'checker_source': OOLONG_SOURCE if dataset.startswith('oolong-') else
            'semantic-query-engines-experiments/repo/experiment_runner/evaluation.py: DerivationEvaluator exact_match_accuracy, SelectionEvaluator/JoinEvaluator f1_score',
            'visible_fields': ['context_window_text', 'question'] if dataset.startswith('oolong-') else
            ['original nlq', 'query-declared input columns'],
            'hidden_fields': ['answer', 'context_window_text_with_labels', 'evaluation columns', 'ground_truth.csv'],
            'limitations': 'oolong-real: only the locally delivered toy_dnd subset; SQE semantic judge and unlabelled aggregation queries remain unscored.'}


def _text(row):
    return row['context_window_text'] + '\n\nQuestion: ' + row['question']


def _load_text(dataset, table, ordinal, identity):
    for i, unit in enumerate(units(dataset, table)):
        if i == ordinal:
            if unit['unit_id'] != identity:
                raise ValueError('Oolong source item identity changed')
            return _text(unit['raw_row'])
    raise ValueError('Oolong source item disappeared')


@lru_cache(maxsize=2)
def _oolong_index(dataset):
    # Store only lightweight answer/locator metadata. Expanded synth contexts
    # exceed 24GB; no corpus-wide text dictionary or source copy is created.
    descriptors = plan(dataset)
    identity = {'plan': descriptors,
                'source_stats': [(str(d['path']), Path(d['path']).stat().st_size, Path(d['path']).stat().st_mtime_ns) for d in descriptors],
                'reader': {str(p.relative_to(UNIFIED)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for name in ('source_adapters','tada_unified') for p in sorted((UNIFIED/name).glob('*.py'))},
                'adapter': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    digest = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache = REPO/'v5_runs/source_cache/oolong_index'
    cache.mkdir(parents=True,exist_ok=True)
    path = cache/(dataset+'-'+digest+'.json')
    with (cache/(dataset+'.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            envelope = json.loads(path.read_text())
            items = envelope['items']
            if envelope.get('identity') != json.loads(json.dumps(identity)) or hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest() != envelope['sha256']:
                raise ValueError('Private Oolong index cache content changed')
        else:
            items = []
            for desc in descriptors:
                for ordinal, unit in enumerate(units(dataset, desc['table_id'])):
                    row = unit['raw_row']
                    if not row.get('question') or not row.get('context_window_text') or row.get('answer') in (None, ''):
                        continue
                    checker = {'kind': 'oolong_synth', 'answer_type': row['answer_type']} if dataset == 'oolong-synth' else {'kind': 'oolong_real'}
                    items.append({'id': unit['unit_id'], 'text': '', 'text_bytes': len(_text(row).encode()),
                                  'text_sha256': hashlib.sha256(_text(row).encode()).hexdigest(),
                                  '_table': desc['table_id'], '_ordinal': ordinal,
                                  'answer': row['answer'], 'checker': checker, 'assets': []})
            envelope = {'identity':identity,'items':items,'sha256':hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()}
            partial=path.with_suffix('.partial')
            partial.write_text(json.dumps(envelope,ensure_ascii=False));partial.replace(path)
    for item in items:
        item['text_loader'] = lambda d=dataset,t=item['_table'],n=item['_ordinal'],u=item['id']: _load_text(d,t,n,u)
    return tuple(items)


def selected_texts(dataset, identities):
    """Read each selected source table once, retaining one original row at a time."""
    index = {item['id']: item for item in _oolong_index(dataset)}
    selected = {}
    for identity in identities:
        item = index[identity]
        selected.setdefault(item['_table'], {})[item['_ordinal']] = identity
    for table, ordinals in selected.items():
        last = max(ordinals)
        emitted = set()
        for ordinal, unit in enumerate(units(dataset, table, limit=last+1)):
            if ordinal in ordinals:
                identity = ordinals[ordinal]
                if unit['unit_id'] != identity:
                    raise ValueError('Oolong selected item identity changed')
                text = _text(unit['raw_row'])
                if hashlib.sha256(text.encode()).hexdigest() != index[identity]['text_sha256']:
                    raise ValueError('Oolong selected input changed after indexing')
                emitted.add(ordinal)
                yield identity, text
        if emitted != set(ordinals):
            raise ValueError('Oolong selected items disappeared')


def _public(row, columns):
    return {column: row[column] for column in columns}


def _sqe_items():
    base = DATA/'raw/analytics_benchmarks'/SQE/'repo'
    queries = yaml.safe_load((base/'experiment_runner/configs/queries.yaml').read_text())['queries']
    descriptors = {d['relative_path']: d for d in plan(SQE)}
    tables = {}
    def rows(path):
        if path not in descriptors:
            return None
        if path not in tables:
            tables[path] = [u['raw_row'] for u in units(SQE, descriptors[path]['table_id'])]
        return tables[path]
    for query in queries:
        qid = query['id']
        corpus = 'nba' if int(qid[1:]) < 34 else 'rotten_tomatoes'
        prefix = f'repo/datasets/{corpus}/quality_exps/'
        kind = query['class_name']
        # Aggregation entries have no delivered original answers/evaluator.
        if kind == 'aggregation':
            continue
        if kind in ('derivation', 'selection'):
            table = query.get('quality_table') or query.get('table')
            data = rows(prefix + table) if table else None
            if not data:
                continue
            cols = query['cols']
            gold_column = query['evaluation_cols'][-1] if kind == 'derivation' else query['filtering_col']
            if any(gold_column not in row or any(c not in row for c in cols) for row in data):
                continue
            public = [_public(row, cols) for row in data]
            if kind == 'derivation':
                # Original evaluation converts both result columns to strings.
                answer = [str(row[gold_column]) for row in data]
                checker = {'kind': 'sqe_derivation'}
                instruction = 'Return a JSON array of generated values in input row order.'
            else:
                key_cols = [c for c in query['evaluation_cols'] if c != gold_column]
                answer = [_public(row, key_cols) for row in data if str(row[gold_column]).lower() == 'true']
                if not answer:
                    continue  # Upstream F1 is zero for an empty gold set.
                checker = {'kind': 'sqe_set_f1', 'mode': 'selection',
                           'negative_rows': [_public(row,key_cols) for row in data if str(row[gold_column]).lower() == 'false']}
                instruction = 'Return a JSON array of matching rows with these keys: ' + ', '.join(key_cols)
            text = query['nlq'] + '\n' + instruction + '\nInputs:\n' + json.dumps(public, ensure_ascii=False)
            yield {'id': qid, 'text': text, 'answer': answer, 'checker': checker, 'assets': []}
        elif kind == 'join':
            left, right, gold = [rows(prefix + query[name]) for name in ('table_left', 'table_right', 'evaluation_table')]
            if left is None or right is None or gold is None or not gold:
                continue
            keys = [query['left_key'], query['right_key']]
            if any(any(key not in row for key in keys) for row in gold):
                continue
            public = {'left': [_public(row, query['cols_left']) for row in left],
                      'right': [_public(row, query['cols_right']) for row in right]}
            answer = [_public(row, keys) for row in gold]
            text = query['nlq'] + '\nReturn the matching row pairs as a JSON array with keys: ' + ', '.join(keys)
            text += '\nInputs:\n' + json.dumps(public, ensure_ascii=False)
            yield {'id': qid, 'text': text, 'answer': answer, 'checker': {'kind': 'sqe_set_f1','mode':'join'}, 'assets': []}


def iter_items(dataset):
    if dataset in ('oolong-synth', 'oolong-real'):
        yield from _oolong_index(dataset)
    elif dataset == SQE:
        yield from _sqe_items()
    else:
        raise ValueError('Unexpected dataset for Oolong/SQE helper')
