"""Validate the designer contract without executing designer code."""
from __future__ import annotations
import json
from pathlib import Path
from .multisource import scenario_sources

TYPES = {'original_label', 'construction_fact', 'derived_quantity', 'deliverable_column'}
PROVISIONS = {'given', 'in-document', 'solver-derived', 'build-internal'}
SCORING = {'label', 'prerequisite', 'input', 'consistency-only', 'unscored'}


def safe_path(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or not value or '\\' in value:
        raise ValueError(f'unsafe relative path: {value!r}')
    return value


def validate_graph(graph, work_items=None, resolved=False):
    if not isinstance(graph, dict) or not isinstance(graph.get('nodes'), list) or not isinstance(graph.get('edges'), list):
        raise ValueError('dependency_graph requires nodes and edges lists')
    nodes = graph['nodes']; ids = [n['id'] for n in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate node id')
    known = set(ids); adjacency = {k: [] for k in ids}
    work_ids = {w['id'] for w in work_items} if work_items else None
    for n in nodes:
        kind = n.get('type', n.get('kind'))
        if kind not in TYPES or n.get('provision') not in PROVISIONS:
            raise ValueError(f'invalid node type/provision: {n["id"]}')
        if n.get('scoring', 'unscored') not in SCORING:
            raise ValueError('invalid scoring status')
        if kind in ('original_label', 'construction_fact') and not n.get('carrier', n.get('carrier_document')):
            raise ValueError('every fact requires a carrier document')
        if kind == 'derived_quantity' and (n.get('scoring') not in ('consistency-only', 'unscored') or 'gold' in n):
            raise ValueError('derived quantity cannot have independent gold')
        if kind == 'original_label' and n.get('scoring', 'label') != 'label':
            raise ValueError('original labels must be scored')
        if work_ids is not None and n.get('work_item') not in work_ids:
            raise ValueError('node references unknown work item')
        if resolved and kind == 'original_label' and not all(k in n for k in ('record', 'question', 'cell')):
            raise ValueError('resolved label requires record, question and cell')
    for e in graph['edges']:
        if e.get('from') not in known or e.get('to') not in known:
            raise ValueError('edge references unknown node')
        adjacency[e['from']].append(e['to'])
    visiting, visited = set(), set()
    def visit(k):
        if k in visiting:
            raise ValueError('dependency graph cycle')
        if k in visited:
            return
        visiting.add(k)
        for v in adjacency[k]:
            visit(v)
        visiting.remove(k); visited.add(k)
    for k in ids:
        visit(k)
    return graph


def validate_spec_structure(spec):
    if not isinstance(spec, dict) or spec.get('version', spec.get('schema_version')) != 5:
        raise ValueError('v5 spec required')
    for field in ('id', 'questions', 'sample_size', 'work_items', 'documents', 'dependency_graph', 'instruction', 'design', 'thresholds'):
        if field not in spec:
            raise ValueError(f'missing spec field: {field}')
    sources = scenario_sources(spec)
    if spec.get('category') not in ('parallel_coding', 'sequential_dependency'):
        raise ValueError('explicit task category required')
    if not isinstance(spec['sample_size'], int) or isinstance(spec['sample_size'], bool) or spec['sample_size'] < 2:
        raise ValueError('sample_size must be integer >= 2')
    question_sets = [spec['questions']]
    if 'sources' in spec:
        if not isinstance(spec['questions'], dict) or set(spec['questions']) != set(sources):
            raise ValueError('questions must map every canonical source to its question IDs')
        question_sets = list(spec['questions'].values())
        sizes = spec.get('sample_sizes', {})
        if not isinstance(sizes, dict) or not set(sizes) <= set(sources) or any(
                not isinstance(n, int) or isinstance(n, bool) or n < 2 for n in sizes.values()):
            raise ValueError('sample_sizes must map sources to integers >= 2')
    if any(not isinstance(qs, list) or not qs or any(not isinstance(q, str) for q in qs)
           or len(set(qs)) != len(qs) for qs in question_sets):
        raise ValueError('nonempty unique questions required')
    work = spec['work_items']
    if not 2 <= len(work) <= 4 or [w.get('order') for w in work] != list(range(1, len(work)+1)):
        raise ValueError('2-4 ordered work items required')
    seen = set()
    for w in work:
        if w['id'] in seen or not set(w.get('depends_on', [])) <= seen:
            raise ValueError('work dependencies must reference earlier items')
        seen.add(w['id'])
    docs = spec['documents']; dids = [d['id'] for d in docs]
    if len(set(dids)) != len(dids):
        raise ValueError('duplicate document id')
    for d in docs:
        safe_path(d['path'])
        if not isinstance(d.get('facts', {}), dict):
            raise ValueError('document facts must be a mapping')
    for key in ('include_all', 'ignore_revisions'):
        value = spec['thresholds'].get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f'predeclared {key} scope F1 bound must be within [0, 1]')
    validate_graph(spec['dependency_graph'], work)
    node_ids = {n['id'] for n in spec['dependency_graph']['nodes']}
    for w in work:
        if w.get('scope_from') and w['scope_from'] not in node_ids:
            raise ValueError('unknown scope prerequisite')
    return spec


def validate_spec(spec, *, adapter=None, screening_report=None):
    """Validate compile inputs; numeric screening is a report under slim policy."""
    validate_spec_structure(spec)
    # The compiler resolves questions against the adapter. Historical pool
    # screening must not introduce a seventh admission gate here.
    return spec


def load_spec(path):
    return validate_spec(json.loads(Path(path).read_text()))


COMPILER_TEMPLATES = ('record_scope', 'record_question', 'question_count')


def validate_compilable(spec):
    """Structural contract of the fixed compiler, checked before any model or build cost.

    Messages are structural only (no labels or records) and are returned to the designer verbatim.
    """
    nodes = spec['dependency_graph']['nodes']
    seen = [n.get('template') for n in nodes]
    bad = sorted({str(t) for t in seen if t not in set(COMPILER_TEMPLATES) | {'document_fact'}})
    if bad:
        raise ValueError('unsupported node template(s) ' + ', '.join(bad) + '; every node needs template '
                         'record_scope, record_question, question_count or document_fact')
    groups = [nodes]
    if 'sources' in spec:
        sources = scenario_sources(spec)
        if any(n.get('source') not in sources for n in nodes if n.get('template') in COMPILER_TEMPLATES):
            raise ValueError('each source template must declare one selected source')
        groups = [[n for n in nodes if n.get('source') == source] for source in sources]
    expected = set()
    for group in groups:
        for t in COMPILER_TEMPLATES:
            count = sum(n.get('template') == t for n in group)
            if (t == 'record_scope' and 'sources' in spec and count <= 1):
                continue
            if count != 1:
                raise ValueError(f'template {t} must appear on exactly one node per source (found {count})')
        current = {n['template']: n for n in group if n.get('template') in COMPILER_TEMPLATES}
        expected.add((current['record_question']['id'], current['question_count']['id']))
        if 'record_scope' in current:
            expected.add((current['record_scope']['id'], current['record_question']['id']))
    templates = {n['template']: n for n in nodes if n.get('template') in COMPILER_TEMPLATES}
    edges = {(e['from'], e['to']) for e in spec['dependency_graph']['edges']}
    if not expected <= edges:
        raise ValueError('edges must include record_scope -> record_question and record_question -> question_count')
    docs = {d['id']: d for d in spec['documents']}
    for node in nodes:
        if node.get('template') == 'document_fact':
            if node.get('carrier') not in docs or node.get('fact_key') not in docs[node['carrier']]['facts']:
                raise ValueError(f'document_fact node {node["id"]} must name a documents[].id as carrier and one '
                                 'of that document\'s facts keys as fact_key')
            if node.get('scoring') not in ('prerequisite', 'input'):
                raise ValueError(f'document_fact node {node["id"]} scoring must be prerequisite or input')
    for w in spec['work_items']:
        if w.get('scope_from') and w['scope_from'] not in {n['id'] for n in nodes if n.get('template') == 'record_scope'}:
            raise ValueError('scope_from must name the record_scope node id')
    return templates


def load_designer_output(directory):
    """Bind the four designer artifacts. The standalone files are authoritative."""
    directory = Path(directory)
    required = {'scenario_spec.json', 'dependency_graph.json', 'instruction.md', 'design.md'}
    present = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if present != required:
        raise ValueError('designer output must contain exactly the four declared artifacts')
    try:
        spec = json.loads((directory / 'scenario_spec.json').read_text())
        graph = json.loads((directory / 'dependency_graph.json').read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid JSON: {exc.msg} at line {exc.lineno}') from None
    # Embedded copies are optional; the standalone files win, so copies can never disagree.
    spec['dependency_graph'] = graph
    for key in ('instruction', 'design'):
        spec[key] = (directory / (key + '.md')).read_text()
    validate_compilable(spec)
    validate_spec(spec)
    return spec
