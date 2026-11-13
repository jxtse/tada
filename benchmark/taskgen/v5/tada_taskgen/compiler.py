"""Deterministic adapter-driven Harbor compiler; no task-authored scoring code."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import tomllib
import copy
import hashlib
import io
import json
import random
import re
import shutil
import sys
import zipfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from .multisource import adapter_for, normalized_spec, scenario_sources, source_of, origin, is_benchmark

REPO = Path(__file__).resolve().parents[2]
SCORING_POLICY = tomllib.loads((REPO / 'v5/config.toml').read_text())['scoring']
WEIGHTS = {key: SCORING_POLICY[key] for key in ('labels', 'scope', 'facts', 'consistency')}
MINIMUM_SCORABLE_QUESTIONS = 3
MINIMUM_RETAINED_QUESTIONS = 5
SAMPLING_POLICY = {'minimum_scorable_questions': MINIMUM_SCORABLE_QUESTIONS,
                   'minimum_retained_questions': MINIMUM_RETAINED_QUESTIONS}
REVIEW_MEMO = 'review_memo.md'


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def write_workbook(sheets, specs, path, *, canary=None):
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.created = wb.properties.modified = datetime(2000, 1, 1)
    if canary:
        wb.properties.description = 'Benchmark canary: ' + canary
    for name, keys, columns in specs:
        ws = wb.create_sheet(name)
        if canary:
            ws.oddHeader.center.text = 'Benchmark canary: ' + canary
        ws.append(columns)
        for row in sheets.get(name, []):
            ws.append([row.get(c) for c in columns])
    stream = io.BytesIO()
    wb.save(stream)
    # openpyxl resets modified on save; normalize OOXML plus ZIP timestamps.
    with zipfile.ZipFile(stream) as src, zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as dst:
        for name in sorted(src.namelist()):
            data = src.read(name)
            if name == 'docProps/core.xml':
                data = re.sub(rb'(<dcterms:modified[^>]*>)[^<]+', rb'\g<1>2000-01-01T00:00:00Z', data)
            z = zipfile.ZipInfo(name, (2000, 1, 1, 0, 0, 0))
            z.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(z, data)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def asset_archive(asset):
    """Seek a declared ZIP member without extracting an archive to disk."""
    with zipfile.ZipFile(asset['source_path']) as archive:
        if asset.get('nested_member'):
            with archive.open(asset['nested_member']) as nested_stream, zipfile.ZipFile(nested_stream) as nested:
                yield nested
        else:
            yield archive


def asset_bytes(asset):
    if asset.get('member'):
        stat = Path(asset['source_path']).stat()
        return _archive_asset_bytes(str(asset['source_path']), asset.get('nested_member'), asset['member'], stat.st_size, stat.st_mtime_ns)
    return Path(asset['source_path']).stat().st_size


@lru_cache(maxsize=4096)
def _archive_asset_bytes(source, nested, member, size, mtime):
    with asset_archive({'source_path': source, 'nested_member': nested}) as archive:
        info = archive.getinfo(member)
        if info.is_dir():
            raise ValueError('benchmark asset member is a directory')
        return info.file_size


def item_checker(adapter, record, question):
    checker = copy.deepcopy(adapter.benchmark_checker(record, question))
    if checker.get('database'):
        asset_path = checker['database']
        if asset_path not in {a['path'] for a in adapter.records()[record].get('assets', [])}:
            raise ValueError('checker database is not a declared public input')
        checker['database'] = str(Path('environment/public/inputs') / adapter.visible_id(record) / asset_path)
    return checker


def roles(spec, source=None):
    return {n['template']: n for n in spec['dependency_graph']['nodes']
            if n.get('template') in ('record_scope', 'record_question', 'question_count')
            and (source is None or 'sources' not in spec or n.get('source') == source)}


def sheet_specs(spec):
    names = spec.get('sheets', {})
    result = [(names.get('scope', 'Scope'), ['Record_ID'], ['Record_ID', 'Included']),
            (names.get('coding', 'Coding'), ['Record_ID', 'Question'], ['Record_ID', 'Question', 'Answer']),
            (names.get('summary', 'Summary'), ['Question'], ['Question', 'Positive_Count'])]
    if any(n.get('template') == 'document_fact' and n.get('scoring') == 'prerequisite' for n in spec['dependency_graph']['nodes']):
        result.append((names.get('facts', 'Fact_Checks'), ['Fact_ID'], ['Fact_ID', 'Value']))
    return result


def compute_gold(sp, label_fn=None, *, adapter=None):
    """Compatibility hook consumed by the unchanged label_flip_check.py."""
    adapter = adapter if adapter is not None else adapter_for(sp['spec'])
    specs = sheet_specs(sp['spec'])
    scope_name, coding_name, summary_name = [x[0] for x in specs[:3]]
    sheets = {name: [] for name, _, _ in specs}
    labels, cells = {}, []
    pool = adapter.label_pool()
    for record in sp['records']:
        rid, vid = record['record_id'], record['visible_id']
        if vid in sp['scope_facts']:
            sheets[scope_name].append({'Record_ID': vid, 'Included': sp['scope_facts'][vid]})
        for q in record['questions']:
            result = label_fn(rid, q) if label_fn else pool[(rid, q)]
            if result is None:
                raise ValueError(f'unscorable label: {rid}:{q}')
            answer = result[0]
            labels[f'{rid}:{q}'] = answer
            sheets[coding_name].append({'Record_ID': vid, 'Question': q, 'Answer': adapter.encode(q, answer)})
            cells.append({'id': f'label:{rid}:{q}', 'sheet': coding_name, 'key': [vid, q], 'column': 'Answer',
                          'record': rid, 'question': q, 'answer': answer,
                          'source': source_of(sp['spec'], question=q), 'origin': origin(sp['spec'], rid, q),
                          'encoding': ([[answer, adapter.encode(q, answer)]] if is_benchmark(adapter, q) else
                                       [[a, adapter.encode(q, a)] for a in adapter.questions()[q]['options']]),
                          **({'kind': 'benchmark_item', 'checker': item_checker(adapter, rid, q)} if is_benchmark(adapter, q) else {}),
                          'work_item': roles(sp['spec'], source_of(sp['spec'], question=q))['record_question']['work_item']})
    for q in sp['spec']['questions']:
        option = None if is_benchmark(adapter, q) else adapter.questions()[q]['options'][0]
        sheets[summary_name].append({'Question': q, 'Positive_Count': sum(
            is_benchmark(adapter, q) or c['answer'] == option for c in cells if c['question'] == q)})
    for node in sp['spec']['dependency_graph']['nodes']:
        if node.get('template') == 'document_fact' and node['scoring'] == 'prerequisite':
            doc = next(d for d in sp['spec']['documents'] if d['id'] == node['carrier'])
            sheets[specs[3][0]].append({'Fact_ID': node['id'], 'Value': doc['facts'][node['fact_key']]})
    return sheets, {'labels': labels, 'cells': cells}


from .compile_errors import CompilationFailure


class InstanceCompilationError(CompilationFailure):
    """An invalid sampled instance is a failed seed, never a resampling request."""


def class_counts(cells):
    counts = defaultdict(Counter)
    for cell in cells:
        counts[cell['question']][str(cell['answer'])] += 1
    return {q: dict(sorted(values.items())) for q, values in sorted(counts.items())}


def guard_instance(sp):
    """Check authoritative original classes, including during synthetic label shuffles.

    Permutation builds are leakage diagnostics; they keep the original sampled
    instance's admission decision, while emitting shuffled labels only as a control.
    """
    adapter = adapter_for(sp['spec']).reference()
    pool = adapter.label_pool()
    cells = [{'question': q, 'answer': pool[(r['record_id'], q)][0]}
             for r in sp['records'] for q in r['questions']]
    counts = class_counts(cells)
    invalid = {q: counts.get(q, {}) for q in sp['spec']['questions'] if not is_benchmark(adapter, q)
               and ( len(counts.get(q, {})) < SCORING_POLICY['instance_min_classes']
               or min(counts[q].values(), default=0) < SCORING_POLICY['instance_min_class_cells'])}
    minimum = sp.get('sampling', {}).get('minimum_retained_questions', MINIMUM_RETAINED_QUESTIONS)
    if invalid or len(sp['spec']['questions']) < minimum:
        raise InstanceCompilationError('instance class guard failed: each question needs >= 2 present classes, each with >= 2 cells',
                                       {'class_counts': counts, 'invalid_questions': invalid,
                                        'retained_question_count': len(sp['spec']['questions']),
                                        'minimum_retained_questions': minimum, 'seed': sp['seed']},
                                       code='class_guard_failed', context={'question_ids': sorted(invalid),
                                        'retained_count': len(sp['spec']['questions']),
                                        'minimum_required': minimum})
    return {'pass': True, 'class_counts': counts, 'seed': sp['seed']}


def resolve(spec, seed):
    requested = copy.deepcopy(spec)
    spec = normalized_spec(spec)
    adapter = adapter_for(spec)
    questions = spec['questions']
    if not set(questions) <= set(adapter.questions()):
        raise ValueError('unknown source question')
    legacy = 'sources' not in spec and spec['source'] in ('MAUD', 'CUAD', 'C2')
    minimum_scorable = MINIMUM_SCORABLE_QUESTIONS if legacy else 1
    minimum_retained = MINIMUM_RETAINED_QUESTIONS if legacy else len(scenario_sources(spec))
    rng = random.Random(seed)
    candidates = sorted(adapter.records())
    available = {r: [q for q in questions if adapter.scorable(r, q)] for r in candidates}
    eligible = [r for r in candidates if len(available[r]) >= minimum_scorable]
    cap = tomllib.loads((REPO / 'v5/config.toml').read_text()).get('compiler', {}).get('workspace_max_bytes', 2 * 1024 ** 3)
    if type(cap) is not int or cap <= 0:
        raise ValueError('workspace_max_bytes must be a positive integer')
    sizes = {}
    if any(is_benchmark(adapter, q) for q in questions):
        for record in eligible:
            public_record = adapter.records()[record]
            text_size = public_record.get('text_bytes')
            if text_size is None:
                text_size = len(public_record['text'].encode())
            sizes[record] = text_size + sum(asset_bytes(a) for a in public_record.get('assets', []))
    oversized = [record for record, size in sizes.items() if size > cap]
    eligible = [r for r in eligible if r not in set(oversized)]
    selected = []
    # Independent fixed-size draws by source use availability only, never labels.
    groups = scenario_sources(spec)
    for source in groups:
        pool_records = [r for r in eligible if 'sources' not in spec or source_of(spec, record=r) == source]
        size = spec.get('sample_sizes', {}).get(source, spec['sample_size'])
        if len(pool_records) < size:
            raise InstanceCompilationError(
                f'need {size} records scorable for at least {minimum_scorable} questions; only {len(pool_records)} eligible',
                {'source': source, 'candidate_count': len(pool_records), 'minimum_scorable_questions': minimum_scorable,
                 'sample_size': size, 'seed': seed, 'oversized_items_dropped': len(oversized), 'workspace_max_bytes': cap})
        selected.extend(rng.sample(pool_records, size))
    if sum(sizes[r] if r in sizes else len(adapter.records()[r]['text'].encode()) for r in selected) > cap:
        raise InstanceCompilationError('sampled inputs exceed the workspace byte cap', {'max_bytes': cap, 'dropped_oversized': len(oversized)})
    records = [{'record_id': r, 'visible_id': adapter.visible_id(r),
                'visible_file': f'agreements/{adapter.visible_id(r)}.txt', 'questions': available[r]} for r in selected]
    pool = adapter.reference().label_pool()
    counts = class_counts([{'question': q, 'answer': pool[(r['record_id'], q)][0]}
                           for r in records for q in r['questions']])
    dropped = {q: {'class_counts': counts.get(q, {}), 'reason': 'requires >= 2 present classes, each with >= 2 cells'}
               for q in questions if not is_benchmark(adapter, q) and (len(counts.get(q, {})) < SCORING_POLICY['instance_min_classes']
               or min(counts.get(q, {}).values(), default=0) < SCORING_POLICY['instance_min_class_cells'])}
    retained = [q for q in questions if q not in dropped]
    sampling = {'minimum_scorable_questions': minimum_scorable,
                'minimum_retained_questions': minimum_retained,
                'candidate_count': len(eligible), 'requested_questions': list(questions),
                'retained_questions': retained, 'dropped_questions': dropped,
                'class_counts_before_filter': counts, 'oversized_items_dropped': len(oversized), 'workspace_max_bytes': cap}
    missing_sources = [source for source in groups if not any(source_of(spec, question=q) == source for q in retained)]
    if len(retained) < minimum_retained or missing_sources:
        raise InstanceCompilationError(
            f'instance retains {len(retained)} questions after class guard; need at least {minimum_retained} and one per source',
            {'seed': seed, 'sampling': sampling, 'missing_sources': missing_sources,
             'sampled_records': [r['record_id'] for r in records]}, code='too_few_retained_questions',
            context={'question_ids': sorted(dropped), 'retained_count': len(retained), 'minimum_required': minimum_retained})
    for record in records:
        record['questions'] = [q for q in record['questions'] if q in retained]
    instance_spec = copy.deepcopy(spec)
    instance_spec['questions'] = retained
    stride = spec.get('parameters', {}).get('scope_exclude_every', 4)
    if not isinstance(stride, int) or stride < 2:
        raise ValueError('scope_exclude_every must be >= 2')
    scoped = [r for r in records if 'sources' not in spec
              or 'record_scope' in roles(spec, source_of(spec, record=r['record_id']))]
    scope = {r['visible_id']: ('No' if i % stride == 0 else 'Yes') for i, r in enumerate(scoped)}
    old = {r['visible_id']: 'Yes' for r in scoped}
    return {'seed': seed, 'spec': instance_spec, 'requested_spec': requested,
            'sampling': sampling, 'records': records, 'scope_facts': scope, 'prior_scope_facts': old}


def public_scoring_policy(spec):
    """Describe fixed scoring from the public schema, without reading gold."""
    specs = sheet_specs(spec)
    scope, coding, summary = [row[0] for row in specs[:3]]
    field = lambda sheet, column: {'sheet': sheet, 'column': column}
    facts = [field(specs[3][0], 'Value')] if len(specs) == 4 else []
    rules = {
        'labels': ([field(coding, 'Answer')],
            'Decode every listed pair, regardless of scope status. For each question, average the '
            'fraction correct within each answer class present in the assigned pairs (balanced accuracy), '
            'after multiplying each cell contribution by its sheet precision. With k equal to the full '
            'number of published options, the question score is max(0, (balanced_accuracy - 1/k) / '
            '(1 - 1/k)). Average question scores equally; k does not shrink when a class is absent.'),
        'scope': ([field(scope, 'Included')] if 'record_scope' in roles(spec) else [],
            'Treat Yes as selected. Precision is correct selections / all selections; recall is correct '
            'selections / all records required by the current notice. Score is their harmonic mean '
            'F1 = 2*precision*recall/(precision+recall), multiplied by scope-sheet precision. '
            'Zero precision and recall give zero; two empty sets give one.'),
        'facts': (facts,
            'Average exact correctness across the document-fact Value cells, each multiplied by its '
            'sheet precision. Values must match the stated value and value type. Scope status is scored '
            'only by the scope component. If a task has no document facts, this component is dropped '
            'and the remaining weights are renormalised to sum to one.'),
        'consistency': ([field(summary, 'Positive_Count')],
            'For each question, count the first listed original option in your own decoded Answer '
            'cells across every listed pair, including records marked No in the scope sheet. '
            'A count earns credit only if it matches your coding and every underlying label for '
            'that question is correct. Multiply by summary-sheet precision and average questions. '
            'Scope errors do not affect label or summary credit.'),
    }
    return {'weights': dict(WEIGHTS), 'components': {
        name: {'weight': WEIGHTS[name], 'inputs': inputs, 'rule': rule}
        for name, (inputs, rule) in rules.items()},
        'summary_population': 'Every listed (Record_ID, Question) pair, regardless of scope status.',
        'extra_rows': 'Extra rows reduce precision: each sheet uses distinct matched keys / '
            '(distinct matched keys + extra rows). Unknown or malformed keys, duplicate occurrences '
            'and data above the header count as extra rows; do not add unassigned pairs.',
        'missing_answers': 'Missing, invalid or conflicting duplicate answers earn no accuracy credit; '
            'missing required answers remain in the accuracy denominators. The total reward is the '
            'weighted sum of the components present (renormalised if a component has no items). Scope errors reduce scope credit only.'}


def review_memo(sp):
    """Publish assigned pairs as business scope, with no answers or class data."""
    lines = ['# Agreement review memo', '',
             'The checklist below defines the assigned questions for each agreement. '
             'The coding register covers every listed (Record_ID, Question) pair regardless of scope status; '
             'enter one coding row even for records marked No in the scope sheet. '
             'Unlisted pairs are outside this register.', '',
             'Record each agreement\'s inclusion status separately in the scope sheet using the current review notice. '
             'Summaries count the first listed original option in your own coding across all listed pairs, '
             'including out-of-scope records; do not filter summaries by scope.', '',
             f"Scoring: labels {WEIGHTS['labels']:.0%}, scope {WEIGHTS['scope']:.0%}, facts {WEIGHTS['facts']:.0%}, "
             f"consistency {WEIGHTS['consistency']:.0%}. Labels use per-question chance-corrected balanced accuracy; "
             'scope uses precision/recall F1. Facts score the document-fact Value cells only. '
             'Extra rows reduce sheet precision; missing answers lose accuracy. Summary consistency uses your '
             'own coded counts and requires all underlying labels for that question to be correct. Scope errors '
             'reduce scope credit only. Full component-to-column mappings and formulas are in '
             'coding_conventions.json under scoring_policy.', '',
             '| Record_ID | Assigned questions |', '|---|---|']
    if 'sources' in sp['spec']:
        lines[2:2] = ['Only the Record_ID values in coding_conventions.json scope_records require scope rows. '
                      'If scope_records is empty, leave the scope sheet empty.', '']
    for record in sp['records']:
        lines.append(f'| {record["visible_id"]} | {", ".join(record["questions"]) or "None"} |')
    return '\n'.join(lines) + '\n'


def carrier_failure(document, expected):
    """Keep expected/extracted values private; expose only failed public fact keys."""
    from .roundtrip import canonical
    attempts = document.get('attempts', [])
    extracted = attempts[-1].get('extracted') if attempts else None
    keys = [key for key, value in expected.items() if not isinstance(extracted, dict)
            or key not in extracted or canonical(extracted[key]) != canonical(value)]
    # Extra extracted fields also fail exact round trip; their untrusted names
    # must not reach feedback. Name the expected keys needing re-extraction.
    if not keys:
        keys = list(expected)
    return CompilationFailure('required fact carrier dropped: ' + document['id'],
        code='carrier_roundtrip_failed', context={'document_id': document['id'], 'fact_keys': sorted(keys)})


def compile_task(spec, seed, out, *, writer=None, extractor=None, doc_client=None, document_archive=None, screening_report=None, client='fake', model_transport=None):
    from tada_taskgen.canary import GUID, text_header, output_contract
    from tada_taskgen.docgen import generate_documents
    from tada_taskgen.roundtrip import canonical
    if client not in ('fake', 'real'):
        raise ValueError('client must be fake or real')
    if client == 'real' and any(value is not None for value in (writer, extractor, doc_client)):
        raise ValueError('Use model_transport to inject a real client transport, not mixed clients')
    if isinstance(spec, (str, Path)):
        spec_path = Path(spec)
        if spec_path.name == 'scenario_spec.json' and (spec_path.parent / 'dependency_graph.json').exists():
            from tada_taskgen.schema import load_designer_output
            spec = load_designer_output(spec_path.parent)
        else:
            spec = json.loads(spec_path.read_text())
    spec = copy.deepcopy(spec)
    from tada_taskgen.schema import validate_spec
    validate_spec(spec, screening_report=screening_report)
    from tada_taskgen.schema import validate_compilable
    templates = validate_compilable(spec)
    expected_edges = set()
    for source in scenario_sources(spec):
        source_roles = roles(spec, source)
        expected_edges.add((source_roles['record_question']['id'], source_roles['question_count']['id']))
        if 'record_scope' in source_roles:
            expected_edges.add((source_roles['record_scope']['id'], source_roles['record_question']['id']))
    sp = resolve(spec, seed)
    instance_guard = guard_instance(sp)
    requested_spec = spec
    spec = sp['spec']
    out = Path(out)
    for folder in ['build', 'environment/public', 'tests', 'solution']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    public = out / 'environment/public'
    adapter = adapter_for(spec)
    specs = sheet_specs(spec)
    scope_name, coding_name, summary_name = [x[0] for x in specs[:3]]
    sheets, info = compute_gold(sp)
    documents = copy.deepcopy(spec.get('documents', []))
    documents += [{'id': 'scope_revision', 'path': 'correspondence/current_review_notice.txt', 'kind': 'notice',
                   'facts': sp['scope_facts']},
                  {'id': 'scope_prior', 'path': 'correspondence/prior_review_notice.txt', 'kind': 'email',
                   'facts': sp['prior_scope_facts']}]
    # Successful real builds are immutable model artifacts on recompilation. An
    # archived report can be reused without credentials, including controller copies.
    saved_archive = out / 'tests/roundtrip_report.json'
    if client == 'real' and document_archive is None and saved_archive.exists():
        document_archive = json.loads(saved_archive.read_text())
    kwargs = {}
    if writer is not None or doc_client is not None:
        kwargs['writer'] = writer or doc_client
    if extractor is not None:
        kwargs['extractor'] = extractor
    if document_archive is not None:
        report = copy.deepcopy(document_archive)
        archived = {d['id']: d for d in report['documents']}
        if client == 'real' and (report.get('fake') is not False or any(d.get('fake') is not False for d in report['documents'])):
            raise ValueError('Real compilation cannot reuse fake document artifacts')
        if (set(archived) != {d['id'] for d in documents} or
                any(d['id'] not in archived or canonical(archived[d['id']]['facts']) != canonical(d['facts'])
                    or archived[d['id']]['path'] != d['path'] or archived[d['id']]['kind'] != d['kind']
                    for d in documents)):
            raise ValueError('archived documents do not match this exact scenario; regenerate explicitly')
    else:
        if client == 'real':
            from tada_taskgen.clients import OpenAICompatibleClient
            kwargs = {role: OpenAICompatibleClient.from_config(role, out, transport=model_transport)
                      for role in ('writer', 'extractor')}
        report = generate_documents(documents, **kwargs)
    doc_by_id = {d['id']: d for d in report['documents']}
    dump(out / 'tests/roundtrip_report.json', report)
    if doc_by_id.get('scope_revision', {}).get('status') != 'verified':
        raise carrier_failure(doc_by_id.get('scope_revision', {'id': 'scope_revision'}), {})
    for node in spec['dependency_graph']['nodes']:
        if node.get('template') == 'document_fact':
            doc = doc_by_id[node['carrier']]
            if doc['status'] != 'verified':
                expected = next(d['facts'] for d in documents if d['id'] == node['carrier'])
                raise carrier_failure(doc, expected)
    # A rebuild cannot retain visible documents from another seed.
    for p in public.iterdir():
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    (public / REVIEW_MEMO).write_text(text_header(review_memo(sp)))
    for doc in report['documents']:
        if doc['status'] != 'verified':
            continue
        dest = public / doc['path']
        if not dest.resolve().is_relative_to(public.resolve()):
            raise ValueError('document path escapes public workspace')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text_header(doc['text']))
    selected_records = {row['record_id']: row for row in sp['records']}
    text_items = adapter.iter_record_texts(list(selected_records)) if hasattr(adapter, 'iter_record_texts') else (
        (record, adapter.records()[record]['text']) for record in selected_records)
    written_records = set()
    for record, text in text_items:
        if record not in selected_records or record in written_records:
            raise ValueError('Unexpected or duplicate selected source text')
        written_records.add(record)
        path = public / selected_records[record]['visible_file']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text_header(text))
    if written_records != set(selected_records):
        raise ValueError('Missing selected source text')
    assets = []
    for record in sp['records']:
        for asset in adapter.records()[record['record_id']].get('assets', []):
            relative = Path('inputs') / record['visible_id'] / asset['path']
            destination = (public / relative).resolve()
            if not destination.is_relative_to(public.resolve()) or destination.exists() or Path(asset['path']).is_absolute():
                raise ValueError('unsafe or duplicate benchmark asset path')
            source_path = Path(asset['source_path']).resolve()
            if not source_path.is_file():
                raise ValueError('benchmark input file is unavailable')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if asset.get('member'):
                with asset_archive(asset) as archive, archive.open(asset['member']) as member, destination.open('wb') as target:
                    shutil.copyfileobj(member, target)
            else:
                shutil.copyfile(source_path, destination)
            assets.append({'path': str(relative), 'record': record['visible_id'], 'bytes': destination.stat().st_size,
                           'sha256': file_sha256(destination)})
    if assets:
        dump(public / 'input_assets.json', {'canary': GUID, 'assets': assets})
    encoding = {q: {'question': adapter.questions()[q]['question'],
                    'source': source_of(spec, question=q),
                    'source_question': q.split('::', 1)[-1] if 'sources' in spec else q,
                    **({'provision': adapter.questions()[q]['provision']} if 'provision' in adapter.questions()[q] else {}),
                    'coding_conventions': adapter.questions()[q]['coding_conventions'],
                    'encoding': [[a, adapter.encode(q, a)] for a in adapter.questions()[q]['options']],
                    'kind': 'benchmark_item' if is_benchmark(adapter, q) else 'record_label',
                    'answer_format': 'JSON value; deterministic original item checker' if is_benchmark(adapter, q) else 'published codebook',
                    'summary_option': None if is_benchmark(adapter, q) else adapter.questions()[q]['options'][0]} for q in spec['questions']}
    public_contract = {'_canary': GUID, 'deliverable_path': '/output/review.xlsx', 'sheets': specs, 'questions': encoding,
                       'applicability_memo': REVIEW_MEMO,
                       'records': [{'visible_id': r['visible_id'], 'visible_file': r['visible_file'], 'questions': r['questions']} for r in sp['records']],
                       'scope_records': list(sp['scope_facts']),
                       'scope_documents': {'current': doc_by_id['scope_revision']['path'], 'prior': doc_by_id['scope_prior']['path']},
                       'source': 'C2' if spec['source'] == 'CUAD' else spec['source'], 'scope_rule': 'Use the current notice. Included means in the review scope.',
                       'coding_rule': 'Code every listed pair regardless of scope status, including records marked No in the scope sheet.',
                       'summary_rule': 'Positive_Count counts the first listed original answer option in your own coding for each question across every listed pair, regardless of scope status.',
                       'scoring_policy': public_scoring_policy(spec)}
    if any(is_benchmark(adapter, q) for q in spec['questions']):
        public_contract['summary_rule'] += ' For benchmark items, count nonempty answers instead; each item must be correct for consistency credit.'
        public_contract['scoring_policy']['components']['labels']['rule'] += (' Benchmark item questions use their original deterministic checker and average item scores, without categorical chance correction. All question scores are averaged equally.')
        public_contract['assets_manifest'] = 'input_assets.json' if assets else None
    public_contract['extra_facts'] = [{'id': n['id'], 'fact_key': n['fact_key'],
        'visible_file': doc_by_id[n['carrier']]['path']} for n in spec['dependency_graph']['nodes']
        if n.get('template') == 'document_fact' and n['scoring'] == 'prerequisite']
    dump(public / 'coding_conventions.json', public_contract)
    write_workbook({}, specs, public / 'review_template.xlsx', canary=GUID)
    gold = {'schema_version': 5, 'source': spec['source'], 'sources': scenario_sources(spec), 'qualified_labels': 'sources' in spec, 'seed': seed,
            'label_aggregation': SCORING_POLICY['label_aggregation'],
            'label_class_counts': class_counts(info['cells']), 'weights': dict(WEIGHTS),
            'sheets': {name: {'key_columns': keys, 'columns': columns,
                       'rows': {'|'.join(str(r[k]) for k in keys): r for r in copy.deepcopy(sheets[name])}} for name, keys, columns in specs},
            'cells': info['cells'], 'facts': [], 'consistency': [], 'scopes': []}
    graph = {'nodes': [], 'edges': []}
    provenance = {'rows': {}}
    record_by_id = {r['record_id']: r for r in sp['records']}
    scope_doc = doc_by_id['scope_revision']
    scenario_dep = {'kind': 'scenario', 'visible_file': scope_doc['path'], 'quote': scope_doc['text']}
    for r in sp['records']:
        vid = r['visible_id']
        if vid not in sp['scope_facts']:
            continue
        source = source_of(spec, record=r['record_id'])
        fact = {'id': f'scope:{vid}', 'sheet': scope_name, 'key': [vid], 'column': 'Included',
                'value': sp['scope_facts'][vid], 'work_item': roles(spec, source)['record_scope']['work_item'],
                'source': source,
                'counted_in': 'scope'}
        gold['facts'].append(fact)
        graph['nodes'].append({'id': fact['id'], 'type': 'construction_fact', 'provision': 'in-document',
                               'scoring': 'prerequisite', 'carrier': scope_doc['path'], 'work_item': fact['work_item'],
                               'value': fact['value'], 'source': source, 'cell': {k: fact[k] for k in ['sheet', 'key', 'column']}})
        provenance['rows'][f'{scope_name}|{vid}'] = {'visible_file': scope_doc['path'], 'source_ids': [], 'dependencies': [scenario_dep]}
    for cell in info['cells']:
        r = record_by_id[cell['record']]
        dep = {'source_id': f'{cell["record"]}:{cell["question"]}', 'label': cell['answer'], 'visible_file': r['visible_file']}
        graph['nodes'].append({'id': cell['id'], 'type': 'original_label', 'provision': 'in-document', 'scoring': 'label',
                               'carrier': r['visible_file'], 'work_item': cell['work_item'], 'record': cell['record'],
                               'question': cell['question'], 'source': cell['source'], 'origin': cell['origin'], 'cell': {k: cell[k] for k in ['sheet', 'key', 'column']}})
        if r['visible_id'] in sp['scope_facts']:
            graph['edges'].append({'from': f'scope:{r["visible_id"]}', 'to': cell['id']})
        provenance['rows']['|'.join([cell['sheet'], *cell['key']])] = {'visible_file': r['visible_file'],
                                                                     'source_ids': [dep['source_id']], 'dependencies': [dep]}
    for q in spec['questions']:
        cells = [c for c in info['cells'] if c['question'] == q]
        check = {'id': f'summary:{q}', 'sheet': summary_name, 'key': [q], 'column': 'Positive_Count',
                 'operation': 'count_answered' if is_benchmark(adapter, q) else 'count_option',
                 'option': None if is_benchmark(adapter, q) else adapter.questions()[q]['options'][0],
                 'dependencies': [c['id'] for c in cells], 'source': source_of(spec, question=q),
                 'work_item': roles(spec, source_of(spec, question=q))['question_count']['work_item']}
        gold['consistency'].append(check)
        graph['nodes'].append({'id': check['id'], 'type': 'derived_quantity', 'provision': 'solver-derived',
                               'scoring': 'consistency-only', 'dependencies': check['dependencies'],
                               'operation': check['operation'], 'work_item': check['work_item'],
                               'cell': {k: check[k] for k in ['sheet', 'key', 'column']}})
        graph['edges'].extend({'from': c['id'], 'to': check['id']} for c in cells)
        deps = [provenance['rows']['|'.join([c['sheet'], *c['key']])]['dependencies'][0] for c in cells]
        provenance['rows'][f'{summary_name}|{q}'] = {'visible_file': deps[0]['visible_file'],
                                                  'source_ids': [d['source_id'] for d in deps], 'dependencies': deps}
    for node in spec['dependency_graph']['nodes']:
        if node.get('template') != 'document_fact':
            continue
        doc = doc_by_id[node['carrier']]
        expanded = {**node, 'carrier': doc['path'], 'value': doc['facts'][node['fact_key']]}
        if node['scoring'] == 'prerequisite':
            fact = {'id': node['id'], 'sheet': specs[3][0], 'key': [node['id']], 'column': 'Value',
                    'value': expanded['value'], 'work_item': node['work_item']}
            gold['facts'].append(fact)
            expanded['cell'] = {k: fact[k] for k in ['sheet', 'key', 'column']}
            provenance['rows'][f'{specs[3][0]}|{node["id"]}'] = {'visible_file': doc['path'], 'source_ids': [],
                'dependencies': [{'kind': 'scenario', 'visible_file': doc['path'], 'quote': doc['text']}]}
        graph['nodes'].append(expanded)
    # Expand every declared dependency edge, retaining relationships to additional facts.
    expansion = {}
    for source in scenario_sources(spec):
        for template, node in roles(spec, source).items():
            population = {'record_scope': gold['facts'], 'record_question': gold['cells'],
                          'question_count': gold['consistency']}[template]
            expansion[node['id']] = [item['id'] for item in population if item.get('source') == source]
    expansion.update({n['id']: [n['id']] for n in spec['dependency_graph']['nodes'] if n.get('template') == 'document_fact'})
    for edge in spec['dependency_graph']['edges']:
        if (edge['from'], edge['to']) in expected_edges:
            continue
        graph['edges'].extend({'from': a, 'to': b} for a in expansion[edge['from']] for b in expansion[edge['to']])
    gold['scopes'] = ([{'work_item': templates['record_question']['work_item'], 'sheet': scope_name, 'column': 'Included',
                       'expected_keys': [[vid] for vid, status in sp['scope_facts'].items() if status == 'Yes']}]
                      if sp['scope_facts'] else [])
    # Derived outputs in reference are examples computed from labels, never independent gold.
    for row in gold['sheets'][summary_name]['rows'].values():
        row.pop('Positive_Count', None)
    for name, keys, columns in specs:
        for column in columns:
            graph['nodes'].append({'id': f'column:{name}:{column}', 'type': 'deliverable_column',
                                   'provision': 'given', 'scoring': 'unscored', 'sheet': name, 'column': column,
                                   'work_item': templates.get(['record_scope', 'record_question', 'question_count'][[x[0] for x in specs].index(name)], templates['record_question'])['work_item'] if name in [x[0] for x in specs[:3]] else next(n['work_item'] for n in spec['dependency_graph']['nodes'] if n.get('template') == 'document_fact' and n['scoring'] == 'prerequisite')})
    contract = {'schema_version': 5, 'sheets': {name: {'key_columns': keys, 'categorical_values':
                ({'Included': ['Yes', 'No']} if name == scope_name else
                 {'Answer': sorted({str(v) for q in encoding.values() for a, v in q['encoding']})} if name == coding_name else {}),
                'out_of_scope_keys': []} for name, keys, columns in specs}}
    dump(out / 'tests/instance_guard.json', {**instance_guard, 'sampling': sp['sampling']})
    for name, obj in [('gold', gold), ('scenario_params', sp), ('validation_contract', contract),
                      ('provenance', provenance), ('dependency_graph', graph), ('scenario_spec', requested_spec)]:
        dump(out / 'tests' / f'{name}.json', obj)
    write_workbook(sheets, specs, out / 'solution/reference.xlsx')
    (out / 'tests/scoring.py').write_text(Path(__file__).with_name('scoring.py').read_text())
    if any(is_benchmark(adapter, q) for q in spec['questions']):
        for module in ('benchmark_checkers.py', 'benchmark_oolong_score.py', 'benchmark_sembench_score.py'):
            (out / 'tests' / module).write_text(Path(__file__).with_name(module).read_text())
        for cell in gold['cells']:
            database = cell.get('checker', {}).get('database')
            if database:
                relative = Path(database).relative_to('environment/public/inputs')
                verifier_copy = out / 'tests/input_assets' / relative
                verifier_copy.parent.mkdir(parents=True, exist_ok=True)
                if not verifier_copy.exists():
                    shutil.copyfile(out / database, verifier_copy)
    (out / 'instruction.md').write_text(text_header(spec['instruction'].rstrip() + '\n\n' + output_contract(specs, public_contract)))
    (out / 'design.md').write_text(spec['design'] + '\n\nThe summary is consistency-only (5%). Keys and Question are identifiers; all Answer cells map bijectively to source labels. Scope errors affect scope/fact credit only.\n')
    family = 'C2' if spec['source'] in ('C2', 'CUAD') else spec['source']
    task_toml = '''schema_version = "1.4"
artifacts = ["/output/review.xlsx"]
[metadata]
canary = "{canary}"
name = "{name}"
family = "{family}"
category = "{category}"
description = "Reversible contract coding and review scope reconciliation"
[agent]
timeout_sec = 28800.0
user = "65532:65532"
[verifier]
timeout_sec = 600.0
environment_mode = "separate"
user = "65532:65532"
[verifier.environment]
network_mode = "no-network"
cpus = 1
memory_mb = 2048
[environment]
network_mode = "no-network"
build_timeout_sec = 1200.0
cpus = 1
memory_mb = 2048
'''
    (out / 'task.toml').write_text(task_toml.format(name=spec['id'], family=family, category=spec['category'], canary=GUID))
    template = (REPO / 'harbor_templates.md').read_text()
    for name in ['environment/Dockerfile', 'tests/Dockerfile']:
        body = re.search(r'`' + re.escape(name) + r'`\n```dockerfile\n(.*?)```', template, re.S).group(1)
        if name == 'tests/Dockerfile' and (out / 'tests/benchmark_checkers.py').exists():
            body += '\nCOPY benchmark_checkers.py benchmark_oolong_score.py benchmark_sembench_score.py /opt/\n'
            if (out / 'tests/input_assets').exists():
                body += '\nCOPY input_assets/ /environment/public/inputs/\n'
        (out / name).write_text(body)
    (out / 'tests/test.sh').write_text('#!/bin/sh\nset -eu\npython /opt/scoring.py /output/review.xlsx /tests/gold.json /logs/verifier\n')
    (out / 'solution/solve.sh').write_text('#!/bin/sh\nset -eu\ncp /solution/reference.xlsx /output/review.xlsx\n')
    # Controller-side build entry points import this versioned compiler, never author code.
    bootstrap = 'import sys\nfrom pathlib import Path\nsys.path.insert(0, ' + repr(str(REPO / 'v5')) + ')\n'
    build_entry = bootstrap + 'from tada_taskgen.compiler import *\nSPEC = ' + repr(requested_spec)
    build_entry += '\nCLIENT = ' + repr(client) + '\nSHEET_SPECS = sheet_specs(SPEC)\n'
    build_entry += (
        'if __name__ == "__main__":\n'
        '    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=7)\n'
        '    a = p.parse_args(); archive = Path(a.out) / "tests/roundtrip_report.json"\n'
        '    if CLIENT == "real" and not archive.exists():\n'
        '        raise ValueError("Real controller rebuild requires archived documents; no live model calls permitted")\n'
        '    compile_task(SPEC, a.seed, a.out, client=CLIENT, document_archive=json.loads(archive.read_text()) if archive.exists() else None)\n')
    (out / 'build/build.py').write_text(build_entry)
    for file, function in [('wrong_answers', 'wrong_main'), ('keyword_baseline', 'keyword_main'), ('sensitivity', 'sensitivity_main')]:
        (out / 'build' / f'{file}.py').write_text(bootstrap + f'from tada_taskgen.baselines import {function}\nif __name__ == "__main__":\n    {function}()\n')
    if sum(path.stat().st_size for path in public.rglob('*') if path.is_file()) > sp['sampling']['workspace_max_bytes']:
        raise InstanceCompilationError('generated workspace exceeds the configured byte cap', {'max_bytes': sp['sampling']['workspace_max_bytes']})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--client', choices=['fake', 'real'], default='fake')
    args = parser.parse_args()
    print(compile_task(args.spec, args.seed, args.out, client=args.client))
