"""Six hard gates for one seed, with optional historical five-seed reports."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from .multisource import scenario_sources

REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parents[1] / 'config.toml'
POLICY = tomllib.loads(CONFIG.read_text())['gates']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def invertibility(task, *, slim=False):
    """Fail closed on malformed packages as well as semantic gate failures."""
    try:
        return _invertibility(task, slim=slim)
    except Exception as exc:
        return {'pass': False, 'failures': [f'malformed invertibility input: {type(exc).__name__}: {exc}']}


def _invertibility(task, *, slim=False):
    from .multisource import adapter_for, origin, scenario_sources, is_benchmark
    from .schema import validate_graph
    from .roundtrip import canonical
    from .scoring import norm_id, hdr
    from .canary import strip_header
    task = Path(task); failures = []
    gold = read(task / 'tests/gold.json'); graph = read(task / 'tests/dependency_graph.json')
    spec = read(task / 'tests/scenario_spec.json'); adapter = adapter_for(spec)
    if (gold.get('sources') != scenario_sources(spec)
            or gold.get('qualified_labels') is not ('sources' in spec)):
        failures.append('gold source namespace differs from declared sources')
    validate_graph(graph, spec['work_items'], resolved=True)
    cells = gold['cells']; facts = gold.get('facts', []); consistency = gold.get('consistency', [])
    nodes = {n['id']: n for n in graph['nodes']}
    label_nodes = {k: n for k, n in nodes.items() if n['type'] == 'original_label'}
    used = [(c['record'], c['question']) for c in cells]
    def coordinate(c):
        return c['sheet'], tuple(c['key']), c['column']
    coordinates = [coordinate(c) for c in cells + facts + consistency]
    normalized_coordinates = [(hdr(c['sheet']), tuple(norm_id(k) for k in c['key']), hdr(c['column']))
                              for c in cells + facts + consistency]
    all_ids = [c['id'] for c in cells + facts + consistency]
    if (len(set(used)) != len(cells) or len(set(coordinates)) != len(coordinates)
            or len(set(normalized_coordinates)) != len(normalized_coordinates)
            or len(set(all_ids)) != len(all_ids) or set(label_nodes) != {c['id'] for c in cells}):
        failures.append('cells and used source labels are not a bijection')
    public = read(task / 'environment/public/coding_conventions.json')['questions']
    pool = adapter.label_pool(); records = adapter.records()
    edges = {(e['from'], e['to']) for e in graph['edges']}
    columns = {(n.get('sheet'), n.get('column')) for n in nodes.values() if n['type'] == 'deliverable_column'}
    known_fact_ids = {c['id'] for c in cells + facts}
    for c in cells:
        pair = (c['record'], c['question']); q = c['question']
        expected_origin = origin(spec, *pair)
        if c.get('origin') != expected_origin or c.get('source') != expected_origin['source']:
            failures.append('source/original label identity differs: ' + c['id'])
        if c['key'] != [adapter.visible_id(c['record']), q]:
            failures.append('label key differs from adapter record/question mapping: ' + c['id'])
        authoritative = pool.get(pair, (None,))[0]
        if (not adapter.scorable(*pair) or type(authoritative) is not type(c['answer'])
                or authoritative != c['answer']):
            failures.append('unknown/excluded/changed label: ' + c['id'])
        item = is_benchmark(adapter, q)
        expected = ([[authoritative, adapter.encode(q, authoritative)]] if item else
                    [[a, adapter.encode(q, a)] for a in adapter.questions()[q]['options']])
        public_expected = [] if item else expected
        if canonical(c['encoding']) != canonical(expected) or canonical(public[q]['encoding']) != canonical(public_expected):
            failures.append('encoding not published/adapter-owned: ' + c['id'])
        if item:
            from .benchmark_checkers import score_answer
            from .compiler import item_checker
            if (c.get('kind') != 'benchmark_item' or canonical(c.get('checker')) != canonical(item_checker(adapter, *pair))
                    or score_answer(authoritative, authoritative, c['checker'], root=task) != 1):
                failures.append('benchmark checker differs from original: ' + c['id'])
        if any(adapter.decode(q, v) != a for a, v in c['encoding']) or len({v for a, v in c['encoding']}) != len(expected):
            failures.append('noninvertible encoding: ' + c['id'])
        n = label_nodes.get(c['id'], {})
        if n.get('source') != c.get('source') or n.get('origin') != c.get('origin'):
            failures.append('graph/source identity differs: ' + c['id'])
        if n.get('record') != c['record'] or n.get('question') != q or n.get('cell') != {k:c[k] for k in ('sheet','key','column')}:
            failures.append('graph/cell mapping differs: ' + c['id'])
        carrier = (task / 'environment/public' / n['carrier']).resolve()
        if not carrier.is_relative_to((task / 'environment/public').resolve()) or not carrier.is_file():
            failures.append('label carrier missing/unsafe: ' + c['id'])
        else:
            original = records[c['record']]
            published = strip_header(carrier.read_bytes().decode('utf-8'))
            expected_hash = original.get('text_sha256')
            if (hashlib.sha256(published.encode()).hexdigest() != expected_hash if expected_hash else published != original['text']):
                failures.append('original labelled text was rewritten: ' + c['id'])
    for fact in facts:
        node = nodes.get(fact['id'], {})
        if (node.get('type') != 'construction_fact' or node.get('scoring') != 'prerequisite'
                or node.get('cell') != {k: fact[k] for k in ('sheet', 'key', 'column')}
                or canonical(node.get('value')) != canonical(fact['value'])):
            failures.append('fact/cell mapping differs: ' + fact['id'])
    for node in nodes.values():
        if node['type'] == 'construction_fact' and node.get('scoring') == 'prerequisite' and node['id'] not in {f['id'] for f in facts}:
            failures.append('scored construction fact has no cell: ' + node['id'])
    for c in consistency:
        row = gold['sheets'][c['sheet']]['rows'].get('|'.join(c['key']), {})
        if c['column'] in row or 'answer' in c or 'value' in c or not c['dependencies'] or not set(c['dependencies']) <= known_fact_ids:
            failures.append('derived output used as independent gold: ' + c['id'])
        node = nodes.get(c['id'], {})
        if node.get('type') != 'derived_quantity' or node.get('scoring') != 'consistency-only':
            failures.append('consistency cell lacks derived node: ' + c['id'])
        if any((dependency, c['id']) not in edges for dependency in c['dependencies']):
            failures.append('derived cell not traced to every prerequisite: ' + c['id'])
    # Every non-key gold value must be one registered label or construction fact.
    registered = {coordinate(c): c for c in cells + facts}
    for name, sheet in gold['sheets'].items():
        for row in sheet['rows'].values():
            key = tuple(str(row[k]) for k in sheet['key_columns'])
            for column in set(row) - set(sheet['key_columns']):
                if (name, key, column) not in registered:
                    failures.append('unregistered independent gold cell: ' + name + '|' + '|'.join(key) + '|' + column)
    for c in cells + facts + consistency:
        if (c['sheet'], c['column']) not in columns:
            failures.append('scored cell has no deliverable column: ' + c['id'])
        row = gold['sheets'][c['sheet']]['rows'].get('|'.join(c['key']))
        if row is None:
            failures.append('cell key is absent from gold row registry: ' + c['id'])
    weights = gold['weights']
    if (any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in weights.values())
            or abs(sum(weights.values()) - 1) > 1e-9
            or (not slim and weights.get('consistency', 0) > POLICY['consistency_weight_max'])):
        failures.append('invalid weights or consistency exceeds frozen low-weight cap')
    visible = [task/'instruction.md', *sorted((task/'environment/public').rglob('*'))]
    derived = {n['id']: str(n['value']) for n in graph['nodes']
               if n['provision'] == 'solver-derived' and 'value' in n and str(n['value'])}
    found = set()
    def scan(stream):
        import codecs
        decoder = codecs.getincrementaldecoder('utf-8')('ignore')
        overlap = max((len(value) for value in derived.values()), default=1) - 1
        tail = ''
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            data = tail + decoder.decode(chunk)
            found.update(key for key, value in derived.items() if value in data)
            tail = data[-overlap:] if overlap else ''
    for p in visible:
        if not p.is_file() or not derived: continue
        if p.suffix in ('.xlsx','.docx'):
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    if name.endswith('.xml'):
                        with z.open(name) as stream:
                            scan(stream)
        else:
            with p.open('rb') as stream:
                scan(stream)
    failures.extend('solver-derived value visible: ' + key for key in sorted(found))
    from openpyxl import load_workbook
    wb = load_workbook(task/'environment/public/review_template.xlsx', read_only=True)
    if any(ws.max_row != 1 for ws in wb): failures.append('visible template is not header-only')
    wb.close()
    return {'pass': not failures, 'labels': len(cells), 'distinct_source_labels': len(set(used)), 'failures': failures}


def roundtrip_gate(task):
    try:
        return _roundtrip_gate(task)
    except Exception as exc:
        return {'pass': False, 'failures': [f'malformed round-trip input: {type(exc).__name__}: {exc}']}


def _roundtrip_gate(task):
    from .roundtrip import canonical
    from .canary import strip_header
    task = Path(task); report = read(task/'tests/roundtrip_report.json'); graph=read(task/'tests/dependency_graph.json')
    failures=[]; verified={}; ids=set(); paths=set()
    for d in report['documents']:
        if d['id'] in ids or d['path'] in paths:
            failures.append('duplicate carrier id/path: ' + d['id'])
        ids.add(d['id']); paths.add(d['path'])
        attempts=d.get('attempts',[])
        if not attempts or len(attempts)>4 or d['writer_model']==d['extractor_model']:
            failures.append('invalid independent regeneration record: '+d['id'])
        if d['status'] == 'dropped':
            if d.get('facts') or d.get('pass'):
                failures.append('dropped carrier retains scored facts: ' + d['id'])
            continue
        if d['status'] != 'verified':
            failures.append('unverified carrier: ' + d['id']); continue
        path=(task/'environment/public'/d['path']).resolve()
        if not path.is_relative_to((task/'environment/public').resolve()) or not path.is_file() or strip_header(path.read_text())!=d['text']:
            failures.append('carrier differs: '+d['id']); continue
        if not attempts or canonical(attempts[-1].get('extracted')) != canonical(d['facts']) or attempts[-1].get('pass') is not True:
            failures.append('blind exact-match verification absent: '+d['id'])
        verified[d['path']]=d
    for n in graph['nodes']:
        if n['type']=='construction_fact':
            d=verified.get(n['carrier'])
            key = n.get('fact_key', n.get('cell', {}).get('key', [n['id']])[0])
            if d is None or 'value' not in n or canonical(d['facts'].get(key, d['facts'].get(n['id']))) != canonical(n['value']):
                failures.append('fact lacks verified carrier: '+n['id'])
    return {'pass':not failures,'verified_documents':len(verified),'dropped_facts':report.get('dropped_facts',[]),'fake':report.get('fake'), 'failures':failures}


def question_screening_report(spec):
    """Read the validated historical measurements without imposing eligibility."""
    from .multisource import adapter_for, origin, scenario_sources, is_benchmark
    from .screening import load_screening, DEFAULT_OUTPUT
    if 'sources' in spec:
        reports = {source: question_screening_report({'source': source, 'questions': spec['questions'][source]})
                   for source in scenario_sources(spec)}
        available = all(row['available'] for row in reports.values())
        return {'status': 'reported_only', 'used_for_admission': False, 'sources': reports,
                'available': available, 'pass': all(row['pass'] for row in reports.values()) if available else None}
    result = {'status': 'reported_only', 'used_for_admission': False,
              'report_source': str(DEFAULT_OUTPUT / (spec['source'] + '.json'))}
    try:
        if not Path(result['report_source']).exists():
            raise FileNotFoundError('historical source measurement is not available')
        measured = load_screening(adapter_for(spec))
        questions = {question: {key: value for key, value in measured['questions'].get(question, {}).items()
                               if key not in ('nb_oof_record_ids', 'nb_oof_folds')}
                     for question in spec['questions']}
        eligible = [question for question, row in questions.items() if row.get('eligible') is True]
        result.update(available=True, pass_=len(eligible) == len(questions),
                      version=measured.get('version'), policy=measured.get('policy'),
                      eligible_questions=eligible, questions=questions)
        result['pass'] = result.pop('pass_')
    except Exception as exc:
        # A missing/stale report is an unavailable measurement, never a task veto.
        result.update(available=False, error=f'{type(exc).__name__}: {exc}')
        result['pass'] = None
    return result


def extensions(task, report):
    task=Path(task); spec=read(task/'tests/scenario_spec.json')
    steps=report['steps']; steps['invertibility']=invertibility(task); steps['roundtrip']=roundtrip_gate(task)
    from .schema import validate_spec
    try:
        validate_spec(spec)
        steps['spec_structure'] = {'pass': True}
    except ValueError as exc:
        steps['spec_structure'] = {'pass': False, 'error': str(exc)}
    steps['question_screening'] = question_screening_report(spec)
    steps['instance_class_guard'] = sparse_instance_gate(task)
    report['verdict'].update({'v5_question_screening': steps['question_screening']['pass'],
                             'v5_spec_structure': steps['spec_structure']['pass'],
                             'v5_instance_class_guard': steps['instance_class_guard']['pass']})
    details=(steps.get('keyword_baseline',{}).get('details') or {})
    scores=details.get('per_work_item', {})
    items={w['id'] for w in spec['work_items']}
    steps['per_work_item_keyword']={'scores':scores,'gate':POLICY['work_item_keyword_max'],
        'pass':items <= set(scores) and all(scores[k]<=POLICY['work_item_keyword_max'] for k in items)}
    report['verdict'].update({'v5_invertibility':steps['invertibility']['pass'], 'v5_roundtrip':steps['roundtrip']['pass'],
                             'per_work_item_keyword_below_gate':steps['per_work_item_keyword']['pass']})
    return evaluate_task(task, report)


def evaluate_task(task, report=None, *, out=None):
    from .hard_gates import evaluate
    return evaluate(task, report, out=out)


def sparse_instance_gate(task):
    """Replay the fixed draw and pruning rule, rather than trusting submitted keys.

    A self-consistent gold, graph and scenario can still remove a difficult
    question. Reconstruct from the requested spec and original source labels so
    only the predeclared class guard is permitted to drop it.
    """
    from .compiler import (resolve, guard_instance, class_counts, review_memo,
                           REVIEW_MEMO, SCORING_POLICY, WEIGHTS)
    from .roundtrip import canonical
    from .canary import strip_header
    task = Path(task)
    try:
        original = read(task/'tests/scenario_spec.json')
        sp = read(task/'tests/scenario_params.json')
        replay = resolve(original, sp['seed'])
        if canonical(sp) != canonical(replay):
            raise ValueError('sampled instance differs from fixed sparse draw or question-drop rule')
        guard = guard_instance(replay)
        guard['sampling'] = replay['sampling']
        if canonical(read(task/'tests/instance_guard.json')) != canonical(guard):
            raise ValueError('instance guard audit differs from the reconstructed instance')
        public = task/'environment/public'
        contract = read(public/'coding_conventions.json')
        expected_records = [{key: r[key] for key in ('visible_id', 'visible_file', 'questions')}
                            for r in replay['records']]
        if contract['records'] != expected_records or set(contract['questions']) != set(replay['spec']['questions']):
            raise ValueError('published applicability differs from exact retained record/question pairs')
        if contract.get('applicability_memo') != REVIEW_MEMO or strip_header((public/REVIEW_MEMO).read_text()) != review_memo(replay):
            raise ValueError('visible review memo does not publish the exact per-record checklist')
        gold = read(task/'tests/gold.json')
        expected_pairs = {(r['record_id'], q) for r in replay['records'] for q in r['questions']}
        if (len(gold['cells']) != len(expected_pairs)
                or {(c['record'], c['question']) for c in gold['cells']} != expected_pairs):
            raise ValueError('gold labels do not cover the exact sampled record/question pairs')
        if class_counts(gold['cells']) != guard['class_counts']:
            raise ValueError('gold class denominators differ from the original sampled instance')
        if gold.get('label_aggregation') != SCORING_POLICY['label_aggregation'] or gold['weights'] != WEIGHTS:
            raise ValueError('gold aggregation/weights differ from frozen Repair R1 policy')
        if gold.get('label_class_counts') != class_counts(gold['cells']):
            raise ValueError('gold class-count metadata differs from original cells')
        return guard
    except Exception as exc:
        return {'pass': False, 'error': str(exc)}


NUMERIC_VERDICTS={'keyword_baseline_below_gate','per_work_item_keyword_below_gate','one_label_error_bounded',
                  'wrong_answers','independent_then_aggregate_below_gate'}


def scope_wrong_answer(name):
    """Return the canonical scope-test threshold key, or None for reward tests."""
    if name == 'include_all':
        return name
    if name == 'ignore_revisions' or name.startswith('ignore_'):
        return 'ignore_revisions'
    return None


def valid_unit(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def evaluate_wrong_answer(name, reward, details, expected, thresholds):
    """v5 fixture verdict, keeping total reward while gating scope tests on F1.

    Scorer details are the complete on-disk output for this exact submission.
    Absent/invalid scope is never substituted by total reward.
    """
    key = scope_wrong_answer(name)
    if key is None:
        lo, hi = expected
        return {'reward': reward, 'expected': expected, 'gate_metric': 'reward', 'gate_value': reward,
                'in_range': valid_unit(reward) and lo - 1e-9 <= reward <= hi + 1e-9}
    bound = thresholds.get(key)
    components = details.get('components') if isinstance(details, dict) else None
    scope = components.get('scope') if isinstance(components, dict) else None
    valid_bound = valid_unit(bound) and bound <= .90
    return {'reward': reward, 'scope_f1': scope, 'expected': [0, bound],
            'gate_metric': 'scope_f1', 'gate_value': scope,
            'in_range': valid_bound and valid_unit(scope) and scope <= bound}


def classify(report, spec):
    status_error = None
    if not isinstance(report.get('verdict'),dict):
        status_error = report.get('verdict','missing verdict')
        report = {**report, 'verdict': {'controller_report_complete': False}, 'steps': report.get('steps', {})}
    verdict=report['verdict']; steps=report['steps']
    from .hard_gates import NAMES
    hard = report.get('hard_gates', {})
    engineering = {name: hard.get(name, {}).get('pass', verdict.get(name)) is True for name in NAMES}
    def metric(value,limit,worst_limit=None):
        valid=isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and 0 <= value <= 1
        cap = min(1.0,limit*POLICY['worst_multiplier']) if worst_limit is None else worst_limit
        return {'value':value,'limit':limit,'pass':valid and value<=limit,
                'worst_limit': cap,
                'within_worst_cap':valid and value<=cap}
    numeric={'keyword':metric(steps.get('keyword_baseline',{}).get('reward'),POLICY['keyword_max'],POLICY['keyword_worst_max'])}
    scores=steps.get('per_work_item_keyword',{}).get('scores',{})
    for w in spec['work_items']:
        numeric['work_item:'+w['id']]=metric(scores.get(w['id']),POLICY['work_item_keyword_max'],POLICY['keyword_worst_max'])
    flips=steps.get('label_flip',{})
    numeric['label_max']=metric(flips.get('max_loss'),POLICY['single_label_max'])
    numeric['label_median']=metric(flips.get('median_loss'),POLICY['single_label_median'])
    wrong=steps.get('wrong_answers',{}).get('results',{})
    for key in ('include_all','ignore_revisions'):
        candidates=[v.get('scope_f1') for name,v in wrong.items() if scope_wrong_answer(name)==key]
        value = max(candidates) if candidates and all(valid_unit(v) for v in candidates) else None
        numeric[key]=metric(value,spec['thresholds'][key])
        numeric[key]['component']='scope_f1'
        numeric[key]['total_rewards']={name:v.get('reward') for name,v in wrong.items() if scope_wrong_answer(name)==key}
    # Historical metrics remain inspectable, but do not participate in admission.
    numeric_pass=all(v['pass'] for v in numeric.values()) and all(v is True for k,v in verdict.items() if k in NUMERIC_VERDICTS)
    eng=all(v is True for v in engineering.values())
    return {'errors': [status_error] if status_error else [],'engineering_pass':eng,'engineering':engineering,'numeric_pass':numeric_pass,'metrics':numeric,'all_pass':eng,
            'hard_gates': hard, 'reported_only': {'metrics': numeric, 'numeric_pass': numeric_pass,
                'legacy_verdict': report.get('reported_only', verdict.get('reported_only', {}))},
            'harbor_status':'not_run' if verdict.get('harbor_oracle_nop') is None else verdict['harbor_oracle_nop']}


def run_controller(run_dir, out, seed=7):
    command=[sys.executable,str(REPO/'controller.py'),str(run_dir),'--no-harbor','--seed',str(seed),'--output',str(out)]
    p=subprocess.run(command,capture_output=True,text=True)
    Path(str(out)+'.stdout.log').write_text(p.stdout+'\n'+p.stderr)
    path=Path(out)/'report.json'
    if p.returncode or not path.exists():
        raise RuntimeError(f'controller failed ({p.returncode}); see {out}.stdout.log')
    return read(path)


def aggregate(instances, config_digest):
    if len(instances)!=5 or len({i['seed'] for i in instances})!=5:
        raise ValueError('exactly five distinct predeclared seeds required')
    all_engineering=all(i['engineering_pass'] for i in instances)
    passes=sum(i['numeric_pass'] for i in instances)
    names=set().union(*(i.get('metrics',{}) for i in instances))
    worst={name:max((i.get('metrics',{}).get(name,{}).get('value') for i in instances if i.get('metrics',{}).get(name,{}).get('value') is not None),default=None) for name in names}
    worst_ok=all(i.get('metrics') and all(m['within_worst_cap'] for m in i['metrics'].values()) for i in instances)
    numeric=passes>=POLICY['minimum_numeric_passes'] and worst_ok
    return {'pass':all_engineering and numeric,'engineering_pass':all_engineering,'numeric_pass':numeric,
            'status':'reported_only','used_for_admission':False,
            'numeric_passes':passes,'required_numeric_passes':POLICY['minimum_numeric_passes'],'worst_values':worst,
            'worst_caps_pass':worst_ok,'instances':instances,'eligible_instances':[i['seed'] for i in instances if i['all_pass']],
            'config_sha256':config_digest,'policy':POLICY,'harbor_trials':'not_run',
            'metric_pass_counts': {name:sum(i.get('metrics',{}).get(name,{}).get('pass') is True for i in instances) for name in names}}


def run_multiseed(spec_path, out, seeds=None, *, client='fake'):
    from .compiler import compile_task, SCORING_POLICY, SAMPLING_POLICY
    from .schema import load_spec, validate_spec
    if client not in ('fake', 'real'):
        raise ValueError('client must be fake or real')
    spec=load_spec(spec_path) if not isinstance(spec_path,dict) else validate_spec(spec_path)
    seeds=tuple(seeds or POLICY['seeds'])
    if tuple(seeds)!=tuple(POLICY['seeds']): raise ValueError('seeds must equal frozen config before measurement')
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    digest=hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    write(out/'frozen_policy.json',{'policy':POLICY,'scoring':SCORING_POLICY,'sampling':SAMPLING_POLICY,'client':client,'config_sha256':digest,
                                  'spec_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()})
    instances=[]
    for seed in seeds:
        run=out/f'seed-{seed}'
        stage='compile'
        try:
            compile_task(spec,seed,run/'task',client=client)
            write(run/'receipt.json',{'family':'C2' if spec.get('source') in ('C2','CUAD') else '+'.join(scenario_sources(spec)),'version':5})
            stage='controller'
            report=run_controller(run,run/'controller',seed)
            path=run/'controller/report.json'
        except Exception as exc:
            # A failed seed is retained, never replaced or retried with another seed.
            from .compile_errors import from_exception
            report={'seed':seed, 'verdict':{'build':False}, 'steps':{},
                    'failure':{'stage':stage,'type':type(exc).__name__,'reason':str(exc),
                               'details':getattr(exc,'details',{})}}
            diagnostic = from_exception(exc, spec)
            if diagnostic is not None:
                report['compile_diagnostic'] = diagnostic
            path=run/'failure.json'; write(path,report)
        row={'seed':seed,'task':str(run/'controller/task') if 'failure' not in report else None,
             'keyword':report['steps'].get('keyword_baseline',{}).get('reward'),'report':str(path),**classify(report,spec)}
        if 'failure' in report: row['failure']=report['failure']
        params = run/'task/tests/scenario_params.json'
        if params.is_file():
            sampled = read(params)
            row['sampling'] = sampled.get('sampling')
            row['scored_pairs'] = sum(len(r['questions']) for r in sampled['records'])
        instances.append(row)
        write(out/'progress.json',instances)
    result=aggregate(instances,digest)
    result['client']=client
    result['scope_fact_numbers_meaningful']=False if client=='fake' else None
    result['scope_fact_measurement_note']=('Fake documents expose facts transparently; scope/fact numbers do not measure semantic difficulty.'
        if client=='fake' else 'Real document client selected; semantic difficulty still requires separate validation.')
    result['compilation_failures']=sum(i.get('failure',{}).get('stage')=='compile' for i in instances)
    result['keyword_gate']={'passes':result['metric_pass_counts'].get('keyword',0),
                            'required_passes':POLICY['minimum_numeric_passes'],
                            'worst_value':result['worst_values'].get('keyword'),
                            'worst_limit':POLICY['keyword_worst_max'],
                            'worst_cap_pass':all(i.get('metrics',{}).get('keyword',{}).get('within_worst_cap') is True for i in instances)}
    result['keyword_gate']['pass']=(result['keyword_gate']['passes']>=POLICY['minimum_numeric_passes'] and result['keyword_gate']['worst_cap_pass'])
    write(out/'report.json',result)
    return result


def run_single(spec_path, out, *, seed=7, client='fake'):
    """Compile exactly one seed for the hard decision, preserving failed artifacts."""
    from .compiler import compile_task
    from .schema import load_spec, validate_spec
    spec = load_spec(spec_path) if not isinstance(spec_path, dict) else validate_spec(spec_path)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    try:
        compile_task(spec, seed, out/'run/task', client=client)
        write(out/'run/receipt.json', {'version': 5, 'family': 'C2' if spec.get('source') in ('C2', 'CUAD') else '+'.join(scenario_sources(spec))})
        report = run_controller(out/'run', out/'controller', seed)
    except Exception as exc:
        from .hard_gates import NAMES
        from .compile_errors import from_exception
        report = {'seed': seed, 'steps': {}, 'pass': False,
                  'failure': {'type': type(exc).__name__, 'reason': str(exc)},
                  'hard_gates': {name: {'pass': False, 'error': 'compilation/controller unavailable'} for name in NAMES},
                  'verdict': {**{name: False for name in NAMES}, 'reported_only': {}}}
        diagnostic = from_exception(exc, spec)
        if diagnostic is not None:
            report['compile_diagnostic'] = diagnostic
    report.update(seed=seed, client=client, decision_seeds=[seed])
    report['task'] = str(out/'controller/task') if (out/'controller/task').exists() else None
    row = {'seed': seed, 'task': report['task'], 'keyword': report.get('steps', {}).get('keyword_baseline', {}).get('reward'),
           'all_pass': report.get('pass') is True, 'hard_gates': report.get('hard_gates', {}), 'report': str(out/'report.json')}
    report['instances'] = [row]
    report['eligible_instances'] = [seed] if row['all_pass'] else []
    write(out/'report.json', report)
    return report


def main():
    p=argparse.ArgumentParser(); source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--spec'); source.add_argument('--existing-run', type=Path)
    p.add_argument('--out',required=True); p.add_argument('--seed',type=int,default=7)
    p.add_argument('--multiseed-report',action='store_true',help='optional five-seed measurements; never an admission decision')
    p.add_argument('--client',choices=['fake','real'],default='fake')
    a=p.parse_args()
    if a.existing_run:
        if a.multiseed_report: p.error('--multiseed-report requires --spec')
        r=run_controller(a.existing_run,Path(a.out),a.seed)
    elif a.multiseed_report:
        r=run_multiseed(a.spec,a.out,client=a.client)
    else:
        r=run_single(a.spec,a.out,seed=a.seed,client=a.client)
    print(json.dumps({k:v for k,v in r.items() if k not in ('instances','policy')}))

if __name__=='__main__': main()
