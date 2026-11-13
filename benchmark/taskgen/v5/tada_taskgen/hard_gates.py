"""The six mechanical gates in design section 0A; all other metrics are reports."""
from __future__ import annotations

import hashlib
import json
import re
import tomllib
import zipfile
from pathlib import Path

NAMES = ('oracle_invertibility', 'trivial_answers', 'deterministic_offline',
         'leakage', 'keyword_baseline', 'output_contract')
LIMIT = .30


def read(path):
    return json.loads(Path(path).read_text())


def visible_text(path):
    path = Path(path)
    if path.suffix.lower() in ('.xlsx', '.docx'):
        with zipfile.ZipFile(path) as archive:
            return '\n'.join(archive.read(name).decode('utf8', 'replace')
                             for name in archive.namelist() if name.endswith('.xml'))
    return path.read_text('utf8', 'replace')


def _contains_bytes(stream, needle):
    tail = b''
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        data = tail + chunk
        if needle in data:
            return True
        tail = data[-(len(needle)-1):] if len(needle) > 1 else b''
    return False


def _json_has_labels(path):
    """Validate JSON and detect private cell objects with bounded memory.

    Large source input tables can be gigabytes; neither values nor an entire
    document are retained. Only the three suspicious object keys are tracked.
    """
    with Path(path).open(encoding='utf-8') as stream:
        buf, pos = '', 0
        def peek():
            nonlocal buf, pos
            if pos == len(buf):
                buf, pos = stream.read(65536), 0
            return buf[pos:pos+1]
        def take():
            nonlocal pos
            ch = peek()
            pos += bool(ch)
            return ch
        def space():
            while peek() and peek() in ' \t\r\n':
                take()
        def string(key=False):
            nonlocal pos
            if take() != '"': raise ValueError('Expected JSON string')
            saved = '"' if key else None
            while True:
                ch = peek()
                if not ch: raise ValueError('Unterminated JSON string')
                if ch == '"':
                    take()
                    return json.loads(saved+'"') if saved is not None else None
                if ch == '\\':
                    token = take() + take()
                    if len(token)!=2 or token[1] not in '"\\/bfnrtu': raise ValueError('Invalid JSON escape')
                    if token[1]=='u':
                        digits=''.join(take() for _ in range(4))
                        if len(digits)!=4 or any(c not in '0123456789abcdefABCDEF' for c in digits): raise ValueError('Invalid Unicode escape')
                        token += digits
                else:
                    match = re.match(r'[^"\\\x00-\x1f]+',buf[pos:])
                    if not match: raise ValueError('Control character in JSON string')
                    token = match.group(0)
                    pos += len(token)
                if saved is not None:
                    saved = saved + token if len(saved)+len(token)<128 else None
        def value(depth=0):
            if depth>512: raise ValueError('JSON nesting exceeds verifier bound')
            space(); ch=peek()
            if ch=='{':
                take();space();keys=set();found=False
                if peek()=='}':take();return False
                while True:
                    key=string(True);space()
                    if take()!=':': raise ValueError('Missing JSON colon')
                    if key in ('record','question','answer'):keys.add(key)
                    found=value(depth+1) or found
                    if found or len(keys)==3:return True
                    space();separator=take()
                    if separator=='}':return False
                    if separator!=',':raise ValueError('Malformed JSON object')
                    space()
            if ch=='[':
                take();space()
                if peek()==']':take();return False
                while True:
                    if value(depth+1):return True
                    space();separator=take()
                    if separator==']':return False
                    if separator!=',':raise ValueError('Malformed JSON array')
            if ch=='"':string();return False
            token=''
            while peek() and peek() not in ' \t\r\n,]}':
                token+=take()
                if len(token)>8192:raise ValueError('JSON scalar exceeds verifier bound')
            if token not in ('true','false','null') and not re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?',token):
                raise ValueError('Invalid JSON value')
            return False
        found=value();space()
        if not found and peek():raise ValueError('Trailing JSON content')
        return found


def canary_check(task):
    task = Path(task)
    config = tomllib.loads(Path(__file__).parents[1].joinpath('config.toml').read_text())
    guid = config.get('canary', {}).get('guid')
    if not guid:
        return {'pass': False, 'missing': ['config.canary.guid']}
    paths = [task/'instruction.md', *sorted((task/'environment/public').rglob('*'))]
    manifest_path = task/'environment/public/input_assets.json'
    manifest = read(manifest_path) if manifest_path.exists() else {}
    assets = {row['path']: row for row in manifest.get('assets', [])} if manifest.get('canary') == guid else {}
    def marked(path):
        relative = str(path.relative_to(task/'environment/public')) if path.is_relative_to(task/'environment/public') else None
        if relative in assets:
            from .compiler import file_sha256
            return file_sha256(path) == assets[relative]['sha256']
        if path.suffix.lower() in ('.xlsx', '.docx'):
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith('.xml'):
                        with archive.open(name) as stream:
                            if _contains_bytes(stream, guid.encode()): return True
            return False
        with path.open('rb') as stream:
            return _contains_bytes(stream, guid.encode())
    missing = [str(path.relative_to(task)) for path in paths if path.is_file() and not marked(path)]
    metadata = tomllib.loads((task/'task.toml').read_text()).get('metadata', {})
    if guid not in json.dumps(metadata):
        missing.append('task.toml metadata')
    return {'pass': not missing, 'missing': missing, 'files_checked': sum(p.is_file() for p in paths)}


def output_contract(task):
    """Check scorer-consumed names/options against the instruction or XLSX template."""
    from openpyxl import load_workbook
    task = Path(task)
    gold = read(task/'tests/gold.json')
    contract = read(task/'tests/validation_contract.json')
    instruction = (task/'instruction.md').read_text()
    wb = load_workbook(task/'environment/public/review_template.xlsx', read_only=True, data_only=True)
    try:
        tokens = set(wb.sheetnames)
        for ws in wb:
            tokens.update(str(value) for row in ws.iter_rows(values_only=True) for value in row if value is not None)
    finally:
        wb.close()
    def declared(value):
        value = str(value)
        return value in tokens or re.search(r'(?<![\w])' + re.escape(value) + r'(?![\w])', instruction) is not None
    required = {('sheet', name) for name in gold['sheets']}
    for name, sheet in gold['sheets'].items():
        required.update(('column', str(column)) for column in sheet['columns'])
        required.update(('column', str(column)) for column in sheet['key_columns'])
    for cell in gold.get('cells', []):
        if cell.get('kind') != 'benchmark_item':
            required.update(('option', str(value)) for _, value in cell['encoding'])
    for sheet in contract['sheets'].values():
        for values in sheet.get('categorical_values', {}).values():
            required.update(('option', str(value)) for value in values)
    missing = [{'kind': kind, 'value': value} for kind, value in sorted(required) if not declared(value)]
    return {'pass': not missing, 'missing': missing, 'requirements_checked': len(required)}


def visible_gold_check(task):
    """Detect copied private answer data, including answer workbooks with arbitrary names."""
    from .scoring import load_book, parse_sheet, norm_id, norm_text
    task = Path(task)
    gold = read(task/'tests/gold.json')
    from .compiler import file_sha256
    private = {file_sha256(task/'tests/gold.json'), file_sha256(task/'solution/reference.xlsx')}
    hits = []
    for path in sorted((task/'environment/public').rglob('*')):
        if not path.is_file():
            continue
        found = file_sha256(path) in private
        if path.suffix.lower() == '.json':
            found = found or _json_has_labels(path)
        if path.suffix.lower() == '.xlsx':
            book, _, _ = load_book(path, gold['sheets'])
            for name, spec in gold['sheets'].items():
                rows, _, _ = parse_sheet(book.get(name, []), spec['columns'], spec['key_columns'])
                cells = [cell for cell in gold['cells'] if cell['sheet'] == name]
                expected = {(tuple(norm_id(v) for v in cell['key']), cell['column']):
                            norm_text(cell['encoding'][0][1] if cell.get('kind') == 'benchmark_item' else dict(cell['encoding'])[cell['answer']]) for cell in cells}
                for row in rows:
                    key = tuple(norm_id(row.get(col)) for col in spec['key_columns'])
                    if any(norm_text(row.get(column)) == value for (target, column), value in expected.items() if target == key):
                        found = True
        if found:
            hits.append(str(path.relative_to(task)))
    return {'pass': not hits, 'answer_data_files': hits}


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return {'pass': False, 'error': f'{type(exc).__name__}: {exc}'}


def score_checks(task, out):
    """Use the audited local scorer, never execute a task-provided judge."""
    from .baselines import wrong_answers
    from .scoring import score
    from .compiler import write_workbook
    from openpyxl import load_workbook
    task, out = Path(task), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gold = read(task/'tests/gold.json')
    gold['_task_root'] = str(task.resolve())
    ref = task/'solution/reference.xlsx'
    scorer = Path(__file__).with_name('scoring.py')
    shared = (task/'tests/scoring.py').read_bytes() == scorer.read_bytes()
    if any(c.get('kind') == 'benchmark_item' for c in gold['cells']):
        for module in ('benchmark_checkers.py', 'benchmark_oolong_score.py', 'benchmark_sembench_score.py'):
            checker = task / 'tests' / module
            shared = shared and checker.is_file() and checker.read_bytes() == Path(__file__).with_name(module).read_bytes()
        from .compiler import file_sha256
        for cell in gold['cells']:
            database = cell.get('checker', {}).get('database')
            if database:
                relative = Path(database).relative_to('environment/public/inputs')
                verifier_copy = task / 'tests/input_assets' / relative
                shared = shared and verifier_copy.is_file() and file_sha256(verifier_copy) == file_sha256(task / database)
                shared = shared and 'COPY input_assets/ /environment/public/inputs/' in (task / 'tests/Dockerfile').read_text()
    if not shared:
        return {'pass': False, 'error': 'task scorer differs from audited offline scorer'}
    first, first_details = score(ref, gold)
    second, second_details = score(ref, gold)
    config = tomllib.loads((task/'task.toml').read_text())
    entrypoint = [line.strip() for line in (task/'tests/test.sh').read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith('#')]
    entrypoint_verified = entrypoint == ['set -eu', 'python /opt/scoring.py /output/review.xlsx /tests/gold.json /logs/verifier']
    offline = entrypoint_verified and config.get('verifier', {}).get('environment', {}).get('network_mode') == 'no-network'
    fixtures = out/'trivial'
    wrong_answers(task, fixtures)
    specs = [(name, row['key_columns'], row['columns']) for name, row in gold['sheets'].items()]
    write_workbook({}, specs, fixtures/'empty.xlsx')
    values = {str(value) for cell in gold['cells'] for _, value in cell['encoding']}
    constants = []
    for index, value in enumerate(sorted(values)):
        wb = load_workbook(ref)
        for sheet, schema in gold['sheets'].items():
            if 'Answer' not in schema['columns']:
                continue
            ws = wb[sheet]
            column = schema['columns'].index('Answer') + 1
            for row in range(2, ws.max_row + 1):
                ws.cell(row, column).value = value
        path = fixtures/f'constant-{index}.xlsx'
        wb.save(path); wb.close()
        constants.append({'value': value, 'reward': score(path, gold)[0]})
    rewards = {name: score(fixtures/f'{name}.xlsx', gold)[0]
               for name in ('empty', 'all_majority')}
    include_reward, include_details = score(fixtures/'include_all.xlsx', gold)
    include_all = {'status': 'reported_only', 'used_for_admission': False,
                   'reward': include_reward, 'scope_f1': include_details['components']['scope']}
    rewards['constant'] = max((row['reward'] for row in constants), default=None)
    checks = {name: {'reward': value, 'limit': LIMIT, 'pass': isinstance(value, (int, float)) and value <= LIMIT}
              for name, value in rewards.items()}
    return {'pass': True, 'reference': {'reward': first}, 'reference_repeat': {'reward': second},
            'identical': first == second and first_details == second_details,
            'offline': shared and offline, 'scorer_source_verified': shared,
            'verifier_entrypoint_verified': entrypoint_verified,
            'include_all': include_all,
            'trivial': {'pass': all(r['pass'] for r in checks.values()),
                'metric': 'total_reward', 'results': checks, 'constant_candidates': constants}}


def permutation_check(task, out):
    """Rebuild from the exact archived documents; no writer/extractor calls."""
    from .compiler import compile_task, file_sha256
    task, out = Path(task), Path(out)
    spec = read(task/'tests/scenario_spec.json')
    params = read(task/'tests/scenario_params.json')
    archive = read(task/'tests/roundtrip_report.json')
    from .multisource import label_permutation
    with label_permutation(991):
        compile_task(spec, params['seed'], out, document_archive=archive)
    def hashes(root):
        paths = [root/'instruction.md', *sorted((root/'environment/public').rglob('*'))]
        return {str(p.relative_to(root)): file_sha256(p) for p in paths if p.is_file()}
    original, permuted = hashes(task), hashes(out)
    changed = sorted(key for key in set(original) | set(permuted) if original.get(key) != permuted.get(key))
    return {'pass': not changed, 'files_compared': len(original), 'label_dependent_files': changed,
            'gold_changed_under_permutation': (task/'tests/gold.json').read_bytes() != (out/'tests/gold.json').read_bytes()}


def evaluate(task, report=None, *, out=None):
    """Return six hard groups; optional prior controller steps remain report-only."""
    from .gates import invertibility
    task = Path(task)
    out = Path(out) if out is not None else task.parent/'hard_gate_checks'
    out.mkdir(parents=True, exist_ok=True)
    report = report if report is not None else {'steps': {}, 'verdict': {}}
    steps = report.setdefault('steps', {})
    checks = _guard(score_checks, task, out)
    steps['slim_score_checks'] = checks
    steps['include_all_reported_only'] = checks.get('include_all',
        {'status': 'unavailable', 'used_for_admission': False})
    inverse = _guard(invertibility, task, slim=True)
    if 'keyword_baseline' not in steps:
        from .baselines import keyword_baseline
        from .scoring import score
        def keyword():
            target = out/'keyword_baseline.xlsx'
            keyword_baseline(task/'environment/public', target)
            reward, details = score(target, read(task/'tests/gold.json'))
            return {'pass': reward <= LIMIT, 'reward': reward, 'details': details}
        steps['keyword_baseline'] = _guard(keyword)
    if 'leak' not in steps:
        from controller import leak_scan
        def scan():
            source = read(task/'tests/gold.json')['source']
            return leak_scan(task, 'C2' if source == 'CUAD' else source)
        steps['leak'] = _guard(scan)
    if 'leak_permutation' not in steps:
        steps['leak_permutation'] = _guard(permutation_check, task, out/'permuted_task')
    if 'extra_rows_nonperfect' not in steps:
        from controller import score as controller_score
        from weekend_checks import extra_rows_nonperfect, extra_rows_tail_gap
        baseline = out/'keyword_baseline.xlsx'
        steps['extra_rows_nonperfect'] = _guard(extra_rows_nonperfect, task, task/'solution/reference.xlsx',
            baseline, steps['keyword_baseline'].get('reward'), out/'extra_rows', controller_score)
        steps['extra_rows_tail_gap'] = _guard(extra_rows_tail_gap, task, task/'solution/reference.xlsx',
            out/'tail_rows', controller_score)
    contract = _guard(output_contract, task)
    canary = _guard(canary_check, task)
    visible_gold = _guard(visible_gold_check, task)
    # Old tail-gap checks demand a strict penalty. The slim rule requires only no gain.
    tail = steps.get('extra_rows_tail_gap', {})
    tail_runs = tail.get('runs', [])
    tail_ok = bool(tail_runs) and all(isinstance(row.get('adjacent'), (int, float))
        and isinstance(row.get('after_gap'), (int, float)) and row['after_gap'] <= row['adjacent'] + 1e-9
        for row in tail_runs)
    reference = checks.get('reference', {}).get('reward')
    keyword = steps['keyword_baseline'].get('reward')
    hard = {
        'oracle_invertibility': {'pass': reference == 1.0 and inverse.get('pass') is True,
                                'reward': reference, 'required': 1.0, 'invertibility': inverse},
        'trivial_answers': checks.get('trivial', {'pass': False, 'error': checks.get('error', 'checks unavailable')}),
        'deterministic_offline': {'pass': checks.get('identical') is True and checks.get('offline') is True
            and steps['extra_rows_nonperfect'].get('pass') is True and tail_ok,
            'repeat_identical': checks.get('identical', False), 'offline': checks.get('offline', False),
            'extra_rows_never_raise': steps['extra_rows_nonperfect'].get('pass', False), 'tail_gap_no_gain': tail_ok},
        'leakage': {'pass': steps['leak'].get('pass') is True and steps['leak_permutation'].get('pass') is True
                    and visible_gold.get('pass') is True,
                    'visible_scan': steps['leak'], 'visible_gold': visible_gold,
                    'label_permutation': steps['leak_permutation']},
        'keyword_baseline': {'pass': isinstance(keyword, (int, float)) and not isinstance(keyword, bool)
                            and 0 <= keyword <= LIMIT, 'reward': keyword, 'limit': LIMIT},
        'output_contract': {'pass': contract.get('pass') is True and canary.get('pass') is True,
                            'contract': contract, 'canary': canary},
    }
    old_verdict = report.get('verdict', {})
    reported = {key: value for key, value in old_verdict.items() if key not in NAMES and key != 'reported_only'} if isinstance(old_verdict, dict) else {'controller_status': old_verdict}
    reported.update(old_verdict.get('reported_only', {}) if isinstance(old_verdict, dict) else {})
    reported['include_all'] = steps['include_all_reported_only']
    report['hard_gates'] = hard
    report['reported_only'] = {'verdict': reported, 'measurements': {key: value for key, value in steps.items()
        if key not in ('slim_score_checks', 'leak', 'leak_permutation', 'keyword_baseline')}}
    report['verdict'] = {**{name: hard[name]['pass'] for name in NAMES}, 'reported_only': reported}
    report['pass'] = all(hard[name]['pass'] is True for name in NAMES)
    report['gate_policy'] = ('Admission policy: '
                             'exactly six hard groups; one seed; include-all reported only')
    return report
