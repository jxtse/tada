"""Controller for pi-designed TADA tasks: build and check without the designer seeing results.

Steps (each recorded in <run>/controller/report.json):
  1. static   - required files, task.toml parses, solver-visible template sanity
  2. build    - run build/build.py in a macOS sandbox (no network; writes only inside the work copy)
  3. leak     - solver-visible files must not contain labels/evidence or private paths
  4. scorer   - reference deliverable must score 1.0; empty/missing deliverable must score 0
  5. wrong    - designer's wrong answers scored against their declared ranges
  6. baseline - designer's keyword baseline scored (gate: < 0.30)
  7. sensitivity - designer's parameter-change assertions
  8. harbor   - `harbor lint`; and, when Docker is healthy, `harbor run` with oracle and nop agents

Usage: controller.py <run_dir> [--seed 7] [--harbor]
"""
from __future__ import annotations

import argparse, json, os, re, shutil, subprocess, sys, time, tomllib
from pathlib import Path
from local_config import get, path, require

HERE = Path(__file__).resolve().parent
PY = str(path('datasets_python'))
HARBOR = str(path('harbor_binary'))
SOURCES = HERE / 'sources'
RUNTIME = path('offline_runtime')
READ_ONLY_DATA = [str(path('legacy_cache')), str(path('quarantine'))]
GATE_BASELINE = 0.30


def sandbox_profile(writable: list[Path]) -> str:
    rules = ['(version 1)', '(allow default)', '(deny network*)', '(allow network* (remote unix-socket))',
             '(deny file-write*)']
    for w in writable:
        rules.append(f'(allow file-write* (subpath "{w.resolve()}"))')
    for dev in ['/dev/null', '/dev/tty', '/dev/dtracehelper']:
        rules.append(f'(allow file-write* (literal "{dev}"))')
    rules.append(f'(allow file-write* (subpath "{Path(os.environ.get("TMPDIR", "/tmp")).resolve()}"))')
    return '\n'.join(rules)


def run_py(args: list[str], cwd: Path, writable: list[Path], log: Path, timeout: int = 1800, extra_env: dict | None = None) -> dict:
    require('datasets_python')
    prof = log.with_suffix('.sb')
    prof.write_text(sandbox_profile(writable))
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(cwd), 'PYTHONPATH': os.pathsep.join([str(HERE), str(SOURCES)]), 'PYTHONDONTWRITEBYTECODE': '1',
           'TMPDIR': os.environ.get('TMPDIR', '/tmp'), 'LANG': 'C.UTF-8', **(extra_env or {})}
    t0 = time.monotonic()
    try:
        p = subprocess.run(['sandbox-exec', '-f', str(prof), PY, *args], cwd=cwd, env=env, capture_output=True,
                           text=True, timeout=timeout)
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        code, out, err = 'timeout', e.stdout or '', e.stderr or ''
    log.write_text(f'$ {" ".join(args)}\nexit={code}\n--- stdout\n{out}\n--- stderr\n{err}')
    return {'exit': code, 'seconds': round(time.monotonic() - t0, 1), 'stdout_tail': out[-1500:], 'stderr_tail': err[-2500:]}


def score(task: Path, deliverable: Path | None, out: Path, log: Path) -> dict:
    """Run the designer's scorer exactly as tests/test.sh would, on the host."""
    out.mkdir(parents=True, exist_ok=True)
    # v5 uses one audited, bounded scorer for every task. Reuse it in-process for the
    # exhaustive matrix while preserving the original subprocess path for legacy tasks.
    gp = task / 'tests/gold.json'
    if gp.exists() and json.loads(gp.read_text()).get('schema_version') == 5:
        sys.path.insert(0, str(HERE / 'v5'))
        from tada_taskgen.scoring import score as generic_score
        if file_digest(task / 'tests/scoring.py') != file_digest(HERE / 'v5/tada_taskgen/scoring.py'):
            return {'exit': 1, 'reward': None, 'stderr_tail': 'v5 scorer differs from shared compiler scorer'}
        parsed_gold = json.loads(gp.read_text())
        if any(cell.get('kind') == 'benchmark_item' for cell in parsed_gold.get('cells', [])):
            for name in ('benchmark_checkers.py', 'benchmark_oolong_score.py', 'benchmark_sembench_score.py'):
                checker = task / 'tests' / name
                if not checker.exists() or file_digest(checker) != file_digest(HERE / 'v5/tada_taskgen' / name):
                    return {'exit': 1, 'reward': None, 'stderr_tail': 'benchmark checker differs from shared implementation'}
        parsed_gold['_task_root'] = str(task)
        reward, details = generic_score(deliverable or task / '__missing__.xlsx', parsed_gold)
        (out / 'reward.txt').write_text(str(reward))
        (out / 'details.json').write_text(json.dumps(details, ensure_ascii=False, indent=1))
        log.write_text('exit=0\nshared v5 bounded scorer\n')
        return {'exit': 0, 'reward': reward, 'expected_output_path': '/output/review.xlsx', 'stderr_tail': ''}
    require('datasets_python', 'offline_runtime')
    test_sh = (task / 'tests/test.sh').read_text()
    m = re.search(r'python\s+/opt/scoring\.py\s+(\S+)\s+/tests/gold\.json', test_sh)
    expected_path = m.group(1) if m else None
    target = deliverable if deliverable else task / '__missing__.xlsx'
    env = {'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(RUNTIME), 'PYTHONDONTWRITEBYTECODE': '1'}
    p = subprocess.run([PY, str(task / 'tests/scoring.py'), str(target), str(task / 'tests/gold.json'), str(out)],
                       capture_output=True, text=True, timeout=600, env=env)
    log.write_text(f'exit={p.returncode}\n{p.stdout}\n{p.stderr}')
    reward = None
    if (out / 'reward.txt').exists():
        try:
            reward = float((out / 'reward.txt').read_text().strip())
        except ValueError:
            reward = None
    return {'exit': p.returncode, 'reward': reward, 'expected_output_path': expected_path,
            'stderr_tail': p.stderr[-1500:]}


def task_category(task: Path) -> str:
    """[metadata] category in task.toml: 'sequential_dependency' (default) or 'parallel_coding'."""
    try:
        return tomllib.loads((task / 'task.toml').read_text()).get('metadata', {}).get('category', 'sequential_dependency')
    except Exception:  # noqa: BLE001
        return 'sequential_dependency'


def family_of(run: Path) -> str:
    family = json.loads((run / 'receipt.json').read_text())['family']
    if family not in SOURCE_FILE_RE:
        sys.path.insert(0, str(HERE / 'v5'))
        from tada_taskgen.adapters import get_adapter
        sources = family.split('+')
        if not 1 <= len(sources) <= 3 or len(set(sources)) != len(sources):
            raise ValueError(f'unsupported source family: {family}')
        for source in sources:
            get_adapter(source)
    return family


def evidence_index(family: str, task: Path | None = None) -> dict:
    """{source_id: [expert evidence span, ...]} for the family's private source module."""
    if task is not None and (task / 'tests/scenario_spec.json').exists():
        spec = json.loads((task / 'tests/scenario_spec.json').read_text())
        if spec.get('version') == 5:
            sys.path.insert(0, str(HERE / 'v5'))
            from tada_taskgen.multisource import adapter_for
            return {f'{record}:{question}': spans for (record, question), (_, spans)
                    in adapter_for(spec).reference().label_pool().items()}
    sys.path.insert(0, str(SOURCES))
    if family == 'C2':
        import c2_sources as m  # noqa
        return {f"{c['contract_id']}:{k}": v for c in m.iter_contracts() for k, v in c['evidence_spans'].items()}
    if family == 'CUADX':
        import cuadx_sources as m  # noqa
        return {f"{c['contract_id']}:{k}": v for c in m.iter_contracts() for k, v in c['evidence_spans'].items()}
    if family == 'B':
        import b_sources as m  # noqa
        return {c['prompt_id']: c['evidence_spans'] for c in m.iter_comparisons(eligible_only=False)}
    if family == 'PREC':
        # No annotator spans exist; the table entry text is the label record. Sentences >40 chars are leak-scanned.
        import prec_sources as m  # noqa
        return {f"{p['overruling']}>{p['overruled']}": [p['loc_text']] for p in m.overruling_pairs()}
    if family == 'NTSB':
        # Private text that must never appear in solver-visible files: the probable-cause statement. The analysis
        # narrative often repeats factual-report sentences verbatim (measured ~10% of pieces), so only its sentences
        # NOT present in the event's own factual report are treated as private.
        import ntsb_sources as m  # noqa
        out = {}
        for e in m.iter_events():
            fact = e['factual']
            novel = [p for p in re.split(r'(?<=[.;])\s+', m.analysis_text(e['ev_id']) or '') if p.strip() not in fact]
            cause = [p for p in re.split(r'(?<=[.;])\s+', m.probable_cause(e['ev_id']) or '') if p.strip() not in fact]
            out[e['ev_id']] = cause + novel
        return out
    if family == 'NDA':
        import nda_sources as m  # noqa
        return {f"{d['doc_id']}:{h}": (m.label(d['doc_id'], h) or ('', []))[1]
                for d in m.iter_documents() for h in m.hypotheses()}
    if family in ('MAUD', 'MAUDB', 'MAUDC'):
        # Pseudonymised evidence segments; the full agreement is a legitimate source file, so these are
        # leak-scanned only in non-agreement files (SOURCE_FILE_RE below).
        import maud_sources as m  # noqa
        return {f'{c}:{dp}': v[1] for (c, dp), v in m._labels().items()}
    raise ValueError(f'unsupported evidence family: {family}')


PRIVATE_STRINGS = ['/' + 'Volumes/', '/' + 'Users/', 'gold.json', 'b_sources', 'c2_sources', 'cuadx_sources', 'private_cache',
                   'Evidence Inference', 'PromptID', 'CUAD', 'master_clauses', 'prec_sources', 'precedentAlteration',
                   'precedent_meta', 'ntsb_sources', 'narr_cause', 'narr_accf', 'Cause_Factor', 'nda_sources',
                   'ContractNLI', 'NotMentioned', 'maud_sources', 'MAUD_', 'MAUD v1']
PRIVATE_STRINGS += [str(get(key)) for key in ('datasets_root', 'unified_root', 'legacy_cache', 'quarantine') if get(key)]
if get('legacy_cache'):
    PRIVATE_STRINGS.append('/'.join(Path(get('legacy_cache')).parts[-2:]))
SOURCE_FILE_RE = {'B': r'(paper|article|pmc|full.?text|literature)',
                  'C2': r'(contract|agreement|data.?room|exhibit|executed|signed|\.txt$|\.htm)',
                  'CUADX': r'(contract|agreement|data.?room|exhibit|executed|signed|\.txt$|\.htm)',
                  'PREC': r'(opinion|decision|case|reporter|u\.?s\.?_?\d|\.txt$|\.htm)',
                  # NTSB private texts must not appear anywhere, including factual-report files.
                  'NTSB': r'(?!x)x',
                  'NDA': r'(nda|agreement|contract|executed|signed|counterpart|\.txt$|\.htm)',
                  'MAUD': r'(merger|agreement|contract|executed|signed|exhibit|\.txt$|\.htm)',
                  'MAUDB': r'(merger|agreement|contract|executed|signed|exhibit|\.txt$|\.htm)',
                  'MAUDC': r'(merger|agreement|contract|executed|signed|exhibit|\.txt$|\.htm)'}


def scan_chunks(path, overlap=4096):
    """Bounded scanning of every input byte, including copied database assets."""
    import codecs
    import zipfile
    def chunks(stream):
        previous = ''
        decoder = codecs.getincrementaldecoder('utf-8')('ignore')
        for raw in iter(lambda: stream.read(1024 * 1024), b''):
            text = decoder.decode(raw)
            yield previous + text
            previous = text[-overlap:]
        yield previous + decoder.decode(b'', final=True)
    if path.suffix.lower() in ('.xlsx', '.docx'):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith('.xml'):
                    with archive.open(name) as stream:
                        yield from chunks(stream)
    else:
        with path.open('rb') as stream:
            yield from chunks(stream)


def leak_scan(task: Path, family: str = 'B') -> dict:
    private_strings = PRIVATE_STRINGS
    spec_path = task / 'tests/scenario_spec.json'
    if spec_path.exists() and json.loads(spec_path.read_text()).get('version') == 5:
        # Dataset identifiers and complete answer vocabularies are public codebook
        # metadata in v5, not per-record answers. Keep every private path/module rule.
        sys.path.insert(0, str(HERE / 'v5'))
        from tada_taskgen.multisource import scenario_sources
        sources = scenario_sources(json.loads(spec_path.read_text()))
        public_terms = ({'CUAD'} if any(s in sources for s in ('CUAD', 'CUADX')) else set())
        if 'NDA' in sources:
            public_terms.add('NotMentioned')
        private_strings = [term for term in PRIVATE_STRINGS if term not in public_terms]
    pieces = set()
    for spans in evidence_index(family, task).values():
        for s in spans:
            for piece in re.split(r'(?<=[.;])\s+', s or ''):
                if len(piece.strip()) > 40:
                    pieces.add(piece.strip())
    public = task / 'environment/public'
    files = [p for p in public.rglob('*') if p.is_file()] if public.exists() else []
    if (task / 'instruction.md').is_file():
        files.append(task / 'instruction.md')
    text_hits, private_hits, label_words = [], [], []
    import zipfile
    for f in files:
        try:
            private_hit, vocabulary_hit = False, False
            for raw in scan_chunks(f):
                private_hit |= any(x in raw for x in private_strings)
                vocabulary_hit |= bool(re.search(r'Label Code|evidence[_ ]span|annotator', raw, re.I))
        except Exception:
            continue
        if private_hit:
            private_hits.append(str(f.relative_to(task)))
        if vocabulary_hit:
            label_words.append(str(f.relative_to(task)))
    # Evidence sentences may appear legitimately inside the paper text itself (that is the source).
    # They must not appear in any non-paper file, where they would be an answer key.
    for f in files:
        rel = str(f.relative_to(task))
        source_patterns = [SOURCE_FILE_RE.get('C2' if source == 'CUAD' else source, SOURCE_FILE_RE['B'])
                           for source in family.split('+')]
        if any(re.search(pattern, rel, re.I) for pattern in source_patterns):
            continue
        hits = set()
        try:
            for raw in scan_chunks(f, max((len(p) for p in pieces), default=4096)):
                hits.update(p for p in pieces if p in raw)
        except Exception:
            pass
        if hits:
            text_hits.append({'file': rel, 'n': len(hits), 'example': sorted(hits)[0][:120]})
    return {'public_files': len(files), 'public_bytes': sum(f.stat().st_size for f in files),
            'evidence_sentences_in_non_paper_files_WARNING': text_hits, 'private_strings': private_hits,
            'label_vocabulary': label_words,
            # Evidence sentences are paper text; a label-free heuristic quoting them is not a leak by itself.
            # The authoritative leak test is the label-permutation rebuild; this scan only blocks private strings.
            'pass': not private_hits}


def file_digest(p: Path) -> str:
    """Frozen weekend criterion is byte identity, including Office metadata."""
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def permutation_leak_test(src: Path, ctl: Path, seed: int, base_public: Path) -> dict:
    """Rebuild with expert labels/evidence shuffled; solver-visible files must not change."""
    alt = ctl / 'perm_task'
    shutil.copytree(src, alt)
    r = run_py(['build/build.py', '--out', str(alt), '--seed', str(seed)], alt, [alt], ctl / 'perm_build.log',
               extra_env={'TADA_LABEL_PERMUTE': '991'})
    if r['exit'] != 0:
        return {'pass': False, 'error': 'permuted build failed', 'run': r}
    a_files = {str(p.relative_to(base_public)): p for p in base_public.rglob('*') if p.is_file()}
    pub_b = alt / 'environment/public'
    b_files = {str(p.relative_to(pub_b)): p for p in pub_b.rglob('*') if p.is_file()}
    only_a, only_b = sorted(set(a_files) - set(b_files)), sorted(set(b_files) - set(a_files))
    changed = sorted(k for k in set(a_files) & set(b_files) if file_digest(a_files[k]) != file_digest(b_files[k]))
    if file_digest(base_public.parent.parent / 'instruction.md') != file_digest(alt / 'instruction.md'):
        changed.append('../instruction.md')
    gold_changed = (alt / 'tests/gold.json').exists() and (alt / 'tests/gold.json').read_bytes() != (ctl / 'task/tests/gold.json').read_bytes()
    shutil.rmtree(alt, ignore_errors=True)
    return {'pass': not (only_a or only_b or changed), 'files_compared': len(a_files),
            'label_dependent_files': changed[:30], 'n_label_dependent': len(changed),
            'only_in_real': only_a[:10], 'only_in_permuted': only_b[:10],
            'gold_changed_under_permutation': gold_changed}


def wrong_answer_result(task, name, scored, expected, score_directory):
    """Preserve legacy reward tests; v5 scope tests use complete scorer details."""
    task, score_directory = Path(task), Path(score_directory)
    if json.loads((task / 'tests/gold.json').read_text()).get('schema_version') == 5:
        sys.path.insert(0, str(HERE / 'v5'))
        from tada_taskgen.gates import evaluate_wrong_answer
        try:
            details = json.loads((score_directory / 'details.json').read_text()) if scored.get('exit') == 0 else {}
        except (OSError, ValueError):
            details = {}
        thresholds = json.loads((task / 'tests/scenario_spec.json').read_text())['thresholds']
        return evaluate_wrong_answer(name, scored['reward'], details, expected, thresholds)
    lo, hi = expected
    return {'reward': scored['reward'], 'expected': expected,
            'in_range': scored['reward'] is not None and lo - 1e-9 <= scored['reward'] <= hi + 1e-9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--no-harbor', action='store_true', help='skip harbor oracle/nop trials')
    ap.add_argument('--output', type=Path, help='new controller output directory; existing outputs are never overwritten')
    a = ap.parse_args()
    run = Path(a.run).resolve()
    family = family_of(run)
    src = run / 'task'
    ctl = (a.output or run / 'controller').resolve()
    if ctl.exists():
        raise FileExistsError(f'preserve existing results: {ctl}; choose a fresh --output')
    ctl.mkdir(parents=True)
    task = ctl / 'task'
    shutil.copytree(src, task)
    # Normalise to the layout verified with plain `harbor run` (no compose overrides; files readable by uid 65532).
    for f in ['environment/docker-compose.yaml', 'tests/docker-compose.yaml']:
        (task / f).unlink(missing_ok=True)
    tmpl = (HERE / 'harbor_templates.md').read_text()
    for target, lang in [('environment/Dockerfile', 'environment/Dockerfile'), ('tests/Dockerfile', 'tests/Dockerfile')]:
        m = re.search(r'`' + re.escape(lang) + r'`\n```dockerfile\n(.*?)```', tmpl, re.S)
        (task / target).write_text(m.group(1))
        if target == 'tests/Dockerfile' and (task / 'tests/benchmark_checkers.py').exists():
            with (task / target).open('a') as handle:
                handle.write('\nCOPY benchmark_checkers.py benchmark_oolong_score.py benchmark_sembench_score.py /opt/\n')
                if (task / 'tests/input_assets').exists():
                    handle.write('COPY input_assets/ /environment/public/inputs/\n')
    require('offline_runtime', 'datasets_python')
    for d in ['environment', 'tests']:
        (task / d).mkdir(exist_ok=True)
        if (task / d / 'runtime').exists():
            shutil.rmtree(task / d / 'runtime')
        shutil.copytree(RUNTIME, task / d / 'runtime')
        if d == 'tests' and (task / 'tests/benchmark_checkers.py').exists():
            # The original Oolong date parser is required in the offline verifier.
            candidates = sorted((Path(PY).parent.parent / 'lib').glob('python*/site-packages'))
            runtime_root = next((p for p in candidates if (p / 'dateutil').is_dir()), None)
            if runtime_root is None:
                raise ValueError('Configured datasets_python has no dateutil verifier runtime')
            shutil.copytree(runtime_root / 'dateutil', task / d / 'runtime/dateutil',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            shutil.copyfile(runtime_root / 'six.py', task / d / 'runtime/six.py')
    rep: dict = {'run': str(run), 'seed': a.seed, 'family': family, 'steps': {},
                 'controller_revision': file_digest(Path(__file__)),
                 'frozen_gate_source': 'Fixed baseline, sensitivity, leakage and output-contract policy'}

    # 1 static
    req = json.loads((HERE / 'required_files.json').read_text())
    missing = [p for p in req if not (src / p).exists()]
    try:
        cfg = tomllib.loads((src / 'task.toml').read_text()); toml_ok, artifacts = True, cfg.get('artifacts')
    except Exception as e:
        toml_ok, artifacts = f'error: {e}', None
    rep['steps']['static'] = {'missing': missing, 'task_toml': toml_ok, 'artifacts': artifacts}

    # 2 build
    b = run_py(['build/build.py', '--out', str(task), '--seed', str(a.seed)], task, [task], ctl / 'build.log')
    gold = task / 'tests/gold.json'
    refs = sorted((task / 'solution').glob('reference.*'))
    b.update({'gold_exists': gold.exists(), 'reference': [r.name for r in refs],
              'public_files': len(list((task / 'environment/public').rglob('*'))) if (task / 'environment/public').exists() else 0})
    rep['steps']['build'] = b
    if b['exit'] != 0 or not gold.exists() or not refs:
        rep['verdict'] = 'build_failed'
        (ctl / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1)); print(json.dumps(rep, ensure_ascii=False, indent=1)); return

    # 3 leak: string scan + label-permutation rebuild
    rep['steps']['leak'] = leak_scan(task, family)
    rep['steps']['leak_permutation'] = permutation_leak_test(src, ctl, a.seed, task / 'environment/public')

    # 4 scorer: reference = 1, missing = 0, empty workbook = ~0
    ref = refs[0]
    s_ref = score(task, ref, ctl / 'score_ref', ctl / 'score_ref.log')
    s_none = score(task, None, ctl / 'score_missing', ctl / 'score_missing.log')
    empty = ctl / f'empty{ref.suffix}'
    if ref.suffix == '.xlsx':
        sys.path.insert(0, str(RUNTIME)); import openpyxl  # noqa
        try:
            wb = openpyxl.load_workbook(ref)
        except Exception as exc:  # noqa: BLE001 — unreadable reference: record and stop (fail closed)
            rep['steps']['scorer'] = {'reference': s_ref, 'missing': s_none, 'pass': False,
                                      'error': f'reference workbook unreadable by openpyxl: {type(exc).__name__}: {exc}'[:500]}
            rep['verdict'] = 'reference_unreadable'
            (ctl / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1))
            print(json.dumps({'verdict': rep['verdict'], 'error': rep['steps']['scorer']['error']}, ensure_ascii=False))
            return
        for ws in wb.worksheets:
            if ws.max_row > 1:
                ws.delete_rows(2, ws.max_row)
        wb.save(empty)
    s_empty = score(task, empty, ctl / 'score_empty', ctl / 'score_empty.log') if empty.exists() else None
    rep['steps']['scorer'] = {'reference': s_ref, 'missing': s_none, 'headers_only': s_empty,
                              'pass': s_ref['exit'] == 0 and s_ref['reward'] == 1.0 and
                                      s_none['exit'] == 0 and s_none['reward'] == 0.0 and
                                      s_empty is not None and s_empty['exit'] == 0 and s_empty['reward'] == 0.0}

    # 4b unknown-row precision: append rows with unknown keys to every sheet of the reference
    if ref.suffix == '.xlsx':
        import openpyxl  # noqa
        wb = openpyxl.load_workbook(ref)
        for ws in wb.worksheets:
            if ws.max_row < 2:
                continue
            last = [c.value for c in ws[ws.max_row]]
            for k in range(max(3, ws.max_row // 5)):
                row = []
                for v in last:
                    if isinstance(v, str) and re.search(r'[A-Z]{2,}[-_]?\d{3,}', v):
                        row.append(re.sub(r'\d+', lambda m: '9' * len(m.group()), v) + f'X{k}')
                    else:
                        row.append(v)
                ws.append(row)
        padded = ctl / f'reference_plus_unknown{ref.suffix}'
        wb.save(padded)
        s_pad = score(task, padded, ctl / 'score_padded', ctl / 'score_padded.log')
        rep['steps']['unknown_rows'] = {'reward': s_pad['reward'], 'pass': s_pad['reward'] is not None and s_pad['reward'] < 0.999}

    # 4d field ablation: every template column is either scored or declared unscored; one bad cell stays local.
    if ref.suffix == '.xlsx':
        import openpyxl  # noqa
        base = s_ref['reward'] or 0.0
        ablation, local = [], []
        wb0 = openpyxl.load_workbook(ref)
        for ws0 in wb0.worksheets:
            if ws0.max_row < 2:
                continue
            for col in range(1, ws0.max_column + 1):
                header = ws0.cell(1, col).value
                for mode in ['blank', 'garbage', 'negated', 'negated_symbol', 'negated_cjk']:
                    wb = openpyxl.load_workbook(ref)
                    ws = wb[ws0.title]
                    changed = 0
                    for r in range(2, ws.max_row + 1):
                        v = ws.cell(r, col).value
                        if mode.startswith('negated'):
                            if v is not None and str(v).strip():
                                prefix = {'negated': 'not ', 'negated_symbol': '≠', 'negated_cjk': '不是 '}[mode]
                                ws.cell(r, col).value = prefix + str(v); changed += 1
                        else:
                            ws.cell(r, col).value = None if mode == 'blank' else 'XX-UNKNOWN'; changed += 1
                    if not changed:
                        continue
                    f = ctl / f'ablate_{len(ablation)}_{mode}{ref.suffix}'
                    wb.save(f)
                    s = score(task, f, ctl / f'score_ablate_{len(ablation)}_{mode}', ctl / f'score_ablate.log')
                    ablation.append({'sheet': ws0.title, 'column': header, 'mode': mode, 'reward': s['reward']})
                # one cell only, with a pathological value: the drop should be small, never a global zero
                wb = openpyxl.load_workbook(ref)
                ws = wb[ws0.title]
                ws.cell(2, col).value = '9' * 400 + '.0'
                f = ctl / f'local_{ws0.title[:10]}_{col}{ref.suffix}'
                wb.save(f)
                s = score(task, f, ctl / 'score_local', ctl / 'score_local.log')
                local.append({'sheet': ws0.title, 'column': header, 'reward': s['reward']})
        unscored = sorted({(a['sheet'], a['column']) for a in ablation
                           if a['mode'] == 'garbage' and a['reward'] is not None and a['reward'] >= base - 1e-9})
        worst_local = min([x['reward'] for x in local if x['reward'] is not None] or [0.0])
        neg_accepted = sorted({(a['sheet'], a['column']) for a in ablation if a['mode'].startswith('negated')
                               and a['reward'] is not None and a['reward'] >= base - 1e-9}
                              - set(unscored))
        rep['steps']['field_ablation'] = {
            'columns_with_no_effect_when_garbled': [list(u) for u in unscored],
            'scored_columns_accepting_negated_values': [list(u) for u in neg_accepted],
            'single_cell_min_reward': worst_local, 'single_cell': local, 'runs': ablation,
            # Unscored columns are allowed only if design.md declares them unscored; the reviewer checks that.
            'pass': worst_local >= 0.8 and not neg_accepted}

    # 4e provenance: each scored key names its source records and the solver-visible file holding the evidence.
    prov_f = task / 'tests/provenance.json'
    if prov_f.exists():
        spans = evidence_index(family, task)
        norm = lambda t: re.sub(r'\W+', ' ', t or '').lower().strip()
        prov = json.loads(prov_f.read_text())
        rows, found, missing_file, examples = 0, 0, 0, []
        cache: dict = {}
        for key, ent in (prov.get('rows') or prov).items():
            rows += 1
            vf = task / 'environment/public' / ent.get('visible_file', '')
            if not vf.is_file():
                missing_file += 1; continue
            if vf not in cache:
                cache[vf] = norm(vf.read_text('utf8', 'ignore'))
            ok = any(norm(s)[:120] in cache[vf] for sid in ent.get('source_ids', []) for s in spans.get(str(sid), []) if s)
            found += ok
            if not ok and len(examples) < 10:
                examples.append(key)
        rep['steps']['provenance'] = {'scored_keys': rows, 'evidence_found_in_visible_file': found,
                                      'visible_file_missing': missing_file, 'not_found_examples': examples,
                                      'share_found': round(found / rows, 3) if rows else None,
                                      'pass': rows > 0 and found / rows >= 0.9,
                                      # Legacy single-visible_file manifest check. Manifests with per-dependency
                                      # visible files are checked by strict_provenance, which alone feeds the
                                      # evidence_present_for_scored_keys gate.
                                      'superseded_by': 'strict_provenance', 'used_for_verdict': False}
    else:
        rep['steps']['provenance'] = {'pass': False, 'error': 'tests/provenance.json missing'}

    # 4c clean rebuild determinism (same seed, fresh copy) - solver-visible files identical
    alt = ctl / 'rebuild_task'
    shutil.copytree(src, alt)
    rb = run_py(['build/build.py', '--out', str(alt), '--seed', str(a.seed)], alt, [alt], ctl / 'rebuild.log')
    pub_a, pub_b = task / 'environment/public', alt / 'environment/public'
    fa = {str(p.relative_to(pub_a)): file_digest(p) for p in pub_a.rglob('*') if p.is_file()}
    fb = {str(p.relative_to(pub_b)): file_digest(p) for p in pub_b.rglob('*') if p.is_file()} if pub_b.exists() else {}
    diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    rep['steps']['determinism'] = {'rebuild_exit': rb['exit'], 'n_diff': len(diff), 'diff': diff[:15], 'pass': rb['exit'] == 0 and not diff}
    shutil.rmtree(alt, ignore_errors=True)

    # 5 wrong answers
    wdir = ctl / 'wrong'
    w = run_py(['build/wrong_answers.py', '--task', str(task), '--out', str(wdir)], task, [task, wdir], ctl / 'wrong.log')
    wres = {}
    exp_file = wdir / 'expected.json'
    if w['exit'] == 0 and exp_file.exists():
        expected = json.loads(exp_file.read_text())
        for wid, rng in expected.items():
            f = next(iter(sorted(wdir.glob(f'{wid}.*'))), None)
            if f is None or f.name == 'expected.json':
                wres[wid] = {'error': 'file missing'}; continue
            s = score(task, f, ctl / f'score_{wid}', ctl / f'score_{wid}.log')
            wres[wid] = wrong_answer_result(task, wid, s, rng, ctl / f'score_{wid}')
    rep['steps']['wrong_answers'] = {'run': w, 'results': wres,
                                     'all_in_range': bool(wres) and all(v.get('in_range') for v in wres.values())}

    # 6 keyword baseline (sees only the public files)
    bl_out = ctl / f'keyword_baseline{ref.suffix}'
    pub_copy = ctl / 'public_ro'
    shutil.copytree(task / 'environment/public', pub_copy)
    blr = run_py([str(task / 'build/keyword_baseline.py'), '--public', str(pub_copy), '--out', str(bl_out)],
                 ctl, [ctl / 'bl_tmp', bl_out.parent], ctl / 'baseline.log')
    sb = score(task, bl_out, ctl / 'score_baseline', ctl / 'score_baseline.log') if bl_out.exists() else None
    details = None
    if sb and (ctl / 'score_baseline/details.json').exists():
        details = json.loads((ctl / 'score_baseline/details.json').read_text())
    rep['steps']['keyword_baseline'] = {'run': blr, 'reward': sb and sb['reward'], 'details': details,
                                        'gate': GATE_BASELINE,
                                        'pass': blr['exit'] == 0 and sb is not None and sb['exit'] == 0 and
                                                sb['reward'] is not None and sb['reward'] <= GATE_BASELINE}

    # 7 sensitivity
    rep['steps']['sensitivity'] = run_py(['build/sensitivity.py', '--task', str(task)], task, [task], ctl / 'sensitivity.log')

    # Controller-owned adversarial gates. Legacy measurements above remain visible,
    # but never substitute for full-denominator negative provenance or exact-instance replay.
    from weekend_checks import (unknown_matrix, strict_provenance, revision_effects, run_binding_and_independent,
                                extra_rows_nonperfect, extra_rows_tail_gap)
    import traceback

    def guarded(fn, *args):
        """A controller exception is a FAILED check with its traceback recorded, never a pass and never a crash."""
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            return {'pass': False, 'controller_error': f'{type(exc).__name__}: {exc}',
                    'traceback': traceback.format_exc()[-2500:]}
    rep['steps']['unknown_value_matrix'] = guarded(unknown_matrix, task, ref, ctl / 'unknown_matrix', score)
    rep['steps']['extra_rows_nonperfect'] = guarded(extra_rows_nonperfect, task, ref,
                                                    bl_out if bl_out.exists() else None, sb and sb['reward'],
                                                    ctl / 'extra_rows_nonperfect', score)
    rep['steps']['extra_rows_tail_gap'] = guarded(extra_rows_tail_gap, task, ref, ctl / 'extra_rows_tail_gap', score)
    rep['steps']['strict_provenance'] = guarded(lambda: strict_provenance(task, family, evidence_index(family, task)))
    rep['steps']['revision_effects'] = guarded(revision_effects, wres)
    pair = guarded(run_binding_and_independent, task, ctl / 'replay', a.seed, family, run_py, score, wres)
    binding, independent = pair if isinstance(pair, tuple) else (pair, pair)
    rep['steps']['exact_instance_binding'] = binding
    rep['steps']['independent_then_aggregate'] = independent
    # Cost of one misread upstream label (added 2026-11-11 for MAUD r3; gate declared before any r3 measurement).
    if family in ('MAUD', 'MAUDB', 'MAUDC') or json.loads(gold.read_text()).get('schema_version') == 5:
        lf_out = ctl / 'label_flip'
        lf = run_py([str(HERE / 'label_flip_check.py'), '--task', str(task), '--out', str(lf_out), '--runtime', str(RUNTIME)],
                    task, [lf_out], ctl / 'label_flip.log', timeout=3600)
        try:
            # Read the full on-disk result: stdout_tail is truncated to 1,500 chars and long reports broke parsing.
            full = json.loads((lf_out / 'label_flip.json').read_text())
            full.pop('all', None)
            rep['steps']['label_flip'] = {**full, 'run_exit': lf['exit']}
        except Exception:  # noqa: BLE001
            rep['steps']['label_flip'] = {'pass': False, 'error': 'label_flip_check failed', 'run': lf}
    (ctl / 'report.partial.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1))

    # 8 harbor
    require('harbor_binary')
    h = subprocess.run([HARBOR, 'lint', '--json', str(task)], capture_output=True, text=True, timeout=300,
                       env={**os.environ, 'COLUMNS': '200'})
    try:
        findings = json.loads(h.stdout or '[]')
    except json.JSONDecodeError:
        findings = h.stdout[-1500:]
    rep['steps']['harbor_lint'] = {'exit': h.returncode, 'findings': findings}
    if not a.no_harbor:
        for p in task.rglob('*'):
            p.chmod(p.stat().st_mode | 0o044 | (0o011 if p.is_dir() else 0))
        trials = {}
        for agent in ['oracle', 'nop']:
            jobs = ctl / 'jobs'
            p = subprocess.run([HARBOR, 'run', '-p', str(task), '-a', agent, '-o', str(jobs), '--job-name', agent,
                                '-n', '1', '-y', '-q'], capture_output=True, text=True, timeout=3600,
                               env={k: v for k, v in os.environ.items() if k != 'PYTHONPATH'})
            res = []
            for r in (jobs / agent).rglob('result.json'):
                if r.parent.name == agent:
                    continue
                d = json.loads(r.read_text())
                res.append({'reward': ((d.get('verifier_result') or {}).get('rewards') or {}).get('reward'),
                            'exception': ((d.get('exception_info') or {}).get('exception_message') or '')[:400]})
            trials[agent] = {'exit': p.returncode, 'trials': res}
        rep['steps']['harbor_trials'] = trials
        rep['steps']['harbor_trials']['pass'] = (
            [t['reward'] for t in trials['oracle']['trials']] == [1.0] and
            [t['reward'] for t in trials['nop']['trials']] == [0.0])

    s = rep['steps']
    rep['verdict'] = {
        'build': s['build']['exit'] == 0, 'leak': s['leak']['pass'], 'leak_permutation': s['leak_permutation']['pass'],
        'scorer': s['scorer']['pass'], 'unknown_rows_penalised': s.get('unknown_rows', {}).get('pass'),
        'single_bad_cell_stays_local': s.get('field_ablation', {}).get('pass'),
        'evidence_present_for_scored_keys': s['strict_provenance']['pass'],
        'unknown_all_values_penalised': s['unknown_value_matrix']['pass'],
        'revision_nonzero_effects': s['revision_effects']['pass'],
        'exact_instance_binding': s['exact_instance_binding']['pass'],
        'extra_rows_never_raise_score': s['extra_rows_nonperfect']['pass'],
        'extra_rows_after_blank_gap_penalised': s['extra_rows_tail_gap']['pass'],
        # Category 'parallel_coding' (Richard, 2026-11-11): the task is parallel per-document coding plus source
        # selection and does not claim sequential dependency; the 0.60 gate is reported, not gated.
        **({'independent_then_aggregate_reported_only': True} if task_category(task) == 'parallel_coding'
           else {'independent_then_aggregate_below_gate': s['independent_then_aggregate']['pass']}),
        **({'one_label_error_bounded': s['label_flip'].get('pass')} if 'label_flip' in s else {}),
        'determinism': s['determinism']['pass'],
        'wrong_answers': s['wrong_answers']['all_in_range'], 'keyword_baseline_below_gate': s['keyword_baseline']['pass'],
        'sensitivity': s['sensitivity']['exit'] == 0, 'harbor_lint': s['harbor_lint']['exit'] == 0 and not findings,
        'harbor_oracle_nop': (s.get('harbor_trials') or {}).get('pass'),
    }
    (ctl / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    if json.loads(gold.read_text()).get('schema_version') == 5:
        sys.path.insert(0, str(HERE / 'v5'))
        from tada_taskgen.gates import extensions
        try:
            extensions(task, rep)
        except Exception as exc:
            rep['steps']['v5_extensions'] = {'pass': False, 'error': f'{type(exc).__name__}: {exc}'}
            from tada_taskgen.hard_gates import NAMES
            rep['reported_only'] = {'verdict': rep['verdict']}
            rep['hard_gates'] = {name: {'pass': False, 'error': 'hard gate evaluation failed'} for name in NAMES}
            rep['verdict'] = {**{name: False for name in NAMES}, 'reported_only': rep['reported_only']['verdict']}
            rep['pass'] = False
    rep['category'] = task_category(task)
    (ctl / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({'category': rep['category'], 'verdict': rep['verdict'], 'baseline': s['keyword_baseline']['reward'],
                      'unscored_columns': s.get('field_ablation', {}).get('columns_with_no_effect_when_garbled'),
                      'negated_values_accepted': s.get('field_ablation', {}).get('scored_columns_accepting_negated_values'),
                      'provenance_share_found': s.get('provenance', {}).get('share_found'),
                      'single_cell_min_reward': s.get('field_ablation', {}).get('single_cell_min_reward'),
                      'scorer': {k: (v or {}).get('reward') for k, v in s['scorer'].items() if k != 'pass'},
                      'wrong': {k: v.get('reward') for k, v in s['wrong_answers']['results'].items()}}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
