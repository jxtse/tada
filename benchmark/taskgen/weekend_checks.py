"""Controller-owned checks; never repairs frozen packages or changes scoring.

Legacy adapters are explicit. Unsupported contracts fail closed rather than awarding
an unmeasured pass. Generated code is executed only through controller.run_py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision_effects(results):
    cases = {k: v for k, v in results.items() if k.startswith('ignore_')}
    ineffective = [k for k, v in cases.items() if v.get('reward') == 1.0]
    return {'pass': bool(cases) and not ineffective, 'ineffective': ineffective,
            'tested': sorted(cases), 'criterion': 'declared revision omissions must lose credit'}


def unknown_matrix(task, ref, out, score):
    """Test every categorical value separately, not a mixed padded submission.

    A controller-independent schema is required for non-legacy task shapes.
    Unknown, known-out-of-scope, duplicate and empty identifiers are all tested.
    """
    import openpyxl
    out.mkdir(parents=True, exist_ok=True)
    gold = json.loads((task / 'tests/gold.json').read_text())
    wb = openpyxl.load_workbook(ref)
    spec_path = task / 'tests/validation_contract.json'
    if spec_path.exists():
        schema = json.loads(spec_path.read_text())['sheets']
    elif 'Schedule' in wb.sheetnames and 'tracks' in gold:
        schema = {'Schedule': {'key_columns': ['Contract Ref'],
                              'categorical_values': {'Review Track': gold['tracks'],
                                                     'Transfer Method': gold['methods']},
                              'out_of_scope_keys': [[k] for k in gold['dataroom_refs'] if k not in gold['rows']]}}
    else:
        return {'pass': False, 'error': 'tests/validation_contract.json missing for this schema'}
    runs = []
    for sheet, spec in schema.items():
        ws = wb[sheet]
        headers = [c.value for c in ws[1]]
        if ws.max_row < 2:
            return {'pass': False, 'error': f'no reference row for {sheet}'}
        base = [c.value for c in ws[2]]
        key_cols = [headers.index(k) for k in spec['key_columns']]
        variants = [('reference_values', None, None)]
        for col, values in spec['categorical_values'].items():
            variants.extend((f'{col}={v}', headers.index(col), v) for v in [*values, '', 'INVALID'])
        keys = [('unknown', ['__UNKNOWN__'] * len(key_cols)),
                ('duplicate', [base[c] for c in key_cols]),
                ('blank', [''] * len(key_cols))]
        keys += [('out_of_scope', k) for k in spec.get('out_of_scope_keys', [])]
        for kind, key in keys:
            for name, col, value in variants:
                row = base.copy()
                for i, k in zip(key_cols, key):
                    row[i] = k
                if col is not None:
                    row[col] = value
                if all(v is None or str(v).strip() == '' for v in row):
                    # A wholly blank row is spreadsheet padding, not a submitted row (review of PREC r1, F07).
                    continue
                # Position matters: scorers that search for the header can silently drop rows above it.
                positions = ['end'] + (['above_header', 'formula_end'] if name == 'reference_values' else [])
                for pos in positions:
                    test = openpyxl.load_workbook(ref)
                    if pos == 'end':
                        test[sheet].append(row)
                    elif pos == 'formula_end':
                        # Same row written as formulas without cached values (what openpyxl produces). A
                        # non-empty row must not escape scoring because the scorer only reads cached values.
                        if all(v is None or str(v).strip() == '' for v in row):
                            continue
                        test[sheet].append([None if v is None or str(v).strip() == '' else
                                            (f'={v}' if isinstance(v, (int, float)) else '="' + str(v).replace('"', '""') + '"')
                                            for v in row])
                    else:
                        test[sheet].insert_rows(1)
                        for j, v in enumerate(row, start=1):
                            test[sheet].cell(1, j).value = v
                    dest = out / f'case_{len(runs):04}.xlsx'
                    test.save(dest)
                    result = score(task, dest, out / f'score_{len(runs):04}', out / f'case_{len(runs):04}.log')
                    reward = result['reward']
                    runs.append({'sheet': sheet, 'kind': kind, 'keys': key, 'value': name, 'position': pos,
                                 'reward': reward,
                                 'pass': result['exit'] == 0 and reward is not None and reward < 1.0})
                continue
    return {'pass': bool(runs) and all(r['pass'] for r in runs), 'cases': len(runs),
            'failures': [r for r in runs if not r['pass']], 'runs': runs}


def extra_rows_nonperfect(task, ref, base_xlsx, base_reward, out, score, n_rows=(1, 30)):
    """Run the extra-row check from two non-perfect starting points: the keyword baseline and a
    constant-majority workbook (every categorical cell set to that column's most common reference value),
    where agreement statistics such as kappa sit at 0 and are most exposed."""
    import collections, openpyxl
    out.mkdir(parents=True, exist_ok=True)
    schema = json.loads((task / 'tests/validation_contract.json').read_text())['sheets']
    wb = openpyxl.load_workbook(ref)
    for sheet, spec in schema.items():
        ws = wb[sheet]
        headers = [c.value for c in ws[1]]
        for col in spec['categorical_values']:
            j = headers.index(col) + 1
            vals = [ws.cell(r, j).value for r in range(2, ws.max_row + 1)]
            common = collections.Counter(v for v in vals if v not in (None, '')).most_common(1)
            for r in range(2, ws.max_row + 1):
                if common:
                    ws.cell(r, j).value = common[0][0]
    const = out / 'constant_majority.xlsx'
    wb.save(const)
    cs = score(task, const, out / 'score_constant', out / 'score_constant.log')
    results = {'keyword_baseline': _extra_rows_from(task, ref, base_xlsx, base_reward, out / 'from_baseline', score, n_rows)}
    if cs['reward'] is not None and cs['reward'] < 1.0:
        results['constant_majority'] = _extra_rows_from(task, ref, const, cs['reward'], out / 'from_constant', score, n_rows)
    return {'pass': all(r['pass'] for r in results.values()), 'starts': results}


def extra_rows_tail_gap(task, ref, out, score, n=30, gap=25000):
    """30 unknown-key rows appended right after the table vs after a long blank gap must score the same:
    a scorer that stops reading after N blank rows (or caps the row index) must not compress the penalty."""
    import openpyxl
    out.mkdir(parents=True, exist_ok=True)
    schema = json.loads((task / 'tests/validation_contract.json').read_text())['sheets']
    runs = []
    for sheet, spec in schema.items():
        res = {}
        for g in (0, gap):
            wb = openpyxl.load_workbook(ref)
            ws = wb[sheet]
            hdr = [c.value for c in ws[1]]
            base = [c.value for c in ws[2]]
            start = ws.max_row + 1 + g
            for i in range(n):
                for j, (h, v) in enumerate(zip(hdr, base), start=1):
                    ws.cell(start + i, j).value = f'__TAIL{i:03}__' if h in spec['key_columns'] else v
            dest = out / f'{sheet.replace(" ", "_")}_{g}.xlsx'
            wb.save(dest)
            res[g] = score(task, dest, out / f'score_{sheet.replace(" ", "_")}_{g}', out / f'{sheet}_{g}.log')['reward']
        ok = res[0] is not None and res[gap] is not None and res[gap] <= res[0] + 1e-9 and res[0] < 1.0
        runs.append({'sheet': sheet, 'adjacent': res[0], 'after_gap': res[gap], 'pass': ok})
    return {'pass': all(r['pass'] for r in runs), 'rows': n, 'gap': gap, 'runs': runs}


def _extra_rows_from(task, ref, base_xlsx, base_reward, out, score, n_rows=(1, 30)):
    """Extra rows must never RAISE a non-perfect score (agreement statistics such as kappa can).

    Starting from a non-perfect workbook (the keyword baseline), append n unknown-key rows carrying each
    declared categorical value (and blank / INVALID), one sheet and one value at a time.
    """
    import openpyxl
    if base_xlsx is None or not Path(base_xlsx).exists() or base_reward is None:
        return {'pass': False, 'error': 'no non-perfect baseline workbook to test'}
    out.mkdir(parents=True, exist_ok=True)
    schema = json.loads((task / 'tests/validation_contract.json').read_text())['sheets']
    refwb = openpyxl.load_workbook(ref)
    runs = []
    for sheet, spec in schema.items():
        rws = refwb[sheet]
        headers = [c.value for c in rws[1]]
        template = [c.value for c in rws[2]] if rws.max_row >= 2 else [None] * len(headers)
        key_cols = [headers.index(k) for k in spec['key_columns']]
        values = [(col, v) for col, vals in spec['categorical_values'].items() for v in [*vals, '', 'INVALID']]
        for col, v in values or [(None, None)]:
            for n in n_rows:
                wb = openpyxl.load_workbook(base_xlsx)
                if sheet not in wb.sheetnames:
                    return {'pass': False, 'error': f'baseline workbook lacks sheet {sheet}'}
                ws = wb[sheet]
                bh = [c.value for c in ws[1]]
                for i in range(n):
                    row = template.copy()
                    for j, kc in enumerate(key_cols):
                        row[kc] = f'__EXTRA{i:03}_{j}__'
                    if col is not None:
                        row[headers.index(col)] = v
                    ws.append([row[headers.index(h)] if h in headers else None for h in bh])
                dest = out / f'case_{len(runs):04}.xlsx'
                wb.save(dest)
                r = score(task, dest, out / f'score_{len(runs):04}', out / f'case_{len(runs):04}.log')
                runs.append({'sheet': sheet, 'column': col, 'value': v, 'rows': n, 'reward': r['reward'],
                             'pass': r['exit'] == 0 and r['reward'] is not None and r['reward'] <= base_reward + 1e-9})
    return {'pass': bool(runs) and all(x['pass'] for x in runs), 'base_reward': base_reward, 'cases': len(runs),
            'failures': [x for x in runs if not x['pass']][:20],
            'max_reward': max((x['reward'] for x in runs if x['reward'] is not None), default=None)}


def strict_provenance(task, family, evidence):
    """Full gold denominator. Negative labels require source identity, not a fake span.

    New manifest rows must contain dependencies with source_id, label, visible_file.
    C2 requires all category judgements for each candidate, including excluded rows;
    source text identity is checked for negative AND positive labels. Cross-record
    applicability is still an independent-review obligation, not proved by presence.
    """
    import re
    import sys
    from controller import SOURCES
    sys.path.insert(0, str(SOURCES))
    norm = lambda s: re.sub(r'\s+', ' ', str(s)).strip()
    gold = json.loads((task / 'tests/gold.json').read_text())
    pf = task / 'tests/provenance.json'
    if not pf.exists():
        return {'pass': False, 'error': 'provenance missing'}
    prov = json.loads(pf.read_text())
    rows = prov.get('rows', prov)

    def gold_keys(g):
        """Scored keys. Preferred: every row of the reference workbook, keyed by validation_contract key_columns
        as 'Sheet|k1|k2' (independent of the author's gold layout). Fallback: legacy {'rows'} / {'sheets'} gold."""
        vcp = task / 'tests/validation_contract.json'
        refs = sorted((task / 'solution').glob('reference.xlsx'))
        if vcp.exists() and refs:
            import openpyxl
            wb = openpyxl.load_workbook(refs[0], read_only=True, data_only=True)
            keys = set()
            for sheet, spec in json.loads(vcp.read_text()).get('sheets', {}).items():
                if sheet not in wb.sheetnames:
                    raise KeyError(f'reference lacks sheet {sheet}')
                rows_ = list(wb[sheet].iter_rows(values_only=True))
                hdr = [str(h).strip() if h is not None else '' for h in rows_[0]]
                idx = [hdr.index(k) for k in spec['key_columns']]
                for r in rows_[1:]:
                    vals = [str(r[i]).strip() if r[i] is not None else '' for i in idx]
                    if any(vals):
                        keys.add('|'.join([sheet, *vals]))
            return keys
        if 'rows' in g:
            return set(g['rows'])
        if 'sheets' in g:
            return {f'{s}|{k}' for s, spec in g['sheets'].items() for k in spec.get('rows', {})}
        raise KeyError('gold has neither rows nor sheets')
    is_v5 = gold.get('schema_version') == 5
    if is_v5:
        from controller import HERE
        sys.path.insert(0, str(HERE / 'v5'))
        from tada_taskgen.multisource import adapter_for, origin
        from tada_taskgen.adapters.base import normalize
        spec = json.loads((task / 'tests/scenario_spec.json').read_text())
        adapter = adapter_for(spec).reference()
        labels = adapter.label_pool()
        source = {}
        for cell in gold['cells']:
            pair = (cell['record'], cell['question'])
            expected_origin = origin(spec, *pair)
            if cell.get('origin', expected_origin) != expected_origin or cell.get('source', expected_origin['source']) != expected_origin['source']:
                continue
            if pair in labels:
                label, spans = labels[pair]
                source[f'{pair[0]}:{pair[1]}'] = {'label': label, 'spans': spans,
                                                'text': adapter.records()[pair[0]]['text']}
        required = gold_keys(gold)
    elif family == 'C2':
        import c2_sources
        source = {f"{c['contract_id']}:{cat}": {'label': val, 'text': c['text']}
                  for c in c2_sources.iter_contracts() for cat, val in c['labels'].items()}
        required = gold_keys(gold)
    elif family == 'CUADX':
        # Same record shape and checks as C2 (Yes: all expert spans visible; No: full contract text visible).
        import cuadx_sources
        source = {f"{c['contract_id']}:{cat}": {'label': val, 'text': c['text']}
                  for c in cuadx_sources.iter_contracts() for cat, val in c['labels'].items()}
        required = gold_keys(gold)
    elif family == 'B':
        import b_sources
        source = {c['prompt_id']: {'label': c['direction'], 'text': b_sources.paper_text(c['pmc'])}
                  for c in b_sources.iter_comparisons()}
        required = set(gold.get('endpoints', {})) | set(gold.get('trials', {}))
    elif family == 'PREC':
        import prec_sources
        pairs = prec_sources.overruling_pairs()
        overruled = {p['overruled'] for p in pairs}
        source = {f"{p['overruling']}>{p['overruled']}":
                  {'label': 'overruled_in_part' if p['in_part'] else 'overruled', 'decision': p['overruling']}
                  for p in pairs}
        required = gold_keys(gold)

        def prec_lookup(sid):
            if sid in source:
                return source[sid]
            if sid.startswith('not>'):
                # pair negative: decision A's majority does not overrule B according to the table
                parts = sid[4:].split('>')
                if len(parts) == 2 and f'{parts[0]}>{parts[1]}' not in source:
                    return {'label': 'not_listed', 'decision': parts[0]}
                return None
            if sid.startswith('none>'):
                cite = sid[5:]
                if cite.split('#')[0] not in overruled:
                    return {'label': 'not_listed', 'decision': cite}
            return None
    elif family == 'NTSB':
        import ntsb_sources
        factual = {e['ev_id']: e['factual'] for e in ntsb_sources.iter_events()}
        required = gold_keys(gold)

        def ntsb_lookup(sid):
            parts = sid.split(':')
            if len(parts) != 3 or parts[0] not in factual:
                return None
            ev, ak, rest = parts
            fs = ntsb_sources.findings(ev)
            if ak == 'none':
                if any(f['category'] == rest and f['cause_factor'] == 'C' for f in fs):
                    return None
                return {'label': 'none', 'text': factual[ev]}
            for f in fs:
                if f['aircraft_key'] == ak and f['no'] == rest:
                    return {'label': f['cause_factor'], 'text': factual[ev]}
            return None
    elif family == 'NDA':
        import nda_sources
        texts = {d['doc_id']: d['text'] for d in nda_sources.iter_documents()}
        hyps = set(nda_sources.hypotheses())
        # Optional declared, label-independent party renaming: tests/rename_map.json
        # {doc_id: {"map": [[old, new], ...]}} applied in order to the source text AND the evidence spans before
        # checking exact presence. The permutation rebuild already proves the map is label-independent.
        rmap_path = task / 'tests/rename_map.json'
        rmap = json.loads(rmap_path.read_text()) if rmap_path.exists() else {}

        def renamed(doc, s):
            for old, new in (rmap.get(str(doc)) or {}).get('map', []):
                s = s.replace(old, new)
            return s
        source = {}
        for doc, t in texts.items():
            for h in hyps:
                lab = nda_sources.label(doc, h)
                if lab:
                    # C2-style record: 'No'-like (NotMentioned) checked by full text, others by all spans
                    source[f'{doc}:{h}'] = {'label': lab[0], 'text': renamed(doc, t),
                                            'spans': [renamed(doc, s) for s in lab[1]]}
        required = gold_keys(gold)
    elif family in ('MAUD', 'MAUDB', 'MAUDC'):
        import maud_sources
        source = {f'{c}:{dp}': {'label': a, 'spans': segs} for (c, dp), (a, segs) in maud_sources._labels().items()}
        required = gold_keys(gold)
    else:
        raise ValueError(f'unsupported provenance family: {family}')
    # Negative scope rows (declared out-of-scope keys) must also carry provenance: excluding a row is a judgement.
    vc = task / 'tests/validation_contract.json'
    negatives_declared = set()
    if vc.exists():
        for sheet, spec in json.loads(vc.read_text()).get('sheets', {}).items():
            for k in spec.get('out_of_scope_keys', []):
                negatives_declared.add('|'.join([sheet, *map(str, k)]) if any('|' in r for r in required)
                                       else '|'.join(map(str, k)))
    required = required | negatives_declared
    missing = sorted(required - set(rows))
    failures, negative, positive = [], 0, 0
    public = (task / 'environment/public').resolve()
    for key in sorted(required & set(rows)):
        entry = rows[key]
        deps = entry.get('dependencies')
        if not deps:
            failures.append({'key': key, 'error': 'complete typed dependencies missing (legacy positive-only manifest)'})
            continue
        for dep in deps:
            sid = str(dep.get('source_id', ''))
            if dep.get('kind') == 'scenario':
                vf = (public / dep.get('visible_file', '')).resolve()
                quote = dep.get('quote', '')
                if not vf.is_relative_to(public) or not vf.is_file() or not quote or norm(quote) not in norm(vf.read_text()):
                    failures.append({'key': key, 'error': 'scenario dependency not grounded'})
                continue
            record = (source.get(sid) if is_v5 else prec_lookup(sid) if family == 'PREC' else ntsb_lookup(sid) if family == 'NTSB'
                      else source.get(sid))
            vf = (public / dep.get('visible_file', '')).resolve()
            if (record is None or record['label'] != dep.get('label')
                    or is_v5 and type(record['label']) is not type(dep.get('label'))):
                failures.append({'key': key, 'source_id': sid, 'error': 'unknown source or wrong label'})
                continue
            if not vf.is_relative_to(public) or not vf.is_file():
                failures.append({'key': key, 'source_id': sid, 'error': 'source file absent/outside public'})
                continue
            text = norm(vf.read_text())
            if is_v5:
                if record['label'] in ('No', 'NotMentioned', '0'):
                    negative += 1
                else:
                    positive += 1
                if norm(record['text']) not in text:
                    failures.append({'key': key, 'source_id': sid, 'error': 'full source text not present'})
                if not all(normalize(span) in normalize(text) for span in record['spans']):
                    failures.append({'key': key, 'source_id': sid, 'error': 'expert evidence not fully grounded'})
                continue
            if family in ('MAUD', 'MAUDB', 'MAUDC'):
                # Every MAUD answer, including "No", has lawyer-selected evidence; all segments must be visible.
                # Use the source module's own admission normalisation (lowercase alphanumerics; MAUD excerpts differ
                # from the agreement text in spacing/punctuation, e.g. "five(5)" vs "five (5)"). Whitespace-only
                # normalisation rejected segments that maud_sources had admitted against the same full text.
                if record['label'] == 'No':
                    negative += 1
                else:
                    positive += 1
                mnorm = maud_sources._norm
                mtext = mnorm(vf.read_text())
                if not all(mnorm(s) in mtext for s in record['spans']):
                    failures.append({'key': key, 'source_id': sid, 'error': 'evidence segments not all in visible file'})
                continue
            if family == 'NDA':
                if record['label'] == 'NotMentioned':
                    negative += 1
                    if norm(record['text']) not in text:
                        failures.append({'key': key, 'source_id': sid, 'error': 'NotMentioned: full NDA text not in visible file'})
                else:
                    positive += 1
                    sp = [s for s in record['spans'] if s.strip()]
                    if not sp or not all(norm(s) in text for s in sp):
                        failures.append({'key': key, 'source_id': sid, 'error': 'evidence spans not all in visible file'})
                continue
            if family == 'NTSB':
                if record['label'] == 'none':
                    negative += 1
                else:
                    positive += 1
                if norm(record['text']) not in text:
                    failures.append({'key': key, 'source_id': sid, 'error': 'full factual report not in visible file'})
                continue
            if family == 'PREC':
                decision = dep.get('decision') or record['decision']
                majority = [o['text'] for o in prec_sources.opinions(decision) if o['type'] == 'majority']
                if record['label'] == 'not_listed':
                    negative += 1
                else:
                    positive += 1
                if not majority or not all(norm(m)[:4000] in text for m in majority):
                    failures.append({'key': key, 'source_id': sid, 'error': 'majority opinion of evidence decision not in visible file'})
                continue
            is_negative = record['label'] == 'No'
            if is_negative:
                negative += 1
                if norm(record['text']) not in text:
                    failures.append({'key': key, 'source_id': sid, 'error': 'negative source full text not present'})
            else:
                positive += 1
                spans = evidence.get(sid, [])
                if not spans or not all(norm(s) in text for s in spans if s):
                    failures.append({'key': key, 'source_id': sid, 'error': 'expert evidence not fully grounded'})
    # No division by number of entries in the author's manifest.
    return {'pass': bool(required) and not missing and not failures, 'gold_keys': len(required - negatives_declared),
            'declared_out_of_scope_keys': len(negatives_declared),
            'manifest_keys': len(rows), 'missing_keys': missing, 'failures': failures,
            'negative_dependencies_checked': negative, 'positive_dependencies_checked': positive,
            'note': 'presence is not unique applicability; independent audit still required'}


LEGACY_REPLAY = {'B', 'C2'}


def run_binding_and_independent(task, out, seed, family, run_py, score, wrong=None):
    """Gate per STATUS 2026-11-07 00:25: per-file gold judgements + public rules, no cross-file facts.

    Legacy B/C2 use controller replay. New families must ship scenario_params.json and a
    wrong-answer fixture `independent_then_filter` built to that definition; the reviewer audits it.
    The 'strong' diagnostic (cross-file facts also given) is reported when available, never gated.
    """
    out.mkdir(parents=True, exist_ok=True)
    build_src = (task / 'build/build.py').read_text()
    legacy = family in LEGACY_REPLAY and (
        (family == 'C2' and 'def compute_all' in build_src and 'def gold_rows' in build_src) or
        (family == 'B' and 'def run_sequence' in build_src and 'EP_HEADERS' in build_src))
    if not legacy:
        params = task / 'tests/scenario_params.json'
        p = json.loads(params.read_text()) if params.exists() else {}
        # Seed may be top-level or nested under "params" (the prompt says "all effective parameters,
        # including seed" without fixing the location; NDA r1 used params.seed). Record where it was found.
        ps, where = p.get('seed'), 'seed'
        if ps is None and isinstance(p.get('params'), dict):
            ps, where = p['params'].get('seed'), 'params.seed'
        try:
            ok = ps is not None and int(ps) == int(seed)
        except (TypeError, ValueError):
            ok = False
        binding = {'pass': params.exists() and ok,
                   'scenario_params_present': params.exists(), 'actual_seed': seed, 'params_seed': ps,
                   'params_seed_location': where if ps is not None else None,
                   'note': 'sensitivity.py must first reproduce gold from these params (checked by its exit code)'}
        fx = (wrong or {}).get('independent_then_filter')
        r = fx.get('reward') if fx else None
        return binding, {'pass': r is not None and r <= .60, 'reward': r, 'gate': .60,
                         'method': 'author fixture independent_then_filter (per-file judgements, no cross-file facts); reviewer audits',
                         'binding': binding['pass']}
    script = Path(__file__).resolve().with_name('weekend_replay.py')
    result = run_py([str(script), '--task', str(task), '--out', str(out), '--seed', str(seed), '--family', family],
                    task, [out], out / 'replay.log')
    rp = out / 'replay.json'
    if result['exit'] != 0 or not rp.exists():
        fail = {'pass': False, 'error': 'controller replay unavailable or failed', 'run': result}
        return fail, fail
    data = json.loads(rp.read_text())
    f = out / 'independent.xlsx'
    sr = score(task, f, out / 'score', out / 'score.log') if f.exists() else {'reward': None}
    if sr['reward'] is None and wrong and 'judge_independent_then_filter' in wrong:
        sr = {'reward': wrong['judge_independent_then_filter'].get('reward')}
    g = out / 'strong.xlsx'
    strong = score(task, g, out / 'score_strong', out / 'score_strong.log')['reward'] if g.exists() else None
    return data['binding'], {'pass': sr['reward'] is not None and sr['reward'] <= .60,
                             'reward': sr['reward'], 'gate': .60, 'method': data['method'] or data.get('gate_note'),
                             'strong_reward_diagnostic': strong, 'strong_method': data.get('strong_method'),
                             'input_gold_sha256': sha(task / 'tests/gold.json'),
                             'binding': data['binding']['pass']}
