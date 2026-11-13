"""Sandboxed controller replay, explicit adapters for previously frozen packages."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

from openpyxl import Workbook


def load(path):
    spec = importlib.util.spec_from_file_location('task_build', path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def c2(task, out, seed, B, gold):
    P = copy.deepcopy(B.PARAMS)
    P['seed'] = seed
    contracts = B.load_contracts()
    model, labels, base, _ = B.compute_all(P, contracts)
    base = {k: {f: v[f] for f in ('entity', 'method', 'track')} for k, v in base.items()}
    binding = {'pass': base == gold['rows'], 'actual_seed': seed, 'script_default_seed': B.PARAMS['seed'],
               'legacy_sensitivity_seed_matches': B.PARAMS['seed'] == seed,
               'scored_fields_match': base == gold['rows'],
               'scenario_params_present': (task / 'tests/scenario_params.json').exists()}
    # The old sensitivity script uses B.PARAMS directly, so it does not test this instance.
    binding['pass'] = binding['pass'] and binding['legacy_sensitivity_seed_matches'] and binding['scenario_params_present']
    def save(rows, name):
        wb = Workbook(); ws = wb.active; ws.title = 'Schedule'
        assert ws is not None
        ws.append(['Contract Ref', 'Group Entity Code', 'Transfer Method', 'Review Track', 'Notes'])
        for key, v in sorted(rows.items()):
            ws.append([key, v['entity'], v['method'], v['track'], ''])
        wb.save(out / name)
    # Gate (STATUS 2026-11-07 00:25): per-contract gold labels + public rules, without the
    # cross-contract link judgement. Diagnostic: also give cross-contract facts.
    save(B.gold_rows(model, labels, dict(P, linked_rule=False)), 'independent.xlsx')
    save(B.gold_rows(model, labels, P), 'strong.xlsx')
    return {'binding': binding, 'method': 'Per-contract gold CUAD labels + public rules, cross-contract linked-counterparty rule not applied',
            'strong_method': 'Same plus cross-contract linking (diagnostic only)',
            'oracle_source': 'expert labels; diagnostic, NOT a cheap baseline'}


def b(task, out, seed, B, gold):
    # Replay exact published seed against current eligibility. No seed search.
    fresh = out / 'rebuild'; fresh.mkdir()
    B.run(str(fresh), seed)
    rebuilt = json.loads((fresh / 'tests/gold.json').read_text())
    fields_match = all(rebuilt[k] == gold[k] for k in ('endpoints', 'trials', 'norms'))
    binding = {'pass': fields_match and (task / 'tests/scenario_params.json').exists(),
               'actual_seed': seed, 'scored_fields_match': fields_match,
               'scenario_params_present': (task / 'tests/scenario_params.json').exists()}
    # Direction oracle per comparison. Sequence, counts and stop reimplemented independently.
    # Scope/order/version are deterministic public metadata, not semantic judgements.
    wb = Workbook(); ws = wb.active; ws.title = 'Endpoint_Review'
    ws.append(B.EP_HEADERS)
    ts = wb.create_sheet('Trial_Summary'); ts.append(B.TR_HEADERS)
    for nct, orderinfo in sorted(gold['aux']['orders'].items()):
        stopped, count, stop = False, 0, 'NONE'
        for eid in orderinfo['applied']:
            result = gold['endpoints'][eid]['result']
            if stopped:
                status = 'Not tested'
            elif result == 'No significant difference':
                status = 'Not significant - sequence stops'; stopped = True; stop = eid
            else:
                status = 'Confirmed'; count += 1
            ws.append([nct, eid, result, status, '', ''])
        ts.append([nct, gold['trials'][nct]['version'], count, stop])
    # Match the package's sheet titles, not an invented schema.
    import openpyxl
    ref = openpyxl.load_workbook(task / 'solution/reference.xlsx', read_only=True)
    ws.title, ts.title = ref.sheetnames[:2]
    ref.close()
    wb.save(out / 'strong.xlsx')
    return {'binding': binding, 'method': None,
            'strong_method': 'Independent direction oracle; independently implemented sequence aggregation. Scope/order/version taken from gold structural metadata (diagnostic only).',
            'gate_note': 'Per-file gate for this legacy design not implemented by controller; author fixture judge_independent_then_filter is reported instead.',
            'oracle_source': 'gold underlying per-comparison directions; diagnostic, NOT a cheap baseline'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--family', choices=['B', 'C2'], required=True)
    a = ap.parse_args()
    B = load(a.task / 'build/build.py')
    gold = json.loads((a.task / 'tests/gold.json').read_text())
    result = {'B': b, 'C2': c2}[a.family](a.task, a.out, a.seed, B, gold)
    (a.out / 'replay.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
