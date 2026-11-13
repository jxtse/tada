"""Measure single-label cost using the deployed v5 aggregation, across sources.

Synthetic flips rebuild submitted answers and their consistency outputs. They do
not compile a new instance or run its class-diversity guard: the original gold
class counts remain the fixed denominators in every measurement.
"""
import json
import statistics
from pathlib import Path
from .adapters import get_adapter
from .multisource import adapter_for
from .compiler import compute_gold, sheet_specs, write_workbook
from .scoring import LABEL_AGGREGATION, score


def label_flip(task, out):
    task, out = Path(task), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gold = json.loads((task / 'tests/gold.json').read_text())
    sp = json.loads((task / 'tests/scenario_params.json').read_text())
    adapter = adapter_for(sp['spec']) if 'sources' in sp['spec'] else get_adapter(sp['spec']['source'])
    pool, specs = adapter.label_pool(), sheet_specs(sp['spec'])
    sheets, _ = compute_gold(sp, adapter=adapter)
    write_workbook(sheets, specs, out / 'self.xlsx')
    self_score, self_details = score(out / 'self.xlsx', gold)
    rows, errors = [], []
    for cell in gold['cells']:
        for alternative in adapter.questions()[cell['question']]['options']:
            if alternative == cell['answer']:
                continue

            def changed(record, question):
                value = pool[(record, question)]
                return (alternative, value[1]) if (record, question) == (cell['record'], cell['question']) else value

            try:
                # compute_gold deliberately has no compile-time class guard.
                submitted, _ = compute_gold(sp, changed, adapter=adapter)
                write_workbook(submitted, specs, out / 'flip.xlsx')
                reward, details = score(out / 'flip.xlsx', gold)
                if 'error' in details:
                    raise ValueError(details['error'])
            except Exception as exc:
                errors.append({'label': cell['id'], 'alt': alternative,
                               'error': f'{type(exc).__name__}: {str(exc)[:300]}'})
                continue
            rows.append({'label': cell['id'], 'question': cell['question'], 'alt': alternative,
                         'loss': round(1 - reward, 9)})
    losses = [row['loss'] for row in rows]
    max_loss = max(losses) if losses else None
    median_loss = statistics.median(losses) if losses else None
    report = {
        'label_aggregation': LABEL_AGGREGATION,
        'gold_class_counts': {q: value['class_counts'] for q, value in self_details.get('per_question', {}).items() if 'class_counts' in value},
        'gold_codebook_option_counts': {q: value['codebook_option_count']
                                        for q, value in self_details.get('per_question', {}).items() if 'codebook_option_count' in value},
        'benchmark_items_reported_only': sum(c.get('kind') == 'benchmark_item' for c in gold['cells']),
        'benchmark_note': 'Open-answer items have no finite alternative-label codebook; this historical diagnostic flips categorical labels only.',
        'self_score': self_score, 'labels_used': len(gold['cells']), 'flips': len(rows),
        'max_loss': max_loss, 'median_loss': median_loss,
        'gate': {'max': .20, 'median': .10},
        'pass': self_score == 1 and bool(losses) and not errors and max_loss <= .20 and median_loss <= .10,
        'worst': sorted(rows, key=lambda row: -row['loss'])[:10], 'errors': errors,
    }
    (out / 'flips.json').write_text(json.dumps(rows, indent=2))
    serialized = json.dumps(report, indent=2)
    (out / 'report.json').write_text(serialized)
    # The shared controller reads this complete file, never truncated stdout.
    (out / 'label_flip.json').write_text(serialized)
    return report
