"""Stable public benchmark marker; never derived from labels or the seed."""
from pathlib import Path
import tomllib

GUID = tomllib.loads((Path(__file__).resolve().parents[1] / 'config.toml').read_text())['canary']['guid']
PREFIX = f'Benchmark canary: {GUID}\n\n'


def text_header(text):
    return PREFIX + strip_header(text)


def strip_header(text):
    return text[len(PREFIX):] if text.startswith(PREFIX) else text


def output_contract(specs, conventions):
    """Publish scorer vocabulary using only public construction information."""
    lines = ['## Workbook output guide', '', 'Save the completed workbook as `/output/review.xlsx`.',
             'Use these exact sheet and column names; retain the identifier columns.']
    for name, keys, columns in specs:
        lines.append(f'- Sheet `{name}`: columns ' + ', '.join(f'`{c}`' for c in columns)
                     + '; row identifiers: ' + ', '.join(f'`{k}`' for k in keys) + '.')
    lines += ['', '`Included` accepts `Yes` or `No`. `Positive_Count` is a nonnegative integer.',
              'Code every assigned Record_ID and Question pair in review_memo.md regardless of scope status; '
              'records marked No still require coding rows. Record scope separately.',
              'For `Answer`, use the encoded option shown below for the matching Question:']
    for question, definition in conventions['questions'].items():
        if definition.get('kind') == 'benchmark_item':
            lines.append(f'- `{question}`: enter the item answer as a JSON value; the original deterministic item checker applies. '
                         'For this Question, Positive_Count counts your nonempty answers (all must be correct for consistency credit).')
            continue
        options = '; '.join(f'{original} → `{encoded}`' for original, encoded in definition['encoding'])
        lines.append(f'- `{question}`: {options}.')
    for fact in conventions['extra_facts']:
        lines.append(f'- Fact_ID `{fact["id"]}`: enter the `{fact["fact_key"]}` from `{fact["visible_file"]}` in `Value`.')
    lines += ['', 'Summaries count the first listed original option for each Question in your own coding '
              'across all assigned pairs, including records marked No; do not filter by scope.',
              'Consult coding_conventions.json for the question definitions and the current notice for scope.']
    return '\n'.join(lines) + '\n'
