"""V5-only MAUD packaging: replace distinctive names, never legal forms or labels."""
from functools import lru_cache
import hashlib
import random
import re

VERSION = 'maud-core-only-v1'
# The separator is part of the retained suffix, including punctuation and case.
# Holdings and Group are business-name tokens, not legal forms.
LEGAL_FORM = re.compile(
    r'(?:,?\s+)(?:Public Limited Company|Limited Liability Company|Limited Partnership|'
    r'Inc\.?|Incorporated|Corp\.?|Corporation|Company|Co\.?|LLC|L\.L\.C\.?|'
    r'LLP|L\.L\.P\.?|L\.P\.?|LP|plc|N\.V\.?|B\.V\.?|S\.A\.?|AG|'
    r'GmbH|Ltd\.?|Limited|Trust)\s*$', re.I)


def core_name(name):
    """Split once, leaving the original suffix bytes in the source text."""
    return LEGAL_FORM.sub('', name).strip(' ,')


def subsidiary_name(name):
    """Reject long prose/heading parses instead of rewriting an agreement section."""
    return (3 <= len(name) <= 80 and len(name.split()) <= 12
            and not re.search(r'\.\s|[;:!?\n]|,\s*and\b|^Subsidiary\s+of\b', name, re.I)
            and bool(LEGAL_FORM.search(name) or re.search(r'\b(?:Sub|Merger|Acquisition|Bidco|MergeCo)\b', name, re.I)))


@lru_cache(maxsize=512)
def mapping(api, record):
    parties = api._parties(record)
    if not parties:
        return ()
    rng = random.Random(int(hashlib.sha256(record.encode()).hexdigest()[:12], 16))
    a, b = rng.sample(api.FIRST, 2)
    names = {'target': f'{a} {rng.choice(api.SECOND)}', 'acquirer': f'{b} {rng.choice(api.SECOND)}'}
    pairs = [(core_name(parties[role]), names[role]) for role in ('target', 'acquirer')]
    # Retain the legacy principal-core aliases (e.g. a bank's shorter name).
    # Holding/Holdings variants occur in the same party's notice address.
    for role in ('target', 'acquirer'):
        core = core_name(parties[role])
        alias = api.SUFFIX.sub('', parties[role]).strip(' ,')
        if len(alias) < len(core) and core.startswith(alias + ' '):
            pairs.append((alias, names[role]))
        if re.search(r'\bHoldings?$', core, re.I):
            variant = core[:-1] if core.lower().endswith('holdings') else core + 's'
            pairs.append((variant, names[role]))
    pairs += [(core_name(name), f'{b} Merger Sub {index + 1}')
              for index, name in enumerate(parties['subs']) if subsidiary_name(name)]
    pairs = [(core, replacement) for core, replacement in pairs
             if len(core) >= 4 and core.lower() not in ('the company', 'parent')]
    # Longest first preserves principal names embedded in longer subsidiary names.
    return tuple((re.compile(r'(?<![A-Za-z])' + re.escape(core).replace(r'\ ', r'\s+') + r'(?![A-Za-z])', re.I), new)
                 for core, new in sorted(pairs, key=lambda row: -len(row[0])))


def pseudonymize(api, record, text):
    for pattern, new in mapping(api, record):
        def replace(match):
            old = match.group()
            return new.upper() if old.isupper() else new.lower() if old.islower() else new
        text = pattern.sub(replace, text)
    return text


@lru_cache(maxsize=16)
def records(api):
    result = {}
    for record in api.eligible_contracts():
        parties = api._parties(record)
        text = (api.ROOT / 'contracts' / f'{record}.txt').read_text(errors='ignore')
        result[record] = {'contract_id': record, 'text': pseudonymize(api, record, text),
                          'target_name': pseudonymize(api, record, parties['target']),
                          'acquirer_name': pseudonymize(api, record, parties['acquirer'])}
    return result


@lru_cache(maxsize=16)
def grounded_labels(api):
    """Use the same transformation for source text and evidence before locating it."""
    raw = api._labels_raw()
    keys = {(row['question'], row['subquestion']): q for q, row in api.deal_points().items()}
    out = {}
    for record, row in records(api).items():
        full = api._norm(row['text'])
        for source in raw[raw.contract_name == record].itertuples():
            key = (record, keys[source.question, source.subquestion])
            if key in api._quarantine():
                continue
            evidence = api._segments(pseudonymize(api, record, str(source.text)))
            if evidence and all(api._norm(segment) in full for segment in evidence):
                out[key] = (str(source.answer), evidence)
    # Match the controller's existing within-question label permutation interface.
    seed = api.os.environ.get('TADA_LABEL_PERMUTE')
    if seed:
        rng = random.Random(int(seed))
        groups = {}
        for (record, question), value in out.items():
            groups.setdefault(question, []).append((record, value))
        out = {}
        for question, rows in sorted(groups.items()):
            values = [v for _, v in rows]
            rng.shuffle(values)
            for (record, _), value in zip(rows, values):
                out[record, question] = value
    return out


def residual_original_names(api):
    """Reuse the private legacy principal-core audit on the new visible texts."""
    residuals = {}
    for record, row in records(api).items():
        parties = api._parties(record)
        count = sum(row['text'].count(api.SUFFIX.sub('', parties[role]).strip(' ,'))
                    for role in ('target', 'acquirer')
                    if len(api.SUFFIX.sub('', parties[role]).strip(' ,')) >= 4)
        if count:
            residuals[record] = count
    return residuals
