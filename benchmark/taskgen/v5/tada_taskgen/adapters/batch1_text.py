"""Original text-label sources read through source_adapters.iter_units.

Each class explicitly selects public narrative fields and original coded labels;
raw rows, source locators, joins, report targets and annotations stay private.
Identity codebooks preserve the original strings (including unknown/absent codes).
No labels are inferred from a heuristic, another label, or a model.
"""
from collections import defaultdict
from functools import cached_property, lru_cache
import hashlib
import json
import random
import re

from .base import SourceAdapter, exclusions, generic_term_predict
from local_config import public_text, require
from .unified_reader import UNIFIED, iter_author, ntsb_relations


@lru_cache(maxsize=32)
def _excluded(path, source, stamp):
    pairs, records = exclusions(path, source)
    if source == 'ntsb_aviation_avall':
        data = json.loads(path.read_text()) if path.exists() else {}
        records |= {str(row['ev_id']) for row in data.get('NTSB_events', [])}
    return pairs, records


class UnifiedTextAdapter(SourceAdapter):
    kind = 'record_label'
    status = 'available'
    checker = 'categorical_exact'
    pseudonymiser_version = 'opaque-record-id-original-text-v1'

    @cached_property
    def _data(self):
        return self.load()

    def records(self):
        return self._data[0]

    @cached_property
    def _questions(self):
        return self.build_questions(self._data[1])

    def questions(self):
        return self._questions

    def build_questions(self, labels):
        options = defaultdict(set)
        for (_, question), (answer, _) in labels.items():
            options[question].add(answer)
        return {q: {'question': 'What is the original coded value for ' + q + '?',
                    'options': sorted(values, key=lambda x: (type(x).__name__, str(x))),
                    'coding_conventions': ['Use the published original code exactly; decimal integer codes decode to integers.']}
                for q, values in sorted(options.items())}

    @cached_property
    def _labels(self):
        labels = dict(self._data[1])
        if self.permutation_seed:
            rng = random.Random(int(self.permutation_seed))
            for question in self.questions():
                keys = sorted(key for key in labels if key[1] == question)
                values = [labels[key] for key in keys]
                rng.shuffle(values)
                labels.update(zip(keys, values))
        return labels

    def label_pool(self):
        pairs, records = self._excluded()
        return {key: value for key, value in self._labels.items() if key not in pairs and key[0] not in records}

    def scorable(self, record, question):
        pairs, records = self._excluded()
        return (record, question) in self._labels and (record, question) not in pairs and record not in records

    def _excluded(self):
        stamp = self.quarantine.stat().st_mtime_ns if self.quarantine.exists() else None
        return _excluded(self.quarantine, self.source, stamp)

    def encode(self, question, answer):
        if not any(type(answer) is type(option) and answer == option for option in self.questions()[question]['options']):
            raise ValueError('Original typed code required')
        return str(answer)

    def decode(self, question, visible):
        if type(visible) is not str:
            raise ValueError('Encoded string required')
        matches = [option for option in self.questions()[question]['options'] if self.encode(question, option) == visible]
        if len(matches) != 1:
            raise ValueError('Unknown or ambiguous encoded code')
        return matches[0]

    def visible_id(self, record):
        if record not in self.records():
            raise KeyError(record)
        return 'record-' + hashlib.sha256((self.source + ':' + record).encode()).hexdigest()[:20]

    def record_map(self):
        result = {record: self.visible_id(record) for record in self.records()}
        if len(set(result.values())) != len(result):
            raise ValueError('Record ID collision')
        return result

    def metadata(self):
        require('unified_root')
        profile = json.loads((UNIFIED / 'output/cards' / self.source / 'profile.json').read_text())
        return {'kind': self.kind, 'status': self.status, 'checker': self.checker,
                'visible_fields': self.visible_fields, 'hidden_fields': self.hidden_fields,
                'scorable_units': len({r for r, _ in self.label_pool()}),
                'scorable_cells': len(self.label_pool()), 'limitations': list(self.limitations),
                'licensing_notes': public_text(profile.get('license', '')),
                'license_evidence': public_text(profile.get('license_evidence', ''))}


ASPECTS = ('Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find',
           'Service#Queue', 'Service#Hospitality', 'Service#Parking', 'Service#Timely',
           'Price#Level', 'Price#Cost_effective', 'Price#Discount', 'Ambience#Decoration',
           'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary', 'Food#Portion',
           'Food#Taste', 'Food#Appearance', 'Food#Recommend')


class ASAPAdapter(UnifiedTextAdapter):
    """ASAP: public review only; hidden original 18 expert aspect codes.

    Source contract: unified/output/cards/asap-chinese-restaurant-absa/label_card.md;
    raw Meituan-Dianping/asap data/{train,dev,test}.csv. star is an unscored
    platform rating. Each string code decodes identically to its original cell.
    """
    source = 'asap-chinese-restaurant-absa'
    visible_fields = ['review']
    hidden_fields = list(ASPECTS) + ['star', 'id', 'split']
    limitations = ('Only the 18 manual aspects are scored; platform star ratings are excluded.',
                   'Original Chinese review prose and named businesses are retained for meaning.')

    def load(self):
        return _asap()

    def build_questions(self, labels):
        return {q: {'question': 'What sentiment is expressed about ' + q.replace('#', ': ') + '?',
                    'options': ['-2', '-1', '0', '1'],
                    'coding_conventions': ['-2: not mentioned; -1: negative; 0: neutral; 1: positive.']}
                for q in ASPECTS}


@lru_cache(maxsize=1)
def _asap():
    records, labels = {}, {}
    for unit in iter_author('asap-chinese-restaurant-absa'):
        text = unit['text'].get('review')
        if not isinstance(text, str) or not text.strip():
            continue
        key = unit['unit_id']
        records[key] = {'text': text}
        for question in ASPECTS:
            value = unit['labels'].get(question, {}).get('value')
            if type(value) is str and value in ('-2', '-1', '0', '1'):
                labels[key, question] = (value, [])
    return records, labels


class BioDEXAdapter(UnifiedTextAdapter):
    """Public original article title/abstract/fulltext; private ICSR target fields.

    Uses the raw BioDEX-ICSR parquet targets via unified reader, not the L0
    article-level aggregated FAERS flags. serious and patientsex are literal
    original target subfields; absent/invalid fields are unscorable, never imputed.
    Source: unified/output/cards/biodex/label_card.md, raw BioDEX-ICSR data/*.parquet.
    """
    source = 'biodex'
    visible_fields = ['title', 'abstract', 'fulltext']
    hidden_fields = ['target', 'serious', 'patientsex', 'drugs', 'reactions', 'split']
    limitations = ('Only original ICSR serious and patientsex target fields are scored; L0 report aggregates are excluded.',
                   'Article text is retained; per-article copyright/licensing restrictions still apply.')
    def load(self):
        return _biodex()

    def build_questions(self, labels):
        return {'serious': {'question': 'Is the case serious under the original FAERS ICSR definition?',
                    'options': ['1', '2'], 'coding_conventions': ['1: serious; 2: not serious.']},
                'patientsex': {'question': 'What patient sex is reported for the case?',
                    'options': ['0', '1', '2'], 'coding_conventions': ['0: unknown; 1: male; 2: female.']}}


@lru_cache(maxsize=1)
def _biodex():
    records, labels = {}, {}
    for table in ('9a0a44745ff28643', '6a1367ed833812d9', '61abb1ae28c3a16a'):
        for unit in iter_author('biodex', view='raw', table=table):
            target = unit['labels'].get('target', {}).get('value')
            if not isinstance(target, str):
                continue
            text = '\n\n'.join(unit['text'].get(k) or '' for k in ('title', 'abstract', 'fulltext'))
            if not text.strip():
                continue
            key = unit['unit_id']
            records[key] = {'text': text}
            for question, allowed in [('serious', ('1', '2')), ('patientsex', ('0', '1', '2'))]:
                matches = re.findall(r'^' + question + r': ([^\n]*)$', target, re.M)
                if len(matches) == 1 and matches[0] in allowed:
                    labels[key, question] = (matches[0], [])
    return records, labels


UDA_FIELDS = {'art': ('Nationality', 'Art_movement', 'Century', 'Birth_country', 'Birth_continent', 'Field', 'Genre', 'Marriage', 'Teaching'),
              'finance': ('principal_activities', 'major_equity_changes', 'major_events', 'business_risks', 'remuneration_policy'),
              'legal': ('case_type', 'verdict', 'nationality_for_applicant', 'plaintiff_current_status', 'defendant_current_status', 'evidence', 'first_judge'),
              'nba_city': ('state_name',), 'nba_manager': ('nationality', 'nba_team'),
              'nba_player': ('nationality', 'team', 'position'), 'nba_team': ('location',)}


@lru_cache(maxsize=1)
def _uda_definitions():
    from .benchmark_reader import DATA, extended_units
    root = DATA / 'raw/analytics_benchmarks/uda-bench-vldb/repo_bench-u/Query'
    mapping = {'Art': 'art', 'finance': 'finance', 'legal_case': 'legal',
               'player': 'nba_player', 'city': 'nba_city', 'manager': 'nba_manager', 'team': 'nba_team'}
    result = {}
    for folder in ('Art', 'Finan', 'Legal', 'Player'):
        path = root / folder / (folder + '_attributes.json')
        for unit in extended_units('uda-bench-vldb', [path], format='json'):
            for original, fields in unit['raw_row'].items():
                for field, description in fields.items():
                    result[mapping[original], field.lower()] = description['description']
    return result


class UDAAdapter(UnifiedTextAdapter):
    """Public source documents; hidden original joined categorical gold attributes.

    Unified reader performs the original id/scenario gold join. Fields tied to
    images, paths, URLs, names/identifiers and open numeric extraction are omitted.
    Each scenario:field string is an unchanged original expert field, not a derived
    judgment. Source: unified/output/cards/uda-bench-vldb/label_card.md and original
    gdrive/ground_truth/{Art,Finance,Legal} plus repo_bench-u/Query/Player tables.
    """
    source = 'uda-bench-vldb'
    visible_fields = ['text']
    hidden_fields = ['gold', 'id', 'scenario', 'source_locator']
    limitations = ('Categorical text-grounded fields only; image-specific and numeric attributes are not scored.',
                   'Original source errors, unknown codes and multi-value strings are retained without relabeling.',
                   'nba_player:is_dead is excluded because the delivered original attributes do not define its code semantics.')
    def load(self):
        return _uda()

    def build_questions(self, labels):
        result = super().build_questions(labels)
        definitions = _uda_definitions()
        for question, metadata in result.items():
            scenario, field = question.split(':', 1)
            metadata['question'] = definitions[scenario, field.lower()]
            metadata['coding_conventions'] = ['Return the original option exactly; original multi-value strings retain their || separators.']
        return result


@lru_cache(maxsize=1)
def _uda():
    records, labels = {}, {}
    for unit in iter_author('uda-bench-vldb'):
        text = unit['text'].get('text')
        if not isinstance(text, str) or not text.strip():
            continue
        scenario = unit['metadata'].get('scenario')
        gold = unit['labels'].get('gold', {}).get('value', {})
        if not isinstance(gold, dict):
            continue
        key = unit['unit_id']
        records[key] = {'text': text}
        for field in UDA_FIELDS.get(scenario, ()):
            value = gold.get(field)
            if type(value) is str and value.strip():
                labels[key, scenario + ':' + field] = (value, [])
    return records, labels


class NTSBAdapter(UnifiedTextAdapter):
    """Public factual narrative narr_accp only; private original event codes.

    Original events.ev_highest_injury/light_cond/wx_cond_basic are one code per
    event. Aircraft damage and occurrence Phase_of_Flight are eligible only where
    exactly one original row exists for that event; no worst-case aggregation.
    Finding C/F identities, cause and analysis are never exposed or scored.
    Source: unified/output/cards/ntsb_aviation_avall/label_card.md; avall.mdb
    events/aircraft/narratives/Occurrences through the unchanged unified reader.
    """
    source = 'ntsb_aviation_avall'
    visible_fields = ['narr_accp']
    hidden_fields = ['ev_highest_injury', 'light_cond', 'wx_cond_basic', 'damage', 'Phase_of_Flight',
                     'narr_accf', 'narr_cause', 'finding_code', 'Cause_Factor', 'ev_id']
    limitations = ('Finding-level cause/factor labels remain excluded because their units are private.',
                   'Damage/phase codes require a unique original aircraft/occurrence row per event.',
                   'Factual narratives under 1500 characters or containing conclusion language are excluded.')
    def load(self):
        return _ntsb()

    def metadata(self):
        result = super().metadata()
        result['available_coded_fields'] = sorted(self.questions())
        if 'Phase_of_Flight' not in self.questions():
            result['limitations'].append('No unique original occurrence-level Phase_of_Flight cells are present in the delivered MDB; dictionary definitions alone are not labels.')
        return result

    def build_questions(self, labels):
        result = super().build_questions(labels)
        definitions = self._data[2] if len(self._data) > 2 else {}
        for question, row in result.items():
            source = definitions.get(question, {})
            if not source:
                raise ValueError('NTSB source codebook missing for ' + question)
            row['question'] = source['question']
            meanings = source['codes']
            if any(str(option).casefold() not in {str(code).casefold() for code in meanings} for option in row['options']):
                raise ValueError('NTSB observed code lacks original definition: ' + question)
            observed = {str(option).casefold(): option for option in row['options']}
            row['options'] = sorted([observed.get(str(code).casefold(), code) for code in meanings], key=str)
            row['coding_conventions'] = ['Original NTSB coded values (stored source spelling retained):'] + [
                f'{observed.get(str(code).casefold(), code)}: {meaning}' for code, meaning in sorted(meanings.items())]
        return result


TELLS = re.compile(r'probable cause|ntsb (?:determines|determined)|contributing to the accident', re.I)


def ntsb_data(rows, definitions=None):
    events, narratives, aircraft, occurrences = {}, defaultdict(list), defaultdict(list), defaultdict(list)
    for unit in rows:
        row = unit['row']
        if unit['relation'] == 'eADMSPUB_DataDictionary':
            if definitions is not None and row.get('Column') in ('ev_highest_injury', 'light_cond', 'wx_cond_basic', 'damage', 'Phase_of_Flight'):
                q = row['Column']; code = row.get('code_iaids'); meaning = row.get('meaning')
                if code is not None and meaning:
                    metadata = definitions.setdefault(q, {'question': row.get('Question_Def') or q, 'codes': {}})
                    metadata['codes'][str(code)] = meaning
            continue
        key = row.get('ev_id')
        if not key:
            continue
        if unit['relation'] == 'events':
            if key in events:
                raise ValueError('Duplicate original NTSB event')
            events[key] = row
        elif unit['relation'] == 'narratives':
            narratives[key].append(row.get('narr_accp') or '')
        elif unit['relation'] == 'aircraft':
            aircraft[key].append(row)
        elif unit['relation'] == 'Occurrences':
            occurrences[key].append(row)
    records, labels = {}, {}
    for key, row in events.items():
        text = '\n\n'.join(narratives[key])
        if len(text.strip()) < 1500 or TELLS.search(text):
            continue
        records[key] = {'text': text}
        values = {q: row.get(q) for q in ('ev_highest_injury', 'light_cond', 'wx_cond_basic')}
        if len(aircraft[key]) == 1:
            values['damage'] = aircraft[key][0].get('damage')
        if len(occurrences[key]) == 1:
            values['Phase_of_Flight'] = occurrences[key][0].get('Phase_of_Flight')
        for question, value in values.items():
            if type(value) in (str, int) and str(value).strip():
                labels[key, question] = (value, [])
    return records, labels


@lru_cache(maxsize=1)
def _ntsb():
    definitions = {}
    records, labels = ntsb_data(ntsb_relations(), definitions)
    return records, labels, definitions
