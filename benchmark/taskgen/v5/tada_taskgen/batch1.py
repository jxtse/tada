"""Author-side batch-1 catalogue; raw rows and private locators are never exported."""
import json
from pathlib import Path

from local_config import path as local_path, public_text
DATASETS = local_path('datasets_root')
SHORTLIST = DATASETS / 'catalog/BATCH1_SHORTLIST.json'
INVENTORY = Path(__file__).resolve().parents[1] / 'batch1_inventory.json'
if SHORTLIST.is_file():
    GROUPS = {group[0]: row['ids'] for group, row in json.loads(SHORTLIST.read_text())['batch1_groups'].items()}
else:
    # The audited public inventory is sufficient for imports and dry metadata.
    GROUPS = {g: [r['dataset_id'] for r in json.loads(INVENTORY.read_text())['datasets'] if r['group'] == g] for g in ('A','B','C')}
IDS = tuple(source for sources in GROUPS.values() for source in sources)
INVENTORY = Path(__file__).resolve().parents[1] / 'batch1_inventory.json'
LEGACY = {'cuad': 'CUAD', 'contract_nli': 'NDA', 'maud': 'MAUD',
          'scdb_supreme_court_database': 'PREC', 'cap_us_reports_scdb': 'PREC', 'evidence_inference': 'B'}
SUMMARIES = {
    'cuad': 'Lawyer-annotated contract provisions; grounded original categorical labels.',
    'contract_nli': 'Contract hypotheses with original entailment, contradiction and not-mentioned labels.',
    'maud': 'Merger agreement provisions with original expert deal-point codes and conventions.',
    'scdb_supreme_court_database': 'Original SCDB precedent-alteration codes joined to CAP opinion text by case identity.',
    'cap_us_reports_scdb': 'CAP opinion text paired with SCDB original precedent-alteration codes; one linked source.',
    'uda-bench-vldb': 'Text-grounded original categorical attributes from UDA scenarios with published code meanings.',
    'sembench': 'Original movie-workload query answers and deterministic movie evaluator; platform ratings are not labels.',
    'biodex': 'Original ICSR seriousness and patient-sex codes inferred from article text; derived aggregates excluded.',
    'evidence_inference': 'Original expert intervention/comparator/outcome effect labels grounded in study text.',
    'ntsb_aviation_avall': 'Factual accident narratives with original event injury, light, weather and unambiguous damage codes.',
    'asap-chinese-restaurant-absa': 'Chinese restaurant reviews with 18 original manually annotated aspect polarities; stars excluded.',
    'oolong-synth': 'Original synthetic long-context questions; unlabelled context visible and original answers hidden.',
    'oolong-real': 'Original long-context questions from the locally delivered toy_dnd subset.',
    'semantic-query-engines-experiments': 'Original derivation, selection and join queries with declared input columns and hidden result tables.',
    'tag-bench': 'Original table-augmented generation questions and database inputs; unanswered aggregation excluded.',
    'bird': 'Original text-to-SQL questions with local SQLite databases and hidden reference SQL.',
    'loft': 'Long-context SQL, retrieval, RAG and in-context classification items with original deterministic metrics.',
    'palimpzest': 'Original Enron email field-extraction answers with source email inputs.',
    'longds-bench': 'Original long-horizon data-science tasks; canonical answer verification requires an LLM judge.',
    'kramabench': 'Original multi-file analysis workload questions and typed answers with official success/F1/RAE metrics.',
    'discoverybench': 'Original discovery hypotheses and variable relationships; official semantic verification requires a judge.',
    'dsbench': 'Original data-analysis and modeling tasks; public answer evaluation uses a judge and competition targets are absent.',
    'lrobench': 'Original relational selection, matching, ordering, clustering and supported multi-operation answers.',
    'spider2': 'Original Spider2-Lite questions with local SQLite inputs and hidden result-table references.',
    'semtpch': 'Original semantic TPC-H workload questions with relational and multimedia inputs and deterministic result metrics.',
    'dabstep': 'The ten original answered development tasks; public task inputs retained and reference answers hidden.',
    'infiagent-dabench': 'Original CSV analysis questions and structured field/value answers with official numeric and string checking.',
    'tablebench': 'Original tabular reasoning questions, table inputs and deterministic category-specific answer metrics.',
    'blade': 'Original analysis-decision multiple-choice items; correct-option indices scored, semantic analysis judging excluded.',
}


def source_id(value):
    if value in IDS:
        return value
    return next((key for key, source in LEGACY.items() if source == value), value)


def public_metadata(adapter):
    """Only field names, method summaries and aggregate counts cross the boundary."""
    identifier = source_id(adapter.source)
    metadata = getattr(adapter, 'metadata', {})
    metadata = dict(metadata() if callable(metadata) else metadata)
    profile_path = DATASETS / 'unified/output/cards' / identifier / 'profile.json'
    if not metadata.get('licensing_notes') and profile_path.is_file():
        profile = json.loads(profile_path.read_text())
        metadata['licensing_notes'] = '; '.join(str(profile[k]) for k in ('license', 'license_evidence') if profile.get(k))
    limitations = list(getattr(adapter, 'limitations', ()))
    extra = metadata.get('limitations', [])
    limitations.extend([extra] if isinstance(extra, str) else extra)
    limitations = list(dict.fromkeys(limitations))
    if INVENTORY.exists():
        row = next((row for row in json.loads(INVENTORY.read_text()).get('datasets', [])
                    if row['dataset_id'] == identifier), {})
        metadata.setdefault('scorable_unit_count', row.get('scorable_unit_count'))
        metadata.setdefault('scorable_cells', row.get('scorable_cell_count'))
    return {'dataset_id': identifier, 'group': next((g for g, ids in GROUPS.items() if identifier in ids), None),
            'kind': getattr(adapter, 'kind', 'record_label'),
            'label_card_summary': metadata.get('label_card_summary', SUMMARIES.get(identifier, metadata.get('summary', 'Original source labels with reversible public encodings.'))),
            'visible_fields': metadata.get('visible_fields', ['text', 'question', 'codebook']),
            'hidden_fields': metadata.get('hidden_fields', ['original_answers', 'evidence_annotations', 'source_locators']),
            'scorable_unit_count': metadata.get('scorable_unit_count', metadata.get('scorable_units')),
            'scorable_cell_count': metadata.get('scorable_cells', metadata.get('scorable_units')
                                             if getattr(adapter, 'kind', '') == 'benchmark_item' else None),
            'checker': metadata.get('checker', 'categorical_exact'),
            'checker_source': metadata.get('checker_source', ''),
            'dropped_counts': metadata.get('dropped_counts', {}),
            'licensing_notes': public_text(metadata.get('licensing_notes', '')),
            'limitations': [public_text(value) for value in limitations],
            **({'paired_dataset_ids': ['scdb_supreme_court_database', 'cap_us_reports_scdb']}
               if adapter.source == 'PREC' else {})}
