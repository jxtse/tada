"""Batch task-space identity, coverage order and public catalogue boundaries."""
from tada_taskgen import combinations
from tada_taskgen.adapters import canonical_source
from tada_taskgen.batch1 import IDS, LEGACY


def test_audited_inventory_covers_the_agreed_task_space():
    import json
    from tada_taskgen.batch1 import INVENTORY, SUMMARIES
    from tada_taskgen.adapters import available_sources
    inventory = json.loads(INVENTORY.read_text())
    rows = {row['dataset_id']: row for row in inventory['datasets']}
    assert set(rows) == set(IDS) == set(SUMMARIES)
    assert {key for key, row in rows.items() if row['status'] != 'available'} == {
        'longds-bench', 'discoverybench', 'dsbench'}
    assert all(row['status'] == 'requires_llm_judge' and row['reason']
               for row in rows.values() if row['status'] != 'available')
    assert len(available_sources()) == 25 and 'CUADX' not in available_sources()
    assert inventory['combination_count'] == len(combinations.enumerate_spaces(available_sources())['combinations']) == 2625
    assert all(row['scorable_unit_count'] > 0 and row['scorable_cell_count'] > 0
               and row['label_card_summary'] and row['checker_source']
               for row in rows.values() if row['status'] == 'available')


def test_batch_shortlist_keeps_all_29_ids_and_case_pair():
    assert len(IDS) == len(set(IDS)) == 29
    assert canonical_source('cuad') == canonical_source('C2') == 'CUAD'
    assert canonical_source('contract_nli') == 'NDA'
    assert canonical_source('cap_us_reports_scdb') == canonical_source('scdb_supreme_court_database') == 'PREC'
    assert canonical_source('oolong-synth') == 'oolong-synth'
    assert len(set(LEGACY.values())) == 5


def test_prefer_unused_sources_before_repeating_an_existing_combination():
    enum = combinations.enumerate_spaces(['CUAD', 'MAUD', 'NDA', 'B'], 2)
    rows = combinations.counts({'final_verdicts': [{'sources': ['CUAD', 'MAUD'], 'verdict': 'discarded'}]}, enum)
    selected = combinations.choose(rows, seed=7, prefer_unused=True)
    assert selected['key'] == 'B+NDA'
    assert selected == combinations.choose(rows, seed=7, prefer_unused=True)
    rows['B+NDA'].update(admitted=2)
    assert combinations.choose(rows, seed=7, prefer_unused=True)['key'] != 'B+NDA'


def test_mixed_case_batch_keys_do_not_create_alias_duplicates():
    enum = combinations.enumerate_spaces(['contract_nli', 'NDA', 'C2', 'cuad', 'oolong-synth'])
    assert enum['available_sources'] == ['CUAD', 'NDA', 'oolong-synth']
    assert len(enum['combinations']) == 7


def test_large_input_scanner_preserves_boundary_matches(tmp_path):
    from controller import scan_chunks
    path = tmp_path / 'input.sqlite'
    path.write_bytes(b'x' * (1024 * 1024 - 4) + b'gold.json' + b'y' * 2000)
    chunks = list(scan_chunks(path))
    assert any('gold.json' in chunk for chunk in chunks)
    assert max(map(len, chunks)) <= 1024 * 1024 + 4096
