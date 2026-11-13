"""Offline tests; real source contracts run only with their configured local inputs."""
import os
import sys
from pathlib import Path
import pytest
from local_config import get, path

REPO = Path(__file__).resolve().parents[2]
# Reuse a configured interpreter's dependencies without changing its installation.
python = path('datasets_python')
runtime = python.parent.parent / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
if python.is_file() and runtime.is_dir():
    sys.path.append(str(runtime))


def pytest_configure(config):
    """Keep every test artifact in a private scratch tree, outside public exports."""
    import tempfile
    sys.dont_write_bytecode = True
    base = REPO / 'v5_runs'
    scratch = base / ('pytest-scratch-' + str(os.getpid()))
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    tempfile.tempdir = str(scratch)
    if not config.option.basetemp:
        config.option.basetemp = str(base / ('pytest-' + str(os.getpid())))
    config.addinivalue_line('markers', 'local_data: requires configured original datasets or source caches')


# These modules have real-source or original-codebook fixtures shared by all tests.
DATA_MODULES = {
    'test_adapters.py', 'test_compiler.py', 'test_designer_check.py', 'test_gates.py',
    'test_intents.py', 'test_label_cost.py', 'test_r6_adapters.py',
    'test_r5_public_policy.py', 'test_slim_redteam.py',
}
DATA_TESTS = {
    'test_disabled_questions.py': {'test_disabled_question_keeps_labels_and_codebook',
        'test_disablement_is_source_scoped', 'test_live_quarantine_disables_dp040_for_new_design',
        'test_designer_check_rejects_disabled_question'},
    'test_screening.py': {'test_maud_non_dedicated_uses_generic_but_dedicated_is_unchanged',
        'test_source_loader_pins_explicit_permutation_and_reference_ignores_environment'},
    'test_r5_maud_pseudo.py': {'test_contract_109_heading_and_principal_private_audit',
        'test_labels_are_unchanged_and_evidence_uses_the_visible_text',
        'test_question_provision_context_without_option_changes', 'test_screening_version_binds_pseudonymiser',
        'test_maud_permutation_retains_public_text_and_scorable_pairs'},
    'test_designer_v5.py': {'test_v5_designer_has_schema_and_no_code_contract'},
    'test_r4_slim_gates.py': {'test_actual_trivial_checks_use_total_and_repeat_details'},
    'test_r4_slim_loop_review.py': {'test_designer_dry_run_uses_only_four_file_tools', 'test_designer_receives_feedback_after_incomplete_initial_output'},
    'test_r5_compile_feedback.py': {'test_actual_failed_carrier_names_only_fields_that_failed_final_read'},
    'test_r6_designer.py': {'test_catalogues_do_not_open_records_or_labels', 'test_prompt_contains_only_assigned_catalogues', 'test_two_source_example_compiles_with_fake_clients'},
    'test_real_clients.py': {'test_real_compile_and_copied_rebuild_never_recall_models',
        'test_real_compile_cannot_reuse_fake_archive', 'test_archive_boolean_integer_mismatch_refused_without_clearing_public'},
    'test_r7_benchmarks.py': {'test_real_original_items_and_visibility',
        'test_dabstep_all_ten_dev_answers_bijective', 'test_benchmark_quarantine_preserves_inputs'},
    'test_r7_sembench.py': {'test_all_local_items_keep_gold_out_of_assets', 'test_port_matches_original_upstream_evaluator'},
}


def pytest_collection_modifyitems(items):
    missing = [key for key in ('datasets_root', 'private_cache', 'legacy_cache', 'quarantine') if not get(key) or not path(key).exists()]
    for item in items:
        filename = Path(str(item.fspath)).name
        name = item.originalname or item.name.split('[')[0]
        needs_data = filename in DATA_MODULES or name in DATA_TESTS.get(filename, set())
        if needs_data:
            item.add_marker(pytest.mark.local_data)
            if missing:
                item.add_marker(pytest.mark.skip(reason='local data unavailable: configure ' + ', '.join(missing) + ' in local_paths.toml'))
        if filename == 'test_intents.py' and not path('onet_root').is_dir():
            item.add_marker(pytest.mark.skip(reason='configured onet_root tables are absent'))
        if filename == 'test_designer_check.py' and not path('datasets_python').is_file():
            item.add_marker(pytest.mark.skip(reason='configured datasets_python interpreter is absent'))
        if filename == 'test_slim_redteam_runtime.py' and not path('datasets_python').is_file():
            item.add_marker(pytest.mark.skip(reason='configured datasets_python interpreter is absent'))


@pytest.fixture(autouse=True)
def isolated_model_slots(tmp_path, monkeypatch):
    """Unit tests exercise real file locks without competing with live solvers."""
    import tada_taskgen.concurrency as concurrency
    monkeypatch.setattr(concurrency, 'LOCK_ROOT', tmp_path / 'model_slots')
    monkeypatch.delenv('TADA_MODEL_SLOT_HELD', raising=False)
    monkeypatch.delenv('TADA_MODEL_SLOT_FAMILY', raising=False)
