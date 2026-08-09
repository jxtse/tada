# Controlled evaluation materials

These scripts record the experiment logic used for the five-arm TABench
controlled matrix and the historical product snapshot.

## Pipeline

1. `run_matrix_v2.py` — executes Full, Operator, Skill-off, Code-only, and
   Direct configurations on a shared model configuration.
2. `build_matrix_units.py` — aligns delivered tables and constructs differing
   cell-level judgment units.
3. `run_v2_judge.py` — requests three votes under both anonymized A/B
   directions.
4. `build_report_v3.py` and `reaggregate_tieband.py` — aggregate the frozen
   judge ledgers, apply the adjacent-level deterministic tie rule, and build
   reporting tables.
5. `build_product_units.py` and `aggregate_product.py` — construct and
   aggregate the dated commercial-product snapshot.

## Configuration

The released runner reads these environment variables:

- `TADA_PROJECT_ROOT`: repository root (defaults to the parent of this folder)
- `TADA_ANTHROPIC_BASE_URL`: Anthropic-compatible endpoint
- `ANTHROPIC_API_KEY`: endpoint credential
- `TADA_JUDGE_BASE_URL`: OpenAI Responses-compatible judge endpoint

The model roles and frozen task-set notes are recorded in
`results/matrix_v2_config_note.json`.

## Scope of this release

The aggregate result tables are included. Raw model traces and 107,694
provider calls are not included because they contain large generated
intermediates and provider-specific telemetry. The human calibration inputs
and all six rater files are fully released in `../calibration_release/`.

The product snapshot predates the final v2 controlled matrix and is retained
only as a dated external-system snapshot. Its commercial backing models could
not be pinned.
