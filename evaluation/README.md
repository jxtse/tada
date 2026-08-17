# Controlled evaluation materials

This directory contains the sanitized experiment logic and aggregate outputs for the final submission protocol.

## Final TABench protocol

The controlled study evaluates 40 cases from `TABench/queries_snapshot_v2.json`; Q11, Q12, and Q18 are boundary cases outside the two-atom contract and are discussed but not scored.

The five configurations are:

1. **TADA-Agent** — coding agent with the released TADA skill and atom APIs.
2. **Semantic operators** — LOTUS semantic-operator baseline.
3. **Claude Code** — the same coding-agent harness without the TADA skill or atom APIs.
4. **Code-only** — code-generation baseline without the TADA atoms.
5. **Direct** — direct model baseline.

`results/final_protocol.json` records the frozen case set, evaluation tracks, reference role, and judge protocol used by the paper. Historical identifiers that remain in script variable names are implementation compatibility keys; the publication-facing names above are authoritative.

## Pipeline

1. `run_matrix_v2.py` executes the controlled configurations.
2. `build_matrix_units.py` aligns delivered tables and constructs differing cell-level judgment units.
3. `run_v2_judge.py` requests three votes under both anonymized A/B directions.
4. `build_report_v3.py` and `reaggregate_tieband.py` aggregate the frozen judge ledgers and apply the one-step tie band for fixed integer-rating tasks.
5. `build_product_units.py` and `aggregate_product.py` reproduce a dated historical commercial-product snapshot; these files are not part of the final five-configuration comparison.

## Metric routing

- Row-aligned cases without rule-based answers use reference-based pairwise judgments on differing cells.
- Non-row-aligned cases without rule-based answers use complete-artifact judgments.
- Q37–Q40, Q42, Q44, and Q45 use absolute accuracy against rule-based answers and do not call the judge.

The held-out reference output provides a common relative comparison scale; it is not ground truth and the three tracks are not combined into a single TABench score.

## Configuration

The released runner reads:

- `TADA_PROJECT_ROOT`: repository root (defaults to the parent of this folder)
- `TADA_ANTHROPIC_BASE_URL`: Anthropic-compatible endpoint
- `ANTHROPIC_API_KEY`: endpoint credential
- `TADA_JUDGE_BASE_URL`: OpenAI Responses-compatible judge endpoint

## Scope

The repository includes aggregate result tables, configuration notes, and the fully released 39-unit human-calibration set. Raw model traces and 107,694 provider calls are omitted because they contain large generated intermediates and provider-specific telemetry. Reproducing model-backed executions requires compatible model access and may incur cost.
