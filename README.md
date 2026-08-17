# TADA-Agent

Official artifact repository for **TADA: Semantic Atoms for Scalable Text Analysis in Data Analytics in the Era of LLM-Based Agents**.

TADA exposes two reusable semantic operations for analytical agents:

- **Tagging** materializes one row-aligned semantic value per input row (`N -> N`).
- **Categorization** discovers and materializes a reusable category table (`N -> M`).

The agent composes these operations with ordinary Python/dataframe code. For large inputs, the public API remains unchanged while the runtime uses chunked execution, category consolidation, and discover-then-assign processing internally.

## Repository structure

| Path | Contents |
|---|---|
| `TADASkill/` | Reusable skill, semantic prompts, and the `ta.tag` / `ta.categorize` implementation |
| `TADAAgent/` | Planner, execution, payload, and repair pipeline |
| `TABench/` | Benchmark inputs, frozen query snapshot, rule-based answers, and checksums |
| `calibration_release/` | Blinded 39-unit human calibration set, six rater files, judge verdicts, and analysis code |
| `evaluation/` | Sanitized controlled-matrix, judging, aggregation scripts, protocol, and aggregate tables |
| `artifact_release/public_benchmarks/` | Recovered CUAD and BioDEX experiment materials, with scope notes |
| `artifact_release/large_scale/` | Final end-to-end large-scale run records, reference labels, and scoring scripts |
| `artifact_release/boundary_diagnostic/` | Availability note for the neural–symbolic diagnostic |

## Quick start

Requires Python 3.10+ and access to an OpenAI-compatible chat-completions endpoint.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r TADAAgent/requirements.txt
```

Configure a model endpoint. Do not commit real credentials.

```bash
export OPENAI_BASE_URL="https://your-provider.example/v1"
export OPENAI_API_KEY="dummy"
export OPENAI_MODEL="your-model"
```

See [`TADAAgent/README.md`](TADAAgent/README.md) for the planner/runtime entry point and [`TADASkill/SKILL.md`](TADASkill/SKILL.md) for the semantic APIs and environment variables.

A credential-free deterministic check reproduces the human-calibration statistics:

```bash
python calibration_release/analyze_calibration.py
```

## TABench and evaluation protocol

`TABench/Query.md` contains the original 26-query release. `TABench/queries_snapshot_v2.json` is the frozen 43-query snapshot (Q01–Q40, Q42, Q44, Q45). The paper scores 40 cases after excluding Q11, Q12, and Q18, whose output contracts fall outside the stated two-atom interface. Q41 and Q43 are distributed only as later source/gold additions under `TABench/expansion/`.

The final controlled study compares TADA-Agent, semantic operators, Claude Code without the TADA APIs, code-only execution, and a direct-model baseline. It routes cases to three separate metrics:

- reference-based row judgments for row-aligned cases without rule-based answers;
- reference-based complete-artifact judgments for non-row-aligned cases without rule-based answers;
- absolute accuracy for Q37–Q40, Q42, Q44, and Q45.

The common reference output provides a relative comparison scale and is not ground truth. See [`evaluation/results/final_protocol.json`](evaluation/results/final_protocol.json) and [`evaluation/README.md`](evaluation/README.md).

## Artifact scope

The repository contains the implementation, prompts, benchmark inputs, rule-based answers, human-calibration package, aggregate TABench results, recovered public-benchmark materials, and final end-to-end large-scale run records.

Some provider-backed evidence is intentionally or unavoidably absent:

- raw TABench provider traces and 107,694 judge calls contain large generated intermediates and provider-specific telemetry;
- the recovered CUAD handoff contains the implementation and smoke evidence but not the final three full prediction files;
- the later six-population candidate-retention trial bundle and the trial-level neural–symbolic diagnostic outputs were not present in the frozen local handoff.

The per-directory README files state these limits explicitly. No missing artifact has been reconstructed or represented as an original run.

## Data and model access

- No API keys or model credentials are included.
- Benchmark CSV files are included for research use under `DATA_LICENSE`.
- Reproducing model-backed runs requires provider access and may incur cost.
- Commercial-product outputs are retained only as a dated historical snapshot; their backing model versions could not be pinned.

## Citation

Citation metadata will be updated after publication. Until then, please cite the accompanying manuscript.

## License

Code is released under the MIT License. Benchmark and calibration data are released under CC BY 4.0; see [`LICENSE`](LICENSE) and [`DATA_LICENSE`](DATA_LICENSE).
