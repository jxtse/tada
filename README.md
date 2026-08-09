# TADA-Agent

Official artifact repository for **TADA: Semantic Atoms for Scalable Text Analysis in Data Analytics in the Era of LLM-Based Agents**.

TADA exposes two reusable semantic operations for analytical agents:

- **Tagging**: materializes one row-aligned semantic value per input row (`N -> N`).
- **Categorization**: discovers and materializes a reusable category table (`N -> M`).

The agent composes these operations with ordinary Python/dataframe code. For large inputs, the public API remains unchanged while the runtime uses chunked execution, category consolidation, and discover-then-assign processing internally.

## Repository structure

| Path | Contents |
|---|---|
| `TADASkill/` | Reusable skill, semantic prompts, and the `ta.tag` / `ta.categorize` implementation |
| `TADAAgent/` | Planner, execution, payload, and repair pipeline |
| `TABench/` | Original benchmark inputs, frozen query snapshot, and the later controlled-matrix expansion |
| `calibration_release/` | Blinded 39-unit human calibration set, six rater files, judge verdicts, and analysis code |
| `evaluation/` | Sanitized controlled-matrix, judging, aggregation scripts, and aggregate result tables |

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
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="your-model"
```

See [`TADAAgent/README.md`](TADAAgent/README.md) for the planner/runtime entry point and [`TADASkill/SKILL.md`](TADASkill/SKILL.md) for the two semantic APIs and their environment variables.

## TABench

`TABench/Query.md` contains the original 26-query release. `TABench/queries_snapshot_v2.json` is the frozen 43-query snapshot used by the controlled study (Q01--Q40, Q42, Q44, Q45). The later source/gold additions and their checksums are under `TABench/expansion/`.

The paper's controlled analysis reports 40 scored cases: the frozen snapshot minus Q11, Q12, and Q18, whose required output contracts fall outside the two-atom interface's stated scope and are discussed as boundary evidence instead. Q41 and Q43 are not part of the frozen snapshot; their inputs and gold labels are distributed under `TABench/expansion/` for transparency only. See the manuscript and `evaluation/results/matrix_v2_config_note.json` for the frozen exclusions.

## Reproducing the human calibration analysis

The calibration analysis uses only the Python standard library:

```bash
python calibration_release/analyze_calibration.py
```

The directory includes the blank form, blinded HTML interface, six completed rater files, the released answer key, and the LLM-judge verdicts used in the paper.

## Evaluation materials

The `evaluation/` directory includes the code path used to:

1. run the five controlled configurations;
2. construct cell-level judgment units;
3. collect three position-swapped votes in each direction;
4. aggregate verdicts and apply the deterministic adjacent-level tie rule; and
5. aggregate the historical commercial-product snapshot.

Private serving addresses were replaced by environment variables. These scripts expect the benchmark inputs, skills, and model endpoints described in their headers; they are released as experiment logic and provenance rather than a one-command cloud reproduction. Aggregate tables and configuration notes are in `evaluation/results/`.

## Data and model access

- No API keys or model credentials are included.
- The benchmark CSV files are included for research reproducibility.
- Reproducing model-backed runs requires provider access and may incur cost.
- Commercial-product results are dated snapshots; their backing model versions could not be pinned.

## Citation

Citation metadata will be updated after publication. Until then, please cite the accompanying manuscript.

## License

Code is released under the MIT License. Benchmark and calibration data are released under CC BY 4.0; see [`LICENSE`](LICENSE) and [`DATA_LICENSE`](DATA_LICENSE).
