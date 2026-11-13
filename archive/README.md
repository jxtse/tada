# Archive

Earlier TADA research artifacts, kept unchanged for provenance. They describe the original planner/executor agent and the first paper submission, not the current agent (`../agent/`) or the task-generation pipeline (`../benchmark/`). Scripts here keep their original layout; paths inside them refer to their pre-archive locations and may need adjusting before reuse.

| Path | Contents |
|---|---|
| `TABench/` | Original 26-query release, frozen 43-query snapshot, answers, and checksums |
| `TADASkill/` | Original skill, `ta.py` operator library, and prompts that the new harness was reviewed against |
| `evaluation/` | Historical controlled-matrix, judging, and aggregation scripts and results |
| `calibration_release/` | Human calibration data and analysis code |
| `artifact_release/` | Historical public-benchmark, large-scale, and diagnostic artifacts |
| `supplemental_material/` | Submitted supplementary PDF and checksum |
| `.env.example` | Environment variables used by the historical experiment runners |

The historical paper protocol scores 40 cases after excluding Q11, Q12, and Q18; see [the evaluation protocol](evaluation/results/final_protocol.json) and [evaluation documentation](evaluation/README.md). Reference outputs provide a relative comparison scale and are not ground truth. The supplementary material is [this PDF](supplemental_material/TADA_supplementary_material.pdf) with its [checksum](supplemental_material/SHA256SUMS.txt).

Benchmark and calibration data are licensed under CC BY 4.0 ([`../DATA_LICENSE`](../DATA_LICENSE)); code under MIT ([`../LICENSE`](../LICENSE)).
