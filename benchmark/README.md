# TADA benchmark

Long-horizon text-analysis tasks built from existing expert-labelled datasets. Every scored cell of a task maps back, by a fixed reversible encoding, to one original expert label or one original benchmark answer; task generation never invents gold answers.

| Path | Contents |
|---|---|
| `taskgen/` | Task-generation pipeline: dataset-combination task space, designer loop with bounded revision, fixed compiler, offline hard gates, review, sandboxed trials, and exclusion-only label quarantine. |
| `tasks/` | Admitted tasks, each with its task card. |

Both directories are populated as the pipeline is published and the first batch is admitted.
