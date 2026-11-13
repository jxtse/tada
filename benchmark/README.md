# TADA benchmark

Long-horizon text-analysis tasks built from existing expert-labelled datasets. Every scored cell of a task maps back, by a fixed reversible encoding, to one original expert label or one original benchmark answer; task generation never invents gold answers.

| Path | Contents |
|---|---|
| [`taskgen/`](taskgen/README.md) | Task-generation pipeline: dataset-combination task space, designer loop with bounded revision, fixed compiler, six offline hard gates, review, sandboxed trials, and exclusion-only label quarantine. |
| `tasks/` | Admitted tasks with their task cards (added as the first batch is admitted). |

Datasets are not included. The pipeline reads the user's local copies through the paths configured in `taskgen/local_paths.toml`; each dataset's own licence applies.
