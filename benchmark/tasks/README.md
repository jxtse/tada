# Admitted tasks (pilot)

Task packages are hosted in a private Hugging Face dataset, because they contain text and
labels from source datasets whose licences have not all been cleared for redistribution:
<https://huggingface.co/datasets/jxtse/tada-bench> (request access from the maintainer).

```bash
hf download jxtse/tada-bench --repo-type dataset --local-dir tada-bench
harbor run -p tada-bench/tasks/<task-id> -a oracle        # reference solution, expect 1.0
```

[`tasks.json`](tasks.json) records each task's sources, difficulty, trial rewards, oracle reward
and a file-tree SHA-256 (sorted `path\0sha256` lines) for verifying a download.

| Task | Sources | Difficulty | DeepSeek-Flash ×2 | Claude Opus ×2 | Oracle |
|---|---|---|---|---|---|
| `maud-pilot` | MAUD | easy | 0.911 / 0.911 | 0.980 / 0.980 | 1.0 |
| `bird-infiagent-oolongsynth` | bird, infiagent-dabench, oolong-synth | medium | 0.747 / 0.696 | 0.905 / 0.884 | 1.0 |
| `dabstep-semtpch-spider2` | dabstep, semtpch, spider2 | hard | 0.438 / 0.420 | 0.413 / 0.413 | 1.0 |
| `prec-loft-lrobench` | PREC, loft, lrobench | hard | 0.673 / 0.592 | 0.689 / 0.670 | 1.0 |
| `infiagent-kramabench-palimpzest` | infiagent-dabench, kramabench, palimpzest | easy | 0.873 / 0.862 | 0.890 / 0.890 | 1.0 |
| `biodex-udabench` | biodex, uda-bench-vldb | hard | 0.691 / 0.743 | 0.736 / 0.749 | 1.0 |
| `blade-udabench` | blade, uda-bench-vldb | hard | 0.644 / 0.681 | 0.668 / 0.670 | 1.0 |
| `b-dabstep-tablebench` | B, dabstep, tablebench | medium | 0.717 / 0.732 | 0.818 / 0.820 | 1.0 |
| `cuad-maud-lrobench` | CUAD, MAUD, lrobench | easy | 0.853 / 0.810 | 0.915 / 0.910 | 1.0 |

Difficulty: easy = at least 3 of 4 trials reach reward 0.8, medium = 2, hard = fewer than 2.
It is a stratum, not an admission threshold.
