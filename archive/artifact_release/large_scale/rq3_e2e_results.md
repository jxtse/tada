# RQ3 audited large-scale results

Quality uses the saved normalized-metrics artifact. Normalized accuracy and macro-F1 divide by the corresponding reference-round pairwise metric. Contested reference rows are excluded from scoring, not from processing. Em dashes are unavailable measurements, never zeroes or estimates.

| Workload | System | valid n | Acc. | Macro-F1 | Norm. Acc. | Norm. Macro-F1 | observed calls | physical tokens | wall min | coverage | scored n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sentiment tagging (32K) | TADA-Agent | 3 | 0.973±0.000 | 0.928±0.003 | 1.017±0.001 | 1.048±0.003 | 57.0±41.6 | 1196222.5±63102.9 | 64.0±87.1 | 1.000±0.000 | 32130 |
| Sentiment tagging (32K) | Claude Code | 3 | 0.817±0.047 | 0.577±0.015 | 0.854±0.049 | 0.652±0.018 | — | — | 16.2±11.5 | — | 32130 |
| Sentiment tagging (32K) | LOTUS | 0 | — | — | — | — | — | — | — | — | — |
| Topic assignment (32K) | TADA-Agent | 3 | 0.942±0.002 | 0.838±0.041 | 1.038±0.002 | 1.051±0.051 | 33.0±0.0 | 1546075.3±25829.8 | 32.6±3.9 | 1.000±0.000 | 31972 |
| Topic assignment (32K) | Claude Code | 2 | 0.820±0.138 | 0.665±0.271 | 0.903±0.151 | 0.833±0.339 | — | — | 37.9±42.4 | — | 31972 |
| Topic assignment (32K) | LOTUS | 0 | — | — | — | — | — | — | — | — | — |
| Feedback categorization (7K) | TADA-Agent | 3 | 0.837±0.015 | 0.829±0.007 | 1.098±0.020 | 1.091±0.009 | 7.0±0.0 | 309655.7±24730.1 | 5.7±3.7 | 1.000±0.000 | 6430 |
| Feedback categorization (7K) | Claude Code | 3 | 0.658±0.013 | 0.605±0.023 | 0.863±0.017 | 0.797±0.030 | — | — | 2.8±1.0 | — | 6430 |
| Feedback categorization (7K) | LOTUS | 0 | — | — | — | — | — | — | — | — | — |

## Telemetry and evidence notes
- TADA uses planned chunk counts of 33/33/7 for sentiment/topic/category. Topic and category have exactly those successful calls in every run.
- TADA sentiment rep1 is retained as an anomalous but valid quality run: 105 observed attempts, 36 failed calls, and 9,876.5 s wall time. Thus 69 attempts succeeded while only 33 chunks were planned; retries make observed attempts differ from the plan. Reps 2 and 3 each record 33 calls, zero failures, and concurrency 3.
- Sentiment token telemetry exists for only 2/3 TADA runs (n_usage=2). Its provider-reported physical totals are reported only over those two instrumented runs; rep1 tokens are unavailable.
- Claude Code calls and token totals cannot be recovered from the retained process artifacts and remain em dashes. Its timed-out topic rep2 is not a valid quality run (10,799.9 s); valid topic n=2.
- No completed `lotus_cat_tm_rep1.json` exists at freeze. The empty partial checkpoint is not evidence. Accordingly the full-scale LOTUS row remains unavailable.
- The only completed LOTUS scale sanity check available to this experiment is a separate 500-row sentiment pilot (94.8% accuracy, 25.7 min, concurrency 4). It is explicitly a pilot, is not entered into the 6,628-row category table, and is not extrapolated.
- LOTUS would be n=1 if completed; no variance, significance, or stability claim is made.
