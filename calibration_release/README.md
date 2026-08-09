# Human Calibration of the LLM Judge (39-Unit Stratified Sample)

Supplementary data and analysis code for the judge-validity calibration
reported in the paper. Six human annotators independently and blindly
labeled 39 judgment units drawn from the row-aligned TA-Bench tasks;
this directory contains the exact stimuli, all raw annotations
(anonymized), the per-unit LLM-judge verdicts, and a script that
reproduces every number in the paper's calibration paragraph and
appendix table.

## Contents

| File | Description |
|---|---|
| `calibration39_key.json` | The 39 judgment units: task query, source text, both candidate answers (anonymized A/B), sampling stratum, and the internal answer key (`_secret_A_is`: whether candidate A is the full-system output). Annotators never saw the fields prefixed with `_`. |
| `calibration39.html` | The exact self-contained annotation form shown to annotators (open in any browser; Chinese task guides, blind A/B presentation, one-click export). |
| `calibration39_blank_form.csv` | Spreadsheet rendition of the same 39 units (blank verdict column), for annotators preferring a tabular workflow. |
| `raters/rater1.csv` … `raters/rater6.csv` | Raw verdicts from the six annotators (`idx, verdict` with `verdict` in `A`/`B`/`tie`). Filenames are anonymized; the name-to-ID mapping is retained privately by the authors. |
| `judge_verdicts.csv` | Per-unit LLM-judge verdict (`A`/`B`/`tie`), aggregated from the raw judge logs under the paper's protocol (anonymized A/B, position swapping, three votes per direction; direction disagreement scores as a tie; adjacent-level differences on anchored rating scales tie by deterministic rule). |
| `analyze_calibration.py` | Reproduces all reported statistics from the files above. |

## Sample design

39 units, stratified by judge confidence: 21 *contested* (per-run or
per-direction judge verdicts conflicted), 12 *low-confidence*, 6
*high-confidence*. Eight task types are covered (translation,
summarization, document revision, typed-field extraction, urgency
triage, and three anchored rating tasks). Candidate sides were
randomized: the full-system output is candidate A in 21 of 39 units.

## Annotation protocol

Annotators saw, per unit: the task instruction, the complete source
row, and the two candidates as anonymous "A" and "B". They chose
*A better*, *B better*, or *tie / both acceptable*. For typed-field
extraction units, only the single highlighted field was judged (all
other fields were identical between candidates). Annotators worked
independently, without access to system identities, judge verdicts, or
each other's answers.

## Reproducing the numbers

```bash
python3 analyze_calibration.py
```

Expected headline output (paper values):

- Human majority reached on 33/39 units.
- Judge matches the human majority on 26/33 (78.8%, Cohen's kappa = 0.68),
  with 2 direction reversals; all other disagreements are
  tie-versus-preference.
- Inter-annotator agreement: mean pairwise raw agreement 0.55,
  Fleiss' kappa = 0.32.
- Judge vs. individual annotators: mean raw agreement 0.60
  (annotator range 0.36–0.77).
- Per-stratum: contested 17/19, low-confidence 4/8, high-confidence 5/6.
- All 6 majority-free units fall on anchored rating tasks with a
  one-step scale difference.

The script has no dependencies beyond the Python 3 standard library.
