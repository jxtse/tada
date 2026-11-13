# TADA task generation v5

v5 separates source labels, designer specifications, compilation, scoring, and
admission. Historical v4-only tools are not included in the public export. Use
fake clients or `--dry-run` for offline validation; real runs require configured services.

## Offline validation

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR="$PWD/v5/.uv-cache" uv run --offline --project v5 --no-sync pytest v5/tests -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=v5 python -m tada_taskgen.screening --source MAUD --source CUAD
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=v5 python v5/compile_task.py \
  --spec v5/specs/maud_screened.json --seed 7 --out v5_runs/example/task
```

Historical R2 screening measures the complete current label pool; under R4 slim
its results are report-only and do not veto compilation or intent generation.
Reports in `v5_runs/screening/` bind the pool, quarantine, question metadata,
keyword implementations and NB implementation hashes. Refresh explicitly after
those inputs change. Screening takes the maximum balanced accuracy of the
available dedicated rule, generic option-token rule, and five-fold stratified
out-of-fold multinomial NB (seed 0, train-fold document-frequency top 5,000 word
unigrams/bigrams, Laplace 1). Its limit is .65, with at least 30 labels, two
observed classes and at least five examples per class. NB uses pseudonymised full
texts; it is a screening proxy only. The solver keyword script stays unchanged.

`maud_screened.json` selects up to nine eligible questions by ascending shortcut
score, then descending label count, then question ID; n=16 stays fixed. Records
must have at least three requested scorable questions before the one random draw.
After sampling, questions without at least two classes and two cells per present
class are dropped and recorded; at least five questions must remain or the seed
fails. No resampling is allowed. The workspace review memo publishes the exact
per-record checklist; missing pairs are not an undisclosed applicability test.

Repair R3 first computes balanced recall within each retained question, then
uses `max(0, (balanced_recall - 1/k) / (1 - 1/k))`, where k is the full
codebook option count, and averages the corrected question scores. When some
codebook classes are absent in the instance, constant answers can retain positive
credit under this exact formula; no additional cap is applied. Weights remain labels
.70, scope .15, facts .10, consistency .05. Per-cell inverse encoding, equality and the bounded reader remain in force.
R4 slim uses exactly six hard groups: reference 1.0 plus invertibility; empty,
all-majority and constant answers each <= .30 total reward;
deterministic offline scoring and no gain from extra rows; visible leakage and
byte-identical permutation rebuilds; fixed keyword baseline <= .30; and an output
contract declaring all scorer-read sheets, columns and options. The output-contract
group also checks the fixed config canary in every public file and task metadata.
Historical scope-F1 thresholds, per-work-item baselines, single-label costs,
provenance/sensitivity and declared wrong-answer ranges remain reported-only.
The owner-approved amendment also makes include-all reported-only, retaining its
total reward and scope F1; it is not one of the trivial-answer checks.

Repair R5 keeps the legacy MAUD source API frozen. Its v5 adapter replaces only
distinctive party-name cores, preserving legal forms and rejecting prose parsed
as subsidiary names. Evidence grounding uses exactly the same text transform;
screening identities include its version and implementation hash. MAUD question
metadata publishes the source provision context before the question text.

The public memo and output guide explicitly require coding and summaries for
every assigned pair regardless of scope. `coding_conventions.json.scoring_policy`
maps the four weighted components to actual workbook columns and explains chance
correction, F1, precision penalties and conditional summary credit. Scope status
is scored once in its own component; empty components are renormalized away.
Scope errors never invalidate label or summary credit.
Known carrier/class-guard compilation failures send only declared document/fact/
question identifiers and aggregate counts to the designer, never extracted
values, source answers, record IDs or private exception prose.

The existing shared environment provides openpyxl/pandas; it is read-only. The local
uv project installs pytest only. The lock uses the PyPI mirror recorded in `uv.lock`.
Tests add the existing spreadsheet runtime through `tests/conftest.py`.

For a single-seed decision and an end-to-end offline scheduler run, use fresh output directories:

```sh
mkdir -p v5_runs/example-tmp
export TMPDIR="$PWD/v5_runs/example-tmp" PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=v5 python -m tada_taskgen.gates \
  --spec v5/specs/maud_screened.json --seed 7 --out v5_runs/example-single-seed
PYTHONPATH=v5 python -m tada_taskgen.scheduler \
  --spec v5/specs/maud_screened.json --out v5_runs/example-scheduler --dry-run
```

Add `--multiseed-report` to the gates command for optional five-seed diagnostics;
they never decide admission. `--existing-run PATH` evaluates a copied/rebuilt
existing run, reusing archived documents without new model requests.

The controller retains its macOS `sandbox-exec` boundary. Nested managed sandboxes
may deny it; execute the same offline command with approved local permissions.
`--no-harbor` skips trials, retains local Harbor lint, and records oracle/nop as
unmeasured. Admission uses the six local hard groups and the four-question review.

## Designer contract

`run_designer.py FAMILY --v5 --dry-run` assembles the prompt without a model call.
The live designer is restricted to four files: `dependency_graph.json`,
`scenario_spec.json`, `instruction.md`, and `design.md`. The scenario schema is
`schemas/scenario.schema.json`; `schema.validate_spec` also enforces graph acyclicity,
ordered dependencies, safe paths, and predeclared bounds.

The current compiler's declarative vocabulary is `record_scope`, `record_question`,
`question_count`, and optional `document_fact`. It supports 2–4 ordered work items,
with template roles bound explicitly to work items. A document fact binds a
`carrier` document ID plus its `fact_key`; it may be a scored prerequisite or input.
Unsupported graph operations fail closed instead of running designer code.
The two regression specs retain source questions and scenario provenance while
normalizing repeated labels into one coding register and summaries into conditional
consistency checks. They do not reproduce every historical scenario motif.

## Models, concurrency and admission

`clients.ModelClient.complete` is the injection boundary. `FakeClient` writes
transparent fixtures and the independent fake extractor reads only their text and
fact names. These documents are intentionally easy for the rule baseline; tests do
not establish semantic difficulty. `config.toml` records future model endpoints and
environment variable names, never credentials. The default `--client fake` makes no real calls. `--client real` enables the
configured writer Chat Completions and extractor Responses transports for the
parent to run; this implementation run tests them only with injected transports.
Scope/fact numbers from fake documents are not meaningful difficulty measurements.

For a later parent-run real-document build, pass `--client real` to either
`v5/compile_task.py` or `python -m tada_taskgen.gates`. The configured env names
are read only on uncached requests; no key files are created. Writer uses
`/v1/chat/completions`, extractor `/v1/responses`. `supported_parameters` is an
explicit capability list: add `temperature` and/or `seed` there together with
values only when the configured endpoint supports them. Both are otherwise
omitted; every sent value is logged and included in the cache identity.

Request/response JSON is stored in the task's `build/model_requests.jsonl`, without
HTTP headers and with the active token redacted. Successful responses use the repository content cache
`v5_runs/model_cache/`, addressed by the model, endpoint, operation and complete
request content. Cache hits work without credentials, including identical
spec/seed compilation into a fresh output directory. Tests use isolated cache
directories and injected transports. HTTP 429/5xx receive at
most four retries after the initial attempt, with 1/2/4/8-second backoff; other
failures stop. Copied controller/permutation builds require archived verified
documents and cannot silently initiate new model requests. Invalid or mismatched
archives fail closed. The writer requests professional prose without JSON or
fact tables; extraction receives only that prose and the field names.

The designer writes four files, then the controller compiles one seed and runs
the six hard groups plus Astra's four-question review (instruction sufficiency,
scoring fairness, realism/factual correctness, and text-analysis difficulty).
Severity >= 2 requests revision within the revision budget. One consolidated feedback message contains only
aggregate hard-gate measurements and public review findings. One initial design
has at most three revisions shared by both kinds of failure. After that budget,
remaining severity-2 review notes are reported; failed hard gates still discard it.

Passing candidates receive DeepSeek x2 and Opus x2 trials. Reward >= .80 counts as
a pass: >=3 passes is easy, 2 medium, <=1 hard; difficulty never rejects a task.
All-four common errors receive blind double rereads with agreement definitions.
Only agreeing high-confidence readings with structural evidence can trigger the
approved exclusion policy. Recompile/regate once after exclusion, and repeat trials
only when scored gold changed. Verdicts are admitted or discarded. The scheduler
maintains 10-verdict/15-trial-set caps, writes a manifest and task cards, and samples
10% of each batch for report-only redteam checks. Cards disclose public-dataset
memorisation risk. Cross-process locks are Opus 1, DeepSeek 3 and Astra 2.
Every production entry point must use those locks; they cannot control unrelated
clients outside this pipeline. Dry-run results never count as measured model scores.

`preaudit.preaudit` creates proposals only. `python -m tada_taskgen.preaudit --help`
shows the separate explicit apply command. It verifies exclusion-only policy,
creates a backup before replacement, and stops with a notification file if the
cumulative per-question exclusion share exceeds 5%. The implementation run never
applies proposals to the global quarantine.

## Dataset combinations (R6)

The available dataset IDs are B, CUAD, CUADX, MAUD, NDA and PREC. CUADX is a
distinct task-space ID even though it uses the same contracts as CUAD. PREC uses
original SCDB precedent-alteration labels, not negatives inferred from missing
LoC pairs. NTSB is excluded because its private finding identities cannot provide
label-independent public units while preserving each original C/F label.

Task spaces are all 41 nonempty subsets of these IDs with at most three sources.
`config.toml` controls the maximum subset size and candidate concurrency (default
2). Each sorted `+`-joined combination permits two admitted tasks; failures do
not fill that quota. Selection uses fewest admissions, then attempts, then a
seeded tie break. O*NET supplies occupational context only. State records the
enumeration/hash and atomically reserves the 10-verdict and 15-trial-set budgets,
including work in flight. Existing prepared candidates keep their original intent.

Multi-source specs declare `sources`, a `questions` map by source, and one
`record_question`/`question_count` template per source. `record_scope` is optional
per source. `sample_size` applies per source, with optional `sample_sizes`
overrides. Every source must retain at least one question satisfying the unchanged
two-classes/two-cells guard. Qualified record/question IDs prevent collisions;
gold retains the original `(source, record, question)` triple. Linking facts stay
in round-tripped carrier documents and summaries remain consistency-only.

Four trial futures preserve their submission order while sharing the existing
DeepSeek 3 / Opus 1 / Astra 2 machine slots. Candidate preparation may overlap
another candidate's trials. Request and trial timeouts remain unchanged; waiting
for a model slot is separate from a running request's timeout. Trial preflight,
all-infrastructure refunds and saved-intent resumption remain in force.

Read-only pilot accounting and a fresh offline smoke run:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=v5 python \
  -m tada_taskgen.scheduler --combinations-report --state v5_runs/pilot/run/state.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=v5 python \
  -m tada_taskgen.scheduler --dry-run --loops 3 --out v5_runs/new-combination-smoke
```

Dry runs preserve actual compile/controller measurements but use explicitly
marked gate/review/trial fixtures for scheduler lifecycle checks. A fixture
admission does not establish real task quality or model difficulty.

Generated acceptance evidence, replay measurements and run reports are written
under `v5_runs/` and are intentionally excluded from the public source export.

## Batch-1 corpus and benchmark items (R7)

`catalog/BATCH1_SHORTLIST.json` in the read-only Datasets project defines all 29
agreed IDs. `tada_taskgen.adapters.batch1_registry` accounts for each ID, including
explicit deterministic-scoring or local-input gaps. Legacy names remain aliases:
CUAD, NDA, MAUD, B and PREC. SCDB codes and CAP opinions form one linked source;
CUADX remains readable for old tasks but is not an extra batch-1 dataset.
`batch1_inventory.json` records the audited counts and source/checker limitations.

New corpus readers use the delivered `source_adapters.iter_units` API. The
versioned repair reader supplies declared asset bindings where the older entry
point lacks them; omitted dev/workload tables use repository-owned descriptors
with the same unified parser. Datasets files, plans and receipts remain read-only.
NTSB uses factual narratives and original event-level codes with source codebook
meanings. Private finding units, cause and analysis remain hidden.

Benchmark items have one original answer per scored cell. Their public records
contain the question and explicitly allowed inputs, with no gold options,
reference SQL/code or hidden tests. The private scorer applies the original
benchmark's deterministic checker. Benchmark item scores are averaged; existing
categorical chance correction remains unchanged. LLM-only evaluation is never
substituted with a guessed deterministic metric. Licensing limitations are
retained in metadata and task cards.

`[compiler].workspace_max_bytes` defaults to 2 GB. Oversized items are counted and
excluded before the fixed draw; a sampled aggregate or generated workspace over
the cap fails explicitly. Binary assets retain their original bytes and are
bound to the public canary manifest by SHA256. SQL verification is read-only,
uses SQLite authorization and an execution budget, and cannot attach databases
or write files. All six hard groups still apply.

`[scheduler].combination_max_size` is 3; its old spelling remains a checked alias.
Seeded selection prioritizes previously unused datasets, then the existing
admission/attempt ordering. Enumeration drift still requires a fresh state;
never silently reuse the R6 pilot state for the expanded task space.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=v5 python \
  -m tada_taskgen.scheduler --datasets-report
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=v5 python \
  -m tada_taskgen.scheduler --combinations-report
```
