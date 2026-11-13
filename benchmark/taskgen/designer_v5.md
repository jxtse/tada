# TADA v5 scenario designer

Design a realistic professional text-analysis assignment. Write exactly four files:
`dependency_graph.json`, `scenario_spec.json`, `instruction.md`, and `design.md`.
The compiler owns build scripts, workbook scoring and Harbor infrastructure. Do not
write code, gold, per-record labels, scoring scripts or any additional files.

## Fixed principles

1. Every scored original-label cell corresponds bijectively to one used dataset
   `(source, record, question)` triple. The adapter's public encoding is bijective: only
   pseudonyms, record IDs, option normalization or translation are allowed. Scoring
   decodes the submitted value and compares it with the unchanged original label.
   Never relabel data or create many-to-one answer transformations. Follow official
   dataset scoring conventions through the adapter.
2. Use 2–4 ordered real work items. A prerequisite may determine a later item's
   scope or inputs. Scope precision/recall is separate from downstream coding, so
   an upstream scope error does not invalidate a correct label. If the work is
   parallel expert coding, declare `parallel_coding`; do not claim dependencies.
3. Preserve labelled source text, except reversible pseudonymisation and packaging.
   Linking facts belong to heterogeneous generated emails, notices, minutes or SOPs.
   The compiler asks a different model to extract each carrier's facts blind and
   requires an exact round trip. Failed facts cannot become scored prerequisites.
4. Derived routes, changes, counts and summaries are never independent gold. Mark
   them unscored or consistency-only, recomputed from the solver's own answers.
   Consistency credit is low weight and only available where underlying labels are
   correct. Do not invent majority-correction caps.

## Instruction and realism contract

Apply TASK_REALISM_SPEC R1–R6 with the approved v5 R1 wording: the opening request
may list ordered work items, but must not give methods, filenames, coding criteria,
thresholds or procedural instructions. Put coding conventions and workplace rules
in workspace documents. The instruction explains the professional purpose,
audience and deliverable. Do not expose benchmark mechanics or answer hints.
The compiler supplies output-location metadata separately.

## Scenario and graph

Use the supplied JSON Schema. A scenario declares its source, question IDs, scale,
ordered work items, document manifest, parameters and sheet layouts. A dependency
graph has original_label, construction_fact, derived_quantity and deliverable_column
node types; provision is given, in-document, solver-derived or build-internal. Every fact
names its carrier. Document origins identify source versus generated text.
Solver-derived values must not appear in any visible file. Every scored cell must
trace to an original label or a round-trip-verified construction fact. Supply the
same graph in the standalone graph file and embedded scenario specification.

Your fixed task space is the supplied dataset combination. Use every assigned
dataset and no others; O*NET provides realistic workplace context and does not
determine the combination. For a multi-source task, declare `sources` (1–3 IDs),
`questions` as a mapping from each source ID to its original question IDs, and
`sample_size` as the number of records to sample from each source. The supplied
catalogues contain exactly the assigned datasets. Original question IDs remain
local to their dataset; the compiler qualifies public identifiers to avoid collisions.

The assigned sources cover the agreed batch-1 text corpora and benchmark items.
Each catalogue includes its label-card summary, visible and hidden fields,
scorable-unit count, verification method and licensing limitations. A source with
`kind: benchmark_item` contributes original questions plus their original input
files. Use its declared `answer` question and the same per-source templates;
the compiler inserts each item's actual question into its record. The solver
returns that item's answer, checked by its original deterministic verifier.
Never invent answer options, reference SQL, reference code or gold answers in
public files. Judge-only parts are unscored context; unavailable sources cannot
be assigned. A mixed task may combine these items with record-label questions,
but no constructed answer may replace either original gold. Keep input files
within the configured 2 GB task workspace cap. CAP+SCDB form one linked source.

## Before measurement

Question-screening measurements are reported to help choose questions; they are
not additional admission gates. The compiler samples once, then drops questions
failing the two classes/two cells per class guard. Legacy single-source MAUD/CUAD
specs require three scorable questions per sampled record and five retained questions.
Combination specs and the new datasets require at least one scorable question per
record and at least one retained question from every assigned dataset. Publish
the retained per-record question checklist in the workspace review memo. Label
scoring applies max(0, (balanced recall - 1/k)/(1 - 1/k)) per question, with k
from the full codebook, then averages questions; weights are
labels .70, scope .15, facts .10, consistency .05. Never resample a failed seed.

Declare every scenario rule before compilation. The controller decides using one
fixed seed and exactly six mechanical checks: reference score 1.0 plus reversible
label encoding; empty/all-majority/constant answers each at most 0.30
total reward; deterministic offline scoring with extra rows unable to raise the
score; no answer leakage and byte-identical public files after label permutation;
keyword baseline at most 0.30 total reward; and an explicit output contract in the
instruction or template. The compiler adds the fixed public canary and checks it.
Other measurements (including multi-seed, per-work-item, single-label cost,
declared wrong-answer ranges, provenance, sensitivity, independent aggregation,
and solver rewards) are reports. Include-all preserves gold codes and widens scope;
its total reward and scope F1 are reported only. Trials assign difficulty after admission checks.
Never change seeds, weights, thresholds or sample rules after observing results.

An independent reviewer considers only instruction sufficiency, scoring fairness,
realism/factual correctness, and substantive text-analysis difficulty. Severity 2
or 3 requires a revision. The controller sends one consolidated feedback message
with failed aggregate checks and review findings. You may revise at most three
times after the initial design; unresolved failures then discard this candidate.
Read the existing four files before revising and address every listed defect.

In `design.md`, cite supplied O*NET task IDs and exact text, explain the deliverable
back-translation, source question coverage, carriers, dependency counterfactual,
scale, coding conventions, anticipated shortcuts, and weakest point. Estimates
are not measurements. No source code or per-record answers are available to you.

## Inputs

{{INPUTS}}

## Schemas

{{SCHEMAS}}

## Compiler contract

The compiler is fixed. It builds the workspace, the scope notices, the coding register
and the summary itself; you choose the scenario, questions, scale, work items, extra
carrier documents and their facts. The legacy single-source shape below uses
these nodes; the multi-source extension follows the example. Every graph node
must carry a `template`:

- exactly one `record_scope` node (construction_fact, in-document, scoring
  `prerequisite`, carrier `scope_revision`): which sampled agreements are in scope.
  The compiler generates the current and prior scope notices (carriers `scope_revision`,
  `scope_prior`); do not list them in `documents`.
- exactly one `record_question` node (original_label, in-document, scoring `label`,
  carrier `agreements/{record}.txt`): one coded cell per assigned (agreement, question).
- exactly one `question_count` node (derived_quantity, solver-derived, scoring
  `consistency-only`): the summary counts.
- any number of `document_fact` nodes (construction_fact, in-document, scoring
  `prerequisite` or `input`) with `carrier` = an id in `documents` and `fact_key` = one
  key of that document's `facts`. Use these for prerequisite facts in your own emails,
  notices, minutes or SOPs.

Edges must include record_scope → record_question and record_question → question_count.
If a work item sets `scope_from`, it must be the record_scope node id. Graph node ids are
yours. The standalone `dependency_graph.json`, `instruction.md` and `design.md` are
authoritative; you may omit those three fields inside `scenario_spec.json`.

Minimal valid shape (values illustrative; design your own scenario):

```json
{{EXAMPLE}}
```

For multi-source scenarios, use exactly one `record_question` and one
`question_count` node per source, each with a `source` field. A `record_scope`
node for a source is optional; where present, connect it to that source's
`record_question`. Connect each coding node to its own summary node. Carrier
`scope_revision` names and agreement paths are resolved within the node's source.
Cross-source workflow links are declared graph edges; factual linking information
belongs in generated `document_fact` carriers with blind round-trip verification.
Use these document facts to relate the work products without deriving new labels.
Summaries remain consistency-only for each source/question. Single-source shapes
above keep working without a `sources` list.

Two-source shape (the four files remain authoritative as above):

```json
{{MULTISOURCE_EXAMPLE}}
```

## Tools

`write_task_file`, `read_task_file`, `list_task_files` and `check_task_files` are
available, confined to the four files. Do not request shell execution, network tools,
source records, gold, or model trials. Write the design early, finish all four files,
then call `check_task_files`; fix every reported error and check again until it returns
ok. It is a format check only: the controller still measures the compiled task.
