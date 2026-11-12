# TADA

TADA is a text analysis harness in which a Pi agent composes two semantic functions with ordinary Python and pandas. Each session owns one persistent Jupyter kernel. Variables, functions, tables, and successful semantic results remain available across turns in that kernel.

- `tag(texts, query)` returns one **string** per input row, aligned to stable row IDs.
- `categorize(texts, query)` returns a reusable category table with host-assigned category IDs.

The bounded first-version checks pass through the actual Pi controller, persistent Jupyter kernel, and configured providers, including multi-turn analysis, compaction, state recovery, and cancellation. Independent review and parent verification support this scoped result. See [verification status](docs/verification.md) for evidence scope and remaining paper-method limitations. Existing paper experiments remain historical artifacts.

## Installation

Use Python 3.11 or later, Node.js 22 or later, `uv`, and a project-local environment. Pi is pinned to version 1.5.0. No global Python or Node installation is needed.

```bash
uv sync --dev
uv run python -m tada_runtime.schemas
npm ci
npm run build
```

Python dependencies are locked in uv.lock and Pi is pinned in package-lock.json. A clean build and actual SDK tests have passed with these dependencies. The tested provider combinations and remaining limitations are described in the verification document.

Create a private profile from the public example, then edit its exact model IDs and API prefixes:

```bash
mkdir -p .tada
cp examples/config.toml .tada/profile.toml
```

API keys are read only from the environment variables explicitly named by the profile. Set those variables through your usual secret-management workflow. Do not put credentials in a command-line argument or tracked file. Agent and operator models have separate configurations; there is no automatic model or protocol fallback.

The agent supports explicitly configured Anthropic Messages, OpenAI Responses, or OpenAI Chat adapters through Pi. Operators implement Responses and Chat adapters. A `base_url` is the API prefix, such as `https://provider.example/v1`. `auth_mode = "optional"` is restricted to explicitly configured local endpoints; it is not proof that an endpoint accepts unauthenticated calls.

A finite request limit and deadline are mandatory for inference. The ledger path is cumulative across sessions, and every main-model, compaction, intent, batch, reduction, and failed or uncertain attempt consumes its reservation. The current implementation caps the ledger at 60 requests and operator concurrency at two. These limits are not a monetary cost estimate or an authorization to spend.

## Run a session

The sample reviews contain synthetic, non-sensitive data:

```bash
uv run tada doctor --config .tada/profile.toml
uv run tada run --config .tada/profile.toml \
  --input reviews=examples/reviews.csv \
  --task "Discover issue categories, materialize category labels, count them by region, and export a CSV." \
  --interactive
```

`doctor` checks configuration, dependencies, and model-list endpoints without requesting inference. `doctor --probe-models` explicitly starts a small model-backed tool probe and requires the same finite budget.

Within the interactive session, enter a follow-up normally, use `/compact` to request explicit context compaction, and `/quit` to close the session. For automation, repeated `--followup` arguments execute in the same session; `--compact-between` requests compaction before each follow-up:

```bash
uv run tada run --config .tada/profile.toml \
  --input reviews=examples/reviews.csv \
  --task "Materialize issue labels and count by region." \
  --followup "Reuse the existing labels and count by month." \
  --compact-between
```

Both interactive use and evaluation drivers use this same entry point. The CLI emits JSON records with a session ID and an explicit completion, cancellation, or failure status. A completed turn is not an assertion that its analytical result is correct.

## Python namespace

The model sees one execution tool, `execute_python`. The kernel exposes `pd`, `np`, `json`, the named input tables, `tag`, `categorize`, `state`, `preview`, `validate_category_references`, `labels_frame`, and `outputs_dir`. Generated Python executes sequentially. CSV inputs use their unique `rid` column as the index; inputs without a `rid` column receive frozen IDs at load time.

```python
categories = categorize(reviews.text, "Discover service issue categories", name="issues")
labels = tag(
    reviews.text,
    "Return exactly one cid from this frozen category table: " + categories.to_json(),
    name="issue_labels",
)
validate_category_references(labels, categories)
reviews["issue"] = labels
state.put("enriched_reviews", reviews)
reviews.groupby(["region", "issue"]).size().to_csv(outputs_dir / "regional_issues.csv")
```

A later cell can regroup `reviews` without any new semantic request. See [the example workflow](examples/workflow.py).

Semantic inputs must be a Series of strings with unique, non-empty, consistently typed row IDs. Missing values and non-text values require explicit preprocessing. Native object, array, numeric, boolean, or null tag values are rejected. A structured extraction can be a JSON **string**, which ordinary Python can explicitly parse afterward.

Large category requests accept an explicit `cardinality={"k": 5, "mode": "exact"}` (or `"at_most"`) contract. Without it, the runtime resolves the requested count in one internal structured call; it never silently defaults arbitrary queries to five categories. The enabled `runtime.large_scale` settings and conservative capacities are exposed in [the example config](examples/config.toml).

When the allowed answers are known, pass them as `labels=` (a list of names, a list of `{"name", "description"}` records, or a `categorize()` table). With the default `multi=False` each value must be exactly one label name. With `multi=True` each value is a JSON-encoded array string of every applicable label, possibly `"[]"`, so several yes/no or topic checks share one pass over each text; `labels_frame(series, labels)` expands it into a boolean table. Supplied labels make the task independent by construction, so no intent request is made. Validated label sets are re-serialized in the caller's label order.

```python
clauses = tag(contracts.text, "Which of these provisions does the agreement contain?",
              labels=["Exclusivity", "Termination for convenience", "Change of control"], multi=True)
flags = labels_frame(clauses, ["Exclusivity", "Termination for convenience", "Change of control"])
```

Rows too long for one request are handled inside `tag` (`runtime.long_rows`). By default only rows longer than `batch_chars`, which previously failed, are cut into overlapping segments that prefer paragraph boundaries; later segments carry the document's opening characters. Segment answers are committed as a child state and aggregated by a fixed rule: with `labels` and `multi=True` the row value is the union of segment label sets; otherwise one extra reduce request per long row combines the ordered segment answers. `long_rows={"aggregate": "collect"}` instead returns a JSON array of the non-empty segment results for free-text extraction. The plan event records the rule, effective configuration, and segments per row. Rows needing more than `max_segments_per_row` segments, segment capacity above `batch_chars`, and joint discovery over long rows are refused before any request.

Tagging without `labels` first detects whether the query has a fixed output domain or requires joint discovery. Joint discovery materializes a child category state before assignments use that frozen table. Batches retain explicit row IDs; incomplete, duplicate, or unknown IDs fail the atom. A failed batch never becomes a partially successful public result. `runtime.batch_retries` (default 1, at most 2; 0 restores strictly one dispatch per request) resends a tag operator request (batch, segment batch, segment reduce or intent) whose response broke the output contract (row coverage, labels, JSON), with a note naming the violated rule, or whose response ended `failed`/`incomplete` or with HTTP 408/429/5xx; every retry is a separate budget-counted request whose attempt record carries `retry_of`. Unknown outcomes, budget exhaustion and cancellation are not retried. Provider failure messages include the upstream error code or incomplete reason. An identical query is classified by the intent request once per session (`fresh=True` re-classifies). Every `execute_python` observation reports `budget` with requests used and remaining and seconds remaining.

Successful semantic operations commit immutable artifacts before returning. A later error in the same cell does not undo that commit. Exact reuse checks the input content and order, query, model parameters, prompt and engine versions, batching plan, cardinality, retention configuration and effective RNG seed. `fresh=True` or `runtime.reuse = "off"` creates a new replicate. An alias only names a state; rebinding an alias preserves its older artifacts.

## Resume and inspect

```bash
uv run tada resume SESSION_ID --task "Inspect the registered states and continue the analysis."
uv run tada log validate SESSION_ID
uv run tada log view SESSION_ID
uv run tada log export SESSION_ID --format html --output review.html
uv run tada log export SESSION_ID --format jsonl --output review.jsonl
```

The HTML view assembles streamed deltas per request and summarizes raw provider transport chunks; add `--include-transport` to inline every raw chunk (large). Blobs over 16 KiB are referenced by path and verified SHA-256 instead of inlined. `events.jsonl` and `blobs/` remain the complete record.

`resume` explicitly acquires the session write lock, validates the Pi checkpoint against the audit prefix, restores committed registered objects, and reports transient namespace loss. It does not restore arbitrary functions, connections, generators, or unregistered variables. Supported snapshots are typed JSON and common flat-index pandas Series/DataFrames, including dtype and category metadata. No pickle or dill is used. CategoricalIndex categories that cannot retain their scalar types through typed JSON are rejected before registration; normalize their dtype explicitly rather than relying on silent conversion. MultiIndex objects must be explicitly flattened before registration. Missing tabular values use JSON null in the portable representation.

The authoritative log is `.tada/sessions/<session-id>/events.jsonl`. Inputs, states, provider payloads, tool output, and checkpoints live beside it. Events distinguish reasoning, text, tool calls, observations, failed attempts, child semantic calls, compaction, and uncertain outcomes. Reported model identity is recorded separately from the requested ID. A gateway alias is not proof of the upstream model identity.

Log commands are offline and read-only with respect to the source session. They never execute saved Python or contact a provider. The HTML viewer embeds escaped content and supports filters for session, turn, step, role, event type, state, and errors, plus text search and a nested request outline; reasoning starts collapsed. JSONL export preserves complete audit events. Readers report an incomplete physical tail without repairing it; only a write-owner resume can reconcile it. Checkpoint mismatches fail explicitly.

Run directories and private profiles under `.tada/` are ignored by Git. Logs can contain user input and should be treated as private. Credential fields and known secret strings are redacted before persistence. Typed streaming channels use bounded buffering to redact known secrets split across fragments. When credentials are present, opaque raw-only transport copies are marked as redacted because encoded fragments cannot safely be published as exact raw data; credential-free transport copies retain their original bytes. This is not a general data-loss prevention boundary.

## Execution boundary

This is a **trusted-local** developer harness. Generated Python has the user's local permissions. A single tool, a stripped Pi loader, and keeping provider credentials out of the kernel environment do not create a security sandbox. Untrusted or unattended workloads require external filesystem, network, and resource isolation that has not been certified here.

Ordinary Python side effects are not transactional. Cancellation interrupts the owned kernel and preserves uncertainty about already dispatched provider calls; uncertain calls are not automatically resent. Recovery restores committed data, not the whole process.

## Tests

```bash
uv run python -m pytest
uv run python -m tada_runtime.schemas --check
npm run schemas:check
npm test
npm run build
```

Tests cover strict validation, row coverage, batching and joint traces, exact reuse, immutable state recovery, durable logging, redaction, checkpoint consistency, and offline export. Real integration acceptance is a separate gate; see [verification](docs/verification.md).

## Research artifacts

This repository also contains the artifacts for **TADA: Semantic Atoms for Scalable Text Analysis in Data Analytics in the Era of LLM-Based Agents**.

| Path | Contents |
|---|---|
| `packages/agent/` | Pi controller, tool adapter, and CLI |
| `python/tada_runtime/` | Supervisor, Jupyter bridge, strict contracts, audit and state storage |
| `python/tada_operators/` | Tagging, categorization, intent, batch and reduce implementation |
| `schemas/` | Generated Pydantic JSON Schemas and content hashes |
| `TADASkill/` | Original skill and operator reference assets; not the new agent entry point |
| `TABench/` | Benchmark inputs, frozen query snapshot, answers, and checksums |
| `calibration_release/` | Human calibration data and analysis code |
| `evaluation/` | Historical controlled-matrix, judging, and aggregation artifacts |
| `artifact_release/` | Historical public benchmark, large-scale, and diagnostic artifacts |
| `supplemental_material/` | Submitted supplementary PDF and checksum |

The old planner/executor agent entry point has been replaced. Historical evaluation scripts may still describe that earlier pipeline; they are retained as evidence, not presented as compatibility wrappers for the new CLI.

The supplementary material is available in [the PDF](supplemental_material/TADA_supplementary_material.pdf), with its [checksum](supplemental_material/SHA256SUMS.txt). The original 26-query release and frozen 43-query snapshot are in `TABench/`; the historical paper protocol scores 40 cases after excluding Q11, Q12, and Q18. See [the evaluation protocol](evaluation/results/final_protocol.json) and [evaluation documentation](evaluation/README.md) for scope and metric definitions. Reference outputs provide a relative comparison scale and are not ground truth.

Large Categorization and Joint Tagging now use the paper-guided pilot/Tag-count/paired-bootstrap retention path. It selects a local candidate budget and sends all proposals plus pilot support to one global reducer; an over-capacity request is explicitly refused. The fixed-taxonomy numeric model is conditional guidance, not a guarantee of open semantic quality. See [retention behavior and configuration](docs/retention.md).

Code is MIT licensed; benchmark and calibration data use CC BY 4.0. See [LICENSE](LICENSE), [DATA_LICENSE](DATA_LICENSE), and [third-party notices](THIRD_PARTY_NOTICES.md). Citation metadata will be updated after publication; until then, cite the accompanying manuscript.

CLI overrides use `--set dotted.path=JSON_VALUE`, for example
`--set 'runtime.batch_rows=8' --set 'models.agent.model="EXACT_MODEL_ID"'`.
They take precedence over the selected project profile and the TADA user profile.
The merged configuration is validated by Pydantic before any provider dispatch.
