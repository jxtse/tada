---
name: TADASkill
description: Use when the user provides a tabular file (CSV/XLSX/TSV) or inline table together with a natural-language question about the text content of one or more columns — e.g., "tag each review's sentiment", "extract the postcode from each address", "summarize these feedback items into topics", "group reviews by organization and find each group's main complaints". Produces a new table with added columns, or a structured category list, by writing and executing a short pandas + ta.tag / ta.categorize Python program.
---

# TADASkill

Repository note: `TADASkill` is the current name of the original
`textanalysis` skill. It lives under `TADASkill/` in this repository,
and the supported primitive contract remains `ta.tag` and `ta.categorize`.

You orchestrate text-analysis-on-tables by writing and running a short Python program that uses two primitives from `lib/ta.py`:

- `ta.tag(data, column_name, query) -> list`  — one value per row (labeling, extraction, translation, per-row transformation).
- `ta.categorize(data, column_name, query) -> list[{"name", "description"}]`  — dataset-level summarization / topic discovery.

These two primitives internally handle batching, intent detection (Independent vs Joint tagging), cross-batch consolidation, and the underlying LLM calls. **Do not try to shortcut them** — they are the only supported way to invoke text analysis from inside this skill.

## When to invoke this skill

Invoke whenever the user's request is text-analysis-over-a-table. Typical cues:

- A file is attached or referenced (CSV/XLSX/TSV/Parquet), OR the user paste-drops a small table, AND
- The question targets the **text content** of at least one column (labeling, extracting, translating, summarizing, clustering, scoring).

Do **not** invoke for purely numeric/statistical questions over tables (use plain pandas for those), nor for free-form text questions without tabular input.

## How to execute a request

1. **Load the table**: `df = pd.read_csv(path)` or `pd.read_excel(path)`. Always keep column headers intact.
2. **Decompose the request** into a short sequence of atomic steps, each of which is either a plain pandas/numpy operation (filter, group, join, percentile, arithmetic) or a single `ta.tag` / `ta.categorize` call.
3. **Write a Python script** at `TADASkill/workspace/run.py` that imports `ta` from the skill's `lib/` folder, runs the steps, and writes the output (usually a new CSV + a concise stdout summary).
4. **Execute it** with `python TADASkill/workspace/run.py`. Stream stderr back to the user unchanged — the library prints brief progress lines (batch counts, Independent vs Joint path) that are useful context.
5. **Report** the output file path plus a short textual summary (e.g., the discovered categories, row counts per tag). Do **not** dump the full result table into the chat unless the user asked for it.

## API rules (mirrors the Excel docs, translated to pandas)

- Always pass the **column name** (string) matching a header in the DataFrame, not a column index.
- The primitives operate on **one text column per call**. Pre-combine columns into one in pandas if a task needs to consider multiple columns together.
- The primitives do **not** filter on other columns. If you want "tag only the negative reviews", first run `ta.tag` for sentiment, then `df_neg = df[df['sentiment'] == 'Negative']`, then `ta.categorize(df_neg, ...)`.
- The primitives do **not** chain. Each analytical step is an explicit call — including the "summarize into main topics → assign each row to a main topic → discover subtopics per main topic" hierarchical pattern. Write it out in Python.
- `ta.tag` returns a `list` aligned one-to-one with the input rows. Append it as a new column: `df['new_col'] = ta.tag(df, 'Review', '...')`.
- `ta.categorize` returns a list of dicts `{"name", "description"}`. It does **not** mutate the DataFrame; to turn it into row-level labels, feed the category names into a follow-up `ta.tag` call.

## Query phrasing tips (pass-through to the primitives)

- For `ta.tag`, the more constrained the query, the better. If you know the exact allowed tags, say so: `"Assign each review to exactly one of: Positive, Neutral, Negative, Mixed."` If the query implies a cardinality or perspective (e.g., `"limit to 7 topics"`, `"from a satisfaction perspective"`), preserve that phrasing — the intent detector looks for those cues to route Independent vs Joint.
- For `ta.categorize`, be explicit about the **decomposition axis**: topic, sentiment, genre, issue type, etc. If the user wants a fixed count, say `"into exactly N categories"`.
- If the user's query mixes per-row labeling and dataset summarization (common!), decompose: one `ta.categorize` followed by one `ta.tag` that uses the discovered category names.

## Reference

- `references/examples.md` — Python translations of the 10 canonical examples from the Excel Text Analysis API docs. Skim these to pick the right decomposition pattern; copy the idiom rather than invent a new one.
- `lib/ta.py` — the primitives' implementation. Read the module docstring if you need to understand input types (DataFrame / Series / list[str] are all accepted).
- `prompts/` — the four system prompts backing the primitives (tagging, categorization, intent detection, consolidation). **Do not read these when deciding how to solve a task** — they are LLM-facing and not part of the orchestrator contract. They are listed here only for completeness.

## Environment

- Backend selection is controlled by `TA_BACKEND` (`auto` / `openai` / `claude`) plus `TA_OPENAI_BASE_URL` or `OPENAI_BASE_URL` for OpenAI-compatible HTTP runs.
- The paper-facing OpenAI-compatible default is `gpt-4.1` with local fallback disabled. Set `TA_BACKEND=openai`, `TA_OPENAI_MODEL=gpt-4.1`, and `TA_ENABLE_LOCAL_FALLBACK=0` for measured TADASkill atom runs. Set `TA_OPENAI_MODEL=gpt-5.5` explicitly only for oracle_silver/reference generation.
- With `TA_BACKEND=openai`, OpenAI-compatible HTTP failures fail closed and do not fall back to the Claude CLI path. Only `TA_BACKEND=auto` may fall back to the historical CLI path after an HTTP backend failure.
- The Claude CLI path is retained for historical smoke tests and local debugging. `TA_MODEL` (default `haiku`) picks the Claude model used by that backend, and `TA_FALLBACK_MODEL` (optional) is used automatically when the primary model fails after retries.
- `TA_MAX_CONCURRENCY` (default 8) caps parallel batch subprocesses.
- `TA_CALL_TIMEOUT_S` (default 300) is the per-subprocess timeout.
- `TA_MAX_RETRIES` (default 2) retries each failed sub-call before escalating.
- `TA_RETRY_BACKOFF_S` (default 1.0) sets exponential retry backoff.
- `TA_TAG_BATCH_ROWS` (default 50) and `TA_CATEGORIZE_BATCH_ROWS` (default 1000) tune per-call row batch sizes.
- `TA_ALLOW_RAW_SINGLE_TAG_CELL=1` permits an explicitly requested raw text cell when a one-row tagging subcall omits the normal JSON envelope; it is off by default and never relaxes multi-row alignment.
- `TA_MIN_SPLIT_BATCH_ROWS` (default 25) is the minimum recursive split size when a batch fails.
- `TA_ENABLE_DEDUP` (default 1) deduplicates repeated text rows in large tagging tasks to reduce token/time cost.
- `TA_DEDUP_MIN_ROWS` (default 2000) is the threshold to activate dedup.
- `TA_ENABLE_CACHE` (default 0) enables local JSON caching of sub-call results in `workspace/ta_cache.json`.

Large-scale behavior now includes retry, adaptive split-on-failure, and optional dedup/cache. For paper-facing benchmark runs, keep `TA_BACKEND=openai`, `TA_OPENAI_MODEL=gpt-4.1`, and `TA_ENABLE_LOCAL_FALLBACK=0`; do not use `TA_FALLBACK_MODEL` as a measured-run downgrade path. `TA_FALLBACK_MODEL` is only for the historical/local Claude CLI path. Tune only one variable at a time when profiling.
