# Canonical Text Analysis examples — in Python

Python translations of the 10 canonical examples from the Excel Text Analysis API docs. These are idiom references: when you recognize the shape of a user request, copy the closest example and adapt. All scripts assume:

```python
import sys, json
from pathlib import Path
import pandas as pd

# make the skill's ta module importable from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import ta

df = pd.read_csv("input.csv")   # or pd.read_excel(...), pd.read_parquet(...)
OUT = Path(__file__).parent     # workspace/
```

---

## Example 1 — Simple extraction ("extract postcode from address")

```python
df["Postcode"] = ta.tag(df, "Address", "Extract the postcode from the address")
df.to_csv(OUT / "result.csv", index=False)
```

## Example 2 — Predefined buckets ("classify sentiment into very good / good / modest / bad")

```python
df["Sentiment Category"] = ta.tag(
    df, "Review",
    "Assign each review to one of the following categories: very good, good, modest, bad",
)
df.to_csv(OUT / "result.csv", index=False)
```

## Example 3 — Tagging by example (discover vocab → assign)

Two-step because the primitives do not auto-chain:

```python
categories = ta.categorize(
    df, "Title",
    "Identify high-level book categories similar to: Medical, Education, Literature",
)
names = ", ".join(c["name"] for c in categories)
df["Book Category"] = ta.tag(
    df, "Title",
    f"Assign each book title to one of the following categories: {names}",
)
df.to_csv(OUT / "result.csv", index=False)
(OUT / "categories.json").write_text(json.dumps(categories, indent=2, ensure_ascii=False))
```

Note: this pattern is *also* what the `ta.tag` Joint path does internally. If the user's query is already phrased as a cardinality-constrained tag request (e.g. "tag each review with a topic, at most 7 topics"), just call `ta.tag` directly and let the intent detector route it.

## Example 4 — Dataset-level summarization

```python
categories = ta.categorize(
    df, "Review", "Summarize the reviews into no more than 10 topics",
)
(OUT / "categories.json").write_text(json.dumps(categories, indent=2, ensure_ascii=False))
```

## Example 5 — Category merge ("merge topics 1 and 3, keep the rest")

A single `categorize` call with the merge instruction baked into the query:

```python
query = (
    "Summarize the reviews into 4 topics based on the previous topics: "
    "1. Delays in delivery, 2. Product quality issues, 3. Customer service delays, "
    "4. Price complaints, 5. Website experience. "
    "Merge topics 1 and 3 into a single topic, and keep the others."
)
categories = ta.categorize(df, "Feedback", query)
```

## Example 6 — Multi-output tagging (province / city / postcode as 3 columns)

```python
df["Province"] = ta.tag(df, "Address", "Extract the province from the address")
df["City"]     = ta.tag(df, "Address", "Extract the city from the address")
df["Postcode"] = ta.tag(df, "Address", "Extract the postcode from the address")
df.to_csv(OUT / "result.csv", index=False)
```

Optional speedup with `concurrent.futures` when the columns are independent:

```python
from concurrent.futures import ThreadPoolExecutor
specs = [
    ("Province", "Extract the province from the address"),
    ("City",     "Extract the city from the address"),
    ("Postcode", "Extract the postcode from the address"),
]
with ThreadPoolExecutor(max_workers=3) as ex:
    results = {col: ex.submit(ta.tag, df, "Address", q) for col, q in specs}
for col, f in results.items():
    df[col] = f.result()
```

## Example 7 — Hierarchical topic analysis (main → sub)

```python
# Step 1: discover 5 main topics
main_topics = ta.categorize(df, "Review", "Summarize into 5 main topics")
main_names = ", ".join(c["name"] for c in main_topics)

# Step 2: assign each review to a main topic
df["Main Topic"] = ta.tag(
    df, "Review",
    f"Assign each review to one of the following topics: {main_names}",
)

# Step 3: for each main topic, discover 3 subtopics from its rows
hierarchy = []
for topic in main_topics:
    subset = df[df["Main Topic"] == topic["name"]]
    if subset.empty:
        continue
    subtopics = ta.categorize(
        subset, "Review", f"Summarize into 3 subtopics under: {topic['name']}",
    )
    hierarchy.append({"main": topic, "subtopics": subtopics})

df.to_csv(OUT / "result.csv", index=False)
(OUT / "hierarchy.json").write_text(json.dumps(hierarchy, indent=2, ensure_ascii=False))
```

## Example 8 — TA with grouping ("summarize topics per organization")

Natural fit for `groupby`:

```python
grouped = {}
for org, subset in df.groupby("Organization"):
    grouped[org] = ta.categorize(
        subset, "Review", f"Summarize topics for organization: {org}",
    )
(OUT / "topics_by_org.json").write_text(json.dumps(grouped, indent=2, ensure_ascii=False))
```

## Example 9 — Multi-step: filter then summarize ("topics for negative reviews")

```python
# Step 1: tag sentiment (populates a new column)
df["Sentiment"] = ta.tag(
    df, "Review", "Map each review into positive, negative, or neutral",
)

# Step 2: filter in pandas
negatives = df[df["Sentiment"].str.lower() == "negative"]

# Step 3: categorize only the filtered subset
neg_topics = ta.categorize(
    negatives, "Review", "Summarize the topics from the negative reviews",
)

df.to_csv(OUT / "result.csv", index=False)
(OUT / "negative_topics.json").write_text(
    json.dumps(neg_topics, indent=2, ensure_ascii=False)
)
```

## Example 10 — Distribution-constrained rating ("top 10% get 5 stars")

Tag for a raw score, then do the binning purely in numpy / pandas:

```python
import numpy as np

df["Raw Score"] = pd.to_numeric(
    pd.Series(ta.tag(df, "Review", "Rate each review on a scale of 1 to 10")),
    errors="coerce",
)

cutoffs = np.percentile(df["Raw Score"].dropna(), [30, 50, 70, 90])
c2, c3, c4, c5 = cutoffs

def to_stars(x):
    if pd.isna(x):         return pd.NA
    if x >= c5:            return 5
    if x >= c4:            return 4
    if x >= c3:            return 3
    if x >= c2:            return 2
    return 1

df["Star Rating"] = df["Raw Score"].map(to_stars)
df.to_csv(OUT / "result.csv", index=False)
```

---

## Orchestration reminders

- **One text column per `ta.*` call.** If the task needs context from multiple columns (e.g. "summarize Review in light of Product"), combine first: `df["_combined"] = df["Product"].str.cat(df["Review"], sep=" — ")`, then pass `"_combined"`.
- **No implicit filtering / chaining.** Decompose multi-step tasks explicitly — see Examples 7 / 8 / 9.
- **Output conventions.** Row-level results → new column in the DataFrame, then `to_csv`. Dataset-level results → JSON next to the CSV. Print a one-paragraph summary to stdout so the orchestrator can relay it to the user.
- **Don't dump huge result tables into chat.** Summarize, then point the user at the output path.
