# Background
Text Categorization is a global text analysis task that organizes a full list of text items into a compact set of high-level categories based on shared semantic patterns. Each batch in this workflow has independently applied categorization to a partitioned subset of data using a unified prompt aligned with the original user query.

You are now tasked with consolidating these batch-level results into a final, coherent set of categories that best represents the entire dataset and remains aligned with the user's intent.

# Input Format
The input is a JSON object with the following structure:
```json
{
  "UserQuery": "string describing the original categorization request",
  "ColumnName": "name of the column being analyzed",
  "BatchCategorySets": [
    {
      "HaveDescription": true,
      "Categories": [
        { "Name": "Late Delivery", "Description": "Issues related to shipping delays" },
        { "Name": "Customer Support", "Description": "Problems with response time" }
      ]
    },
    {
      "HaveDescription": true,
      "Categories": [
        { "Name": "Shipping Delay", "Description": "Complaints about estimated arrival" },
        { "Name": "Service Quality", "Description": "Poor help or unresolved queries" }
      ]
    }
  ]
}
```

# Task Objective
Merge all batch-level category sets into a single, clean, globally consistent list of categories that align with the user's query. Your final categories must preserve the semantic intent of the original query and the batch results while removing duplication and improving clarity.

# Output Format
The output must reuse the exact format used in batch-level categorization:
```json
{
  "TaskType": "Categorization",
  "OutputLanguage": "detected output language (e.g., en-US)",
  "Domain": "inferred data domain",
  "Perspective": {
    "NumCategories": number of categories,
    "CategoryWords": ["word1", "word2", "..."]
  },
  "HaveDescription": true or false,
  "Categories": [
    {
      "Name": "category name 1",
      "Description": "optional description if needed"
    }
  ],
  "Note": "optional rationale"
}
```

# Consolidation Guidelines

## 1. Preserve Query-Level Constraints (Highest Priority)
- Parse `UserQuery` and preserve hard constraints exactly:
  - explicit number bounds (e.g., exactly 5, no more than 7)
  - naming style constraints
  - label-length constraints (e.g., each label <= 5 words)
  - perspective constraints (e.g., safety lens, UX lens)
- These constraints are global and must hold after merging all batches.

## 2. Cluster and Normalize Across Batches
- Group semantically similar categories across batches.
- Merge synonymous or overlapping names (e.g., `Shipping Delay` and `Late Delivery`).
- Produce one globally stable vocabulary that can be reused for downstream tagging.

## 3. Name Stability for Joint Tagging
- Final category names must be:
  - mutually distinct,
  - semantically separable,
  - concise and deterministic.
- Avoid near-duplicates that cause row-level assignment ambiguity.

## 4. Description Policy
- If descriptions are used, use them for all categories.
- If not needed, omit all descriptions.
- Keep `HaveDescription` consistent with actual output.

## 5. Final Cardinality
- If user requests fixed or bounded category count, enforce it strictly.
- Otherwise choose natural global grouping and avoid unnecessary fragmentation.

## 5.5 Stability Constraints for Joint Tagging Workflows
When this consolidated category list will be used as a vocabulary for downstream `tag()` calls (Joint workflow):
- **Category names must follow consistent naming**: use PascalCase, snake_case, or lowercase exclusively; avoid mixed case, special chars (except `_`), or punctuation.
- **Category names should be compact**: prefer single words or short 2-word phrases. If longer necessary, abbreviate consistently.
- **Category names must not overlap**: ensure no category name is a prefix or substring of another, to prevent ambiguity in post-processing normalization.
- **Category list must be explanatory**: ensure every item in the input data can reasonably map to one of these categories. Use "Other" or "Miscellaneous" only as last resort and document the reason.
- **Preserve this list exactly**: between consolidation and `tag()` phases, do NOT rename, reorder, or drop categories. Consistency is critical for row-level correctness.

## 6. Sort and Validate
- Sort by global importance/frequency/clarity.
- Use `Other` only when unavoidable and place it last.
- Ensure output can be used directly as a shared vocabulary for `tag` in Joint workflows.

# Output Language Policy
- If `UserQuery` specifies an output language, use it.
- Otherwise match the language of `UserQuery`.
- Batch content language does not override this.

## 10. Handling Problem-Centric Queries with Mixed Sentiment Data

**Key Principle**: Category NAMES should reflect TOPICS, not PROBLEMS. This allows positive and negative comments to coexist naturally.

When a UserQuery asks to "identify problems," "find issues," or "discover pain points," but the input data contains mixed positive and negative comments:

**Rules**:
1. **Remove problem-centric suffixes** from category names:
   - "Performance Issues" → "Performance"
   - "Reliability & Stability Issues" → "Reliability & Stability"
   - "Usability Problems" → "Usability"
   - "Pricing Concerns" → "Pricing"
   - "Feature Limitations" → "Features"
   - "Quality Issues" → "Product Quality"
   - "Support Problems" → "Support"
   - "Delivery Problems" → "Delivery & Logistics"

2. **Why**: Input data naturally mixes "Performance is amazing" and "Performance is slow" about the same topic. 
   Problem-named categories force praise to be miscategorized as problems.

3. **Result**: Both praise and criticism coexist in the same neutral category—downstream tagging can distinguish sentiment if needed.

**Example**:
```
Query: "Identify main problem categories in customer reviews"
Data sample: 
  - "99.9% uptime—impressive!" (rating: 3-5, positive about reliability)
  - "Data loss incidents eroded trust" (rating: 1-2, negative about reliability)
Same topic, mixed sentiment

Wrong output categories:
  "Reliability & Stability Issues" → both praise and criticism forced into "Issues"

Correct output categories:
  "Reliability & Stability" → naturally accommodates both praise and criticism
```

**Implementation**:
- When merging batch results, check each category name
- If it ends with "Issues", "Problems", "Concerns", "Limitations", "Gaps", "Errors", "Failures":
  - Remove the suffix and its preceding character (usually a space or &)
  - Normalize the remaining name
- Keep the category name semantic and clear, just without problem-centric framing
