# Paper-guided candidate retention

Large Categorization and the shared-domain stage of Joint Tagging use a pilot to choose a local candidate budget. The public results remain a category DataFrame and a row-aligned string Series. The agent still has only `execute_python`; planning runs inside the operators.

## Request contract and routing

Give the category cardinality explicitly when it is known:

```python
categories = categorize(
    reviews.text,
    "Discover the two most prominent service issue categories.",
    cardinality={"k": 2, "mode": "exact"},
    name="issues",
)
labels = tag(
    reviews.text,
    "Assign each review one of at most two recurring issue labels discovered across the corpus.",
    cardinality={"k": 2, "mode": "at_most"},
    name="shared_labels",
)
```

`exact` requires exactly k final categories; `at_most` permits a nonempty underfilled result. Neither fills a missing category with an invented placeholder. The optional contract is validated by Python/Pydantic through the kernel bridge. Large requests without an explicit cardinality use one internal structured interpretation call. General discovery without a requested cardinality is explicitly unsupported in the paper-guided large path; it does not silently become top-5.

The runtime enables this path for inputs meeting the configured large-input threshold. Small inputs retain the existing generic discovery path. Independent Tagging keeps its own row batching and never adds a retention pilot. The generic small-input hierarchy is separate from paper-guided global consolidation.

## Execution and evidence

1. Freeze the input snapshot, a capacity-feasible target chunk size, seed and balanced random partition. For N rows and target n, B=ceil(N/n); chunk sizes differ by at most one, with actual minimum floor(N/B). A capacity failure refuses the request rather than changing the partition after planning.
2. Sample the pilot uniformly. One discovery request explores a taxonomy beyond the requested final k. One assignment request assigns every pilot rid to an immutable pilot cid. The host validates complete coverage and computes integer counts; the model does not estimate frequencies.
3. Draw U profiles from `Dirichlet(counts + 0.5)`. For each profile, draw V balanced multinomial count tables. All candidate r values share each profile and table. The reference is the table's realized total-count top-k. Sum only the locally retained counts; ties use the frozen category order.
4. For each r, calculate successes/V within each outer profile, then take the explicitly configured lower empirical quantile across U profiles. Select the smallest candidate meeting the target. The curve is not smoothed and need not be monotone.
5. Ask every actual mapper for up to selected r prominent candidates, ordered by prominence. Validate the upper bound and record underfills. Pass **all** mapper lists, pilot taxonomy/counts/support and the original query into **one global semantic reducer**. Reject an over-capacity reducer input before dispatch. There is no hidden hierarchical fallback.
6. The reducer performs semantic consolidation only. Every local candidate gets a host-assigned id `bBrR` (list B, local rank R); pilot categories use their cid. The reducer must return a complete partition of all ids into consolidated categories (Pydantic rejects missing, unknown or duplicated ids). It does not choose the final k.
7. The host then selects the final categories deterministically (`local-rank-borda-v1`): in each local list a consolidated category scores `r + 1 − best_rank` of its members; scores are summed over lists. Ties break by exact pilot count, then reducer order. The top k are returned (`exact` requires k consolidated categories to exist). The full ranking is stored as typed `rank` evidence. Pilot support is a tie-breaker, never the primary signal, because a 128-row pilot can badly underweight a large category.
8. For Joint Tagging, materialize this final category state, then assign all rows against the same frozen vocabulary using the existing independent assignment mechanism.

No observed pilot tail and no tested candidate meeting the quantile target are separate fallback outcomes. Each uses an exposed configured local budget and remains marked as a fallback. An open mapper budget is never silently shrunk to the number of pilot categories. Zero observations for a known category and an undiscovered category are distinct; the implementation does not invent unseen-category support.

Typed evidence retains pilot rows, taxonomy, assignments and symbolic counts; partition rid lists; planner inputs, profiles and success counts; per-r diagnostics; mapper underfills and actual candidate volume; global-reducer inputs and result; versions, timings and request usage. Results and their evidence are immutable. Exact reuse includes the request contract, planner configuration, seeds, partition policy and prompt/engine versions. Fresh replicates record new derived randomness. Cancellation is checked throughout CPU planning and provider stages and before commit.

## What the numbers mean

The fixed-taxonomy kernel assumes one canonical label per row, exact local counts and exact retained-count consolidation. The plug-in theorem diagnostic is

`1 - B * (C-r) * sum(exp(-n_min * (p_i-p_(r+1))**2 / 2), i=1..k)`.

Raw and clipped values are both retained with applicability and vacuity flags. A full cutoff tie invalidates the strict-gap theorem premise. Retaining all C categories is an exact special case in the numeric kernel. Pilot plug-in bounds and profile quantiles are not calibrated deployment confidence bounds.

Open semantic discovery adds taxonomy drift, unseen categories, incorrect pilot assignments, mapper underfills and semantic consolidation errors. A numeric retention success does not certify end-to-end LLM quality. Mapper proposals are not exact counts; their occurrence in several batches is not substituted for pilot support.

## Configuration and numeric verification

`runtime.large_scale` controls activation, chunk/pilot capacities, response text limits and the planner. All are exposed engineering settings. Serialized provider instructions, JSON schema, evidence and output reserve count toward the conservative byte capacity. This estimate is a declared resource policy, not a universal byte-to-token conversion.

The paper-controlled experiment has six fixed populations of N=32,000, C=20, k=5, n=1,000 and pilot=1,000; U=100, V=20, alpha=.05, target=.95, online r=5..10 and evaluation r=5..20. Its N/C/k are fixture properties, not runtime defaults. The population weights and largest-remainder allocation follow the current appendix. Original author seeds and scripts were unavailable; this implementation records its engineering seeds, RNG, sampler and quantile convention and does not claim identical published table values.

The offline evaluation draws uniform partitions **without replacement from each fixed population**. It compares naive r=k, pilot-selected r and retain-all using the same partitions and full-population reference. The stable-tail oracle is an evaluation-only statistic and is never passed to the online selector. Numeric experiments require no model calls. Measured results and their execution commands belong in the private acceptance evidence, separate from historical paper artifacts.

Run the complete disclosed count experiment in an ignored local directory:

```bash
uv run python -m tada_operators.retention_evaluation \
  --output .tada/retention-evaluation \
  --trials 5000 --populations 2000 8000 32000 --seed 20261110 \
  --max-cpu-seconds 600 --max-wall-seconds 900
```

The runner refuses to overwrite existing evidence. Each case saves its pilot, decision, full unsmoothed recovery curve, confidence diagnostics, times, memory measures and compressed paired count tables. It records partial progress if cancelled or budget-limited. Evaluation gold is held out from the planner.

The implementation validation ran all 18 cases at 5,000 partitions each. At N=32,000, two of the six pilot-selected budgets fell below the 95% held-out recovery target; one slightly underperformed naive r=k. These observed failures remain in the private evidence. They demonstrate why the reported pilot quantile must not be presented as a calibrated guarantee or a promise that quality never decreases with scale.
