"""Example cells for the harness namespace; input `reviews` is a DataFrame."""
categories = categorize(reviews.text, "Discover the main service issue categories", name="issues")
labels = tag(reviews.text, "Assign exactly one cid from this frozen table: " + categories.to_json(), name="issue_labels")
validate_category_references(labels, categories)
reviews["issue"] = labels
state.put("enriched_reviews", reviews)
summary = reviews.groupby(["region", "issue"]).size().rename("count")
summary.to_csv(outputs_dir / "regional_issues.csv")
# A later turn can change aggregation without a new semantic call.
monthly = reviews.groupby(["month", "issue"]).size()
