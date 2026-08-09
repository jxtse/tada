# TA-Bench Queries

| ID | Query |
|---|---|
| Q01 | Label each row in the `customer_feedback` column as Positive / Negative / Neutral. |
| Q02 | Assign each row in `customer_notes` a priority of High / Medium / Low based on urgency. |
| Q03 | Flag each row in `contract_clause` as Legally Risky or Clean. |
| Q04 | Group the rows in `user_review` by the issues they describe -- discover the groupings yourself. |
| Q05 | First split the reviews into 3 broad categories, then within each, identify sub-clusters. |
| Q06 | Return the rows in `meeting_notes` that mention a need for follow-up action. |
| Q07 | Find the rows in `customer_comment` that imply the customer may churn, even if not stated explicitly. |
| Q08 | Find the rows where the content of `notes` contradicts the `status` column (marked as Closed). |
| Q09 | Extract the names of Party A and Party B from each row in `contract_notes`, writing each into a separate new column. |
| Q10 | From each row in `incident_report`, extract: Time, Location, Affected Scope, and Root Cause -- each into its own column. |
| Q11 | This cell contains a free-form project description. Convert it into a structured table with columns: Project Name, Owner, Budget, Deadline. |
| Q12 | Complete the `revised_document` column for rows where it is blank. Each revision should incorporate the supervisor's feedback and any new experiment evidence into the original document. |
| Q13 | For each `product_id` group, summarize the key topics mentioned in the `user_review` column. |
| Q14 | What are the key differences between VIP and regular customer feedback? |
| Q15 | Summarize the main topics in `support_ticket` per quarter and describe how they evolved from Q1 to Q4. |
| Q16 | What is the average `rating` of reviews that are semantically negative? |
| Q17 | Which products have an abnormally high return rate? What do their `customer_reason` entries say? |
| Q18 | The Statements table contains public statements made by company executives. The Emails table contains internal employee communications. For each email, identify which executive statement it contradicts (return the StatementID), or leave blank if there is no contradiction. |
| Q19 | Sort the rows in `review` from most positive to most negative sentiment. |
| Q20 | Sort the rows in `notes` by their relevance to the current analysis theme. |
| Q21 | Find the rows in `notes` that are semantically most different from all other rows. |
| Q22 | In the `feedback` column, which topics appeared in the last month that were not present in earlier months? |
| Q23 | Normalize all spelling variants in the `country` column to the canonical country name. |
| Q24 | Write a summary of at most 20 words for each row in `description`, storing results in a new `summary` column. |
| Q25 | Translate the `notes` column (mixed Chinese / English) into standard English, writing results to `notes_en`. |
| Q26 | Using both the `description` and `category` columns, infer a numeric `priority` score for each row and write it to a new column. |
