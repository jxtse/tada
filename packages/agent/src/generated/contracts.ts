// Generated from Pydantic schemas. Do not edit.
// config sha256: fcacc983bd05775d508f6c1a06cc83919f83fac61571c0b218e4e73bdd5ecaa0
// execute_python sha256: 6fcd63ffb6b179829d8578ec4cf7589f95782530e89a7ccae9b5589ba1964c6c
// rpc_initialize_result sha256: 6ff2a9e06995f71fe5a054527eb54295b151a5311f458bfac81e3478d9b5a028
// rpc_prepare_tool_result sha256: d3c87fa8b5f1272fd47ecd19920e4f3831cafe8e0a626eb455c6dc066ab685bc
export type BudgetConfig = {"deadline_seconds": number;"ledger_path": string;"max_requests": number};
export type LargeScaleConfig = {"capacity_policy"?: "serialized-utf8-plus-output-reserve-v1";"context_budget_bytes"?: number;"enabled"?: boolean;"engine_version"?: "open-retention-v1";"general_discovery_policy"?: "refuse-unbounded-large-v1";"max_description_chars"?: number;"max_name_chars"?: number;"output_reserve_bytes"?: number;"pilot_max_categories"?: number;"pilot_policy"?: "explore-without-final-k-v1";"pilot_rows"?: number;"planner"?: RetentionConfig;"reducer_budget_bytes"?: number;"reducer_policy"?: "one-global-call-or-refuse-v1";"target_chunk_rows"?: number;"threshold_rows"?: number};
export type LoggingConfig = {"format"?: "jsonl";"level"?: "full";"max_unflushed_events"?: 0;"record_provider_exposed_reasoning"?: true};
export type LongRowsConfig = {"boundary"?: "paragraph" | "line" | "chars";"enabled"?: boolean;"head_chars"?: number;"max_segments_per_row"?: number;"overlap_chars"?: number;"policy"?: "segment-aggregate-v1";"segment_chars"?: number;"trigger"?: "over_batch_chars" | "over_segment_chars"};
export type ModelsConfig = {"agent": ProviderConfig;"operators": ProviderConfig};
export type ProviderConfig = {"api_key_env"?: string | null;"auth_mode"?: "required" | "optional";"auth_placeholder"?: string | null;"base_url": string;"max_output_tokens"?: number;"model": string;"protocol": "anthropic-messages" | "openai-responses" | "openai-chat";"reasoning_effort"?: "minimal" | "low" | "medium" | "high" | "xhigh" | null;"temperature"?: number | null;"timeout_seconds"?: number};
export type RetentionConfig = {"alpha"?: number;"candidates"?: Array<number> | null;"inner_draws"?: number;"max_seconds"?: number;"max_simulation_cells"?: number;"max_table_cells"?: number;"max_work_units"?: number;"no_tail_budget"?: number;"outer_draws"?: number;"prior_half_count"?: 0.5;"quantile_method"?: "inverse_ecdf";"r_max"?: number;"rng"?: "numpy-PCG64-seedsequence-v1";"sampler"?: "per-inner-iid-multinomial-balanced-v1";"seed"?: number;"target"?: number;"tie_policy"?: "canonical-domain-order-v1";"version"?: "paired-jeffreys-retention-v1"};
export type RuntimeConfig = {"batch_chars"?: number;"batch_retries"?: number;"batch_rows"?: number;"cell_timeout_seconds"?: number;"large_scale"?: LargeScaleConfig;"long_rows"?: LongRowsConfig;"mode"?: "trusted-local";"operator_concurrency"?: number;"output_view_chars"?: number;"reduce_fan_in"?: number;"reuse"?: "exact" | "off"};
export type Config = {"budget"?: BudgetConfig | null;"logging"?: LoggingConfig;"models": ModelsConfig;"runtime"?: RuntimeConfig;"schema_version"?: "1.0"};
export type ExecuteArgs = {"code": string};
export type JsonValue = unknown;
export type InitializeResult = {"agent_api_key": string | null;"config": Config;"namespace_summary": string;"restored_entries": Array<JsonValue>;"restored_messages": Array<JsonValue>;"session_dir": string;"session_id": string;"tool_schema": Record<string, JsonValue>};
export type BlobRef = {"media_type"?: string;"path": string;"redacted"?: boolean;"sha256": string;"size": number};
export type ErrorInfo = {"details"?: Array<string>;"kind": string;"message": string;"raw_ref"?: BlobRef | null};
export type ToolAdmission = {"code"?: string | null;"error"?: ErrorInfo | null;"status": "accepted" | "rejected";"tool_call_id": string};
