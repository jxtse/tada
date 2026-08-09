import pandas as pd
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent, RunContext
import json
from typing import Any

# Absolute stand-alone imports
from core.payload import TadaAgentDeps, TadaAgentResult, TadaAgentToolResult
from core.prompts import SYSTEM_PROMPT
from core.executor import execute_and_fix
from core.ta_api import TextAnalysis
from core.llm_config import configure_openai_environment, resolve_llm_config

# =========================================================
# 1. System/Model Init for Open Source Portability
# =========================================================
_llm_config = resolve_llm_config()
_model_name = _llm_config.model

# Configure an explicit OpenAI-compatible provider so custom endpoints work
# with current pydantic-ai releases.
configure_openai_environment(_llm_config)
_pydantic_model = OpenAIChatModel(
    _model_name,
    provider=OpenAIProvider(
        base_url=_llm_config.base_url,
        api_key=_llm_config.api_key,
    ),
)

# =========================================================
# 2. TADA Program Planner (Core Agent)
# =========================================================
tada_planner = Agent(
    model=_pydantic_model,
    deps_type=TadaAgentDeps,
    output_type=TadaAgentResult, 
    system_prompt=SYSTEM_PROMPT,
)

@tada_planner.system_prompt
def inject_runtime_context(ctx: RunContext[TadaAgentDeps]) -> str:
    """Injects the strictly formatted tabular schema to avoid Information Bottleneck."""
    
    # Safe DataFrame sampling to avoid context blowing up
    df_sample = ctx.deps.MainTable.astype(str).head(3).to_dict(orient='records')
    schema = ctx.deps.MainTable.dtypes.astype(str).to_dict()
    
    return f"""
### TABULAR CONTEXT INJECTED ###
Current DataFrame Schema:
{schema}

Sample Rows (First 3):
{json.dumps(df_sample, indent=2, ensure_ascii=False)}

Please generate valid Python utilizing the `TA` object to fulfill the user's requirement.
"""

# =========================================================
# 3. Execution Pipeline
# =========================================================
async def run_tada(df: pd.DataFrame, query: str) -> TadaAgentToolResult:
    """
    Main entry point for TADA.
    """
    deps = TadaAgentDeps(MainTable=df)
    print(f"\n[TADA Planner] Constructing operator program for query: '{query}'")
    
    # Pass 1: Translate semantic intent to an executable operator program.
    try:
        run_result = await tada_planner.run(query, deps=deps)
        output_payload = run_result.data if hasattr(run_result, 'data') else run_result.output
    except Exception as e:
        message = f"Pydantic AI failed to map query to operator program: {e}"
        print(f"[TADA Planner Error] {message}")
        return TadaAgentToolResult(is_success=False, message=message)
        
    print(f"[TADA Planner] Operator program generated. Target Table name: {output_payload.table_name}")
    print(f"--- GENERATED SCRIPT ---\n{output_payload.code}\n------------------------")

    if not output_payload.code or output_payload.code.strip() == "":
        return TadaAgentToolResult(is_success=False, message="No script generated.")

    # Pass 2: Execute generated program with repair.
    exec_res = await execute_and_fix(output_payload.code, df)

    def _is_sentiment_query(q: str):
        return any(k in q.lower() for k in ["sentiment", "emotion", "positive", "negative"])

    def _find_text_column(cols: list[str]):
        for c in ["Verbatim", "review", "text"]:
            for col in cols:
                if c.lower() in col.lower(): return col
        return cols[0]

    # Pass 3: Strict Structural Sanity Handlers
    if exec_res.isSuccess:
        out_df = _coerce_final_result_to_dataframe(exec_res.table_item)
        new_cols = [c for c in out_df.columns if c not in df.columns]
        
        # Pydantic deterministic fallback for highly corrupt models
        if _is_sentiment_query(query) and len(new_cols) > 0:
            target_col = new_cols[0]
            # Check if execution safely returned garbage mostly
            unknowns = sum(1 for val in out_df[target_col] if str(val).lower() in ("unknown", "null", "", "none"))
            if unknowns / len(out_df) > 0.8:
                print(f"[TADA Fallback] Neural schema collapse detected. Enforcing deterministic override pass.")
                ta = TextAnalysis()
                txt_col = _find_text_column(out_df.columns)
                fixed_labels = ta.Tagging(out_df[txt_col].tolist(), txt_col, "Identify feeling: Positive, Neutral, or Negative.")
                out_df[target_col] = pd.Series(fixed_labels[:len(out_df)])

        print(f"[TADA Success] Pipeline completed. Extracted semantics into formal columns: {new_cols}")
        return TadaAgentToolResult(is_success=True, augmented_columns=new_cols, final_df=out_df)
    else:
        print(f"[TADA Failed] Ultimate halting Error:\n{exec_res.ErrorMessage}")
        return TadaAgentToolResult(is_success=False, message=exec_res.ErrorMessage)


def _coerce_final_result_to_dataframe(value: Any) -> pd.DataFrame:
    """Normalize aggregate/multi-table planner outputs into an auditable DataFrame."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.reset_index()
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for key, item in value.items():
            if isinstance(item, pd.DataFrame):
                for idx, row in item.reset_index(drop=True).iterrows():
                    rendered = row.astype(object).where(pd.notna(row), None).to_dict()
                    rows.append({"section": str(key), "row_number": idx + 1, **rendered})
            elif isinstance(item, pd.Series):
                rendered = item.astype(object).where(pd.notna(item), None).to_dict()
                rows.append({"section": str(key), "row_number": 1, **rendered})
            elif isinstance(item, (list, tuple)):
                for idx, element in enumerate(item, start=1):
                    rows.append({"section": str(key), "row_number": idx, "value": json.dumps(element, ensure_ascii=False, default=str)})
            else:
                rows.append({"section": str(key), "row_number": 1, "value": str(item)})
        return pd.DataFrame(rows)
    if isinstance(value, (list, tuple)):
        return pd.DataFrame({"value": [json.dumps(item, ensure_ascii=False, default=str) for item in value]})
    return pd.DataFrame({"value": [str(value)]})
