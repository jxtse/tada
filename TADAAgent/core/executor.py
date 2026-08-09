import traceback
import pandas as pd
from typing import Tuple
from .ta_api import TextAnalysis
from .llm_config import resolve_llm_config

class ExecResult:
    def __init__(self, isSuccess: bool, table_item: pd.DataFrame=None, ErrorMessage: str=None):
        self.isSuccess = isSuccess
        self.table_item = table_item
        self.ErrorMessage = ErrorMessage

async def execute_and_fix(code: str, table_df: pd.DataFrame, max_retries: int = 3) -> ExecResult:
    """
    State-Aware Auto-Healing Loop
    Captures pandas/python exceptions and autonomously utilizes an LLM to patch the AST.
    """
    current_code = code
    for attempt in range(max_retries):
        try:
            # Re-init fresh state space bounds for safe execution
            env = {
                "pd": pd,
                "df": table_df.copy(),
                "final_result": None,
                "TextAnalysis": TextAnalysis
            }
            
            # Setup Neuro-Symbolic Semantic API Hooks
            setup_code = "TA = TextAnalysis()\n"
            full_code = setup_code + current_code
            
            # Evaluate generated symbolic graph
            exec(full_code, env)
            
            if env.get("final_result") is not None:
                return ExecResult(True, table_item=env["final_result"])
            else:
                raise ValueError("Code executed but 'final_result' was not assigned a DataFrame.")
                
        except Exception as e:
            err_msg = traceback.format_exc()
            if attempt == max_retries - 1:
                return ExecResult(False, ErrorMessage=err_msg)
            
            print(f"[Auto-Healing] Python Error Caught: {str(e)}. Generating AST fix... ({attempt+1}/{max_retries-1})")
            
            # Synchronous minimal call to heal code
            try:
                from openai import OpenAI
                config = resolve_llm_config()
                
                client = OpenAI(base_url=config.base_url, api_key=config.api_key)
                model_name = config.model
                
                schema = table_df.dtypes.astype(str).to_dict()
                
                fix_prompt = (
                    f"The following semantic pipeline code on dataframe schema {schema} failed:\n"
                    f"```python\n{current_code}\n```\n\n"
                    f"ERROR TRACEBACK:\n{err_msg}\n\n"
                    "Please output ONLY the corrected Python pandas code containing the fix. "
                    "You must assign the corrected DataFrame to `final_result`."
                )
                
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": fix_prompt}],
                    temperature=0.0
                )
                
                fixed_code = resp.choices[0].message.content
                # Strip artifacts
                if "```python" in fixed_code:
                    fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
                elif "```" in fixed_code:
                    fixed_code = fixed_code.split("```")[1].split("```")[0].strip()
                    
                current_code = fixed_code
            except Exception as llm_e:
                print(f"Auto-healing network/LLM failed: {llm_e}")
                return ExecResult(False, ErrorMessage=err_msg)
                
    return ExecResult(False, ErrorMessage="Max retries reached without successful dataframe generation.")
