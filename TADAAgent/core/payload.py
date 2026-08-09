from pydantic import BaseModel
import pandas as pd
from typing import Optional, List, Any

# Structural Mapping constraints
class ColumnInfo(BaseModel):
    column_name: str
    column_type: str
    value_set: Optional[List[str]] = None

# Planner Output definitions
class TadaAgentResult(BaseModel):
    title_description: str
    table_name: str
    preaug_cols: Optional[List[ColumnInfo]] = None
    code: str
    warning: Optional[str] = None

# Context Dependencies for Execution
class TadaAgentDeps:
    MainTable: pd.DataFrame
    def __init__(self, MainTable: pd.DataFrame):
        self.MainTable = MainTable

# Final Output Envelope
class TadaAgentToolResult:
    def __init__(self, is_success: bool, message: Optional[str]=None, augmented_columns: Optional[List[str]]=None, final_df: Optional[pd.DataFrame]=None):
        self.is_success = is_success
        self.message = message
        self.augmented_columns = augmented_columns if augmented_columns is not None else []
        self.final_df = final_df
