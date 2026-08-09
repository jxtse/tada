# TADAAgent

TADAAgent translates a natural-language request over a pandas DataFrame into a
program that composes semantic operators with ordinary pandas operations. The
program must assign its output to `final_result`; execution failures may be
repaired for up to three attempts.

The agent uses the canonical implementation in the sibling `TADASkill/`
directory. Batching, retries, joint-tagging routing, and taxonomy consolidation
are internal to the two atom APIs and are not planned by the agent.

## Setup

```bash
python -m pip install -r requirements.txt
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1"
```

## Usage

Run from this directory:

```python
import asyncio
import pandas as pd
from tada_planner import run_tada

df = pd.DataFrame({"review": ["Excellent.", "Arrived broken."]})
result = asyncio.run(
    run_tada(df, "Label each review as Positive, Neutral, or Negative.")
)
print(result.final_df)
```

The endpoint must support OpenAI-compatible chat completions.
