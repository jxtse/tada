"""Explicitly gated live test through the product CLI, Pi, and persistent kernel.

Set TADA_LIVE_CONFIG only after authorizing its finite cumulative ledger.
"""
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest
from tada_runtime.audit import read_events, validate_session

@pytest.mark.skipif(not os.environ.get('TADA_LIVE_CONFIG'),reason='live routes require explicitly authorized finite config')
def test_real_multiturn_analysis_compaction_reuse_and_recovery(tmp_path):
    session=tmp_path/'live-session'
    first='''Use the reviews input. Discover three main issue categories with categorize(..., name="issues").
Assign a cid from that exact category table using tag(..., name="issue_labels") and validate_category_references.
Create enriched=reviews.copy(); enriched["issue"]=labels and state.put("enriched", enriched).
Save regional_counts.csv under outputs_dir with columns region,issue,count using ordinary pandas groupby.
After the semantic result is committed deliberately execute raise RuntimeError("acceptance-recoverable").
Recover in a subsequent execute_python cell, retrieve state.get("issue_labels"), and verify equality with the saved labels.
Do not repeat the completed semantic calls. Print RECOVERED only after the equality assertion succeeds.'''
    followup='''Use the existing enriched semantic column to save monthly_counts.csv with columns month,issue,count.
Do not call tag or categorize. Verify total count equals the number of input rows and that state.get("issue_labels") still matches.'''
    final='''A new semantic dimension is now required: use tag to assign positive or negative sentiment,
name="sentiment", to reviews.text. Save final_rows.csv containing rid,text,issue,sentiment,region,month,product.
Verify exactly one row per original rid and cid references against state.get("issues").'''
    result=subprocess.run([sys.executable,'-m','tada_runtime.cli','run','--config',os.environ['TADA_LIVE_CONFIG'],
        '--input','reviews=examples/reviews.csv','--session-dir',str(session),'--task',first,
        '--followup',followup,'--followup',final,'--compact-between'],text=True,capture_output=True,timeout=1800)
    (tmp_path/'cli.stdout').write_text(result.stdout);(tmp_path/'cli.stderr').write_text(result.stderr)
    assert result.returncode==0,result.stderr
    report=validate_session(session);assert report.valid and not report.unknown_requests
    events,_=read_events(session)
    assert any(e.type=='compaction/end' and e.payload.status=='success' for e in events)
    assert any(e.type=='model/request' and e.payload.role=='compaction' for e in events)
    assert any(e.type=='cell/end' and e.payload.error and 'acceptance-recoverable' in e.payload.error.message for e in events)
    calls=[e for e in events if e.type=='tool/call'];assert len(calls)>=4
    assert {e.payload.name for e in calls}=={'execute_python'}
    turns=[e for e in events if e.type=='turn/start'];assert len(turns)==3
    middle=[e for e in events if turns[1].seq<e.seq<turns[2].seq]
    assert not any(e.type=='atom/start' for e in middle)
    for name in ('regional_counts.csv','monthly_counts.csv'):
        assert pd.read_csv(session/'outputs'/name)['count'].sum()==6
    rows=pd.read_csv(session/'outputs'/'final_rows.csv')
    assert len(rows)==6 and rows.rid.is_unique and set(rows.rid)=={f'r{i}' for i in range(1,7)}
    assert set(rows.sentiment)<={'positive','negative'}
    assert rows.set_index('rid').loc['r1','sentiment']=='positive'
    assert rows.set_index('rid').loc['r2','sentiment']=='negative'
    assert rows.issue.notna().all()
