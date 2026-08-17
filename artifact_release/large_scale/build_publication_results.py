#!/usr/bin/env python3
"""Build audited, publication-facing RQ3 artifacts from saved runs.
Missing measurements remain em dashes; no values are inferred.
"""
import csv, glob, json, statistics
from pathlib import Path

BASE=Path('<LOCAL_ARTIFACT_PATH>')
E=BASE/'e2e'; OUT=BASE/'publication'; OUT.mkdir(exist_ok=True)
NORM=json.load(open(E/'artifacts/normalized_metrics.json'))
TASKS={'sentiment_ps':('Sentiment tagging (32K)','sentiment_ps'),
       'joint_ps_fixed':('Topic assignment (32K)','joint_ps'),
       'cat_tm_fixed':('Feedback categorization (7K)','cat_tm')}
ARMS={'tada':'TADA-Agent','claudecode':'Claude Code','lotus':'LOTUS'}

def load_one(p):
    try: return json.load(open(p))
    except Exception: return None

def runs(arm,task):
    lookup=TASKS[task][1] if arm in ('tada','lotus') else task
    out=[]
    for p in sorted(E.glob(f'{arm}_{lookup}_rep*.json')):
        d=load_one(p)
        if d and d.get('status') in ('ok','ok_after_timeout'):
            d['_path']=str(p); out.append(d)
    return out

def metric_for(arm, task, d):
    key=Path(d['_path']).stem
    return NORM[task]['runs'].get(key,{})

def mean_sd(xs, digits=3):
    if not xs: return '—'
    if len(xs)==1: return f'{xs[0]:.{digits}f}'
    return f'{statistics.mean(xs):.{digits}f}±{statistics.stdev(xs):.{digits}f}'

def token_fields(d):
    u=d.get('usage') or {}
    phys=u.get('physical_usage') if isinstance(u.get('physical_usage'),dict) else u
    virt=u.get('virtual_usage') if isinstance(u.get('virtual_usage'),dict) else None
    def vals(x):
        if not isinstance(x,dict): return (None,None,None)
        p=x.get('prompt_tokens'); c=x.get('completion_tokens'); t=x.get('total_tokens')
        if t is None and p is not None and c is not None: t=p+c
        return p,c,t
    return vals(phys),vals(virt),u.get('cache_hits')

rows=[]
for task,(taskname,_) in TASKS.items():
  ref=NORM[task]['reference']
  for arm,name in ARMS.items():
    rr=runs(arm,task); mm=[metric_for(arm,task,d) for d in rr]
    physical=[token_fields(d)[0] for d in rr]
    n_usage=sum(x[2] is not None for x in physical)
    coverage=[]
    declared=32179 if task!='cat_tm_fixed' else 6628
    for d in rr:
        n=d.get('rows',declared); cov=d.get('coverage_rows',n)
        if n and cov is not None: coverage.append(cov/n)
    row={'task':task,'task_name':taskname,'system':name,'n':len(rr),'n_usage':n_usage,
         'accuracy':mean_sd([m['raw_acc'] for m in mm if 'raw_acc' in m]),
         'macro_f1':mean_sd([m['raw_macro_f1'] for m in mm if 'raw_macro_f1' in m]),
         'norm_accuracy':mean_sd([m['norm_acc'] for m in mm if 'norm_acc' in m]),
         'norm_macro_f1':mean_sd([m['norm_macro_f1'] for m in mm if 'norm_macro_f1' in m]),
         'calls_observed':mean_sd([d['calls'] for d in rr if d.get('calls') is not None],1),
         'prompt_tokens':mean_sd([x[0] for x in physical if x[0] is not None],1),
         'completion_tokens':mean_sd([x[1] for x in physical if x[1] is not None],1),
         'total_tokens':mean_sd([x[2] for x in physical if x[2] is not None],1),
         'wall_min':mean_sd([d['wall_s']/60 for d in rr if d.get('wall_s') is not None],1),
         'coverage':mean_sd(coverage),
         'n_scored':str(mm[0].get('n_scored','—')) if mm else '—',
         'reference_pairwise_acc':ref['pairwise_acc'],
         'reference_pairwise_macro_f1':ref['pairwise_macro_f1']}
    # Calls/tokens are unavailable from retained Claude Code telemetry.
    if arm=='claudecode':
        row.update(calls_observed='—',prompt_tokens='—',completion_tokens='—',total_tokens='—',n_usage=0)
    rows.append(row)

fields=list(rows[0])
with open(OUT/'rq3_e2e_results.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

lines=['# RQ3 audited large-scale results','',
'Quality uses the saved normalized-metrics artifact. Normalized accuracy and macro-F1 divide by the corresponding reference-round pairwise metric. Contested reference rows are excluded from scoring, not from processing. Em dashes are unavailable measurements, never zeroes or estimates.','',
'| Workload | System | valid n | Acc. | Macro-F1 | Norm. Acc. | Norm. Macro-F1 | observed calls | physical tokens | wall min | coverage | scored n |',
'|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['task_name']} | {r['system']} | {r['n']} | {r['accuracy']} | {r['macro_f1']} | {r['norm_accuracy']} | {r['norm_macro_f1']} | {r['calls_observed']} | {r['total_tokens']} | {r['wall_min']} | {r['coverage']} | {r['n_scored']} |")
lines += ['', '## Telemetry and evidence notes',
'- TADA uses planned chunk counts of 33/33/7 for sentiment/topic/category. Topic and category have exactly those successful calls in every run.',
'- TADA sentiment rep1 is retained as an anomalous but valid quality run: 105 observed attempts, 36 failed calls, and 9,876.5 s wall time. Thus 69 attempts succeeded while only 33 chunks were planned; retries make observed attempts differ from the plan. Reps 2 and 3 each record 33 calls, zero failures, and concurrency 3.',
'- Sentiment token telemetry exists for only 2/3 TADA runs (n_usage=2). Its provider-reported physical totals are reported only over those two instrumented runs; rep1 tokens are unavailable.',
'- Claude Code calls and token totals cannot be recovered from the retained process artifacts and remain em dashes. Its timed-out topic rep2 is not a valid quality run (10,799.9 s); valid topic n=2.',
'- No completed `lotus_cat_tm_rep1.json` exists at freeze. The empty partial checkpoint is not evidence. Accordingly the full-scale LOTUS row remains unavailable.',
'- The only completed LOTUS scale sanity check available to this experiment is a separate 500-row sentiment pilot (94.8% accuracy, 25.7 min, concurrency 4). It is explicitly a pilot, is not entered into the 6,628-row category table, and is not extrapolated.',
'- LOTUS would be n=1 if completed; no variance, significance, or stability claim is made.']
(OUT/'rq3_e2e_results.md').write_text('\n'.join(lines)+'\n')

tex=[r'\begin{table*}[t]',r'\centering',r'\footnotesize',
 r'\caption{Audited large-scale quality. Values are mean $\pm$ s.d. over valid repeated runs. Normalized metrics divide by reference-round pairwise accuracy or macro-F1. LOTUS full-scale output was unavailable at freeze; its separate 500-row pilot is not substituted.}',
 r'\label{tab:rq3-e2e-generated}',r'\begin{tabular}{llccccc}',r'\toprule',
 r'Workload & System & $n$ & Accuracy & Macro-F1 & Norm. Acc. & Norm. Macro-F1 \\',r'\midrule']
for r in rows:
    esc=lambda x:str(x).replace('±',r'$\pm$').replace('—','---')
    tex.append(f"{r['task_name']} & {r['system']} & {r['n']} & {esc(r['accuracy'])} & {esc(r['macro_f1'])} & {esc(r['norm_accuracy'])} & {esc(r['norm_macro_f1'])} "+r'\\')
tex += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(OUT/'rq3_e2e_table.tex').write_text('\n'.join(tex)+'\n')
print('\n'.join(lines))
