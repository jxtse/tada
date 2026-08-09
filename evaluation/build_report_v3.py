#!/usr/bin/env python3
"""Convert pairwise judge outputs into reader-friendly per-method performance
scores with mean ± std error bars, and rebuild the HTML report (v3).

Score definition (one number, higher = better):
  row-level quality parity with the TADA anchor =
    (units judged at-least-as-good-as TADA + units identical to TADA) / all units
  where all units = identical + judged + missing rows (missing counts as failure).
Every pair has the same denominator (3200 row-field units), so scores are
directly comparable. TADA's own r2/r3 self-comparison gives the effective
ceiling of the metric (benign wording variation keeps even TADA below 100%).
"""
import csv, json, html, statistics
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).parent
RUN = HERE / "full26_runs" / "full26_20260722"
OUT = Path.home() / ".hermes/workspace/out/tabench26_report.html"

matrix = list(csv.DictReader(open(HERE / "matrix_result.csv")))
w3 = {r["qid"]: r for r in csv.DictReader(open(RUN / "winrate_3vote.csv"))}
cases = json.load(open(RUN / "case_scores.json"))
det = json.load(open(RUN / "deterministic.json"))

def esc(s): return html.escape(str(s))

# ---- per-pair score ----
scores = defaultdict(list)   # method -> [(repeat, score, total)]
for r in matrix:
    m, rep = r["pair"].rsplit("_r", 1)
    eq, tie, cand, anch, miss = (int(r["equal_auto"]), int(r["tie_judged"]),
                                 int(r["cand_win"]), int(r["anchor_win"]), int(r["missing_rows"]))
    total = eq + tie + cand + anch + miss
    good = eq + tie + cand
    scores[m].append((rep, 100.0 * good / total, total))

ORDER = [("skill_on", "TADA（自身重复轮，即指标天花板）", "#0f766e"),
         ("skill_off", "Skill-off：同 agent，无 TADA 原语", "#1d4ed8"),
         ("code_only", "Code-only：只许确定性代码", "#b45309"),
         ("direct", "Direct：单次调用，无工具", "#b91c1c")]

stat = {}
for m, _, _ in ORDER:
    vals = [s for _, s, _ in scores[m]]
    stat[m] = {"mean": statistics.mean(vals),
               "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
               "vals": sorted(vals), "n": len(vals)}

# ---- SVG bar chart with error bars ----
W, H, ML, MR, BH, GAP = 880, 300, 300, 70, 44, 18
x0, x1 = 50.0, 100.0
def X(v): return ML + (v - x0) / (x1 - x0) * (W - ML - MR)
bars = ""
y = 30
for m, label, color in ORDER:
    s = stat[m]
    bx, bw = X(x0), X(s["mean"]) - X(x0)
    lo, hi = s["mean"] - s["std"], s["mean"] + s["std"]
    bars += f'<rect x="{X(x0):.1f}" y="{y}" width="{bw:.1f}" height="{BH}" rx="4" fill="{color}" opacity="0.85"/>'
    if s["std"] > 0:
        cy = y + BH / 2
        bars += (f'<line x1="{X(lo):.1f}" y1="{cy}" x2="{X(hi):.1f}" y2="{cy}" stroke="#1a1d23" stroke-width="2"/>'
                 f'<line x1="{X(lo):.1f}" y1="{cy-7}" x2="{X(lo):.1f}" y2="{cy+7}" stroke="#1a1d23" stroke-width="2"/>'
                 f'<line x1="{X(hi):.1f}" y1="{cy-7}" x2="{X(hi):.1f}" y2="{cy+7}" stroke="#1a1d23" stroke-width="2"/>')
    for v in s["vals"]:
        bars += f'<circle cx="{X(v):.1f}" cy="{y+BH/2}" r="3.5" fill="#fff" stroke="#1a1d23" stroke-width="1.5"/>'
    txt = f'{s["mean"]:.1f}%' + (f' ± {s["std"]:.1f}' if s["std"] else "")
    bars += (f'<text x="{ML-12}" y="{y+BH/2+5}" text-anchor="end" font-size="13.5" fill="#1a1d23">{esc(label)}</text>'
             f'<text x="{X(hi)+10 if s["std"] else X(s["mean"])+10:.1f}" y="{y+BH/2+5}" font-size="13" font-weight="700" fill="#1a1d23">{txt}</text>')
    y += BH + GAP
grid = ""
for gv in (50, 60, 70, 80, 90, 100):
    grid += (f'<line x1="{X(gv):.1f}" y1="20" x2="{X(gv):.1f}" y2="{y-6}" stroke="#e5e7eb" stroke-width="1"/>'
             f'<text x="{X(gv):.1f}" y="{y+14}" text-anchor="middle" font-size="11.5" fill="#6b7280">{gv}%</text>')
svg = (f'<svg viewBox="0 0 {W} {y+30}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">'
       f'{grid}{bars}</svg>')

# ---- simple summary table ----
sum_rows = ""
NOTES = {"skill_on": "两轮自身重复得 93–94 分：即使同一方法重跑，措辞级微差也让分数到不了 100。这就是本指标的实际天花板。",
         "skill_off": "三轮波动最大（±5.4）：没有固定原语时，质量取决于 agent 当轮即兴写的方案。",
         "code_only": "规则代码无法做语义判断，且三轮间规则各不相同。",
         "direct": "分数低的主因是每轮丢 461–643 行输出（缺行按 0 分计入）。"}
for m, label, color in ORDER:
    s = stat[m]
    rng = f'{s["vals"][0]:.1f} – {s["vals"][-1]:.1f}'
    sum_rows += (f'<tr><td><span class="dot" style="background:{color}"></span><b>{esc(label.split("：")[0].split("（")[0])}</b>'
                 f'<div class="mdesc">{esc(label.split("：")[-1] if "：" in label else "")}</div></td>'
                 f'<td class="num"><b>{s["mean"]:.1f}</b></td><td class="num">± {s["std"]:.1f}</td>'
                 f'<td class="num">{rng}</td><td class="num">{s["n"]}</td><td>{esc(NOTES[m])}</td></tr>')

# ---- ON/OFF per-case simplified: per-case ON win share among decided units ----
NAMES = {
 "Q01":"Sentiment labeling","Q02":"Urgency scoring","Q03":"Legal risk annotation","Q04":"Open semantic clustering",
 "Q05":"Hierarchical clustering","Q06":"Semantic predicate filter","Q07":"Implicit churn filter","Q08":"Cross-column contradiction",
 "Q09":"NER extraction","Q10":"Multi-field event extraction","Q11":"Text-to-table","Q12":"Missing value imputation",
 "Q13":"Per-group summarization","Q14":"Comparative summarization","Q15":"Temporal aggregation","Q16":"Text→numeric aggregation",
 "Q17":"Anomaly + text explain","Q18":"Cross-table semantic join","Q19":"Sentiment ranking","Q20":"Relevance ranking",
 "Q21":"Semantic outlier detection","Q22":"Emerging topic detection","Q23":"Text standardization","Q24":"Per-row summarization",
 "Q25":"Translation","Q26":"Inference-based priority"}

onoff_rows = ""
for i in range(1, 27):
    q = f"Q{i:02d}"
    if q in w3:
        r = w3[q]
        on, off, tie = int(r["on_win"]), int(r["off_win"]), int(r["tie"])
        total = on + off + tie
        net = on - off
        pct_same = 100.0 * tie / total
        if net > 0: verdict, cls = f"TADA 更好（多赢 {net} 行）", "pos"
        elif net < 0: verdict, cls = f"无 skill 更好（多赢 {-net} 行）", "neg"
        else: verdict, cls = "打平", ""
        bar_on = 100.0 * on / total; bar_off = 100.0 * off / total
        stack = (f'<div class="stack"><span style="width:{bar_on:.1f}%;background:#0f766e"></span>'
                 f'<span style="width:{pct_same:.1f}%;background:#e5e7eb"></span>'
                 f'<span style="width:{bar_off:.1f}%;background:#b91c1c"></span></div>')
        onoff_rows += (f'<tr><td class="qid">{q}</td><td>{esc(NAMES[q])}</td><td>{stack}</td>'
                       f'<td class="num">{on}</td><td class="num">{tie}</td><td class="num">{off}</td>'
                       f'<td class="{cls}">{verdict}</td></tr>')

NAMES2 = NAMES


# rebuild onoff_rows now that NAMES exists
onoff_rows = ""
for i in range(1, 27):
    q = f"Q{i:02d}"
    if q in w3:
        r = w3[q]
        on, off, tie = int(r["on_win"]), int(r["off_win"]), int(r["tie"])
        total = on + off + tie
        net = on - off
        if net > 0: verdict, cls = f"TADA 更好（多赢 {net} 行）", "pos"
        elif net < 0: verdict, cls = f"无 skill 更好（多赢 {-net} 行）", "neg"
        else: verdict, cls = "打平", ""
        stack = (f'<div class="stack"><span style="width:{100*on/total:.1f}%;background:#0f766e"></span>'
                 f'<span style="width:{100*tie/total:.1f}%;background:#e5e7eb"></span>'
                 f'<span style="width:{100*off/total:.1f}%;background:#b91c1c"></span></div>')
        onoff_rows += (f'<tr><td class="qid">{q}</td><td>{esc(NAMES[q])}</td><td>{stack}</td>'
                       f'<td class="num">{on}</td><td class="num">{tie}</td><td class="num">{off}</td>'
                       f'<td class="{cls}">{verdict}</td></tr>')
    else:
        d = det.get(q, {})
        note = {"filter": "两侧选出完全相同的行集 → 打平", "numeric": "两侧算出完全相同的数值 → 打平",
                "mapping": "映射 24/25 一致 → 基本打平"}.get(d.get("status", ""), "整体评分（见附表）")
        onoff_rows += (f'<tr class="dim"><td class="qid">{q}</td><td>{esc(NAMES[q])}</td>'
                       f'<td><div class="stack"><span style="width:100%;background:#e5e7eb"></span></div></td>'
                       f'<td colspan="3" class="num">—</td><td>{esc(note)}</td></tr>')

case_rows = ""
for q in sorted(cases):
    cs = cases[q]
    t_on = sum(v for v in cs.get("skill_on", {}).values() if isinstance(v, (int, float)))
    t_off = sum(v for v in cs.get("skill_off", {}).values() if isinstance(v, (int, float)))
    arrow = "TADA 更好" if t_on > t_off else ("无 skill 更好" if t_off > t_on else "打平")
    case_rows += (f"<tr><td class='qid'>{q}</td><td>{esc(NAMES[q])}</td>"
                  f"<td class='num'>{t_on}/20</td><td class='num'>{t_off}/20</td><td>{arrow}</td></tr>")

doc = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TABench RQ1 · 性能与方差 · 2026-07-24</title>
<style>
:root {{ --ink:#1a1d23; --sub:#6b7280; --line:#e5e7eb; --bg:#fafafa; --card:#fff;
 --green:#0f766e; --blue:#1d4ed8; --amber:#b45309; --red:#b91c1c; }}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:-apple-system,"PingFang SC","Segoe UI",sans-serif; color:var(--ink); background:var(--bg); line-height:1.55; padding:40px 20px; }}
.wrap {{ max-width:1080px; margin:0 auto; }}
h1 {{ font-size:25px; letter-spacing:-.02em; }}
.sub {{ color:var(--sub); margin:6px 0 24px; font-size:14px; }}
.panel {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:22px; margin:14px 0; }}
h2 {{ font-size:17px; margin:30px 0 10px; }}
h2 .anno {{ font-weight:400; font-size:13px; color:var(--sub); margin-left:8px; }}
table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; font-size:13.5px; }}
th {{ text-align:left; font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--sub); background:#f7f7f8; padding:9px 12px; border-bottom:1px solid var(--line); }}
td {{ padding:9px 12px; border-bottom:1px solid #f0f0f1; vertical-align:top; }}
tr:last-child td {{ border-bottom:none; }}
tr.dim td {{ color:var(--sub); }}
.num {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
.qid {{ font-weight:700; white-space:nowrap; }}
.mdesc {{ font-size:12px; color:var(--sub); font-weight:400; }}
.pos {{ color:var(--green); font-weight:600; }} .neg {{ color:var(--red); font-weight:600; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:99px; margin-right:8px; }}
.stack {{ display:flex; height:14px; border-radius:99px; overflow:hidden; min-width:140px; }}
.stack span {{ display:block; height:100%; }}
.legend {{ font-size:12.5px; color:var(--sub); margin:8px 0 0; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin:0 4px 0 12px; vertical-align:-1px; }}
.note {{ background:var(--card); border:1px solid var(--line); border-left:3px solid var(--amber); border-radius:8px; padding:14px 16px; font-size:13.5px; margin:12px 0; }}
.foot {{ color:var(--sub); font-size:12.5px; margin-top:30px; border-top:1px solid var(--line); padding-top:14px; }}
.mono {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; color:var(--sub); }}
ul {{ padding-left:20px; }} li {{ margin:4px 0; }}
</style></head><body><div class="wrap">

<h1>TABench RQ1 · 四种方法的性能与方差</h1>
<div class="sub">26 个任务 × 3 轮独立重复 · 同一输入（SHA-256 锁定）· 同一模型 claude-opus-4.8 · 盲评 judge（gpt-5.5，匿名 A/B × 位置对调 × 3 票）· 2026-07-22 → 07-24</div>

<div class="panel">
<h2 style="margin-top:0">主结果 · 质量得分（越高越好）<span class="anno">条形 = 3 轮均值，误差线 = ±1 标准差，白点 = 每轮实际得分</span></h2>
{svg}
<div class="legend">得分含义：与 TADA 基准逐行盲比后，该方法「不差于 TADA」的行占比（相同 + 判平 + 判胜）。缺行按失败计。所有方法分母相同（3,200 行×字段/轮）。</div>
</div>

<table>
<tr><th>方法</th><th>均值</th><th>标准差</th><th>3 轮范围</th><th>轮数</th><th>解读</th></tr>
{sum_rows}
</table>

<div class="note"><b>怎么读：</b>TADA 自身重跑也只能得 93–94 分（同义改写造成的良性差异），所以 <b>94 分就是满分线</b>。Skill-off 均值低 7 分且三轮波动 ±5.4（最好一轮 92.4、最差 81.7）——没有固定原语时质量靠运气。Code-only 低 16 分，Direct 低 34 分且波动小——它是稳定地差（每轮固定丢约 500 行）。</div>

<h2>TADA（skill ON）vs 同 agent 无 skill（OFF）· 逐任务<span class="anno">绿 = TADA 赢的行 · 灰 = 两侧相同或判平 · 红 = OFF 赢的行</span></h2>
<table>
<tr><th>Case</th><th>任务</th><th>行级对比</th><th>ON 胜</th><th>平</th><th>OFF 胜</th><th>结论</th></tr>
{onoff_rows}
</table>
<div class="legend"><i style="background:#0f766e"></i>TADA 胜 <i style="background:#e5e7eb"></i>相同/平 <i style="background:#b91c1c"></i>无 skill 胜　——大部分行是灰色：两种方法在多数行上产出等价结果，胜负集中在少数难行上。</div>

<h2>整体评分任务（无法逐行对比的 8 题）</h2>
<table>
<tr><th>Case</th><th>任务</th><th>TADA</th><th>无 skill</th><th>结论</th></tr>
{case_rows}
</table>

<h2>三句话总结</h2>
<ul>
<li><b>性能排序</b>：TADA 94（天花板）＞ Skill-off 86.5 ± 5.4 ＞ Code-only 77.8 ± 3.9 ＞ Direct 60.1 ± 1.7。</li>
<li><b>方差含义</b>：TADA 是唯一可复现的方法（±0.5）；Skill-off 的 ±5.4 意味着单轮结果不可信，必须多轮验证。</li>
<li><b>边界条件</b>：26 题中 3 题（摘要 Q24、翻译 Q25、相关性打分 Q20）OFF 略优——自由文本生成上强模型不受接口约束时措辞更自然；结构化推断任务（Q10、Q26）TADA 大幅领先。</li>
</ul>

<div class="note"><b>配置披露：</b>历史 4 月 run 的 host（claude-opus-4.6-1m）已不可用；本结果为 2026-07 新基线（全方法统一 claude-opus-4.8），非 4 月复现。产品型 baseline（ShortcutAI 等）输入不可锁定，另作带日期附录，不进本协议。</div>

<div class="foot">数据：<span class="mono">tabench_rq2/</span> matrix_result.csv · winrate_3vote.csv · 原始判分 matrix_judge_raw.jsonl（66,486 条）/ judge_raw.jsonl（10,720 条）全量落盘 · 输入 manifest 27 文件 SHA-256 · build_report_v3.py 自动生成</div>
</div></body></html>"""

OUT.write_text(doc)
print(f"written: {OUT} ({len(doc)//1024} KB)")
for m, label, _ in ORDER:
    s = stat[m]
    print(f"{m:10s} mean={s['mean']:.1f} std={s['std']:.1f} vals={[f'{v:.1f}' for v in s['vals']]}")
