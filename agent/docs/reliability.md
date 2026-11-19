# 可靠性、预算与长任务处理（feature/tada-reliability-budget-preflight）

本次改动来自第 1–9 题的运行复盘。复盘发现，框架版落后于直接跑的主要原因不是模型判断错了，而是大批量语义调用存在以下问题：

- 跑不完；
- 中途结果存不住；
- 失败后只能全部重来；
- 预算和时间都耗在重复劳动上。

下面逐条说明七个改进点：改了什么、对应哪道题的问题、怎么使用。所有新行为都会写入审计，并计入预算。

## 1. 大批量调用"失败不白干"

**背景**：
- 第 7 题：78 批中已有 70 批成功，却因为一个 HTTP 500 全部作废。
- 第 9 题：两次整批调用（各 105 次请求）都因为一个分段出现 500 而作废。
- 第 1、4 题：超时后原样重跑，分别多出 318 次和 417 次重复请求。

**改动**：
- **批次缓存**（`engine.py`）：每个批次成功后，结果立刻写入审计 blob，并记录一条 `atom/batch_commit` 事件。缓存键由三部分组成：调用身份（`request_identity`）、角色和完整 prompt，此外还带上批次位置。只有"同一个调用再次发起"时才会命中缓存，同一调用里内容碰巧相同的两个请求不会互相复用。
  - 恢复会话（resume）时，从审计日志重建缓存。
  - 再次调用时只补跑缺失的批次。`atom/plan.preflight.cached_batches` 记录本次复用了多少批。
  - 使用 `fresh=True`、`runtime.reuse="off"` 或 `runtime.batch_cache=false` 时不复用缓存。
- **传输层重试**（`providers.py`、`engine.py`）：
  - 以下情况抛出 `TransientProviderFailure`，按 `transport_backoff_seconds × 2^k` 退避后重试，最多 `runtime.transport_retries` 次（默认 2）：HTTP 408/409/425/429/5xx、`httpx.TransportError`、流在没有终止事件时断开。
  - 每次重试都是一个独立的请求，单独计入预算，角色带后缀 `-transport-retry`，并通过 `retry_of` 关联上一个请求。
  - 其他 4xx 错误、预算耗尽和取消都不会重试。
- **快速失败**（`engine._run_batches`）：任意一个批次出现不可恢复的错误后，立刻停止派发还在排队的批次，不再让它们白白发出去再被丢弃。
- **输出截断时拆批**：如果响应因为 `max_output_tokens` 被截断（Responses API 返回 `incomplete_details.reason=max_output_tokens`，或 Chat 接口返回 `length`），会抛出 `OutputTruncated`，错误信息里写明上限。多行批次会对半拆开重试，最多拆 `split_on_truncation_depth` 层（默认 3）。
- **账本写入重试**（`budget.py`）：第 9 题出现过 `PermissionError`，原因是宿主机挂载目录上的 rename 被短暂拒绝。现在会有界地重试，避免整次调用因此作废。

## 2. 解决"30 分钟跑不完、后台任务卡住前台"

**背景**：
- 第 1、2、4、6、7 题都出现过同一个循环：一个大 tag 在 30 分钟的 cell 里跑不完，被取消，然后原样重来。
- 第 2、4 题中，主控自己开的后台线程长期占着全局锁，前台任务排队，每次都要等满 30 分钟。

**改动**：
- **并发参数真正生效**（`supervisor.py`）：Provider 构造时读取 `runtime.operator_concurrency`。之前不论配置多少，实际都只有 1 个并发。上限由 2 放宽到 8。
- **同步调用自动分片**：在 cell 截止时间前 `cell_deadline_margin_seconds`（默认 60 秒）停止派发新批次，抛出 `OperatorIncomplete`，并报告"已完成 X/N 批"。已完成的批次留在缓存里，用完全相同的参数再调用一次即可接着跑。这样不会再被 cell 超时强行打断。
- **异步任务**：`tag_async(...)` 和 `categorize_async(...)` 返回一个 job，支持 `status()`、`result(timeout=...)`、`cancel()`，`jobs()` 可以列出所有任务。
  - job 在 supervisor 内按时间片执行，每片最长 120 秒。每片结束后释放全局锁，前台调用可以插进来执行。
  - 每个时间片至少会派发 1 个批次，避免任务一直不前进（活锁）。
- **客户端放弃即取消**：每次 `tag`/`categorize` 调用都有自己的 `call_id` 和取消令牌（`LinkedCancellation`）。内核端的等待超过 `cell_timeout_seconds + 120` 秒后，会通知服务端取消对应的工作，不再留下继续消耗预算的"孤儿"调用。会话关闭时，所有 job 和调用一起取消。
- **超时反馈带进度**：每个 observation 的 `operators` 字段列出本 cell 内的语义调用，以及仍在运行的 job 的进度（done/total、所处阶段、状态）。

## 3. 让主控看得到预算，并在调用前预检

**背景**：
- 第 4 题：一个调用计划了 1133 批，超过了整题 800 次的预算。
- 第 6 题：只剩约 58 次预算时，主控发起了需要 80 批的审计。
- 第 6、9 题：第一段都把预算用完了，却还没写出工作簿。

**改动**：
- 每个 observation 都带有 `budget` 字段：`used`、`limit`、`left`、`seconds_to_deadline`。剩余不足 20% 时附加 `warning`。在 Python 里也可以调用 `budget()` 查看。
- **调用前预检**（`runtime.budget_preflight`，默认开启）：生成分批计划后，先估算还需要的请求数（未缓存的批次，加上 intent 和 reduce）。超过剩余预算时抛出 `PreflightRefused`，并给出缩小规模的建议，**不会发出任何请求**。估算结果写入 `atom/plan.preflight`，其中预计耗时按本会话的历史请求耗时中位数和并发数计算。
- `tag(..., dry_run=True)`：只返回估算结果（计划请求数、已缓存批次、预计秒数、是否放得进预算），不发请求，也不写 atom 事件。

## 4. 长文档"先定位、再判断"，并减少调用开销

**背景**：
- 第 1、4、9 题中，90% 以上的请求都花在全文逐段扫描上。
- 第 8 题：每篇文章的比较对象都不同，只能逐篇调用，因此多出 116 次 intent 请求。

**改动**：
- `long_rows={"locate": [正则, ...], "locate_window": 1, "locate_on_miss": "all"|"none"}`：只发送正文命中这些正则（不区分大小写）的分段，以及每侧 `locate_window` 个相邻分段。
  - 没有任何命中时，`all` 退回全文扫描，`none` 表示不读任何分段。
  - prompt 中会说明这些分段是被挑出来的。
  - 每行"保留的分段数/总分段数"记录在 `atom/plan.long_rows.located_segments_per_row`。
  - 分段数上限按保留下来的分段计算；`max_segments_per_row` 现在也可以按调用单独覆盖。
- `row_context=Series`：每行可以带一段自己的参数，例如本篇的"干预 / 对照 / 结局"。它会随行一起进入请求，也会带进该行的每个分段和归并步骤。这样多行可以合在一次批量调用里完成，不必逐行调用。
- `mode="independent"`：跳过 intent 请求。使用 `row_context`、`images` 或 `aggregate="collect"` 时会自动视为 independent。
- 同一个 query 字符串在同一会话内的 intent 判断结果会缓存复用。
- 兼容性：新增字段只在设置了值时才写入序列化结果，所以不用这些功能的调用，其 prompt 字节和复用身份与旧版完全相同，旧会话的结果仍然可以复用。

## 5. 交付物校验

**背景**：
- 第 5、8 题各有一格超过评分器 2000 字符的上限。
- 第 8 题约 18 道表格题核心数值正确，但答案带了单位或说明。
- 第 2 题把 `False` 写成了 `false`。

**改动**：
- 新增 `check_deliverable(path, overrides=None)`，检查 xlsx 或 csv，但不修改文件。检查内容包括：
  - 单元格长度是否超过上限；
  - 是否有 NaN 或 inf；
  - 答案列是否为空；
  - 答案列是否写成了句子（超过 `short_answer_max_words` 个词，而且不是 JSON）。
- 默认规则来自 `runtime.deliverable_rules`，评测适配层可以按 benchmark 注入，例如 `max_cell_chars=2000`。

## 6. 主控系统提示词：工作纪律

**背景**：
- 第 6、7、8 题的全量复核共花了约 600 次请求，只改对了 2–3 格。
- 第 9 题擅自扩大了 exclusivity 的定义，第 6、7 题改写了案件类型的定义。
- 第 7 题超时后原样重试。
- 第 8 题的答案格式没有对齐评分器。

**改动**：在 `controller.ts` 的系统提示中加入 `SYSTEM_PROMPT_RULES`，主要包括：
- 看 budget 和 operators 字段，大调用前先用 `dry_run` 估算；
- 一有完整初版就先写出交付物；
- 被时间截断的调用要原样重复，或者改用 `tag_async`；
- 长文档用 `locate`，每行参数不同时用 `row_context`；
- 只做定向复核，并基于已经抽取的证据重判；
- 原样引用 codebook 定义，不扩大、不改写，也不自加判定规则；
- 只交规范答案（大小写、精度按题目要求）；
- 提交前运行 `check_deliverable`，并核对自己记下的疑点。

## 7. 图片通道

**背景**：第 3 题有 4 道题需要看商品图片。PIL 确实打开过图片，但 40 次模型请求里没有一次带上了图片。

**改动**：
- **cell 里显示的图片**：PNG、JPEG、GIF、WEBP 会进入 observation 的 `images` 字段（最多 `observation_images` 张，单张不超过 `observation_image_max_bytes`）。`controller.ts` 把它们转成 Pi 的原生 image content，交给主控模型；text JSON 里只保留 `images_attached` 计数。`observation_images>0` 时，主控模型声明支持 `['text','image']` 输入；设为 0 即可关闭。
- **`tag(images=Series)`**：传入一列图片文件路径，每行图片会以 Responses 的 `input_image`（Chat 接口为 `image_url`）附在请求里，并附上 `IMAGE for rid=...` 标签。每批最多 `max_images_per_batch` 张。图片的 sha256 计入行快照，因此也计入复用身份。带图片的行不能再分段，过长时会直接拒绝。

## 新增配置（`runtime`）

| 配置 | 默认值 | 说明 |
|---|---|---|
| `operator_concurrency` | 1（上限改为 8） | 同时在途的算子 HTTP 请求数 |
| `transport_retries` / `transport_backoff_seconds` | 2 / 5.0 | 临时错误的重试次数和退避基数（秒） |
| `split_on_truncation_depth` | 3 | 输出被截断时批次对半拆分的最大层数 |
| `batch_cache` | true | 批次缓存 |
| `cell_deadline_margin_seconds` | 60 | 同步调用在 cell 截止前多少秒停止派发 |
| `budget_preflight` | true | 计划超出剩余预算时，在发请求前拒绝 |
| `observation_images` / `observation_image_max_bytes` | 4 / 4,000,000 | 转发给主控的图片张数和单张大小上限 |
| `max_images_per_batch` | 8 | `tag(images=...)` 每批最多附带的图片数 |
| `deliverable_rules` | 见 `DeliverableRules` | `check_deliverable` 的默认规则 |

## 新增审计内容

- `atom/batch_commit`：一个批次结果被持久化（缓存键、角色、行数、结果 blob）。
- `atom/plan.preflight`：调用前的估算（总批次、已缓存批次、计划请求数、预计秒数、剩余预算）。
- `atom/end.status="incomplete"`：同步调用在 cell 截止前停下（批次已缓存）。
- 重试请求的角色带后缀 `-transport-retry`。
- observation 新增 `budget`、`operators`、`images` 字段。

## 测试

- `tests/test_reliability.py`：覆盖重试、快速失败、缓存与恢复、截止时间分片、截断拆批、预检与 `dry_run`、`locate`、`row_context`、intent 跳过与缓存、图片、Provider 错误分类、预算查询、取消令牌、`check_deliverable`。
- `tests/test_jobs.py`：覆盖 supervisor 层面的异步任务时间片、任务取消、客户端取消、预算视图、并发接线。
- `packages/agent/test/tool-result.test.mjs`：覆盖图片转发和系统提示中的纪律条目。
