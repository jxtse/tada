# TADA 九题实验结果与迁移快照（2026-11-18）

本包对应 `jxtse/tada` 的 `jev` 分支中 `results/nine-tasks-20261118/`，保存九题实验的结果、运行现场和继续运行所需数据；仓库原有代码仍保留在原位置。

**9 道题中，5 道完成评分；历史上 9 道都启动过。当前没有实验继续执行。**

本包包含两轮实验的结果、历史尝试、原始请求和响应、费用账本、session/state/blob、冻结源码与补丁、官方评分与 oracle 验收、完整九题输入及必要的共享运行组件。它是九题实验的迁移快照，不是整个研究工作区的备份。

**完整性标志：只有 `archives/manifest.json` 已生成且校验通过，归档才算准备完成。该文件尚不存在时，不要把当前目录当作完整迁移包。** 本地准备状态见 `PREPARATION_STATUS.json`。

## 已有结果

| 题目 | 得分 | 条件 |
|---|---:|---|
| BLADE / UDABench | 0.660802 | 原实验 |
| BioDEX / UDABench | 0.699111 | 原实验 |
| DABStep / SemTPC-H / Spider2 | 0.440196 | 原实验 |
| InfiAgent / KramaBench / Palimpzest | 0.871402 | 原实验 |
| B-DABStep / TableBench | 0.696955 | 原实验 |
| BIRD / InfiAgent / OolongSynth | 无有效成绩；原 8 小时已用尽 | main 重跑 |
| MAUD | 无提交；原 8 小时已用尽 | main 重跑 |
| PREC / LOFT / LRO-Bench | 中断；须核定结束计时和剩余预算 | main 重跑 |
| CUAD / MAUD / LRO-Bench | 此轮尚未启动；旧轮尝试另行保留 | main 重跑 |

可直接查看 [`results/task_status.json`](results/task_status.json)、[`results/task_status.csv`](results/task_status.csv)、[`results/official_scores`](results/official_scores) 和原始收尾报告。原实验五题与 main 四题采用不同代码条件，不能合称同一版本的九题成绩，不能把缺失成绩记为 0 或据此计算九题均值。

## 在另一台电脑恢复文件

需要 Git 和 Python 3.12 或更新版本。获取 `jev` 分支后，进入本结果包子目录执行：

```powershell
git clone --branch jev --single-branch https://github.com/jxtse/tada.git
Set-Location tada\results\nine-tasks-20261118
python restore.py --verify-only
python restore.py "C:\TADA-restored"
```

已有该仓库时，先获取最新 `jev`，再进入 `results/nine-tasks-20261118`；无需再次克隆。也可在任意目录用结果包内 `restore.py` 的完整路径执行，它会自动定位旁边的 `archives/manifest.json`。

目标目录必须尚不存在，父目录必须存在。第一条命令校验所有分卷的大小和 SHA256；第二条命令先校验，再安全解包。解压工具不会启动实验、调用模型或修改原始记录。也可在其他操作系统指定一个不存在的目标目录。

分卷保存在 `archives/`，每卷至多 48 MiB，普通 Git 克隆即可取得，不需要 Git LFS。清单见 `archives/manifest.json`。解压后的主要目录：

- `experiments/tada_bench_codex61_sol_20261116`：原九题实验，包含五题有效成绩及所有历史尝试。
- `experiments/tada_bench_main_codex61_sol_20261117`：四题 main 重跑现场、冻结源码、计时纠正和接续工具。
- `tada-bench`：完整九题题包，包括输入、原生评分器、自带 runtime 和参考解。
- `tada-bench-download/DOWNLOAD_COMPLETE.json`：原始数据完整性收据；评分程序仍依赖它。
- `experiments/harbor_rerun_20261112/harness`：当前 solver 依赖的共享模型桥接组件。

题库版本为 `jxtse/tada-bench@d820cd784fc7e9ace28e77f716bc7bfb23cb0813`。main 源版本为 `2d53ad1d9a2bc091a947332c75717021f4749af2`，运行条件是已记录兼容补丁的 `main-2d53ad1-sol-equivalent-v1`。九题 oracle 已全部取得 1.0，无需重复运行。

## 接续前必须处理的实际状态

1. **BIRD、MAUD 已用尽原时限。** 不要重新发放 8 小时。以 `official_timeout_correction_20261118.json`、`maud_closeout_20261118.json` 为准；旧 suite 中的基础设施状态不覆盖这些收尾结论。
2. **PREC 在北京时间 2026-11-18 10:36:39 中断。** 最后 suite 累计值是 7773.890 秒，但 `phase-001.json` 仍缺结束计时。先核定实际已用时，再决定怎样在剩余官方预算内接续。现场没有 `pi-checkpoints/latest.json`，不能保证从原执行点恢复。
3. **CUAD 是本轮 main 未启动。** 旧实验已经存在一次尝试，不要删除或改写旧记录。
4. 接续入口是 `experiments/tada_bench_main_codex61_sol_20261117/tools/resume_remaining_with_task_budget.py`。不带 `--run` 只检查。目前因 MAUD 预算耗尽和 PREC 中断状态，检查可能不通过；迁移文件完成不等于立即可以直接 `--run`。不要重启旧 `harness/run_main_suite.py`，它可能再选到 BIRD。

## 环境与凭据

凭据文件和 Python 字节码缓存不在归档中。题包自带的 `runtime` 则完整保留。使用本结果包内的 `runtime.env.example` 在新机创建解压后主实验的 `runconfig/runtime.env.local`，配置新机本地模型接口并重新登录。模型条件保留 `gpt-6.1-sol` / high / 1050000 context；不要把新机登录信息提交到 Git。

Docker 镜像并非普通项目文件，**本包不包含 Docker 镜像二进制**。原验收镜像为 `tada-bench:main-2d53ad1-sol-equivalent-v1`，ID `sha256:0986ceca1d8d198825678662f1cda21be187aa1c3630f88723dc2bd394b6ab1a`。优先从原机导出并在新机导入该镜像；若从归档内 Dockerfile 重建，需要另做新机环境验收并保留原验收记录，不能改写旧 image ID 冒充同一镜像。详细依赖和验收限制见 [`MIGRATION_DETAILS.md`](MIGRATION_DETAILS.md)。

历史日志中的绝对路径与进程号作为证据原样保留。不要全局替换整个归档中的路径；仅调整新机实际启动配置。

## 保存位置与数据来源

本包按本次明确授权保存到公开仓库 `jxtse/tada` 的 `jev` 分支。题库来源为 gated 数据集，题库 README 中记载的上游许可状态及各项原始许可声明随题包保留；此次存放不改变这些原始声明。实际上传完成状态以远端分支中的文件和提交为准。
