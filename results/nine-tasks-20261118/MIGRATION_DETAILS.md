# 九题实验迁移依赖与继续运行限制

2026-11-18。本文对应公开仓库 `jxtse/tada` 的 `jev` 分支中 `results/nine-tasks-20261118/` 结果包，只覆盖九题实验的迁移依赖与接续限制，不是整个研究工作区的备份。本文基于只读代码审阅整理，不启动模型、不更改历史证据。

在本结果包目录执行 `python restore.py --verify-only` 校验分卷，再执行 `python restore.py "C:\TADA-restored"` 还原到尚不存在的目标目录。脚本根据自身位置寻找 `archives/manifest.json`，放在仓库子目录不会改变解压后的工作区目录关系。

## 必须保留的目录关系

以下项目应在同一工作区中按原相对路径还原：

- `experiments/tada_bench_main_codex61_sol_20261117/`：当前 main 四题的完整现场、冻结源码、构建材料、配置、报告和保护启动器。
- `experiments/tada_bench_codex61_sol_20261116/`：旧九题记录，含已完成五题、历史尝试、九题 oracle 与基础构建来源。
- `experiments/harbor_rerun_20261112/harness/`：当前 solver 复制共享运行组件，preflight 还会校验其中 `host_model_bridge.py`、`bridge_protocol.py` 与当前副本一致。
- `tada-bench/`：完整九题题包，包括公开输入、`tests`、题包自带 `runtime`、`solution`、`task.toml` 和 `tasks.json`。不要按 runtime 或 cache 名称统一删减。
- `tada-bench-download/DOWNLOAD_COMPLETE.json`：官方评分器读取它校验题包文件 SHA256。

两套实验中的 `runs/<task>/attempt-*` 应保留完整目录：原始请求/响应及 ready/claim/done、session events/states/blobs、账本、execution、output、runconfig、manifest。结果文件单独存在不足以恢复审计与预算。

## 新机模型配置

主实验协调器显式读取 `experiments/tada_bench_main_codex61_sol_20261117/runconfig/runtime.env.local`。在新机创建本地私密配置，提供 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`，连接已重新登录的新机模型接口。原配置及凭据不应提交 GitHub。桥接代码的 cloudgpt 默认路径被此显式参数覆盖，不必恢复那个旧目录。

模型条件保留 `gpt-6.1-sol`、推理强度 `high`、上下文 1050000。模型服务是否可用需要在新机单独验收，历史 `api_preflight.json` 不等于新机已经连通。

主体脚本以 `__file__` 推导工作区，但历史记录中的绝对路径、恢复授权证据路径、容器名称（含 run_dir 哈希）仍与原机器相关。不要对冻结源码或历史 JSON 做全局路径替换；若要更换位置，建立单独的迁移映射和适配记录。

## Docker 镜像

原 main 镜像标签：`tada-bench:main-2d53ad1-sol-equivalent-v1`。

原验收 image ID：`sha256:0986ceca1d8d198825678662f1cda21be187aa1c3630f88723dc2bd394b6ab1a`。

最准确的延续方式是从旧机 `docker save` 导出该镜像，在新机 `docker load` 后核对 image ID。普通项目目录中的 build 材料不是 Docker 镜像本体。

如必须重建，主构建入口为 `experiments/tada_bench_main_codex61_sol_20261117/build/main-v1/Dockerfile`，上下文即该目录。它基于 `tada-bench:codex61-sol-runtime-base`；旧实验 `build/Dockerfile.tada` 包含该基础运行环境的构建材料。依赖包括 Python 3.12、Node.js 22、uv、锁文件中的 Python/Node 包，构建需要访问配置的包源。官方 verifier 可从每题 `tests` 中的 Dockerfile 重建。

重建后的 image ID 通常变化，而 `harness/runtime_revision.py` 和 preflight 会严格比对原镜像、源码归档 SHA256、构建收据及 READY 封印。不能简单改写旧 READY 或构建收据冒充原镜像。应另存迁移后的构建与验证证据，明确新的运行条件/适配。

## 继续运行前的两个实际限制

1. BIRD 和 MAUD 的原 28800 秒均已耗尽，不重新发放 8 小时。
2. PREC 在 Docker 中断后仍有 running phase，缺少可靠结束计时，且原 session 无 `pi-checkpoints/latest.json`。不能声称可从原执行点恢复。应先使用已有日志、账本及可取得的容器终止证据核定实际已用时并追加纠正记录，保留原现场；任何后续尝试只用剩余预算。本轮 main 的 CUAD 尚未启动，旧轮尝试仍保留。

现有入口 `tools/resume_remaining_with_task_budget.py` 默认只检查，不派发；只有 `--run` 才可能运行。当前它固定选择 MAUD、PREC、CUAD，默认检查把已耗尽的 MAUD 和未结清的 PREC 判断为未就绪，因此返回未就绪属于预期。不要把它描述为解压后即可一键继续。也不要启动旧 `harness/run_main_suite.py`，因为旧选择逻辑可能重新选择 BIRD。

软件和镜像准备好后，先运行保护入口的默认检查，读取新产生的预算报告；在迁移适配和 PREC 计时核定完成前，不执行 `--run`。

旧五题与 main 四题保持独立代码条件；五题成绩可保留，不能据此计算声称全部来自 main 的九题均分。
