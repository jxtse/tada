# TADA

Artifacts for **TADA: Semantic Atoms for Scalable Text Analysis in Data Analytics in the Era of LLM-Based Agents**.

| Directory | Contents |
|---|---|
| [`agent/`](agent/README.md) | The TADA agent: a Pi agent that composes two semantic functions, `tag` and `categorize`, with ordinary Python and pandas in one persistent Jupyter kernel. Includes paper-guided retention and map-reduce for large inputs. |
| [`benchmark/`](benchmark/README.md) | The long-horizon text-analysis benchmark: the task-generation pipeline and the tasks it admits. |
| [`archive/`](archive/README.md) | Earlier artifacts kept for provenance: TABench, the original skill, evaluation, calibration, and supplementary material. |

## Quick start (agent)

```bash
cd agent
uv sync --dev
npm ci
npm run build
uv run pytest -q
```

Configuration, running sessions, and the execution boundary are described in [`agent/README.md`](agent/README.md).

## License

Code is MIT licensed ([LICENSE](LICENSE)); benchmark and calibration data use CC BY 4.0 ([DATA_LICENSE](DATA_LICENSE)). Third-party design references are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Citation metadata will be updated after publication; until then, cite the accompanying manuscript.
