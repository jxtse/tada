# Task generation

This pipeline builds analytical tasks from combinations of local datasets. Each
scored cell maps back to one original expert label or benchmark answer. The
compiler produces the public workspace, private verifier, provenance and
reference solution from the same scenario specification.

Candidates pass six hard gates: oracle invertibility, trivial-answer resistance,
deterministic offline execution, leakage checks, keyword baselines and the output
contract. A bounded design/review loop precedes remote solver trials. Common
errors can trigger blind re-reading and exclusion-only quarantine; labels are
never rewritten. Admission applies the configured quality and combination quotas.
LLM-judge-only datasets are listed in the inventory and excluded from the current
task space.

## Layout

| Path | Contents |
|---|---|
| `v5/tada_taskgen/` | Adapters, compiler, scoring, gates, review loop and scheduler |
| `v5/specs/`, `v5/schemas/` | Scenario examples and validated contracts |
| `v5/tests/` | Offline tests with fake model and trial clients |
| `sources/` | Original-label source interfaces retained by the adapters |
| `solver/` | Local and remote trial launchers and solver agent |
| `controller.py` | Package assembly and independent checks |
| `designer_v5.md` | Scenario-design contract |
| `local_paths.example.toml` | Portable configuration template |
| `tools/export_public.py` | Allowlisted source export and publication scan |

## Configuration

Use Python 3.11 or later and `uv`. Runtime and tests require `openpyxl`, `PyYAML`
and `python-dateutil`; source readers additionally use their provisioned dataset
environment, including `pandas`. The `v5/` lockfile pins the minimal pytest
harness; it does not install the external dataset reader. Real design also needs
the Pi runtime, and remote trials need SSH, Docker and Harbor on the configured
host.

```sh
cp local_paths.example.toml local_paths.toml
```

Edit the paths, model routes and remote trial settings for your installation.
`TADA_<UPPERCASE_KEY>` environment variables override the corresponding local-file
keys. Relative paths resolve from this directory. The local file is also read by
restricted-environment subprocesses, so configuration does not depend on inherited
environment variables. Missing operational settings fail with the key name.

Model IDs, API styles and budgets are in `v5/config.toml`; its localhost URLs are
examples. Credentials belong in the environment variables named by the model
configuration, never in tracked files.

Datasets and the delivered unified reader are external inputs. Configure the
reader and original source caches, O*NET tables, quarantine file and dataset Python
interpreter in `local_paths.toml`. Dataset licences and redistribution restrictions
apply to the user's local copies. The batch inventory records the available parts
and their deterministic verification methods; it is not a dataset download.

## Offline checks

From this directory, with the required Python dependencies available:

```sh
python -c "import tada_taskgen"
PYTHONDONTWRITEBYTECODE=1 uv run --project v5 --offline --no-sync \
  pytest v5/tests -q -rs -p no:cacheprovider
```

Create the pytest environment and add the test runtime dependencies first:

```sh
uv sync --project v5 --locked
uv pip install --python v5/.venv/bin/python openpyxl PyYAML python-dateutil
```

These extra runtime packages are not pinned by the minimal pytest lockfile.
Tests requiring local datasets, original source caches or a provisioned interpreter
skip with explicit reasons when those inputs are absent. Pure contract, scoring,
configuration and mocked orchestration tests remain runnable. Offline tests do not
start real model requests or Harbor trials. Actual model difficulty and admission
require separate real-run evidence.

## Run the pipeline

Inspect the available source combinations after configuring local inputs:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m tada_taskgen.scheduler --datasets-report
PYTHONDONTWRITEBYTECODE=1 python -m tada_taskgen.scheduler --combinations-report
```

Start an explicitly authorized real run in a fresh output directory:

```sh
mkdir -p v5_runs/batch/tmp
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/v5_runs/batch/tmp" \
  python -m tada_taskgen.scheduler --out v5_runs/batch/run --loops 10
```

The configured dataset interpreter should be used when its source-reader
packages are needed. The scheduler retains bounded revisions, shared model slots,
trial preflight, infrastructure-failure refunds and saved-candidate recovery.
Do not reuse a run directory after changing its source-combination enumeration.

The public export includes code, contracts and tests. It intentionally excludes
local configuration, credentials, datasets, private caches, frozen tasks, run
outputs and internal development notes. Those remain in the live working copy.
Export preparation does not commit, publish, or start a trial:

```sh
python tools/export_public.py --out v5_runs/public_export/taskgen
```
