The controller supplies `environment/runtime/` and `tests/runtime/` (vendored `openpyxl`, `et_xmlfile`) and copies `tests/gold.json` and `solution/reference.*` from your build. Keep these templates as written; change only the lines marked `# EDIT`. They run unchanged with plain `harbor run` (agent and verifier in separate no-network containers, non-root user). Do not add `docker-compose.yaml` files.

`task.toml`
```toml
schema_version = "1.4"
artifacts = ["/output/<DELIVERABLE_FILE>"]   # EDIT: exact output path from instruction.md

[metadata]
name = "<task-name>"                          # EDIT (must not start with a digit)
family = "<family>"                           # EDIT
description = "<one line>"                    # EDIT

[agent]
timeout_sec = 28800.0
user = "65532:65532"

[verifier]
timeout_sec = 600.0
environment_mode = "separate"
user = "65532:65532"

[verifier.environment]
network_mode = "no-network"
cpus = 1
memory_mb = 2048

[environment]
network_mode = "no-network"
build_timeout_sec = 1200.0
cpus = 1
memory_mb = 2048
```

`environment/Dockerfile`
```dockerfile
FROM python:3.12-slim-bookworm
COPY runtime/ /opt/runtime/
ENV PYTHONPATH=/opt/runtime PYTHONDONTWRITEBYTECODE=1
COPY public/ /public/
RUN mkdir -p /output /logs/agent /logs/verifier /logs/artifacts \
 && chown -R 65532:65532 /output /logs \
 && chmod -R a+rX /public /opt && chmod -R a-w /public /opt
WORKDIR /output
```

`tests/Dockerfile`
```dockerfile
FROM python:3.12-slim-bookworm
COPY runtime/ /opt/runtime/
COPY scoring.py /opt/scoring.py
COPY gold.json /tests/gold.json
COPY test.sh /tests/test.sh
ENV PYTHONPATH=/opt/runtime PYTHONDONTWRITEBYTECODE=1
RUN mkdir -p /output /logs/agent /logs/verifier /logs/artifacts \
 && chown -R 65532:65532 /output /logs \
 && chmod -R a+rX /opt /tests && chmod -R a-w /opt /tests
WORKDIR /output
```

`tests/test.sh`
```sh
#!/bin/sh
set -eu
python /opt/scoring.py /output/<DELIVERABLE_FILE> /tests/gold.json /logs/verifier   # EDIT: deliverable path
```

`solution/solve.sh`
```sh
#!/bin/sh
set -eu
cp /solution/reference.<ext> /output/<DELIVERABLE_FILE>   # EDIT
```
