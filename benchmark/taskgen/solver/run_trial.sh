#!/bin/sh
# Usage: run_trial.sh <task-dir> <jobs-dir> <job-name> [model]
# Run one frozen-package Harbor trial using configured endpoints and runtime.
set -u
cd "$(dirname "$0")/.."
T=$1; J=$2; N=$3; M=${4:-deepseek-flash}
# A shared parent holds the model slot for the complete Harbor process lifetime.
case "$M" in
  *opus*) MODEL_FAMILY=opus ;;
  *deepseek*) MODEL_FAMILY=deepseek ;;
  *) echo "unsupported model $M" >&2; exit 2 ;;
esac
if [ "${TADA_MODEL_SLOT_HELD:-}" != "1" ] || [ "${TADA_MODEL_SLOT_FAMILY:-}" != "$MODEL_FAMILY" ]; then
  exec python3 v5/tada_taskgen/concurrency.py --model "$M" -- sh solver/run_trial.sh "$@"
fi
[ -f "$T/task.toml" ] || { echo "no task.toml in $T" >&2; exit 2; }
mkdir -p "$J"
[ -e "$J/$N" ] && { echo "job $J/$N exists; refusing to overwrite" >&2; exit 2; }
CONFIG=$(python3 solver/trial_config.py local "$M") || exit 2
eval "$CONFIG"
unset CONFIG
export TADA_SOLVER_MODEL="$M" TADA_SOLVER_ENDPOINT="$ENDPOINT" TADA_SOLVER_KEY_ENV="$KEYVAR"
export "$KEYVAR=$KEY"
unset KEY
if [ "$M" = "claude-opus-5-5" ]; then
  export TADA_SOLVER_MAX_TOKENS=64000
fi
sha=$(cd "$T" && find . -type f ! -path './environment/runtime/*' ! -path './tests/runtime/*' | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | cut -c1-64)
printf '{"task": "%s", "task_tree_sha256": "%s", "model": "%s", "started": "%s"}\n' "$T" "$sha" "$M" "$(date -u +%FT%TZ)" > "$J/$N.meta.json"
env -u PYTHONPATH PYTHONPATH=$PWD/solver:$PWD "$HARBOR" run -p "$T" \
  -a tada_solver_agent:MintShellAgent -o "$J" --job-name "$N" -n 1 -y -q --skip-credential-checks > "$J/$N.log" 2>&1
python3 - "$J/$N" <<'EOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for r in root.rglob('result.json'):
    if r.parent == root: continue
    d = json.loads(r.read_text())
    print(root.name, 'reward', json.dumps((d.get('verifier_result') or {}).get('rewards')),
          'exc', ((d.get('exception_info') or {}).get('exception_message') or '')[:300])
for s in root.rglob('summary.json'):
    print(root.name, s.read_text().replace('\n', ' '))
EOF
