#!/bin/sh
# Usage: run_trial_remote.sh <task-dir> <jobs-dir> <job-name> [model]
# Run detached Harbor trials on the configured SSH host. Credentials travel only
# on SSH stdin; the shared model slot is held throughout launch, polling and sync.
set -u
cd "$(dirname "$0")/.."
T=$1; J=$2; N=$3; M=${4:-deepseek-flash}
case "$M" in
  *opus*) MODEL_FAMILY=opus ;;
  *deepseek*) MODEL_FAMILY=deepseek ;;
  *) echo "unsupported model $M" >&2; exit 2 ;;
esac
if [ "${TADA_MODEL_SLOT_HELD:-}" != "1" ] || [ "${TADA_MODEL_SLOT_FAMILY:-}" != "$MODEL_FAMILY" ]; then
  exec python3 v5/tada_taskgen/concurrency.py --model "$M" -- sh solver/run_trial_remote.sh "$@"
fi
[ -f "$T/task.toml" ] || { echo "no task.toml in $T" >&2; exit 2; }
CONFIG=$(python3 solver/trial_config.py remote "$M") || exit 2
eval "$CONFIG"
unset CONFIG
mkdir -p "$J"
[ -e "$J/$N" ] && { echo "job $J/$N exists; refusing to overwrite" >&2; exit 2; }
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30"
sha=$(cd "$T" && find . -type f ! -path './environment/runtime/*' ! -path './tests/runtime/*' | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | cut -c1-64)
printf '{"task": "%s", "task_tree_sha256": "%s", "model": "%s", "host": "%s", "started": "%s"}\n' "$T" "$sha" "$M" "$H" "$(date -u +%FT%TZ)" > "$J/$N.meta.json"
RN="$N-$(echo "$sha" | cut -c1-8)-$(date +%s)"
LOG="$J/$N.log"
rsync -a --delete -e "$SSH" "$T/" "$H:$R/tasks/$sha/" >> "$LOG" 2>&1 || { echo "task sync failed" >> "$LOG"; exit 3; }
rsync -a -e "$SSH" solver/tada_solver_agent.py local_config.py "$H:$R/solver/" >> "$LOG" 2>&1 || exit 3
rsync -a -e "$SSH" solver/remote_sitecustomize.py "$H:$R/solver/sitecustomize.py" >> "$LOG" 2>&1 || exit 3
# Quote non-secret settings as one remote shell assignment each.
SETTINGS=$(python3 - "$M" "$ENDPOINT" "$KEYVAR" "$MIRROR" <<'PY'
import shlex,sys
model,endpoint,keyvar,mirror=sys.argv[1:]
values={'TADA_SOLVER_MODEL':model,'TADA_SOLVER_ENDPOINT':endpoint,'TADA_SOLVER_KEY_ENV':keyvar,'TADA_REMOTE_MIRROR_IMAGE':mirror}
if model=='claude-opus-5-5':values['TADA_SOLVER_MAX_TOKENS']='64000'
print(' '.join(k+'='+shlex.quote(v) for k,v in values.items()))
PY
) || exit 2
printf '%s\n' "$KEY" | $SSH "$H" "read -r K; export $KEYVAR=\"\$K\"; unset K; cd $REMOTE_ROOT && \
  setsid nohup env PATH=\"\$PWD/.venv/bin:/usr/local/bin:/usr/bin:/bin\" HOME=\"\$HOME\" USER=\"\$USER\" LANG=en_US.UTF-8 \
    TMPDIR=\"\$PWD/tmp\" DOCKER_CONFIG=$REMOTE_DOCKER PYTHONPATH=\"\$PWD/solver\" \
    $KEYVAR=\"\$$KEYVAR\" $SETTINGS \
    sh -c 'mkdir -p tmp; .venv/bin/harbor run -p tasks/$sha -a tada_solver_agent:MintShellAgent -o jobs --job-name $RN -n 1 -y -q --skip-credential-checks > logs/$RN.log 2>&1; echo \$? > logs/$RN.exit' \
    > /dev/null 2>&1 < /dev/null &" >> "$LOG" 2>&1 || { echo "remote launch failed" >> "$LOG"; exit 3; }
unset KEY
echo "remote job $RN on $H" >> "$LOG"
while :; do
  sleep 60
  code=$($SSH "$H" "cat $REMOTE_ROOT/logs/$RN.exit 2>/dev/null" 2>/dev/null)
  [ -n "$code" ] && break
done
rsync -a -e "$SSH" "$H:$R/jobs/$RN/" "$J/$N/" >> "$LOG" 2>&1
$SSH "$H" "cat $REMOTE_ROOT/logs/$RN.log" >> "$LOG" 2>/dev/null
exit "$code"
