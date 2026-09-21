#!/bin/bash
# litiGPT post-training watchdog.
#
# Waits for the training process to exit, packages the adapter into a small
# tarball, records the final metrics, and then tries to stop the pod so it
# stops billing. Meant to be launched detached (setsid nohup) so it survives
# the SSH session that started it.
#
# It waits on an explicit PID rather than matching a command-line pattern.
# That is deliberate: a pkill-style pattern match can catch the watching
# shell itself, which has already happened once on this project.
#
# Stopping the pod (rather than terminating it) keeps the volume, so the
# archive survives. Be aware that resuming a stopped pod needs a free GPU on
# the same host and is not guaranteed — pull anything you cannot lose BEFORE
# the pod stops.
#
# usage: post_training_watchdog.sh <training-pid> [--no-stop]

set -u

PID="${1:?usage: post_training_watchdog.sh <training-pid> [--no-stop]}"
MODE="${2:-}"

WORK=/workspace
OUT=$WORK/litiGPT/models/litigpt_top30_lora
ARCHIVE=$WORK/litigpt_final.tar.gz
LOG=$WORK/watchdog.log
# RUNPOD_POD_ID is set in the container's own environment, but an SSH session
# gets a fresh one and does NOT inherit it - verified empty over ssh on
# 2026-09-21. RunPod also writes it to /etc/rp_environment, which is readable
# from any session, so fall back to that.
#
# There is deliberately no hardcoded default. This used to name a specific pod,
# and that pod was later terminated: the watchdog would have "stopped" a pod
# that no longer existed, reported success, and left the real one billing.
POD_ID="${RUNPOD_POD_ID:-}"
if [ -z "$POD_ID" ] && [ -r /etc/rp_environment ]; then
    POD_ID=$(sed -n 's/^export RUNPOD_POD_ID="\(.*\)"$/\1/p' /etc/rp_environment)
fi

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "watchdog started; waiting on PID $PID (mode=${MODE:-stop})"

while kill -0 "$PID" 2>/dev/null; do
    sleep 60
done

log "training process $PID exited"
sleep 45   # let the trainer flush its final save and close MLflow

# ---- locate the adapter to keep -------------------------------------------
if [ -f "$OUT/adapter_model.safetensors" ]; then
    SRC="$OUT"
    log "final adapter present at $SRC"
else
    SRC=$(ls -1d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    log "no final adapter written; falling back to latest checkpoint: ${SRC:-none}"
fi

# ---- slim archive: adapter + tokenizer, no optimizer state ----------------
# optimizer.pt is ~140MB and is only needed to resume training, not to run
# the model, so it stays out of the archive.
if [ -n "${SRC:-}" ] && [ -d "$SRC" ]; then
    tar czf "$ARCHIVE" -C "$SRC" \
        --exclude='optimizer.pt' \
        --exclude='rng_state.pth' \
        --exclude='scheduler.pt' \
        . 2>>"$LOG" \
        && log "archive written: $(du -h "$ARCHIVE" 2>/dev/null | cut -f1) from $SRC" \
        || log "ERROR: archive failed"
else
    log "ERROR: no adapter directory found; nothing archived"
fi

# ---- final metrics into the log -------------------------------------------
python - >> "$LOG" 2>&1 <<'PY'
import sqlite3
try:
    c = sqlite3.connect('/workspace/mlflow.db')
    rid = c.execute('select run_uuid from runs order by start_time desc limit 1').fetchone()[0]
    ev = c.execute("select step,value from metrics where run_uuid=? "
                   "and key like '%eval_loss%' order by step", (rid,)).fetchall()
    tr = c.execute("select step,value from metrics where run_uuid=? and key='loss' "
                   "order by step desc limit 3", (rid,)).fetchall()
    print('  final eval_loss: ' + ', '.join('%d:%.4f' % r for r in ev))
    print('  final train loss: ' + ', '.join('%d:%.4f' % r for r in reversed(tr)))
except Exception as e:
    print('  could not read metrics: %r' % (e,))
PY

date -u '+%Y-%m-%d %H:%M:%S' > "$WORK/TRAINING_DONE"
log "wrote $WORK/TRAINING_DONE"

# ---- stop the pod ---------------------------------------------------------
if [ "$MODE" = "--no-stop" ]; then
    log "--no-stop given; pod left running and STILL BILLING"
    exit 0
fi

if [ -z "$POD_ID" ]; then
    log "ERROR: could not determine the pod id (RUNPOD_POD_ID unset and"
    log "       /etc/rp_environment unreadable). Pod left RUNNING and STILL BILLING."
elif runpodctl get pod >/dev/null 2>&1; then
    log "stopping pod $POD_ID"
    runpodctl stop pod "$POD_ID" >> "$LOG" 2>&1 \
        && log "stop command accepted" \
        || log "ERROR: stop command failed"
else
    log "runpodctl has no usable API key; pod left RUNNING and STILL BILLING."
    log "To enable auto-stop, run on the pod before training ends:"
    log "  runpodctl config --apiKey <your-runpod-api-key>"
fi
