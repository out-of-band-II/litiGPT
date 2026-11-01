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
# usage: [NTFY_TOPIC=<topic>] post_training_watchdog.sh <training-pid> [--no-stop]
#
# The adapter's location is model.output_dir in whichever config trained the
# run, so hardcoding it here makes a second definition that drifts from the
# first. It did: this script looked in litiGPT/models/litigpt_top30_lora while
# config.runpod.yaml wrote to /workspace/models/reddit_bot_lora, which would
# have logged "nothing archived" and stopped the pod anyway. Set
# LITIGPT_OUTPUT_DIR to say where it is; otherwise it is found by search.
#
# Set NTFY_TOPIC to get the run on a phone through ntfy.sh. training_notifier.py
# sends the start, halfway and training-outcome messages; this script sends the
# archive and pod-stop ones itself, because only it knows them, and it sends
# the stop message BEFORE stopping, since the stop takes every process on the
# pod down with it. Unset, nothing is sent. A notification is time-bounded and
# cannot fail the script: it must never stand between a finished run and the
# pod stop.

set -u

PID="${1:?usage: post_training_watchdog.sh <training-pid> [--no-stop]}"
MODE="${2:-}"

# Overridable only so the script can be rehearsed off the pod.
WORK="${LITIGPT_WORK:-/workspace}"
LOG=$WORK/watchdog.log
NOTIFIER_LOG=$WORK/notifier.log
HERE="$(cd "$(dirname "$0")" && pwd)"
# The run's log, packed into the archive with mlflow.db. runpod_guide.md
# launches training with "> /workspace/train.log"; say so if it went elsewhere.
TRAIN_LOG="${LITIGPT_TRAIN_LOG:-$WORK/train.log}"
# training_notifier.py reads the same log and posts to the same server.
export LITIGPT_TRAIN_LOG="$TRAIN_LOG"
export NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
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

notify() {  # notify <title> <priority> <tags> <message>
    [ -n "${NTFY_TOPIC:-}" ] || return 0
    curl -fsS -o /dev/null --max-time 10 \
        -H "Title: $1" -H "Priority: $2" -H "Tags: $3" \
        --data-binary "$4" "$NTFY_SERVER/$NTFY_TOPIC" 2>>"$LOG" \
        || log "WARNING: notification failed: $1"
}

log "watchdog started; waiting on PID $PID (mode=${MODE:-stop})"

NOTIFIER=""
if [ -n "${NTFY_TOPIC:-}" ]; then
    if [ -f "$HERE/training_notifier.py" ]; then
        NOTIFIER="$HERE/training_notifier.py"
        python -u "$NOTIFIER" "$PID" >> "$NOTIFIER_LOG" 2>&1 < /dev/null &
        log "notifications on; progress notifier is PID $!"
    else
        log "WARNING: no training_notifier.py beside this script; only archive and stop messages"
    fi
fi

while kill -0 "$PID" 2>/dev/null; do
    sleep 60
done

log "training process $PID exited"
sleep 45   # let the trainer flush its final save and close MLflow

if [ -n "$NOTIFIER" ]; then
    timeout 60 python "$NOTIFIER" --report-exit >> "$NOTIFIER_LOG" 2>&1 < /dev/null \
        || log "WARNING: training outcome not sent; see $NOTIFIER_LOG"
fi

# ---- locate the adapter to keep -------------------------------------------
# Newest directory holding a real adapter, preferring a final save over a
# checkpoint. The trainer writes the final adapter into output_dir itself, not
# into an output_dir/final subdirectory - that subdirectory was an older
# layout and the docs referred to it for a while after it stopped existing.
find_adapter() {
    find "$WORK" -maxdepth 5 -name adapter_model.safetensors "$@" \
        -printf '%T@ %h\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
}

OUT="${LITIGPT_OUTPUT_DIR:-}"
if [ -n "$OUT" ]; then
    log "output dir given: $OUT"
elif OUT=$(find_adapter -not -path '*/checkpoint-*') && [ -n "$OUT" ]; then
    log "output dir found by search: $OUT"
fi

if [ -n "${OUT:-}" ] && [ -f "$OUT/adapter_model.safetensors" ]; then
    SRC="$OUT"
    log "final adapter present at $SRC"
else
    SRC=$(find_adapter)
    log "no final adapter written; falling back to latest checkpoint: ${SRC:-none}"
fi

# ---- name the archive by version ------------------------------------------
# Every archive used to be litigpt_final.tar.gz, so a second run on the same
# volume replaced the first. The manifest's version is unique per run, and
# `python -m litigpt.registry import` unpacks into models/<version>/.
VERSION=""
if [ -n "${SRC:-}" ] && [ -f "$SRC/litigpt_manifest.json" ]; then
    VERSION=$(python -c 'import json, sys; print(json.load(open(sys.argv[1])).get("version") or "")' \
        "$SRC/litigpt_manifest.json" 2>>"$LOG")
fi
[ -n "$VERSION" ] || log "WARNING: no version in the manifest; archive named litigpt_final"
ARCHIVE="$WORK/litigpt_${VERSION:-final}.tar.gz"

# ---- slim archive: adapter + tokenizer + run record, no optimizer state ----
# optimizer.pt is ~140MB and is only needed to resume training, not to run
# the model, so it stays out of the archive. mlflow.db and the training log go
# in under run_artifacts/, where they used to be copied by hand.
EXTRA=()
for f in "$WORK/mlflow.db" "$TRAIN_LOG"; do
    [ -f "$f" ] && EXTRA+=(-C "$(dirname "$f")" "$(basename "$f")")
done
# What the stop message says about the adapter, and how loudly.
ARCHIVE_NOTE="Nothing archived; see $LOG."
STOP_PRIORITY=high
if [ -n "${SRC:-}" ] && [ -d "$SRC" ]; then
    if tar czf "$ARCHIVE" \
        --exclude='optimizer.pt' \
        --exclude='rng_state.pth' \
        --exclude='scheduler.pt' \
        --transform='s,^\([^./][^/]*\)$,run_artifacts/\1,' \
        -C "$SRC" . "${EXTRA[@]}" 2>>"$LOG"; then
        SIZE=$(du -h "$ARCHIVE" 2>/dev/null | cut -f1)
        log "archive written: $ARCHIVE ($SIZE) from $SRC"
        ARCHIVE_NOTE="Archived $(basename "$ARCHIVE") ($SIZE)."
        STOP_PRIORITY=default
    else
        log "ERROR: archive failed"
        ARCHIVE_NOTE="Archive FAILED; the adapter is still in $SRC. See $LOG."
    fi
else
    log "ERROR: no adapter directory found; nothing archived"
fi

# ---- final metrics into the log -------------------------------------------
python - "$WORK/mlflow.db" >> "$LOG" 2>&1 <<'PY'
import sqlite3, sys
try:
    c = sqlite3.connect('file:%s?mode=ro' % sys.argv[1], uri=True)
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
BILLING="The pod is STILL BILLING: stop it from the RunPod console."
if [ "$MODE" = "--no-stop" ]; then
    log "--no-stop given; pod left running and STILL BILLING"
    notify "litiGPT: pod left running" high money_with_wings \
        "$ARCHIVE_NOTE --no-stop was given. $BILLING"
    exit 0
fi

if [ -z "$POD_ID" ]; then
    log "ERROR: could not determine the pod id (RUNPOD_POD_ID unset and"
    log "       /etc/rp_environment unreadable). Pod left RUNNING and STILL BILLING."
    notify "litiGPT: pod STILL BILLING" urgent money_with_wings \
        "$ARCHIVE_NOTE The watchdog could not find the pod id. $BILLING"
elif runpodctl get pod >/dev/null 2>&1; then
    # Before the stop, not after: nothing on the pod runs once it stops.
    notify "litiGPT: stopping pod" "$STOP_PRIORITY" stop_button \
        "$ARCHIVE_NOTE Stopping pod $POD_ID."
    log "stopping pod $POD_ID"
    if runpodctl stop pod "$POD_ID" >> "$LOG" 2>&1; then
        log "stop command accepted"
    else
        log "ERROR: stop command failed"
        notify "litiGPT: pod STILL BILLING" urgent money_with_wings \
            "$ARCHIVE_NOTE runpodctl stop failed. $BILLING"
    fi
else
    log "runpodctl has no usable API key; pod left RUNNING and STILL BILLING."
    log "To enable auto-stop, run on the pod before training ends:"
    log "  runpodctl config --apiKey <your-runpod-api-key>"
    notify "litiGPT: pod STILL BILLING" urgent money_with_wings \
        "$ARCHIVE_NOTE runpodctl has no API key on the pod. $BILLING"
fi
