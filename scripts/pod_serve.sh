#!/bin/bash
# Serve a litiGPT chat interface from a RunPod pod.
#
# Run this ON the pod. It starts the app detached, so it keeps running after
# the SSH session that launched it goes away, and writes a PID file so it can
# be stopped again without pattern-matching process names.
#
#   ssh litigpt-pod
#   bash /workspace/litiGPT/scripts/pod_serve.sh blind --auth me:secret
#   bash /workspace/litiGPT/scripts/pod_serve.sh stop
#
# Two ways to reach it, and the choice matters:
#
#   SSH tunnel (private, the default here)
#       The app binds to 127.0.0.1 and is reachable only through the tunnel.
#       From your own machine:
#           ssh -N -L 7861:localhost:7861 litigpt-pod
#       then open http://localhost:7861
#
#   RunPod HTTP proxy (public — pass --host 0.0.0.0 and use --auth)
#       https://<POD_ID>-<PORT>.proxy.runpod.net
#       The port must be in the pod's exposed HTTP ports. The proxy URL is
#       public to anyone who has it: the pod id is obscurity, not access
#       control, and RunPod's own documentation says so. This app serves a
#       model impersonating real, named people, so do not expose it without
#       --auth. The proxy also drops requests after 100 seconds, which is
#       ample on a GPU and not on CPU.
#
# usage: pod_serve.sh <blind|chat|stop> [extra args passed to the app]

set -euo pipefail

MODE="${1:-blind}"
shift || true

REPO="${LITIGPT_REPO:-/workspace/litiGPT}"
PORT="${LITIGPT_PORT:-7861}"
HOST="${LITIGPT_HOST:-127.0.0.1}"
ADAPTER="${LITIGPT_ADAPTER:-$REPO/models/litigpt_top30_lora}"
BASE_MODEL="${LITIGPT_BASE_MODEL:-microsoft/phi-3-mini-4k-instruct}"
CONFIG="${LITIGPT_CONFIG:-config.top30.yaml}"

RUN_DIR=/workspace/serve
PID_FILE=$RUN_DIR/app.pid
LOG_FILE=$RUN_DIR/app.log

mkdir -p "$RUN_DIR"

stop_app() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No PID file at $PID_FILE; nothing to stop."
        return 0
    fi
    local pid
    pid=$(cat "$PID_FILE")
    # Stop by recorded PID, never by pattern. Matching on a command line also
    # matches the shell that is doing the matching, which has killed a session
    # on this project before.
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" && echo "Stopped PID $pid"
        sleep 2
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    else
        echo "PID $pid is not running."
    fi
    rm -f "$PID_FILE"
}

if [ "$MODE" = "stop" ]; then
    stop_app
    exit 0
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running as PID $(cat "$PID_FILE"). Stop it first:"
    echo "  bash $0 stop"
    exit 1
fi

cd "$REPO"

# HF_HOME must point at the persistent volume. The container's own filesystem
# is wiped when the pod restarts, and re-downloading the base model is 7.6GB
# every time otherwise.
export HF_HOME=/workspace/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

case "$MODE" in
    blind)
        APP=(python -u -m litigpt.interface.blind_eval
             --model "$ADAPTER"
             --base-model "$BASE_MODEL"
             --config "$CONFIG"
             --host "$HOST" --port "$PORT")
        ;;
    oracle)
        # Control condition: no model is loaded at all.
        APP=(python -u -m litigpt.interface.blind_eval
             --oracle
             --config "$CONFIG"
             --host "$HOST" --port "$PORT")
        ;;
    chat)
        APP=(python -u -m litigpt.interface.gradio_app
             --model "$ADAPTER"
             --base-model "$BASE_MODEL"
             --config "$CONFIG"
             --port "$PORT")
        ;;
    *)
        echo "Unknown mode: $MODE (expected blind, oracle, chat or stop)" >&2
        exit 2
        ;;
esac

if [ "$MODE" != "chat" ] && [ -d "$ADAPTER" ] && [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
    echo "WARNING: $ADAPTER has no adapter_model.safetensors." >&2
    echo "         Point LITIGPT_ADAPTER at a checkpoint directory instead." >&2
fi

# -u keeps the log readable live. Without it Python block-buffers stdout when
# it is redirected, and nothing appears until the process exits.
echo "Starting: ${APP[*]} $*"
setsid nohup "${APP[@]}" "$@" > "$LOG_FILE" 2>&1 < /dev/null &
APP_PID=$!
echo "$APP_PID" > "$PID_FILE"

echo "PID $APP_PID  log: $LOG_FILE"
echo "Waiting for the server to answer on port $PORT..."

for _ in $(seq 1 180); do
    if ! kill -0 "$APP_PID" 2>/dev/null; then
        echo "Process exited during startup. Last lines:" >&2
        tail -30 "$LOG_FILE" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
    if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null; then
        echo "Up."
        if [ "$HOST" = "0.0.0.0" ]; then
            echo "Public (if $PORT is an exposed HTTP port):"
            echo "  https://${RUNPOD_POD_ID:-<POD_ID>}-${PORT}.proxy.runpod.net"
        else
            echo "Bound to localhost. From your machine:"
            echo "  ssh -N -L ${PORT}:localhost:${PORT} litigpt-pod"
            echo "  then open http://localhost:${PORT}"
        fi
        echo "Stop with: bash $0 stop"
        exit 0
    fi
    sleep 2
done

echo "Still not answering after 6 minutes. Check $LOG_FILE." >&2
exit 1
