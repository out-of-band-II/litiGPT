#!/usr/bin/env bash
#
# Bootstrap a RunPod GPU pod for litiGPT training.
#
# Designed for RunPod's stock PyTorch templates — no custom image or registry
# push required. Everything persistent lives under the network volume at
# /workspace so a pod can be destroyed and recreated without losing work.
#
# Usage (from an SSH session on the pod):
#   export HF_TOKEN=hf_...            # required for gated models (Llama)
#   bash scripts/runpod_bootstrap.sh            # full run: setup + train
#   bash scripts/runpod_bootstrap.sh setup      # deps only
#   bash scripts/runpod_bootstrap.sh train      # training only
#
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/litiGPT}"
CONFIG="${LITIGPT_CONFIG:-config.runpod.yaml}"
STAGE="${1:-all}"

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Caches on the network volume -------------------------------------------
# Base weights are multi-GB; keeping the HF cache on /workspace means a pod
# restart re-uses them instead of re-downloading.
export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE/.cache/pip}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

setup() {
  log "GPU check"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — is this a GPU pod?"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

  log "Installing litiGPT and dependencies"
  cd "$REPO_DIR"
  python -m pip install --quiet --upgrade pip setuptools wheel
  # Editable install so the package resolves from the repo checkout.
  python -m pip install --quiet -e .
  # tensorboard is an optional reporting backend the trainer auto-detects.
  python -m pip install --quiet tensorboard

  log "Verifying the training stack"
  python - <<'PY'
import torch
print(f"torch            {torch.__version__}")
print(f"cuda available   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device           {torch.cuda.get_device_name(0)}")
    print(f"bf16 supported   {torch.cuda.is_bf16_supported()}")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"vram             {total:.1f} GiB")
import bitsandbytes, peft, transformers, trl  # noqa: F401
print(f"transformers     {transformers.__version__}")
print(f"peft             {peft.__version__}")
print(f"trl              {trl.__version__}")
print(f"bitsandbytes     {bitsandbytes.__version__}")
PY

  if [[ -n "${HF_TOKEN:-}" ]]; then
    log "Authenticating to Hugging Face"
    python -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
  else
    warn "HF_TOKEN not set — gated bases (meta-llama/*) will fail to download."
  fi
}

check_data() {
  local cfg="$REPO_DIR/$CONFIG"
  [[ -f "$cfg" ]] || die "Config not found: $cfg"
  local train_dir
  train_dir=$(python - "$cfg" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print((cfg.get("data") or {}).get("training_dir", "data/training"))
PY
)
  if [[ ! -f "$train_dir/train.jsonl" ]]; then
    die "No training data at $train_dir/train.jsonl.
Upload prepared data to the pod first, e.g. from your workstation:
  runpodctl send data/training
or run the earlier pipeline stages here:
  python -m litigpt.pipeline --step extract    --config $CONFIG
  python -m litigpt.pipeline --step preprocess --config $CONFIG"
  fi
  log "Training data found at $train_dir"
}

train() {
  cd "$REPO_DIR"
  check_data
  log "Starting training (config: $CONFIG)"
  # MLflow writes to the volume so runs survive pod teardown.
  export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:$WORKSPACE/mlruns}"
  python -m litigpt.pipeline --step train --config "$CONFIG"
  log "Adapters written to $(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print((cfg.get("model") or {}).get("output_dir", "models/reddit_bot_lora"))
PY
)"
  cat <<'MSG'

Retrieve the trained adapter before terminating the pod:
  runpodctl send /workspace/models/reddit_bot_lora
Or push it to the Hub:
  python -c "from peft import PeftModel; ..."  # see cloud_training_guide.md
MSG
}

case "$STAGE" in
  setup) setup ;;
  train) train ;;
  all)   setup; train ;;
  *)     die "Unknown stage '$STAGE' (expected: setup | train | all)" ;;
esac

log "Done."
