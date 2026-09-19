#!/usr/bin/env bash
#
# End-to-end smoke test for WSL (or any Linux box).
#
# Runs the whole pipeline on a deliberately tiny cohort so that anything broken
# surfaces here, in minutes and for free, rather than on a paid RunPod GPU.
# This is the rehearsal for the RunPod run: same code path, same configs, a
# fraction of the data.
#
#   bash scripts/wsl_smoke_test.sh            # env + data + validate (no GPU needed)
#   bash scripts/wsl_smoke_test.sh --train    # also run a real 1-epoch train
#   bash scripts/wsl_smoke_test.sh --clean    # remove smoke artefacts and exit
#
set -euo pipefail

CONFIG="${LITIGPT_CONFIG:-config.smoke.yaml}"
VENV="${VENV:-.venv-wsl}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DO_TRAIN=0

for a in "$@"; do
  case "$a" in
    --train) DO_TRAIN=1 ;;
    --clean) rm -rf "$REPO/data/smoke" "$REPO/models/smoke_lora"; echo "removed smoke artefacts"; exit 0 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

cd "$REPO"
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m  FAIL\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
step "1/6  Environment"
grep -qi microsoft /proc/version 2>/dev/null && ok "running under WSL" || ok "running on native Linux"

case "$REPO" in
  /mnt/*) echo "  note: repo lives on a Windows mount ($REPO)."
          echo "        Filesystem I/O here is slow; for the real 30-user run"
          echo "        consider cloning into the WSL filesystem (~/litiGPT).";;
esac

command -v python3 >/dev/null || die "python3 not found — apt install python3 python3-venv"
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
ok "python3 $PYV"
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' \
  || die "python >= 3.11 required (pyproject requires-python)"

# A Windows .venv cannot be reused from Linux — its interpreter is a .exe.
if [[ ! -d "$VENV" ]]; then
  step "     creating $VENV (first run only)"
  python3 -m venv "$VENV" || die "venv creation failed — apt install python3-venv"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
ok "venv active: $(python -c 'import sys; print(sys.prefix)')"

step "2/6  Dependencies"
if ! python -c "import litigpt, polars, transformers" 2>/dev/null; then
  echo "  installing (first run takes a few minutes)..."
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -e .
fi
python - <<'PY'
import importlib
mods = ["polars", "transformers", "peft", "trl", "datasets", "yaml", "pydantic", "torch"]
missing = [m for m in mods if not importlib.util.find_spec(m)]
if missing:
    raise SystemExit("missing packages: " + ", ".join(missing))
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu: {torch.cuda.get_device_name(0)}")
PY
ok "imports resolve"

step "3/6  Config loads"
python - "$CONFIG" <<'PY'
import sys
from litigpt.config import Config
c = Config.from_yaml(sys.argv[1])
print(f"  cohort      top {c.data.top_n_users} users, cap {c.data.max_pairs_per_user}")
print(f"  model       {c.model.base_model}")
print(f"  output      {c.data.training_dir}")
PY
ok "$CONFIG valid"

# ---------------------------------------------------------------------------
step "4/6  Extract + preprocess"
RAW=$(python -c "from litigpt.config import Config;print(Config.from_yaml('$CONFIG').data.raw_dir)")
COMMENTS=$(python -c "from litigpt.config import Config;print(Config.from_yaml('$CONFIG').data.comments_filename)")
[[ -f "$RAW/$COMMENTS" ]] || die "raw corpus not found at $RAW/$COMMENTS"
ok "raw corpus present"

python -m litigpt.pipeline --step extract    --config "$CONFIG"
python -m litigpt.pipeline --step preprocess --config "$CONFIG"
ok "pipeline stages completed"

# ---------------------------------------------------------------------------
step "5/6  Validate dataset"
# --strict: in a smoke test any surviving markup is a real regression, since the
# whole point is to catch it before the paid run.
python scripts/validate_dataset.py --config "$CONFIG" --min-per-user 50 --strict \
  || die "dataset validation failed — fix before running on RunPod"
ok "dataset validation passed"

step "     Sample of what the model will actually see"
python - "$CONFIG" <<'PY'
import json, sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
p = Path(cfg["data"]["training_dir"]) / "train.jsonl"
with p.open(encoding="utf-8") as fh:
    ex = json.loads(fh.readline())
for m in ex["messages"]:
    body = m["content"].replace("\n", "\n              ")
    print(f"  {m['role']:>9}: {body[:300]}")
PY

# ---------------------------------------------------------------------------
if [[ "$DO_TRAIN" -eq 1 ]]; then
  step "6/6  Training (1 epoch, tiny model)"
  python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" \
    || die "no CUDA device visible. On WSL2 install the Windows NVIDIA driver with
       WSL support; the driver is NOT installed inside WSL. Re-run without
       --train to validate everything up to training."
  python -m litigpt.pipeline --step train --config "$CONFIG"
  ok "training completed"
else
  step "6/6  Training  (skipped — pass --train to run it)"
fi

printf '\n\033[1;32mSMOKE TEST PASSED\033[0m\n'
cat <<'MSG'

Everything up to training is verified against the real corpus. Next:
  bash scripts/wsl_smoke_test.sh --train        # prove the training path too
  bash scripts/wsl_smoke_test.sh --clean        # drop the smoke artefacts

Then the real cohort, same code path, config.top30.yaml.
MSG
