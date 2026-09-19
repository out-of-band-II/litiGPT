# Cloud GPU Training (RunPod)

Training litiGPT on a rented GPU. Two paths: a bootstrap script on a stock
RunPod template (fastest), or a pinned container image (reproducible).

Colab and Kaggle remain documented in [colab_kaggle_guide.md](colab_kaggle_guide.md);
RunPod is the option without a session time limit.

---

## Why RunPod over Colab/Kaggle

| | Colab Free | Kaggle | RunPod |
|---|---|---|---|
| Session cap | ~12 h, pre-emptible | 9 h/week quota | none |
| GPU choice | whatever is assigned | P100 / T4 ×2 | you pick |
| Persistent storage | Drive mount | /kaggle/working | network volume |
| Cost | free / $10 mo | free | per-second billing |

A 3-epoch run on ~2k comments finishes well inside Colab's window, so RunPod
earns its cost when you want a **specific** GPU, a larger base model, or
unattended multi-run sweeps.

---

## Path A — bootstrap script (recommended)

No image build, no registry.

**1. Create the pod.** RunPod console → Pods → Deploy.
- Template: any official **PyTorch** template
- GPU: RTX 4090 (24 GB) for 7–8B bases; RTX A5000 (24 GB) is usually cheaper
- **Network volume: 50 GB mounted at `/workspace`** — without it the pod's disk
  is ephemeral and you lose weights and adapters on teardown
- Expose SSH

**2. Clone and bootstrap.**

```bash
cd /workspace
git clone <your-repo-url> litiGPT
cd litiGPT

export HF_TOKEN=hf_...          # only needed for gated bases (meta-llama/*)
bash scripts/runpod_bootstrap.sh setup
```

`setup` verifies the GPU, installs the package editable, and prints the resolved
torch/CUDA/bf16 state. Stop here if anything looks wrong — a `cuda available
False` line means the pod has no GPU attached and training would silently fall
back to CPU.

**3. Get data onto the pod.** From your workstation:

```bash
runpodctl send data/training          # prints a one-time code
```

On the pod:

```bash
runpodctl receive <code>
```

Or run the earlier stages on the pod, if the raw dumps are there:

```bash
python -m litigpt.pipeline --step extract    --config config.runpod.yaml
python -m litigpt.pipeline --step preprocess --config config.runpod.yaml
```

**4. Train.**

```bash
bash scripts/runpod_bootstrap.sh train
```

Adapters land in `/workspace/models/reddit_bot_lora`, MLflow runs in
`/workspace/mlruns` — both on the network volume.

**5. Retrieve results before terminating.**

```bash
runpodctl send /workspace/models/reddit_bot_lora
```

Terminating a pod without copying the adapter out loses it unless it sits on the
network volume. Check the path before you click terminate.

---

## Path B — pinned container

Use when the environment must not drift between runs.

```bash
docker build -f Dockerfile.runpod -t <registry>/litigpt-training:latest .
docker push <registry>/litigpt-training:latest
```

RunPod → Deploy → Custom Container:
- Image: `<registry>/litigpt-training:latest`
- Volume mount path: `/workspace`
- Env: `HF_TOKEN`, optionally `MLFLOW_TRACKING_URI`

The image's default command runs the `train` stage. Override to `bash` for an
interactive session.

---

## Configuration

`config.runpod.yaml` is tracked and points at `/workspace`. It defaults to
`microsoft/phi-3-mini-4k-instruct` — no gated-repo access, and it fits
comfortably on a 24 GB card.

Batch sizing by VRAM at `max_seq_length: 512`:

| GPU | VRAM | `batch_size` | `gradient_accumulation_steps` |
|---|---|---|---|
| RTX 4090 / A5000 | 24 GB | 8 | 4 |
| A40 / A6000 | 48 GB | 16 | 2 |
| A100 | 80 GB | 32 | 1 |

Keep `batch_size × gradient_accumulation_steps` at 32 so the effective batch —
and therefore the learning-rate schedule — stays comparable across GPUs.

---

## Troubleshooting

**CUDA OOM.** Halve `batch_size` and double `gradient_accumulation_steps`. If it
still OOMs, drop `max_seq_length` to 384.

**`401` pulling the base model.** `HF_TOKEN` is unset or the account hasn't
accepted the model's licence. Accept it on the model's Hub page first.

**Re-downloading weights every run.** `HF_HOME` must resolve under `/workspace`.
The bootstrap script sets this; a manual `python -m litigpt.pipeline` call in a
fresh shell does not.

**`bf16 supported False`.** Pre-Ampere GPU (e.g. V100). The trainer falls back to
fp16 automatically — no action needed, but expect slightly slower steps.
