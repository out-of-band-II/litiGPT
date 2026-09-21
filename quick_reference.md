# litiGPT quick reference

One page. Fuller explanations in [readme.md](readme.md).

---

## Setup

```bash
uv sync                                                    # environment
uv pip install torch --extra-index-url \
  https://download.pytorch.org/whl/cu128 --reinstall        # GPU torch
cp .env.example .env                                        # only for the bot
```

---

## Pipeline

```bash
python -m litigpt.pipeline --step extract    --config config.top30.yaml
python -m litigpt.pipeline --step preprocess --config config.top30.yaml
python -m litigpt.pipeline --step train      --config config.top30.yaml
python -m litigpt.pipeline --step eval       --config config.top30.yaml
python -m litigpt.pipeline --step deploy     --config config.top30.yaml
python -m litigpt.pipeline --step all        --config config.top30.yaml
```

`--log-level DEBUG` on any of them.

### Around the pipeline

```bash
python -m litigpt.data.preliminary                       # zstd/jsonl -> parquet
python scripts/validate_dataset.py --config config.top30.yaml
python scripts/validate_dataset.py --config config.top30.yaml --strict
bash scripts/wsl_smoke_test.sh                           # full rehearsal, minutes
```

---

## Chat

```bash
python launch_chat.py --model models/litigpt_top30_lora/final                  # Gradio  :7860
python launch_chat.py --model models/litigpt_top30_lora/final --share          # public link
python launch_chat.py --interface ollama --model <adapter> --port 5000         # streaming :5000
python launch_chat.py --interface blind  --model <adapter> --classifier        # blind eval :7861
python launch_chat.py --interface blind  --oracle                              # ceiling, no GPU
python launch_chat.py --interface blind  --model <adapter> --seed 42           # reproducible draw
```

Add `--base-model` only if you trained on something other than
`microsoft/phi-3-mini-4k-instruct`.

Direct module launch takes the same flags:

```bash
python -m litigpt.interface.gradio_app --model <adapter>
python -m litigpt.interface.ollama     --model <adapter> --host 0.0.0.0
python -m litigpt.interface.blind_eval --model <adapter>
```

---

## RunPod

```bash
ssh litigpt-pod                                    # never the generic alias
bash scripts/runpod_bootstrap.sh setup             # verify GPU, install package
bash scripts/runpod_bootstrap.sh train             # run training
runpodctl send data/training                       # from workstation, prints a code
bash scripts/pod_serve.sh blind                    # serve detached (blind|chat|stop)
setsid nohup bash scripts/post_training_watchdog.sh &   # archive + stop the pod
```

The watchdog needs `runpodctl config --apiKey ...` set on the pod, or it
archives the adapter and leaves the GPU billing. Details in
[runpod_guide.md](runpod_guide.md).

---

## MLflow

```bash
export MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlflow.db   # a file store is rejected
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --port 5000
```

---

## Tests

```bash
uv run pytest
uv run pytest tests/test_prompts.py -v
```

---

## Configs

| File | Use |
|---|---|
| `config.default.yaml` | annotated template, every key |
| `config.top30.yaml` | the real 30-persona run |
| `config.smoke.yaml` | tiny rehearsal |
| `config.runpod.yaml` | top30 with `/workspace` paths |
| `config.yaml` | yours, gitignored |

Keys not defined in [litigpt/config.py](litigpt/config.py) are ignored
silently. Check spelling there.

---

## Knobs worth knowing

| Symptom | Setting |
|---|---|
| OOM | `training.batch_size` down, `gradient_accumulation_steps` up |
| Run too long | `training.num_epochs`, `early_stopping_patience` |
| Eval dominates runtime | `training.eval_steps` up |
| One author drowns the rest | `data.max_pairs_per_user` |
| Replies copy the parent | `data.strip_quoted_text` |
| Bot too quiet / loud | `bot.reply_probability`, `bot.cooldown_seconds` |
| Which persona answers | `user_classification.strategy` (`random` \| `keyword`) |

---

## Checks after a run

```bash
# Which modules the adapter really contains (adapter_config.json records the
# request, not the match).
python -c "import safetensors.torch as st; t=st.load_file('adapter_model.safetensors'); print(sorted({k.split('.')[-3] for k in t}))"
```

---

## Key files

| Path | What |
|---|---|
| [litigpt/prompts.py](litigpt/prompts.py) | the one definition of prompts and thread format |
| [litigpt/pipeline.py](litigpt/pipeline.py) | every entry point |
| [litigpt/config.py](litigpt/config.py) | config schema |
| [litigpt/training/trainer.py](litigpt/training/trainer.py) | QLoRA, loss masking, module detection |
| [litigpt/eval/attribution.py](litigpt/eval/attribution.py) | TF-IDF authorship scoring |
| [litigpt/inference/classifier.py](litigpt/inference/classifier.py) | persona selection strategies |
