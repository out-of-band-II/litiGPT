# Claude AI Project Context

Context for Claude (or another AI assistant) working on litiGPT.

## What this is

litiGPT fine-tunes one QLoRA adapter on a cohort of real Reddit users from an
Italian subreddit, so that a single model can answer as any of them, and then
measures whether the personas are actually distinguishable.

The measurement is the point. Loss curves say the adapter fit the data; the
blind evaluation says whether a human can tell alice from bob. Work that
improves the first without the second is not obviously progress.

**All prompts are in Italian.** The username is always in the system prompt —
in training and at inference, with one persona or thirty. There is no
single-user mode and no `multi_user` flag.

## Architecture

Source lives in `litigpt/`, organized by function. Diagrams and the format
contracts are in [architecture_diagram.md](architecture_diagram.md).

- **`data/`** — `preliminary.py` (zstd/jsonl → parquet, run once),
  `extraction.py` (cohort selection, thread indexing, parent walking),
  `preprocessing.py` (cleaning, formatting, prompt/completion split)
- **`training/`** — `trainer.py` (QLoRA, loss masking, target detection),
  `tracking.py` (MLflow)
- **`inference/`** — `generator.py`, `classifier.py` (persona selection)
- **`interface/`** — `gradio_app.py`, `ollama.py`, `blind_eval.py`
- **`eval/`** — `attribution.py` (TF-IDF + logistic regression authorship)
- **`deployment/`** — `reddit_bot.py`
- **`prompts.py`** — the one definition of prompts and thread format
- **`model_utils.py`** — loading, quantization, dtype detection
- **`config.py`** — Pydantic schema
- **`pipeline.py`** — orchestration and every CLI entry point

## Things that have bitten before

These are not hypothetical. Each one shipped, trained or served without error,
and was found by reading two files side by side.

**LoRA target modules must be detected, not named.** phi-3 fuses q/k/v into
`qkv_proj` and gate/up into `gate_up_proj`. A hardcoded seven-name Llama list
matched two, so PEFT adapted `o_proj` and `down_proj`, said nothing, and still
wrote all seven into `adapter_config.json` — the file on disk disagreed with
the weights. Nothing reached the attention queries or keys. Leave
`lora.target_modules` empty; the trainer now raises on a name that matches
nothing. Verify what an adapter really contains:

```bash
python -c "import safetensors.torch as st; t=st.load_file('adapter_model.safetensors'); print(sorted({k.split('.')[-3] for k in t}))"
```

**Loss must be masked to the reply.** Scoring the whole templated example
trains the model to reproduce the system prompt and the parent comments. The
reply was 33 of 231 tokens in one measurement, so 86% of the signal went to
text the model gets for free at inference. The dataset is split into
`prompt`/`completion` so TRL masks it. A knock-on: an example whose prompt
fills `max_seq_length` has nothing left to learn from, hence
`drop_unlearnable` and the move to 1024.

**The thread format is defined once.** `litigpt/prompts.py::render_thread`.
Every copy drifted — Gradio labelled turns "assistant", Ollama labelled both
sides "user"/"assistant", the Reddit bot emitted `Post: {title}` — English
labels that appear nowhere in training data built from real usernames. Feeding
the model a shape it never saw degrades output and raises nothing.
`tests/test_prompts.py::TestNoSecondImplementation` guards this by reading
source. **Add any new thread-rendering module to its `MODULES` list.**

**Both of the above fail silently.** Loss falls, checkpoints save, eval curves
look healthy. When a fine-tune produces fluent output that does not answer the
question, check the mask and the adapted modules before blaming model size.

## Persona selection is deliberately not learned

`inference/classifier.py` has a `UserSelector` protocol, `RandomUserSelector`,
and `KeywordUserSelector`. That is all of it, by design.

There is no TF-IDF router, no `HybridUserSelector`, no `UserClassifier`, and no
`models/user_classifier.pkl`. Documentation used to describe all four at
length; none ever existed. The only TF-IDF in the project is
`eval/attribution.py`, which *measures* persona separation for the blind
evaluation — it does not route.

To add a strategy: one class implementing
`select_user(context, available_users) -> Optional[str]`, returning `None` to
fall back to `DEFAULT_USERNAME`; then a branch in `pipeline.run_deployment`
and a `user_classification.strategy` value. The likely next one reads an
explicit request out of a mention ("answer as tommyrugby").

## Configuration

Pydantic models in `config.py`. Load with `Config.from_yaml(path)`.

| File | Use |
|---|---|
| `config.default.yaml` | annotated template, every key; ships in the container |
| `config.top30.yaml` | the real 30-persona run, reasoning written out per value |
| `config.smoke.yaml` | tiny end-to-end rehearsal |
| `config.runpod.yaml` | top30 with `/workspace` paths |
| `config.yaml` | user-specific, gitignored |

**Keys absent from `config.py` are ignored silently.** A typo in YAML is
invisible. When adding a config value, add it to the Pydantic model first.

## Common tasks

**Debugging a training run:** confirm the dataset exists and validates
(`scripts/validate_dataset.py --config ...`), confirm GPU availability,
confirm the adapter's real modules, confirm the loss mask. Then look at the
MLflow SQLite DB rather than the training log — Python block-buffers stdout
when redirected, so loss lines do not reach the file live.

**Adding a base model:** test the chat template first, then adjust
`batch_size`. `DEFAULT_BASE_MODEL` in `model_utils.py` is
`microsoft/phi-3-mini-4k-instruct`; it was a gated Llama repo, which meant an
auth failure for anyone who launched an interface without `--base-model`.

**Adding an interface:** import from `prompts.py`. Do not rebuild the prompt
or thread format. Add the module to the test guard.

## Style

- Classes PascalCase, functions and modules snake_case
- `logging`, not `print`, in library code; `tqdm` for progress
- Catch specific exceptions, log with context, degrade or re-raise
- Comments explain why, especially where a value was chosen empirically —
  `config.top30.yaml` is the model for this

## Testing

```bash
uv run pytest
```

The suite targets failures that do not announce themselves: a LoRA target list
matching no module, an interface rebuilding the thread format, a persona name
leaking into a blind round. Tests that only check things which would already
have raised are not worth much here.

`TestNoSecondImplementation` inspects source rather than behaviour on purpose —
the interfaces need a loaded model to exercise, so the guard is against a
second implementation existing at all.

Before any paid GPU run: `bash scripts/wsl_smoke_test.sh`.

## Infrastructure

Training runs on RunPod. `Dockerfile.runpod` is the pinned image;
`scripts/runpod_bootstrap.sh` on a stock PyTorch template is the lower-friction
path. `Dockerfile.training` is the older generic CUDA image and is not what
current runs use.

The stock RunPod PyTorch image does not work as shipped. Working combination:

```
torch 2.11.0+cu128   transformers 4.57.6   trl 0.29.1   peft 0.21.0
datasets 4.8.5       accelerate 1.15.0     bitsandbytes 0.50.2
mlflow 3.16.1        torchvision/torchaudio UNINSTALLED
```

- transformers 5.x needs torch ≥2.5; on older torch it silently disables torch
  and then dies on `NameError: name 'nn' is not defined`
- the image's torchvision breaks any upgraded torch, surfacing as a misleading
  "cannot import BloomPreTrainedModel"
- **MLflow ≥3 refuses a file store.** Use `sqlite:///...`

**Pods never stop themselves.** A finished job leaves the GPU billing
indefinitely. `scripts/post_training_watchdog.sh` archives the adapter and
stops the pod, but needs an API key configured on the pod first. See
[runpod_guide.md](runpod_guide.md).

Reference run: 30 personas, ~54k examples, phi-3-mini r=32, 3h35m on an RTX
4090 at $0.74/hr. eval_loss converged at roughly two epochs; a third bought
0.0014 for about an hour of GPU, which is why configs say 2 with early
stopping.

## Documentation

- [readme.md](readme.md) — setup, pipeline, interfaces, troubleshooting
- [quick_reference.md](quick_reference.md) — one-page command sheet
- [architecture_diagram.md](architecture_diagram.md) — structure and contracts
- [cloud_training_guide.md](cloud_training_guide.md) — RunPod training paths
- [runpod_guide.md](runpod_guide.md) — operating and debugging pods
- [blind_eval_guide.md](blind_eval_guide.md) — evaluation protocol
- `CLAUDE.md` — this file

## Environment

```bash
REDDIT_CLIENT_ID=            # only needed for the bot
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=
REDDIT_USERNAME=
REDDIT_PASSWORD=
MLFLOW_TRACKING_URI=         # sqlite:///... ; a file store is rejected
HF_TOKEN=                    # only for gated bases
```

## Commands

```bash
python -m litigpt.pipeline --step {extract|preprocess|train|eval|deploy|all} --config config.top30.yaml
python -m litigpt.data.preliminary
python scripts/validate_dataset.py --config config.top30.yaml
bash scripts/wsl_smoke_test.sh

python launch_chat.py --interface {gradio|ollama|blind} --model models/litigpt_top30_lora
python launch_chat.py --interface blind --oracle        # ceiling, no GPU needed

uv run pytest
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --port 5000
```

## Known limits

1. ~100+ comments per persona minimum
2. Context capped at 3–5 parents by the token budget
3. 12GB+ VRAM to train
4. Reddit API rate limits
5. Adapter quality is bounded by how distinguishable the authors are at all —
   run `--oracle` before reading any score

## Ethics

This reproduces how real, named people write.

- Every bot reply carries a disclaimer; do not remove it
- Follow subreddit rules on bots
- Blind-eval logs hold impersonations of real users and are gitignored
- Do not use this to put words in someone's mouth where it could pass as
  genuine

---

**Last Updated**: September 2026
