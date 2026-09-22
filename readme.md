# litiGPT

Fine-tune a language model on real Reddit users, then test whether anyone can
tell which one it is imitating.

litiGPT takes a subreddit dump, builds a training set of context/reply pairs
for a cohort of prolific authors, fine-tunes a single QLoRA adapter that learns
all of them at once, and serves it behind three interfaces — a chat UI, a
streaming API, and a blind evaluation where you guess which persona is
answering. It targets an Italian subreddit, so every prompt the model sees is
in Italian.

One model holds every persona. The username is always part of the system
prompt, in training and at inference, so switching personas is a prompt change
rather than a model swap.

---

## Contents

- [How it works](#how-it-works)
- [Install](#install)
- [Running the pipeline](#running-the-pipeline)
- [Chat interfaces](#chat-interfaces)
- [Blind evaluation](#blind-evaluation)
- [Configuration](#configuration)
- [Training on a rented GPU](#training-on-a-rented-gpu)
- [Deploying to Reddit](#deploying-to-reddit)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Ethics](#ethics)

---

## How it works

```
subreddit dump (parquet/jsonl)
  │
  ├─ extract      pick the cohort, pull each author's comments with their parents
  ├─ preprocess   clean, render threads, split prompt/completion, write train+val
  ├─ train        QLoRA on one adapter that holds every persona
  ├─ eval         blind rounds, TF-IDF attribution, MLflow curves
  └─ deploy       optional: a bot that answers in a subreddit
```

Two details carry most of the quality:

**Loss is masked to the reply.** The prompt — system message plus parent
comments — is labelled `-100` and contributes nothing. Without this the model
spends most of its training signal learning to reproduce context it will be
given for free at inference. In one measured run the reply was 33 of 231
tokens, so 86% of the gradient was going to the wrong place.

**LoRA target modules are detected, not named.** Projection names are
architecture-specific: phi-3 fuses q/k/v into `qkv_proj` and gate/up into
`gate_up_proj`. A hardcoded Llama list silently matched two of seven names and
trained a third of the intended parameters with nothing on the attention
queries or keys. Leave `lora.target_modules` empty; training now refuses to
start if a name you supply matches no module.

---

## Install

Python 3.11+. [uv](https://docs.astral.sh/uv/) manages the environment.

```bash
uv sync
```

That resolves a CPU-capable torch. For GPU training, reinstall torch against
your CUDA version afterwards:

```bash
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu128 --reinstall
```

Reddit credentials are only needed for the bot. Copy `.env.example` to `.env`
and fill it in if you want one.

---

## Running the pipeline

Each step reads the same config and writes what the next one expects.

```bash
python -m litigpt.pipeline --step extract    --config config.top30.yaml
python -m litigpt.pipeline --step preprocess --config config.top30.yaml
python -m litigpt.pipeline --step train      --config config.top30.yaml
python -m litigpt.pipeline --step eval       --config config.top30.yaml
```

`--step all` chains them. `--step deploy` starts the Reddit bot.

If your raw data is zstd-compressed JSONL rather than parquet, convert it
first — a separate pass because it is slow and you only do it once. This
converts everything under `data/raw`, picking columns from each filename:

```bash
python -m litigpt.data.preliminary
```

Before spending GPU time, check what preprocessing produced:

```bash
python scripts/validate_dataset.py --config config.top30.yaml
```

It catches the cheap-to-fix, expensive-to-discover problems: personas with too
few examples, a cohort so imbalanced the model collapses onto the loudest
author, and markup that survived cleaning.

### Rehearse locally first

```bash
bash scripts/wsl_smoke_test.sh
```

Runs every stage on three personas and 200 pairs against a 135M base. It takes
minutes, costs nothing, and exercises the same code path as the real run — so
breakage surfaces here rather than on a rented GPU.

---

## Chat interfaces

All three load a base model plus your adapter and talk to it through the same
prompt builder the training data was written with. Launch any of them with:

```bash
python launch_chat.py --interface {gradio|ollama|blind} --model models/litigpt_top30_lora
```

| Interface | Default port | What it is |
|---|---|---|
| `gradio` | 7860 | Chat UI with persona picker and sampling controls. `--share` for a public link. |
| `ollama` | 5000 | Terminal-styled web UI with token streaming, over Flask. |
| `blind` | 7861 | Blind persona evaluation — see below. |

`--base-model` defaults to `microsoft/phi-3-mini-4k-instruct`, matching the
configs. Override it if you trained on something else; the adapter will not
load against the wrong base.

You can also run a module directly, which takes the same arguments minus the
launcher's dispatch:

```bash
python -m litigpt.interface.gradio_app --model models/litigpt_top30_lora
python -m litigpt.interface.ollama     --model models/litigpt_top30_lora --port 5000
```

Serving from a RunPod pod instead of locally: `scripts/pod_serve.sh
<blind|chat|stop>` starts an interface detached, so it survives the SSH session
that launched it. It defaults to port 7870 and localhost deliberately — read
its header before exposing it, since RunPod's HTTP proxy is public to anyone
with the URL and this serves a model impersonating real, named people.

---

## Blind evaluation

The model answers as a persona drawn at random and you guess which. This is the
measurement that matters — loss curves say the adapter fit the data, not that
the personas are distinguishable.

```bash
# The ceiling first: real held-out comments, no model, no GPU.
python launch_chat.py --interface blind --oracle

# Then the model, with the TF-IDF attributor guessing alongside you.
python launch_chat.py --interface blind --model models/litigpt_top30_lora --classifier
```

Run `--oracle` before reading any model score. It replays the persona's own
comments, so it establishes how much identity the text carries at all — if
humans cannot pick the real author apart, a model that also cannot is not
failing.

Full protocol, scoring, and how to read the disagreements between human and
classifier: [blind_eval_guide.md](blind_eval_guide.md).

---

## Configuration

Validated by Pydantic models in [litigpt/config.py](litigpt/config.py). Keys
that do not exist there are accepted and ignored, so check spelling against
that file rather than against another YAML.

| File | Use |
|---|---|
| [config.default.yaml](config.default.yaml) | Annotated template, every key with its default. Ships in the container. |
| [config.top30.yaml](config.top30.yaml) | The real 30-persona run, with the reasoning for each value written out. |
| [config.smoke.yaml](config.smoke.yaml) | Tiny end-to-end rehearsal. |
| [config.runpod.yaml](config.runpod.yaml) | Same as top30 with `/workspace` paths for a RunPod network volume. |

`config.yaml` is gitignored — copy the template there for local edits, or pass
`--config` explicitly.

There is no single-user mode. One persona is just `target_usernames` with one
entry, and the prompt still names them.

---

## Training on a rented GPU

litiGPT trains on RunPod. Two paths, both in
[cloud_training_guide.md](cloud_training_guide.md): a bootstrap script on a
stock PyTorch template, or the pinned image in `Dockerfile.runpod`.

Operating a pod day to day — SSH, moving data, watching a run, and the watchdog
that stops the pod when training ends so it stops billing — is in
[runpod_guide.md](runpod_guide.md). Read the watchdog section before starting a
long run: a finished job leaves the GPU idle at full price indefinitely.

Reference point: 30 personas, ~54k training examples, phi-3-mini with r=32
took 3h35m on an RTX 4090 at $0.74/hr.

---

## Deploying to Reddit

```bash
python -m litigpt.pipeline --step deploy --config config.top30.yaml
```

That is the only entry point; it builds the bot from `bot:` and
`user_classification:` in your config. The bot streams new comments, decides
whether to answer, renders the thread through the shared prompt builder, and
appends a disclaimer to every reply.

Which persona it answers as comes from `user_classification.strategy`:
`random` picks from `bot.available_users`, `keyword` routes on topic keywords.
Both live in [litigpt/inference/classifier.py](litigpt/inference/classifier.py)
behind a `UserSelector` protocol — `select_user(context, available_users)`
returning `None` to fall back. A new strategy is one class implementing that.

There is deliberately no model-based persona router.

Run it where a GPU is: the adapter loads in 4-bit, and CPU inference is slow
enough to be impractical for a live bot.

---

## Tests

```bash
uv run pytest
```

The suite targets failures that do not announce themselves — a LoRA target list
matching no module, an interface rebuilding the thread format its own way, a
persona name leaking into a blind round. Each of those once passed review and
trained or served without error.

`tests/test_prompts.py::TestNoSecondImplementation` reads source rather than
behaviour on purpose: the interfaces need a loaded model to exercise, so the
guard is against a second implementation existing at all. If you add a module
that renders threads, add it to that list.

---

## Troubleshooting

**Out of memory.** Lower `training.batch_size` and raise
`gradient_accumulation_steps` to keep the effective batch constant. Then
`max_seq_length` — but note that shortening it drops examples whose prompt
fills the window, since there is no reply left to learn from.

**Fluent output that does not answer the question.** Check the loss mask before
blaming model size. Verify which modules the adapter actually contains rather
than trusting `adapter_config.json`, which records what was requested, not what
matched:

```bash
python -c "import safetensors.torch as st; t=st.load_file('adapter_model.safetensors'); print(sorted({k.split('.')[-3] for k in t}))"
```

**Every persona sounds the same.** Usually too little data per author, or a
cohort so imbalanced the model collapsed onto the loudest. Run
`scripts/validate_dataset.py` and set `data.max_pairs_per_user`.

**MLflow refuses to start.** Recent versions reject a file store. Use SQLite:
`export MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlflow.db`.

**Bot posts nothing.** Check `.env` credentials, then `reply_probability` and
`cooldown_seconds` — the defaults are deliberately quiet.

---

## Ethics

This trains on real, named people and reproduces how they write.

- Every bot reply carries a disclaimer. Do not remove it.
- Follow the subreddit's rules on bots, and ask first if they are unclear.
- Blind-evaluation logs contain generated impersonations of real users. They
  are gitignored; keep them that way.
- Do not use this to put words in someone's mouth where it could be taken as
  genuine.
