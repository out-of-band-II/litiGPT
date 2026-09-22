# Handoff: repository review, September 2026

**Transient working note.** Not part of the documentation set — delete it once
the items below are done or triaged. Nothing else references it.

This is the output of a read-through of the whole repository on
2026-09-22, on branch `cleanup/consolidate-docs-and-deploy-path`. It was
produced by reading source, not by running the pipeline. What was actually
executed: `uv run pytest` (62 passed) and one simulation of the MLflow run
name against `config.top30.yaml`. Everything else is a claim from reading two
files side by side, which is the method that found the defects CLAUDE.md
already records — treat it accordingly, and verify before acting where it
matters.

Items are ranked by how much they move the thing this project says it
measures. 1–3 are worth doing together; they are one file apart and 2 and 3
are what make 1's output findable afterwards.

---

## 1. `--step eval` does not measure anything

`litigpt/pipeline.py:284` prints three test contexts and blocks on `input()`.
The contexts are in English — `"user1: What's your opinion on Python vs
JavaScript?"` — in a project where every prompt is Italian by design, so they
are off-distribution for the adapter besides.

Meanwhile `litigpt/eval/attribution.py:303` `held_out_ceiling` has **zero
callers**. Its module docstring advertises a "scorer for automated
persona-separation runs" that exists nowhere as an entry point. The only
consumer of the eval module is `blind_eval`'s optional `--classifier` flag,
which uses `AuthorAttributor` but never the ceiling function.

The consequence: the project's stated thesis — loss curves are not progress,
separability is — has no automated reading. The only way to answer "did this
run help?" is a human blind-eval session.

**Suggested shape.** Make `--step eval` non-interactive and have it emit
numbers:

1. Fit the attributor on `train.jsonl` (`AuthorAttributor.from_training_jsonl`).
2. Report `held_out_ceiling` on `val.jsonl` — how separable the cohort is at
   all, given these features.
3. Generate one reply per persona across N held-out contexts, and report top-1
   and top-3 attribution accuracy against chance and against that ceiling.
4. Log all of it to MLflow beside the loss.

Mostly wiring things that already exist. The caveat in the `attribution.py`
module docstring — classifier trained on real text, asked to judge generated
text — belongs in whatever this prints, so the number is never read as an
absolute.

This is also the item that unblocks the open question about the base model
being the ceiling: swapping phi-3-mini for something else currently produces
no number comparable to the existing run.

## 2. Every MLflow run is named identically, with no cohort

`litigpt/pipeline.py:153` builds both the run name and the `users` tag from
`data.target_usernames`. All three real configs (`config.top30.yaml`,
`config.runpod.yaml`, `config.smoke.yaml`) leave that empty and select the
cohort with `top_n_users`. Verified against `config.top30.yaml`:

```
run_name  -> 'train__phi-3-mini-4k-instruct'
users tag -> ''
```

CLAUDE.md directs you to the MLflow DB rather than the training log, and the
DB currently cannot tell two cohorts apart. The per-user sample counts are
logged as metrics (`user_<name>_samples`), so the cohort is recoverable with
effort — but the run name and the tag, which are what you actually scan, are
empty.

**Fix.** The fallback already exists eight lines up the same file:
`run_preprocessing` resolves the cohort with `load_user_metadata(processed_dir)`
at `litigpt/pipeline.py:77`. Use the same call here.

## 3. The default MLflow URI is the one documented to fail

`file:./mlruns` appears in four places:

- `litigpt/pipeline.py:146`
- `litigpt/pipeline.py:308`
- `litigpt/training/tracking.py:19`
- `litigpt/training/trainer.py:403`

CLAUDE.md states that MLflow ≥3 refuses a file store, and `pyproject.toml`
pins `mlflow>=2.8.0`, so the installed version is free to be one that rejects
it. Forgetting `MLFLOW_TRACKING_URI` on a multi-hour paid GPU run loses the
tracking for that run.

**Fix.** Default to `sqlite:///mlflow.db` in one place, or fail fast with the
reason stated. Defaulting to a value the documentation says does not work is
worse than having no default.

## 4. The train/val split is unseeded, unstratified, and leaks threads

`litigpt/data/preprocessing.py:310-318`:

```python
def split_data(self, data, train_ratio=0.9):
    import random
    random.shuffle(data)
```

No seed — while `create_multi_user_training_pairs`, directly above it, takes a
deliberate `seed: int = 0` for exactly this reason. Three consequences:

- **Not reproducible.** `val.jsonl` changes on every preprocess run, so
  `eval_loss` is not comparable across runs, and a cached attributor can be
  scoring a split it was not built for (see item 5).
- **Not stratified by username.** `scripts/validate_dataset.py` already
  anticipates this with its "users absent from val, so their personas go
  unmeasured" warning — which treats the symptom rather than the cause.
- **Threads span the split.** Pairs are shuffled individually, so two comments
  from the same thread can land on opposite sides with overlapping parent
  context. Val loss is optimistic by an unknown amount.

**Fix.** Seed it and stratify by username. Splitting on thread id rather than
on pair is the more honest fix and the larger change.

## 5. The attributor cache has no fingerprint

`litigpt/interface/blind_eval.py:962` calls `AuthorAttributor.load_or_train`
against a fixed path (`models/author_attributor.joblib`) and loads whatever
joblib file it finds there. Change the cohort, or re-run preprocess (see item
4), and the blind evaluation silently scores with a classifier fit on
different people.

This is the same shape as the LoRA target-modules defect already recorded in
CLAUDE.md: nothing raises, nothing looks wrong, the numbers just quietly mean
something else.

**Fix.** Store the cohort list and a hash of `train.jsonl` in the saved
payload; retrain on mismatch instead of loading.

## 6. Generation is a second implementation in every interface

`tests/test_prompts.py::TestNoSecondImplementation` guards the *thread format*.
It does not guard the layer directly below it.

- `litigpt/interface/gradio_app.py:110` hardcodes `top_p=0.9`, `top_k=50`,
  `repetition_penalty=1.1` and ignores `config.inference` entirely.
- `litigpt/interface/ollama.py:115` does the same.
- Both reimplement tokenize → generate → decode rather than calling
  `RedditBotInference`.

So a blind-eval round and a Gradio chat are not exercising the same model
behaviour — which matters when one of them is the measurement instrument. It
is the drift this project already got burned by, one layer down.

**Fix.** Route both through `litigpt/inference/generator.py` and read
`config.inference`. Consider extending the source-reading guard to the
sampling path, in the spirit of the existing one.

## 7. No CI

`.github/` does not exist. The test suite and the pinned ruff ruleset are both
good, and nothing runs either automatically. The guard tests only guard if
something runs them.

**Fix.** A workflow running `ruff check` and `pytest` on push.

---

## Smaller items

- `scripts/validate_dataset.py:121-123` hardcodes `512` in both the check and
  the printed message. `max_seq_length` has been 1024 since the loss-mask fix,
  so it reports truncation that will not happen.
- `litigpt/training/trainer.py:392-443` — the `__main__` block duplicates the
  training entry point, with the stale file-store URI and the empty-cohort run
  name of items 2 and 3. Dead and actively misleading; delete it.
- `litigpt/inference/classifier.py` and `litigpt/inference/generator.py`
  `__main__` demo blocks use English contexts and `alice`/`bob`/`charlie`.

---

## Repository state at handoff

- Branch `cleanup/consolidate-docs-and-deploy-path`, tracking
  `origin/cleanup/consolidate-docs-and-deploy-path`.
- Working tree clean, nothing unpushed, at commit `34c63bf` plus this note.
- `uv run pytest` — 62 passed.
- No code changes were made during the review. Every item above is untouched.
