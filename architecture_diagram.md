# litiGPT architecture

How the pieces fit, and where the format contracts between them live.

---

## Training path

```
data/raw/*.parquet
  litigi_comments.parquet, litigi_submissions.parquet
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│ litigpt/data/extraction.py                                │
│                                                           │
│  pick the cohort      target_usernames, or top_n_users    │
│  index threads        comments_by_id, posts_by_id         │
│  walk parents         get_context_for_comment(max=3..5)   │
└───────────────────────────┬───────────────────────────────┘
                            │  context items: dicts with
                            │  author + body/selftext
                            ▼
┌───────────────────────────────────────────────────────────┐
│ litigpt/data/preprocessing.py                             │
│                                                           │
│  clean_text           unescape, strip markup, drop quotes │
│  _format_context  ──▶ render_thread()   ◀── the contract  │
│  cap per user         max_pairs_per_user                  │
│  split                prompt / completion, train / val    │
└───────────────────────────┬───────────────────────────────┘
                            │  data/training/{train,val}.jsonl
                            ▼
┌───────────────────────────────────────────────────────────┐
│ litigpt/training/trainer.py                               │
│                                                           │
│  4-bit base load      bitsandbytes NF4                    │
│  detect targets       from the model, not a name list     │
│  drop_unlearnable     prompt fills window -> no reply     │
│  SFTTrainer           completion-only loss                │
│  early stopping       patience + threshold on eval_loss   │
└──────────┬──────────────────────────────┬─────────────────┘
           │                              │
           ▼                              ▼
   model.output_dir/             litigpt/training/tracking.py
   adapter_model.safetensors      └─▶ MLflow (SQLite backend)
   litigpt_manifest.json
```

### The training manifest

`litigpt_manifest.json`, from [litigpt/manifest.py](litigpt/manifest.py), is
written into `model.output_dir` before the model loads. The trainer copies it
into every checkpoint, and it is completed after the final save. It records:

- the cohort, read from `train.jsonl`, with examples per user
- sha256 hashes of `train.jsonl` and `val.jsonl`
- the resolved config
- the git commit, or null on a pod with no `.git`
- library versions
- the modules the safetensors actually contain

It exists because the adapter used to carry no record of who it impersonates.
Because it sits inside the adapter directory, the RunPod archive picks it up
with no extra step.

For an adapter trained before the manifest existed, backfill it. The backfill
refuses if the local data disagrees with the run's MLflow record:
`python -m litigpt.manifest --model <dir> --config <cfg> --mlflow-db <db>`.

---

## The two contracts

Everything downstream of preprocessing has to agree with it on two things.
Both are defined once, in [litigpt/prompts.py](litigpt/prompts.py), and both
have been broken before by a module quietly reimplementing them.

```
build_system_prompt(username)
    "Sei {username}, un utente di Reddit. Rispondi nello stile e nel
     tono di scrittura di {username}."

    Always names the persona, in training and at inference, including
    when there is only one. DEFAULT_USERNAME = "anonimo" when there is
    none to name.

render_thread([(speaker, text), ...])
    "\n".join(f"{speaker}: {text}")

    Speakers are real Reddit usernames. A post is rendered exactly like
    a comment -- its author and selftext, no title, no "Post:" label.
    English role labels appear nowhere in the training data.
```

`tests/test_prompts.py::TestNoSecondImplementation` reads the source of every
module that renders threads and fails if one builds the string itself. It
covers preprocessing, the three interfaces, and the Reddit bot.

---

## Serving path

```
                    model.output_dir/  +  base model
                                 │
                                 ▼
                   litigpt/model_utils.py
                   load_model_and_tokenizer
                     4-bit, dtype detection, PEFT merge
                   resolve_available_users
                     the adapter's manifest; users_metadata.json
                     only for adapters that predate it
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
   interface/gradio_app   interface/ollama    interface/blind_eval
   chat UI, :7860         Flask + SSE, :5000  guess the persona, :7861
            │                    │                    │
            └────────────────────┴────────────────────┘
                                 │
                        inference/generator.py
                        RedditBotInference
                          build_system_prompt + render_thread
                          generate, strip, return
```

The blind evaluator additionally draws on:

```
litigpt/eval/attribution.py
  AuthorAttributor    word + char TF-IDF -> logistic regression
                      guesses alongside the human, and scores how far
                      apart the personas are at all
```

That is the only TF-IDF in the system, and it measures — it does not route.

---

## Reddit bot

```
subreddit comment stream (skip_existing, pause_after=-1)
        │
        ├── new comment ──▶ should_respond()
        │                     own comment? deleted? seen? score?
        │                     trigger keywords? probability? cooldown?
        │                           │ yes
        │                           ▼
        │                   get_comment_context()
        │                     walk parents, render_thread
        │                           │
        │                           ▼
        │                   select_user_for_context()
        │                     UserSelector protocol:
        │                       RandomUserSelector   pick from available
        │                       KeywordUserSelector  topic -> persona
        │                       (returns None -> random from available)
        │                           │
        │                           ▼
        │                   generate, append disclaimer, reply
        │
        └── stream idle ──▶ poll mentions if due (mention_poll_seconds)
```

Persona selection is deliberately not learned. A new strategy is one class
with `select_user(context, available_users) -> Optional[str]`; the natural
next one reads an explicit request out of a mention.

Entry point is `python -m litigpt.pipeline --step deploy`, which builds all of
the above from config.

---

## Where things live

| Concern | Module |
|---|---|
| Raw conversion (zstd/jsonl → parquet) | `data/preliminary.py` |
| Cohort selection, thread indexing | `data/extraction.py` |
| Cleaning, formatting, splitting | `data/preprocessing.py` |
| Prompt and thread format | `prompts.py` |
| Model loading, quantization | `model_utils.py` |
| QLoRA training | `training/trainer.py` |
| Experiment tracking | `training/tracking.py` |
| Generation | `inference/generator.py` |
| Persona selection | `inference/classifier.py` |
| Authorship scoring | `eval/attribution.py` |
| Chat, streaming, blind eval | `interface/` |
| Reddit bot | `deployment/reddit_bot.py` |
| Config schema | `config.py` |
| Orchestration and every CLI entry point | `pipeline.py` |
