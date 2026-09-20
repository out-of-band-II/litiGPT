# Blind Persona Evaluation

Chat with the model while it impersonates a persona chosen at random, then
guess who it was. This is the human counterpart to an automated persona
score, and it answers the question the loss curve cannot: *does the adapter
actually sound like thirty different people, or like one person with thirty
labels?*

```bash
# 1. Ceiling first. Real comments, no GPU, no adapter.
python launch_chat.py --interface blind --oracle --config config.top30.yaml

# 2. Then the model, with the classifier guessing alongside you.
python launch_chat.py --interface blind \
    --model models/litigpt_top30_lora \
    --base-model microsoft/phi-3-mini-4k-instruct \
    --config config.top30.yaml \
    --classifier
```

Opens on <http://127.0.0.1:7861>. Add `--share` for a public link, which is
how you reach it when the model is on a RunPod box.

## Run the ceiling first

This is the part worth insisting on. A score of "I got 40% at four options"
means nothing on its own, because you do not yet know what is achievable.
`--oracle` replaces the model's replies with the persona's **real held-out
comments** from `val.jsonl`. Same interface, same guessing task, same
scoreboard — only the text is genuine rather than generated.

That gives you the ceiling: how distinguishable these thirty people are from
their own writing, judged by you. Everything the model scores is then read as
a fraction of that. The scoreboard computes it for you once both conditions
have rounds at the same number of options:

> **A 4 opzioni** il modello arriva a 50% contro un ceiling di 80% sui
> commenti reali — trattiene circa il 45% del segnale identificabile.

Oracle mode needs no GPU and no adapter, so it can be run while training is
still going. The trade-off is that a replayed comment is not a reply, so the
conversation does not cohere — you are judging voice, not relevance. That is
the narrower question, but it is also the one the model is being tested on.

Oracle rounds are logged with `source: "oracle"` and a reserved model path, so
they can never be averaged into a model's numbers.

## Reference points

Measured on this cohort, for real held-out comments, by the TF-IDF attributor:

| Task | Machine top-1 | top-3 | Chance |
|---|---|---|---|
| 30-way (open mode) | 36.4% | 56.5% | 3.3% |
| 4-way (default round) | 68.5% | — | 25.0% |

A linear model on character and word n-grams gets 36.4% of thirty-way
attributions right — eleven times chance. So the personas *are* separable.
Whether the fine-tune preserved that separation is what you are measuring.

Humans and the classifier fail differently, and disagreement is the
interesting case rather than a bug. The classifier keys on spelling habits and
topic lock-in that a reader glides past; a reader picks up on argumentative
posture that no n-gram captures. Run with `--classifier` and the scoreboard
tracks both on identical text.

## Reading the scoreboard

Rounds are grouped by source and by number of options, never pooled — a
four-way round and a thirty-way round are different experiments. Each row
carries an exact binomial test, `P(X >= k)` under blind guessing:

| Verdict | Meaning |
|---|---|
| clearly above chance | p < 0.01 |
| above chance | p < 0.05 |
| suggestive, not conclusive | p < 0.20 |
| not distinguishable from guessing | p >= 0.20 |

Expect to need roughly 10–15 four-way rounds before anything reaches
significance, and considerably more in thirty-way mode. The confidence slider
is worth using honestly: it separates "I knew it" from "I got lucky", and it
is in the log for later.

## What stays hidden

The secret never reaches the browser. It lives in Gradio session state and in
the prompt, both server-side, and nothing renders it until the guess is
locked. Two narrower channels are handled explicitly:

- **Self-identification is redacted.** Asked *chi sei?*, the model will often
  answer with its own name — a giveaway with no stylistic content. The
  persona's own name is masked from what you see, in bare, `u/name` and
  `/u/name` forms. Whether the redaction fired is recorded and shown at the
  reveal, because a model that volunteers its name is itself a finding.
- **Other people's names are deliberately not redacted.** Who a user argues
  with is part of how they write. Blanking that would remove signal, not a
  leak.

Masking cannot catch every self-reference — a persona called `Generale_Zod`
that signs off as "Zod" slips through. It closes the common channel.

## The log is the output

Every completed round is appended to `data/eval/blind_eval.jsonl`: the secret,
your ranked guesses, confidence, the full transcript, the classifier's
ranking, and the round's parameters. That file is the actual artefact — a
human-labelled dataset of how identifiable each persona is, which no automated
metric produces.

It is **gitignored**, like everything else under `data/`, because it contains
generated impersonations of real, named people. Back it up yourself.

The scoreboard reloads prior rounds for the same `model_path` on startup, so
closing the tab does not reset your progress, and two different adapters keep
separate scores.

## Options

| Flag | Effect |
|---|---|
| `--oracle` | Real comments instead of generation. No GPU, no adapter. |
| `--classifier` | TF-IDF attributor guesses alongside you. First run trains and caches it (~2 min). |
| `--seed N` | Reproducible persona draw, for comparing two adapters on the same sequence. |
| `--share` | Public Gradio link — how you reach a run hosted on a cloud GPU. |
| `--no-4bit` | Full precision instead of 4-bit. |

In-app: **Modalita** switches between multiple choice (2–8 options) and open
thirty-way guessing; the settings accordion holds temperature, reply length,
your handle in the thread, and the name-masking toggle.

Your handle defaults to `utente` rather than a cohort member's name on
purpose. Using a real member's name would invite the persona's feelings about
that specific person into the reply, which is a confound in a test about
style.
