"""
Blind persona evaluation.

Chat with the model while it impersonates a persona chosen at random, whose
identity is withheld, then guess who it was. This is the human counterpart to
the automated persona-separation score: a linear attributor can be fooled by
surface n-grams a person would never notice, and it can equally miss a
resemblance a person finds obvious, so the two measure different things and
disagreeing is informative rather than a bug.

What the round protects against
-------------------------------
The secret never reaches the browser. It lives in Gradio session state and in
the prompt, both of which stay server-side; nothing renders it until the guess
is locked in. Two narrower leaks are handled explicitly:

* **Self-identification.** Asked "chi sei?" the model will often answer with
  its own name. That is a giveaway with no stylistic content, so the persona's
  own name is redacted from what is displayed. Whether the redaction fired is
  recorded, because a model that volunteers its name is itself a finding.
* **Other people's names are deliberately *not* redacted.** Naming other
  members of the cohort is real idiolect — who a user argues with is part of
  how they write — and blanking it would remove signal rather than a leak.

Every completed round is appended to a JSONL log, which is the actual output of
this tool: a human-labelled dataset of how identifiable each persona is.
"""

import json
import logging
import math
import random
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from litigpt.prompts import DEFAULT_HUMAN_HANDLE, render_thread

logger = logging.getLogger(__name__)

NAME_PLACEHOLDER = "[nome rimosso]"

# How many prior exchanges are replayed into the prompt. Training examples
# carried 3-5 parent comments, so staying near that keeps the chat closer to
# the distribution the adapter actually saw.
DEFAULT_CONTEXT_TURNS = 3

# DEFAULT_HUMAN_HANDLE is imported from litigpt.prompts, alongside the thread
# renderer, so the handle and the format it appears in cannot drift apart.


# ----------------------------------------------------------------------
# Round state
# ----------------------------------------------------------------------


@dataclass
class BlindRound:
    """
    One blind round. Held in `gr.State`, which is server-side session storage —
    `secret` is never serialised to the client.
    """

    round_id: str
    secret: str
    choices: list[str]
    mode: str
    started_at: str
    turns: list[dict[str, str]] = field(default_factory=list)
    guessed: bool = False
    self_id_attempts: int = 0

    @property
    def n_choices(self) -> int:
        return len(self.choices)

    @property
    def chance(self) -> float:
        return 1.0 / self.n_choices if self.n_choices else 0.0


@dataclass
class Scoreboard:
    """Completed rounds for this session, plus any replayed from the log."""

    records: list[dict] = field(default_factory=list)


# ----------------------------------------------------------------------
# Leak control
# ----------------------------------------------------------------------


def mask_self_reference(
    text: str,
    username: str,
    placeholder: str = NAME_PLACEHOLDER,
) -> tuple[str, bool]:
    """
    Redact a persona's own name from its output.

    Matches the bare handle, `u/handle` and `/u/handle`, case-insensitively.
    Returns the masked text and whether anything was replaced.

    This cannot catch every self-reference — a persona called `Generale_Zod`
    that signs off as "Zod" slips through, as does one that describes itself
    without naming itself. It closes the common channel, not every channel.
    """
    if not text or not username:
        return text, False

    pattern = re.compile(
        r"(?<![\w-])(?:/?u/)?" + re.escape(username) + r"(?![\w-])",
        re.IGNORECASE,
    )
    masked, count = pattern.subn(placeholder, text)
    return masked, count > 0


# ----------------------------------------------------------------------
# Responders — what the persona answers with
# ----------------------------------------------------------------------


class ModelResponder:
    """The real condition: the fine-tuned adapter generates the reply."""

    kind = "model"

    def __init__(self, bot):
        self.bot = bot

    def __call__(self, secret: str, context: str, temperature: float,
                 max_tokens: int) -> str:
        return self.bot.generate_response(
            context=context,
            username=secret,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
        )


class OracleResponder:
    """
    The control condition: replies are the persona's *real* held-out comments,
    drawn at random, not generated.

    This measures the ceiling — how identifiable these thirty people are from
    their own writing, judged by this human, through this interface. It is the
    number every model score should be read against, because "40% correct" is
    uninterpretable until you know whether the real comments score 45% or 95%.

    The trade-off is honest and worth stating: a replayed comment is not a
    reply, so the conversation does not cohere. The guesser is judging voice
    rather than relevance, which is the narrower question but also the one the
    model is actually being tested on.
    """

    kind = "oracle"

    def __init__(self, samples: dict[str, list[str]], rng: random.Random):
        self.samples = samples
        self.rng = rng
        self._used: dict[str, set] = {}

    def __call__(self, secret: str, context: str, temperature: float,
                 max_tokens: int) -> str:
        pool = self.samples.get(secret) or []
        if not pool:
            return "(nessun commento reale disponibile per questa persona)"

        # Avoid repeating a comment inside one round; reset when exhausted.
        used = self._used.setdefault(secret, set())
        available = [i for i in range(len(pool)) if i not in used]
        if not available:
            used.clear()
            available = list(range(len(pool)))

        index = self.rng.choice(available)
        used.add(index)
        return pool[index]


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------


def binomial_at_least(k: int, n: int, p: float) -> float:
    """
    P(X >= k) for X ~ Binomial(n, p): the chance of doing at least this well by
    guessing blindly. Computed exactly; n here is a handful of rounds, not a
    size where the normal approximation would be needed.
    """
    if n <= 0 or k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(
        math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
        for i in range(k, n + 1)
    )


def _significance_note(correct: int, total: int, chance: float) -> str:
    """Plain-language read on whether a score beats guessing."""
    if total == 0:
        return ""
    p_value = binomial_at_least(correct, total, chance)
    if p_value < 0.01:
        verdict = "clearly above chance"
    elif p_value < 0.05:
        verdict = "above chance"
    elif p_value < 0.20:
        verdict = "suggestive, not conclusive"
    else:
        verdict = "not distinguishable from guessing"
    return f"{verdict} (p={p_value:.3f})"


# ----------------------------------------------------------------------
# The evaluation app
# ----------------------------------------------------------------------


class BlindEvalInterface:
    def __init__(
        self,
        model_path: str,
        base_model: str,
        available_users: Sequence[str],
        log_path: str = "data/eval/blind_eval.jsonl",
        reference_jsonl: str | None = None,
        attributor=None,
        load_in_4bit: bool = True,
        seed: int | None = None,
        source: str = "model",
        responder=None,
    ):
        if len(available_users) < 2:
            raise ValueError(
                "Blind evaluation needs at least 2 personas; got "
                f"{len(available_users)}. Run the extract step first so that "
                "users_metadata.json exists."
            )

        self.model_path = model_path
        self.base_model = base_model
        self.available_users = sorted(available_users)
        self.log_path = Path(log_path)
        self.attributor = attributor
        self.rng = random.Random(seed)
        # Kept separate so that drawing illustrative quotes for the reveal does
        # not perturb the persona draw — with --seed set, the sequence of
        # personas stays reproducible no matter what gets rendered.
        self.display_rng = random.Random(seed if seed is None else seed + 1)

        # Real comments. Used after the reveal so the guesser can calibrate
        # against what the persona actually sounds like, and as the source of
        # replies in oracle mode.
        self.reference_samples: dict[str, list[str]] = {}
        if reference_jsonl and Path(reference_jsonl).exists():
            self.reference_samples = self._load_reference_samples(reference_jsonl)

        if responder is not None:
            self.responder = responder
        elif source == "oracle":
            if not self.reference_samples:
                raise ValueError(
                    "Oracle mode needs real comments, but none were loaded from "
                    f"{reference_jsonl!r}. Run the preprocess step first."
                )
            logger.info(
                "Oracle mode: replying with real comments from %d personas "
                "(no model loaded)", len(self.reference_samples),
            )
            self.responder = OracleResponder(self.reference_samples, self.rng)
        else:
            from litigpt.inference.generator import RedditBotInference

            logger.info("Loading model for blind evaluation...")
            self.responder = ModelResponder(RedditBotInference(
                model_path=model_path,
                base_model=base_model,
                use_lora=True,
                load_in_4bit=load_in_4bit,
            ))
            logger.info("Model ready")

        self.source = getattr(self.responder, "kind", source)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_reference_samples(self, path: str, per_user: int = 40) -> dict[str, list[str]]:
        from litigpt.eval.attribution import load_labelled_responses

        samples: dict[str, list[str]] = {}
        try:
            for username, response in load_labelled_responses(
                path, restrict_to=self.available_users
            ):
                bucket = samples.setdefault(username, [])
                if len(bucket) < per_user:
                    bucket.append(response)
        except OSError as e:
            logger.warning("Could not load reference samples from %s: %s", path, e)
        return samples

    def load_prior_rounds(self) -> list[dict]:
        """
        Replay earlier rounds for this same adapter from the log, so the
        scoreboard reflects everything judged against this model rather than
        resetting every time the app restarts.
        """
        if not self.log_path.exists():
            return []

        records = []
        try:
            with open(self.log_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("model_path") == self.model_path:
                        records.append(record)
        except OSError as e:
            logger.warning("Could not read prior rounds from %s: %s", self.log_path, e)

        if records:
            logger.info("Replayed %d prior rounds for this model", len(records))
        return records

    # ------------------------------------------------------------------
    # Round mechanics
    # ------------------------------------------------------------------

    def start_round(self, mode: str, n_choices: int) -> BlindRound:
        secret = self.rng.choice(self.available_users)

        if mode == "open":
            choices = list(self.available_users)
        else:
            n_choices = max(2, min(int(n_choices), len(self.available_users)))
            decoys = [u for u in self.available_users if u != secret]
            choices = self.rng.sample(decoys, n_choices - 1) + [secret]
            self.rng.shuffle(choices)

        return BlindRound(
            round_id=uuid.uuid4().hex[:12],
            secret=secret,
            choices=choices,
            mode=mode,
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def build_context(self, round_state: BlindRound, message: str, human_handle: str,
                      context_turns: int = DEFAULT_CONTEXT_TURNS) -> str:
        """
        Render the exchange as a Reddit thread, matching the training format:
        each prior comment is prefixed with its author, and the model's own
        reply is the bare continuation.

        The persona's real name appears here, prefixing its own past turns,
        exactly as it did in training. This string is a prompt, not output — it
        is never rendered to the browser before the reveal.
        """
        turns: list[tuple[str, str]] = []
        for turn in round_state.turns[-context_turns:]:
            turns.append((human_handle, turn["user"]))
            turns.append((round_state.secret, turn.get("bot_raw", "")))
        turns.append((human_handle, message))
        return render_thread(turns)

    def generate_reply(
        self,
        round_state: BlindRound,
        message: str,
        human_handle: str,
        temperature: float,
        max_tokens: int,
        mask_name: bool,
    ) -> tuple[str, str]:
        """Generate one reply. Returns (raw, displayed)."""
        context = self.build_context(round_state, message, human_handle)

        raw = self.responder(
            secret=round_state.secret,
            context=context,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        shown = raw
        if mask_name:
            shown, fired = mask_self_reference(raw, round_state.secret)
            if fired:
                round_state.self_id_attempts += 1

        return raw, shown

    # ------------------------------------------------------------------
    # Scoring and logging
    # ------------------------------------------------------------------

    def score_round(
        self,
        round_state: BlindRound,
        guesses: Sequence[str],
        confidence: int,
        temperature: float,
        rater: str = "",
    ) -> dict:
        """Build the log record for a finished round."""
        guesses = [g for g in guesses if g]
        primary = guesses[0] if guesses else None

        machine = None
        if self.attributor is not None:
            # The classifier judges only what the model actually produced, and
            # is given the same candidate set the human had.
            generated = "\n\n".join(
                t["bot_raw"] for t in round_state.turns if t.get("bot_raw")
            )
            result = self.attributor.predict(
                generated,
                restrict_to=round_state.choices if round_state.mode != "open" else None,
            )
            if result.ranking:
                machine = {
                    "top1": result.top1,
                    "confidence": round(result.top1_confidence, 4),
                    "top3": result.top_k(3),
                    "correct": result.top1 == round_state.secret,
                    "rank_of_truth": result.rank_of(round_state.secret),
                }

        return {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "round_id": round_state.round_id,
            "model_path": self.model_path,
            "base_model": self.base_model,
            # "model" rounds and "oracle" rounds answer different questions and
            # must never be averaged together; the scoreboard keeps them apart.
            "source": self.source,
            # Who was guessing. With several people sharing one instance their
            # rounds land in the same log, and pooling strangers' scores would
            # be meaningless — one careful rater and one clicking at random
            # average to something that describes neither.
            "rater": rater or "anonimo",
            "mode": round_state.mode,
            "n_choices": round_state.n_choices,
            "chance": round(round_state.chance, 4),
            "secret": round_state.secret,
            "choices": round_state.choices,
            "guesses": list(guesses),
            "guess": primary,
            "correct": primary == round_state.secret,
            "correct_top3": round_state.secret in guesses[:3],
            "confidence": int(confidence),
            "n_turns": len(round_state.turns),
            "self_id_attempts": round_state.self_id_attempts,
            "temperature": float(temperature),
            "transcript": [
                {"user": t["user"], "bot": t["bot_raw"]} for t in round_state.turns
            ],
            "machine": machine,
        }

    def append_log(self, record: dict) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error("Could not append to %s: %s", self.log_path, e)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_reveal(self, record: dict) -> str:
        secret = record["secret"]
        verdict = "**Indovinato.**" if record["correct"] else "**Sbagliato.**"
        if not record["correct"] and record["correct_top3"]:
            verdict = "**Sbagliato**, ma era nei tuoi primi tre."

        lines = [
            f"### Era **{secret}**",
            "",
            verdict,
            "",
            (
                f"La tua risposta: {record['guess'] or '-'}"
                f" | {record['n_choices']} opzioni (caso: {record['chance']:.1%})"
                f" | {record['n_turns']} scambi"
                f" | sicurezza {record['confidence']}/5"
            ),
        ]

        if record["self_id_attempts"]:
            lines += [
                "",
                (
                    f"Il modello ha provato a dire il proprio nome "
                    f"{record['self_id_attempts']}x — nascosto durante la chat."
                ),
            ]

        machine = record.get("machine")
        if machine:
            mark = "corretto" if machine["correct"] else "sbagliato"
            lines += [
                "",
                "---",
                "",
                (
                    f"**Il classificatore** ha detto **{machine['top1']}** "
                    f"({machine['confidence']:.0%}) — {mark}."
                ),
            ]
            if machine.get("rank_of_truth"):
                lines.append(
                    f"Ha messo {secret} al posto #{machine['rank_of_truth']}."
                )

        # In oracle mode the whole conversation was already real comments, so
        # showing more of them adds nothing.
        samples = None if self.source == "oracle" else self.reference_samples.get(secret)
        if samples:
            picked = self.display_rng.sample(samples, min(2, len(samples)))
            lines += ["", "---", "", f"**{secret}** davvero (dal set di validazione):", ""]
            for sample in picked:
                snippet = sample if len(sample) <= 320 else sample[:317] + "..."
                lines.append(f"> {snippet}")
                lines.append("")

        return "\n".join(lines)

    def render_scoreboard(self, board: Scoreboard) -> str:
        if not board.records:
            return (
                "*Nessun round completato.* Il punteggio compare qui, "
                "raggruppato per numero di opzioni."
            )

        groups: dict[tuple[str, int], list[dict]] = {}
        for record in board.records:
            key = (record.get("source", "model"), record["n_choices"])
            groups.setdefault(key, []).append(record)

        labels = {"model": "modello", "oracle": "reali (ceiling)"}

        lines = ["| Fonte | Opzioni | Round | Top-1 | Top-3 | Caso | Verdetto |",
                 "|---|---|---|---|---|---|---|"]

        accuracy: dict[tuple[str, int], float] = {}
        for key in sorted(groups, key=lambda k: (k[1], k[0])):
            source, n_choices = key
            records = groups[key]
            total = len(records)
            top1 = sum(1 for r in records if r["correct"])
            top3 = sum(1 for r in records if r["correct_top3"])
            chance = 1.0 / n_choices
            accuracy[key] = top1 / total
            lines.append(
                f"| {labels.get(source, source)} | {n_choices} | {total} | "
                f"{top1}/{total} ({top1/total:.0%}) | "
                f"{top3}/{total} ({top3/total:.0%}) | {chance:.1%} | "
                f"{_significance_note(top1, total, chance)} |"
            )

        # The comparison that actually matters: how much of the identifiable
        # signal in the real writing survives into the generated text.
        for (source, n_choices) in list(accuracy):
            if source != "model":
                continue
            ceiling = accuracy.get(("oracle", n_choices))
            if ceiling is None:
                continue
            got, chance = accuracy[(source, n_choices)], 1.0 / n_choices
            headroom = ceiling - chance
            retained = (got - chance) / headroom if headroom > 0 else 0.0
            lines += [
                "",
                (
                    f"**A {n_choices} opzioni** il modello arriva a {got:.0%} contro "
                    f"un ceiling di {ceiling:.0%} sui commenti reali — trattiene "
                    f"circa il {max(0.0, retained):.0%} del segnale identificabile."
                ),
            ]

        machine_records = [r for r in board.records if r.get("machine")]
        if machine_records:
            machine_correct = sum(1 for r in machine_records if r["machine"]["correct"])
            human_correct = sum(1 for r in machine_records if r["correct"])
            lines += [
                "",
                (
                    f"**Tu vs classificatore** sugli stessi {len(machine_records)} round: "
                    f"tu {human_correct}, lui {machine_correct}."
                ),
            ]

        # Per-persona breakdown, only once it would say something. The two
        # lists are split down the middle rather than taken as head and tail,
        # so they can never name the same persona as both best and worst.
        per_user: dict[str, list[bool]] = {}
        for record in board.records:
            per_user.setdefault(record["secret"], []).append(record["correct"])

        spotted = sorted(
            ((u, sum(v), len(v)) for u, v in per_user.items() if len(v) >= 2),
            key=lambda item: (item[1] / item[2], item[2]),
            reverse=True,
        )
        if len(spotted) >= 4:
            half = min(3, len(spotted) // 2)
            best = ", ".join(f"{u} {c}/{n}" for u, c, n in spotted[:half])
            worst = ", ".join(f"{u} {c}/{n}" for u, c, n in spotted[-half:])
            lines += [
                "",
                f"**Piu riconoscibili:** {best}",
                "",
                f"**Meno riconoscibili:** {worst}",
            ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Gradio wiring
    # ------------------------------------------------------------------

    def create_interface(self):
        import gradio as gr
        import torch

        max_choices = min(8, len(self.available_users))

        # Generating on CPU runs at a couple of tokens a second, so a 256-token
        # default would mean minutes per reply. Short replies are also closer to
        # what the training data looks like, so this costs little.
        on_cpu = self.source == "model" and not torch.cuda.is_available()
        default_max_tokens = 96 if on_cpu else 256

        with gr.Blocks(title="litiGPT — valutazione alla cieca") as demo:
            round_state = gr.State(value=None)
            board_state = gr.State(value=Scoreboard(records=self.load_prior_rounds()))

            if self.source == "oracle":
                banner = (
                    "# Valutazione alla cieca — *condizione di controllo*\n"
                    "Le risposte sono **commenti reali** della persona, presi dal "
                    "set di validazione: non sono generate e non rispondono a "
                    "quello che scrivi. Serve a misurare il ceiling — quanto sono "
                    "distinguibili queste persone dalla loro scrittura vera."
                )
            else:
                banner = (
                    "# Valutazione alla cieca\n"
                    "Il modello impersona qualcuno del cohort, ma non ti dice chi. "
                    "Chatta quanto vuoi, poi indovina."
                )
                if on_cpu:
                    banner += (
                        "\n\n*Nessuna GPU rilevata: la generazione gira su CPU, "
                        "circa 1-3 token al secondo. Ogni risposta richiede "
                        "qualche decina di secondi.*"
                    )
            gr.Markdown(banner)

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=460, label="Conversazione")
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Scrivi qualcosa...",
                            show_label=False,
                            scale=4,
                            autofocus=True,
                        )
                        send = gr.Button("Invia", variant="primary", scale=1)
                    status = gr.Markdown("*Premi «Nuovo round» per iniziare.*")

                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### Round")
                        mode = gr.Radio(
                            choices=[
                                ("Scelta multipla", "multiple_choice"),
                                (f"Aperta ({len(self.available_users)} utenti)", "open"),
                            ],
                            value="multiple_choice",
                            label="Modalita",
                        )
                        n_choices = gr.Slider(
                            minimum=2, maximum=max_choices, value=min(4, max_choices),
                            step=1, label="Numero di opzioni",
                            info="Solo per la scelta multipla",
                        )
                        new_round = gr.Button("Nuovo round", variant="primary")

                    with gr.Group():
                        gr.Markdown("### Chi era?")
                        guess1 = gr.Dropdown(choices=[], label="Prima scelta",
                                             interactive=True, filterable=True)
                        with gr.Accordion("Seconda e terza scelta (facoltative)",
                                          open=False):
                            guess2 = gr.Dropdown(choices=[], label="Seconda",
                                                 interactive=True, filterable=True)
                            guess3 = gr.Dropdown(choices=[], label="Terza",
                                                 interactive=True, filterable=True)
                        confidence = gr.Slider(
                            minimum=1, maximum=5, value=3, step=1,
                            label="Quanto sei sicuro?",
                            info="1 = tiro a caso, 5 = certo",
                        )
                        submit_guess = gr.Button("Conferma e rivela", interactive=False)

                    reveal = gr.Markdown("")

                    with gr.Accordion("Impostazioni", open=False):
                        temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05,
                                                label="Temperature")
                        max_tokens = gr.Slider(
                            50, 512, value=default_max_tokens, step=25,
                            label="Token massimi",
                            info=("Su CPU ogni token costa: tieni basso"
                                  if on_cpu else "Lunghezza massima risposta"),
                        )
                        human_handle = gr.Textbox(
                            value=DEFAULT_HUMAN_HANDLE, label="Il tuo handle",
                            info="Come appari nel thread passato al modello",
                        )
                        mask_name = gr.Checkbox(
                            value=True, label="Nascondi il nome della persona",
                            info="Impedisce al modello di identificarsi",
                        )
                        rater = gr.Textbox(
                            value="", label="Chi sta giocando",
                            info=("Registrato con ogni round. Compilato in "
                                  "automatico se hai fatto il login."),
                        )

            with gr.Accordion("Punteggio", open=True):
                scoreboard = gr.Markdown(
                    self.render_scoreboard(board_state.value)
                )

            # -- handlers ------------------------------------------------

            def on_new_round(mode_value, n_choices_value, board):
                state = self.start_round(mode_value, n_choices_value)
                options = sorted(state.choices)
                blank = gr.update(choices=options, value=None)
                n_done = len(board.records) if board else 0
                return (
                    [],
                    state,
                    blank, blank, blank,
                    "",
                    gr.update(interactive=True),
                    3,
                    (
                        f"**Round {n_done + 1}** — {state.n_choices} opzioni. "
                        f"Chatta, poi indovina."
                    ),
                )

            def on_send(message, history, state, handle, temp, max_tok, mask):
                history = history or []
                if state is None:
                    return message, history, state, "*Prima avvia un round.*"
                if state.guessed:
                    return (message, history, state,
                            "*Round chiuso. Avviane uno nuovo.*")
                if not message or not message.strip():
                    return "", history, state, gr.update()

                message = message.strip()
                raw, shown = self.generate_reply(
                    state, message, handle or DEFAULT_HUMAN_HANDLE,
                    temp, max_tok, mask,
                )
                state.turns.append({"user": message, "bot_raw": raw, "bot_shown": shown})

                history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": shown},
                ]
                return "", history, state, f"*{len(state.turns)} scambi.*"

            def on_guess(state, board, g1, g2, g3, conf, temp, who,
                         request: gr.Request):
                if state is None:
                    return ("*Nessun round attivo.*", gr.update(), board, state,
                            gr.update())
                if state.guessed:
                    return ("*Gia rivelato.*", gr.update(), board, state,
                            gr.update(interactive=False))
                if not g1:
                    return ("*Scegli almeno la prima opzione.*", gr.update(),
                            board, state, gr.update())

                # Prefer the login name when the app is served with auth, so
                # shared instances attribute rounds without anyone typing it.
                who = (who or "").strip() or getattr(request, "username", "") or ""

                state.guessed = True
                record = self.score_round(state, [g1, g2, g3], conf, temp, who)
                self.append_log(record)
                board.records.append(record)

                return (
                    self.render_reveal(record),
                    self.render_scoreboard(board),
                    board,
                    state,
                    gr.update(interactive=False),
                )

            new_round.click(
                on_new_round,
                [mode, n_choices, board_state],
                [chatbot, round_state, guess1, guess2, guess3, reveal,
                 submit_guess, confidence, status],
            )

            send_inputs = [msg, chatbot, round_state, human_handle,
                           temperature, max_tokens, mask_name]
            send_outputs = [msg, chatbot, round_state, status]
            send.click(on_send, send_inputs, send_outputs)
            msg.submit(on_send, send_inputs, send_outputs)

            submit_guess.click(
                on_guess,
                [round_state, board_state, guess1, guess2, guess3,
                 confidence, temperature, rater],
                [reveal, scoreboard, board_state, round_state, submit_guess],
            )

        return demo

    def launch(self, share: bool = False, server_port: int = 7861,
               server_name: str = "127.0.0.1", auth=None):
        demo = self.create_interface()

        if auth is None and server_name not in ("127.0.0.1", "localhost"):
            logger.warning(
                "Serving on %s with no authentication. Anyone who can reach "
                "this port can chat with a model impersonating real, named "
                "people. Pass --auth user:pass unless the port is reachable "
                "only through an SSH tunnel.", server_name,
            )

        demo.launch(
            share=share,
            server_port=server_port,
            server_name=server_name,
            auth=auth,
            show_error=True,
        )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main():
    import argparse

    from litigpt.config import Config
    from litigpt.model_utils import DEFAULT_BASE_MODEL, resolve_available_users

    parser = argparse.ArgumentParser(
        description="Blind persona evaluation for a multi-user litiGPT adapter"
    )
    parser.add_argument("--model", help="Path to the LoRA adapter")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--oracle", action="store_true",
        help="Control condition: reply with the persona's real held-out "
             "comments instead of generating. Needs no GPU and no adapter. "
             "Run this first to establish the ceiling your model scores "
             "should be read against.",
    )
    parser.add_argument("--log", default="data/eval/blind_eval.jsonl",
                        help="Where completed rounds are appended")
    parser.add_argument("--classifier", action="store_true",
                        help="Also have the TF-IDF attributor guess each round")
    parser.add_argument("--classifier-cache", default="models/author_attributor.joblib")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Load in full precision instead of 4-bit")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the persona draw (for reproducible sessions)")
    parser.add_argument("--share", action="store_true", help="Public Gradio link")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address. 0.0.0.0 to serve beyond localhost, "
                             "which a RunPod HTTP port needs")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--auth", default=None,
        help="Require a login: 'user:pass', or several separated by commas. "
             "The login name is recorded with each round. Use this whenever "
             "the app is reachable by anyone but you — a RunPod HTTP port and "
             "a --share link are both public to whoever has the URL.",
    )

    args = parser.parse_args()

    auth = None
    if args.auth:
        auth = []
        for pair in args.auth.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                parser.error(f"--auth entry {pair!r} is not 'user:pass'")
            user, _, password = pair.partition(":")
            if not user or not password:
                parser.error(f"--auth entry {pair!r} needs both a user and a password")
            auth.append((user, password))

    if not args.oracle and not args.model:
        parser.error("--model is required unless --oracle is given")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config.from_yaml(args.config)
    # Oracle rounds have no adapter, so they draw from the local extraction.
    users = resolve_available_users(args.model, config.data.processed_dir)
    if not users:
        raise SystemExit(
            f"No users found: {args.model or 'the oracle'} has no training "
            f"manifest and {config.data.processed_dir}/users_metadata.json is "
            "missing. Run: python -m litigpt.pipeline --step extract"
        )

    train_jsonl = str(Path(config.data.training_dir) / "train.jsonl")
    val_jsonl = str(Path(config.data.training_dir) / "val.jsonl")

    attributor = None
    if args.classifier:
        from litigpt.eval.attribution import AuthorAttributor

        if not Path(train_jsonl).exists():
            raise SystemExit(
                f"--classifier needs {train_jsonl}. Run the preprocess step first."
            )
        attributor = AuthorAttributor.load_or_train(
            cache_path=args.classifier_cache,
            training_jsonl=train_jsonl,
        )

    interface = BlindEvalInterface(
        # Oracle rounds are tagged with a reserved path so that replaying the
        # log never mixes ceiling rounds into a model's scoreboard.
        model_path=args.model or "(oracle)",
        base_model=args.base_model,
        available_users=users,
        log_path=args.log,
        reference_jsonl=val_jsonl,
        attributor=attributor,
        load_in_4bit=not args.no_4bit,
        seed=args.seed,
        source="oracle" if args.oracle else "model",
    )

    interface.launch(share=args.share, server_port=args.port,
                     server_name=args.host, auth=auth)


if __name__ == "__main__":
    main()
