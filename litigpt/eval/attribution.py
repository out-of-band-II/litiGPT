"""
Authorship attribution over the persona cohort.

A plain linear model — TF-IDF over word and character n-grams, then logistic
regression — that answers "which member of the cohort wrote this?".

It exists to serve two jobs:

  * the machine baseline in the blind chat evaluation, so that a human guess
    and a model guess are made against the exact same generated text;
  * the scorer for automated persona-separation runs, where the model answers
    one held-out context once per persona and we check whether the text it
    produced is attributable to the persona it was told to be.

An important caveat travels with every number this produces: the classifier is
trained on *real* comments but is usually asked to judge *generated* ones. That
is a distribution shift, and it cuts both ways — generated text can be easier
(the model may exaggerate a persona's tics) or harder (it may regress to a
bland average of the cohort). Read its accuracy as a relative signal between
personas, not as an absolute measure of how convincing an impersonation is.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Text shorter than this carries almost no idiolect; "lol" is nobody's
# signature. Both training and scoring drop anything below it.
MIN_ATTRIBUTION_CHARS = 25


@dataclass
class AttributionResult:
    """A ranked guess at who wrote a piece of text."""

    ranking: List[Tuple[str, float]]

    @property
    def top1(self) -> Optional[str]:
        return self.ranking[0][0] if self.ranking else None

    @property
    def top1_confidence(self) -> float:
        return self.ranking[0][1] if self.ranking else 0.0

    def top_k(self, k: int) -> List[str]:
        return [name for name, _ in self.ranking[:k]]

    def rank_of(self, username: str) -> Optional[int]:
        """1-based rank of `username`, or None if it is not in the ranking."""
        for i, (name, _) in enumerate(self.ranking, start=1):
            if name == username:
                return i
        return None


class AuthorAttributor:
    """TF-IDF + logistic regression authorship classifier."""

    def __init__(self, pipeline, classes: Sequence[str]):
        self.pipeline = pipeline
        self.classes = list(classes)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        labels: Sequence[str],
        max_word_features: int = 100_000,
        max_char_features: int = 200_000,
        seed: int = 0,
    ) -> "AuthorAttributor":
        """Fit the classifier on (text, author) pairs."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length")
        if not texts:
            raise ValueError("no training text supplied")

        # Word n-grams catch vocabulary and topic; character n-grams inside
        # word boundaries catch spelling and punctuation habits, which is where
        # most of the individual signal lives (perche vs perche accented, un po
        # with the wrong accent, spacing around punctuation).
        features = FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
                max_features=max_word_features,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                min_df=3,
                max_features=max_char_features,
            )),
        ])

        # class_weight balanced because the cohort is capped but not exactly
        # even, and we would rather not let the most prolific users absorb
        # every ambiguous case.
        clf = LogisticRegression(
            max_iter=2000,
            C=4.0,
            class_weight="balanced",
            random_state=seed,
        )

        pipeline = Pipeline([("features", features), ("clf", clf)])

        logger.info("Fitting attributor on %d samples, %d authors",
                    len(texts), len(set(labels)))
        pipeline.fit(list(texts), list(labels))
        logger.info("Attributor trained")

        return cls(pipeline, pipeline.named_steps["clf"].classes_)

    @classmethod
    def from_training_jsonl(
        cls,
        path: str,
        max_per_class: int = 1500,
        seed: int = 0,
        **train_kwargs,
    ) -> "AuthorAttributor":
        """
        Train from a ChatML training file, using assistant turns as samples and
        the `username` field as the label.

        `max_per_class` subsamples each author so fitting stays under a minute
        or so. The cap is applied by random choice rather than by taking the
        head, because the file is written in extraction order and the head of a
        user's block is not a fair sample of their writing.
        """
        by_author: Dict[str, List[str]] = {}

        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                username = record.get("username")
                response = _assistant_text(record)
                if not username or not response:
                    continue
                if len(response) < MIN_ATTRIBUTION_CHARS:
                    continue

                by_author.setdefault(username, []).append(response)

        if not by_author:
            raise ValueError(f"no usable training samples found in {path}")

        rng = random.Random(seed)
        texts: List[str] = []
        labels: List[str] = []
        for author, samples in sorted(by_author.items()):
            if 0 < max_per_class < len(samples):
                samples = rng.sample(samples, max_per_class)
            texts.extend(samples)
            labels.extend([author] * len(samples))

        logger.info("Loaded %d samples across %d authors from %s",
                    len(texts), len(by_author), path)

        return cls.train(texts, labels, seed=seed, **train_kwargs)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        text: str,
        restrict_to: Optional[Sequence[str]] = None,
    ) -> AttributionResult:
        """
        Rank the cohort by how likely each member is to have written `text`.

        `restrict_to` narrows the ranking to a subset — used by the blind
        evaluation so the classifier faces the same multiple-choice question
        the human was given, rather than an easier or harder one.
        """
        text = (text or "").strip()
        if len(text) < MIN_ATTRIBUTION_CHARS:
            return AttributionResult(ranking=[])

        probabilities = self.pipeline.predict_proba([text])[0]
        ranking = sorted(
            zip(self.classes, (float(p) for p in probabilities)),
            key=lambda pair: pair[1],
            reverse=True,
        )

        if restrict_to:
            allowed = set(restrict_to)
            subset = [(name, p) for name, p in ranking if name in allowed]
            total = sum(p for _, p in subset)
            if total > 0:
                subset = [(name, p / total) for name, p in subset]
            ranking = subset

        return AttributionResult(ranking=ranking)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import joblib

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "classes": self.classes}, target)
        logger.info("Saved attributor to %s", target)

    @classmethod
    def load(cls, path: str) -> "AuthorAttributor":
        import joblib

        payload = joblib.load(path)
        return cls(payload["pipeline"], payload["classes"])

    @classmethod
    def load_or_train(
        cls,
        cache_path: str,
        training_jsonl: str,
        **kwargs,
    ) -> "AuthorAttributor":
        """Load a cached attributor, or train one and cache it if absent."""
        cache = Path(cache_path)
        if cache.exists():
            try:
                logger.info("Loading cached attributor from %s", cache)
                return cls.load(str(cache))
            except Exception as e:  # pragma: no cover - corrupt cache
                logger.warning("Could not load cached attributor (%s); retraining", e)

        attributor = cls.from_training_jsonl(training_jsonl, **kwargs)
        try:
            attributor.save(str(cache))
        except Exception as e:  # pragma: no cover - read-only dir
            logger.warning("Could not cache attributor: %s", e)
        return attributor


def _assistant_text(record: dict) -> str:
    """Pull the assistant turn out of a ChatML record."""
    for message in record.get("messages", []):
        if message.get("role") == "assistant":
            return (message.get("content") or "").strip()
    return ""


def load_labelled_responses(
    path: str,
    restrict_to: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str]]:
    """Read (username, assistant_response) pairs from a ChatML jsonl file."""
    allowed = set(restrict_to) if restrict_to else None
    samples: List[Tuple[str, str]] = []

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            username = record.get("username")
            response = _assistant_text(record)
            if not username or len(response) < MIN_ATTRIBUTION_CHARS:
                continue
            if allowed is not None and username not in allowed:
                continue
            samples.append((username, response))

    return samples


def held_out_ceiling(
    attributor: "AuthorAttributor",
    path: str,
    limit: int = 1500,
    seed: int = 0,
    restrict_to: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Score the attributor on *real* held-out comments.

    This is the reference point every generated-text number should be read
    against: it is how well this cohort can be told apart at all, given these
    features and this much text per sample. A generated-text score near this
    ceiling means the impersonation carries about as much identity as the real
    writing does; a score near chance means it carries none.
    """
    samples = load_labelled_responses(path, restrict_to=restrict_to)

    n_classes = len(set(restrict_to)) if restrict_to else len(attributor.classes)
    chance = 1.0 / n_classes if n_classes else 0.0

    if not samples:
        return {"n": 0, "top1": 0.0, "top3": 0.0, "chance": chance}

    rng = random.Random(seed)
    if 0 < limit < len(samples):
        samples = rng.sample(samples, limit)

    top1 = top3 = 0
    for username, response in samples:
        result = attributor.predict(response, restrict_to=restrict_to)
        if result.top1 == username:
            top1 += 1
        if username in result.top_k(3):
            top3 += 1

    return {
        "n": len(samples),
        "top1": top1 / len(samples),
        "top3": top3 / len(samples),
        "chance": chance,
    }
