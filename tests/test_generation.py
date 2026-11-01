"""
The short-reply floor, and the single generation path that carries it.

15% of the replies the top-30 adapters trained on are under five words, and a
short chat message draws them most. In chat they read as the model having
nothing to say, so replies under inference.min_reply_words are resampled.

Two ways for that to go wrong without raising anything. The selection rule
could quietly favour the longest candidate, pushing every persona toward its
most verbose register. Or an interface could keep its own model.generate
call, so the floor applies in one place and not in another -- the same drift
the thread format suffered when each interface had its own copy of it.
"""

import json
from pathlib import Path
from typing import ClassVar

import pytest
import torch

from litigpt.config import InferenceConfig
from litigpt.inference.generator import generate_reply, pick_reply
from litigpt.interface.blind_eval import BlindEvalInterface

ROOT = Path(__file__).resolve().parent.parent


class TestPickReply:
    def test_first_qualifying_candidate_wins_over_a_longer_one(self):
        # Rejection sampling, not "longest": the second candidate is longer
        # but the first already says enough.
        candidates = ["lol", "questa è una risposta vera", "questa è una risposta molto più lunga e prolissa"]
        assert pick_reply(candidates, 5) == "questa è una risposta vera"

    def test_falls_back_to_the_longest_when_none_qualifies(self):
        assert pick_reply(["no", "eh?", "boh non so"], 5) == "boh non so"

    def test_floor_of_zero_takes_the_first(self):
        assert pick_reply(["no", "una risposta lunga di sei parole"], 0) == "no"

    def test_strips_whitespace(self):
        assert pick_reply(["  uno due tre quattro cinque \n"], 5) == "uno due tre quattro cinque"

    def test_no_candidates(self):
        assert pick_reply([], 5) == ""


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, replies):
        self.replies = replies

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "PROMPT"

    def __call__(self, prompt, **kwargs):
        class _Batch(dict):
            def to(self, device):
                return self
        return _Batch(input_ids=torch.zeros((1, 4), dtype=torch.long))

    def decode(self, ids, skip_special_tokens):
        # The generated part of row i carries i in its first position.
        return self.replies[int(ids[0])]


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        n = kwargs["num_return_sequences"]
        prompt = kwargs["input_ids"]
        generated = torch.arange(n).unsqueeze(1)
        return torch.cat([prompt.repeat(n, 1), generated], dim=1)


SAMPLING = {"max_new_tokens": 16, "temperature": 0.8, "top_p": 0.9,
            "top_k": 50, "repetition_penalty": 1.1}


class TestGenerateReply:
    def test_samples_candidates_in_one_batch_and_skips_short_ones(self):
        model = _FakeModel()
        tok = _FakeTokenizer(["lol", "sono cinque parole qui adesso", "altro"])
        reply = generate_reply(model, tok, [], min_words=5, candidates=3, **SAMPLING)
        assert reply == "sono cinque parole qui adesso"
        assert len(model.calls) == 1
        assert model.calls[0]["num_return_sequences"] == 3

    def test_floor_off_means_one_plain_sample(self):
        model = _FakeModel()
        tok = _FakeTokenizer(["lol"])
        assert generate_reply(model, tok, [], min_words=0, candidates=3, **SAMPLING) == "lol"
        assert model.calls[0]["num_return_sequences"] == 1

    def test_single_candidate_is_returned_even_if_short(self):
        model = _FakeModel()
        tok = _FakeTokenizer(["lol"])
        assert generate_reply(model, tok, [], min_words=5, candidates=1, **SAMPLING) == "lol"


class TestOracleHeldToTheSameFloor:
    """The oracle replays real comments as the ceiling. If the model's short
    replies are resampled and the oracle's are not, the two conditions stop
    measuring the same thing."""

    def test_reference_samples_below_the_floor_are_dropped(self, tmp_path):
        val = tmp_path / "val.jsonl"
        rows = [
            # 4 words, but past the attribution loader's own 25-char minimum,
            # so only the word floor can be what drops it.
            ("alice", "assolutamente d'accordo con te!!!"),
            ("alice", "questa invece è una risposta abbastanza lunga"),
            ("bob", "anche bob scrive qualcosa di sensato qui"),
        ]
        val.write_text("\n".join(json.dumps({
            "username": u,
            "messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": "c"},
                         {"role": "assistant", "content": r}],
        }, ensure_ascii=False) for u, r in rows), encoding="utf-8")

        interface = BlindEvalInterface(
            model_path="x", base_model="x", available_users=["alice", "bob"],
            log_path=str(tmp_path / "log.jsonl"), reference_jsonl=str(val),
            responder=lambda **kw: "", inference=InferenceConfig(min_reply_words=5),
        )
        assert interface.reference_samples["alice"] == [
            "questa invece è una risposta abbastanza lunga"
        ]


class TestOneGenerationPath:
    """Source guard, like TestNoSecondImplementation in test_prompts: these
    modules need a loaded model to exercise, so check that none of them
    calls model.generate itself instead of going through generate_reply."""

    MODULES: ClassVar[list[str]] = [
        "litigpt/interface/gradio_app.py",
        "litigpt/interface/ollama.py",
        "litigpt/interface/blind_eval.py",
        "litigpt/deployment/reddit_bot.py",
        "litigpt/pipeline.py",
    ]

    @pytest.mark.parametrize("module", MODULES)
    def test_no_direct_model_generate(self, module):
        source = (ROOT / module).read_text(encoding="utf-8")
        assert "model.generate(" not in source, (
            f"{module} calls model.generate directly; use "
            "litigpt.inference.generator.generate_reply so the short-reply "
            "floor applies"
        )
