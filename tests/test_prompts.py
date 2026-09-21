"""
The thread format is a contract between training and every interface.

Training builds its context strings in one place, from real Reddit usernames.
Every interface has to reproduce that format at inference time, and when one
of them does not, nothing raises -- the model is simply handed input unlike
anything it was trained on and answers worse. That failure is invisible to the
loss curve, to the logs, and to a quick look at the output, which is why it
survived in two interfaces at once until the files were read side by side.

These tests pin the format down.
"""

import pytest

from litigpt.prompts import (
    DEFAULT_HUMAN_HANDLE,
    build_system_prompt,
    render_thread,
)


class TestRenderThread:
    def test_basic_format(self):
        assert render_thread([("alice", "ciao")]) == "alice: ciao"

    def test_joins_with_newlines_preserving_order(self):
        assert render_thread(
            [("alice", "primo"), ("bob", "secondo")]
        ) == "alice: primo\nbob: secondo"

    def test_skips_empty_text_rather_than_emitting_a_bare_label(self):
        # A deleted or empty-bodied comment must not become "bob:" on its own,
        # which would teach the model that a name followed by nothing is a
        # legitimate turn.
        assert render_thread(
            [("alice", "primo"), ("bob", ""), ("carol", "terzo")]
        ) == "alice: primo\ncarol: terzo"

    def test_empty_input_is_empty_string(self):
        assert render_thread([]) == ""

    def test_accepts_a_generator(self):
        # preprocessing passes a generator expression over its context items.
        assert render_thread(
            (u, t) for u, t in [("alice", "x")]
        ) == "alice: x"

    def test_text_containing_a_colon_is_not_mangled(self):
        assert render_thread(
            [("alice", "guarda qui: davvero")]
        ) == "alice: guarda qui: davvero"

    def test_multiline_text_is_passed_through(self):
        # Reddit comments contain newlines. The renderer must not try to be
        # clever about them: the training data does not.
        assert render_thread(
            [("alice", "riga1\nriga2")]
        ) == "alice: riga1\nriga2"


class TestInterfacesAgree:
    """
    The three interfaces build their turn lists from different in-memory
    shapes. Given the same conversation they must produce the same string.
    """

    PERSONA = "Tom_Hadar"
    EXPECTED = "utente: ciao\nTom_Hadar: boh\nutente: e allora?"

    def test_gradio_shape(self):
        history = [
            {"role": "user", "content": "ciao"},
            {"role": "assistant", "content": "boh"},
        ]
        got = render_thread(
            [
                (
                    DEFAULT_HUMAN_HANDLE if t["role"] == "user" else self.PERSONA,
                    t["content"],
                )
                for t in history
            ]
            + [(DEFAULT_HUMAN_HANDLE, "e allora?")]
        )
        assert got == self.EXPECTED

    def test_ollama_shape(self):
        # Regression: this used to emit "user:" and "assistant:", English role
        # labels that appear nowhere in the training data.
        history = [
            {"role": "user", "content": "ciao"},
            {"role": "assistant", "content": "boh"},
        ]
        got = render_thread(
            [
                (
                    DEFAULT_HUMAN_HANDLE if m["role"] == "user" else self.PERSONA,
                    m["content"],
                )
                for m in history
            ]
            + [(DEFAULT_HUMAN_HANDLE, "e allora?")]
        )
        assert got == self.EXPECTED
        assert "assistant:" not in got
        assert "user:" not in got

    def test_blind_eval_shape(self):
        turns = [{"user": "ciao", "bot_raw": "boh"}]
        pairs = []
        for t in turns:
            pairs.append((DEFAULT_HUMAN_HANDLE, t["user"]))
            pairs.append((self.PERSONA, t.get("bot_raw", "")))
        pairs.append((DEFAULT_HUMAN_HANDLE, "e allora?"))
        assert render_thread(pairs) == self.EXPECTED

    def test_preprocessing_shape_is_the_ground_truth(self):
        # Training context comes from Reddit items keyed by author, and this is
        # the shape the other three are imitating.
        items = [
            {"author": DEFAULT_HUMAN_HANDLE, "body": "ciao"},
            {"author": self.PERSONA, "body": "boh"},
            {"author": DEFAULT_HUMAN_HANDLE, "body": "e allora?"},
        ]
        got = render_thread(
            (i.get("author", "unknown"), i.get("body") or i.get("selftext", ""))
            for i in items
        )
        assert got == self.EXPECTED


class TestPreprocessingContextUnchanged:
    """
    preprocessing._format_context was rewritten to call render_thread. Its
    output defines the training data, so it has to be byte-identical to the
    hand-rolled loop it replaced.
    """

    @staticmethod
    def _original(context_items):
        formatted = []
        for item in context_items:
            author = item.get("author", "unknown")
            body = item.get("body") or item.get("selftext", "")
            if body:
                formatted.append(f"{author}: {body}")
        return "\n".join(formatted)

    @pytest.mark.parametrize(
        "items",
        [
            [],
            [{"author": "alice", "body": "x"}],
            [{"author": "alice", "body": ""}],
            [{"author": "alice", "selftext": "il post"}],
            [{"body": "senza autore"}],
            [
                {"author": "alice", "body": "primo"},
                {"author": "bob", "body": ""},
                {"author": "carol", "selftext": "terzo"},
            ],
        ],
    )
    def test_matches_the_original_loop(self, items):
        from litigpt.data.preprocessing import RedditDataPreprocessor

        new = render_thread(
            (i.get("author", "unknown"), i.get("body") or i.get("selftext", ""))
            for i in items
        )
        assert new == self._original(items)


class TestNoSecondImplementation:
    """
    The tests above pin the format, but they build their turn lists the way
    each interface does rather than calling into it -- those methods need a
    loaded model. So they would still pass if an interface quietly went back
    to formatting the string itself.

    These check the source instead. The defect being guarded against is not a
    wrong value, it is a second implementation existing at all.
    """

    MODULES = [
        "litigpt/data/preprocessing.py",
        "litigpt/interface/gradio_app.py",
        "litigpt/interface/ollama.py",
        "litigpt/interface/blind_eval.py",
        # The bot was missing from this list until 2026-09-21, and it was the
        # one module actually posting to a subreddit. It rendered parents as
        # f"{author}: {body}" by hand and posts as f"Post: {title}\n{selftext}"
        # -- an English label and a title that appear nowhere in the training
        # data. Guarding the interfaces and not the deployment target is how
        # that survived the fix that removed the same bug from the interfaces.
        "litigpt/deployment/reddit_bot.py",
    ]

    @staticmethod
    def _source(relative_path):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        return (root / relative_path).read_text(encoding="utf-8")

    @pytest.mark.parametrize("module", MODULES)
    def test_uses_the_shared_renderer(self, module):
        assert "render_thread" in self._source(module), (
            f"{module} should build thread context with "
            "litigpt.prompts.render_thread, not by hand"
        )

    @pytest.mark.parametrize("module", MODULES)
    def test_no_hand_rolled_speaker_interpolation(self, module):
        """
        Catches the f-string forms the copies used: f"{speaker}: {content}",
        f"user: {msg}", f"utente: {message}" and so on.
        """
        source = self._source(module)
        offenders = [
            line.strip()
            for line in source.splitlines()
            # An f-string whose interpolation is immediately followed by ": {"
            # is the thread format being rebuilt inline.
            if '}: {' in line and line.lstrip().startswith(("f\"", "'", '"', "lines.append", "conversation.append", "formatted.append"))
        ]
        assert not offenders, (
            f"{module} appears to format thread lines by hand: {offenders}"
        )

    @pytest.mark.parametrize("module", MODULES)
    def test_no_english_role_labels(self, module):
        """
        Regression: ollama.py emitted "user:" and "assistant:" for two years
        of this repo's life. Those strings appear nowhere in the training
        data, which is built from real Reddit usernames.

        Checked across every module, not just the one that had the bug --
        catching it in one file is what let it survive in the other.

        Deliberately narrow: it matches the label at the *start* of an
        f-string, so unrelated formatting like f"data: {payload}" in the
        Flask streaming code is not swept up.
        """
        source = self._source(module)
        for bad in ('f"user: ', "f'user: ", 'f"assistant: ', "f'assistant: "):
            assert bad not in source, (
                f"{module} uses the English role label {bad!r}; the thread "
                "format is built from usernames"
            )

    def test_the_handle_is_defined_once(self):
        """
        DEFAULT_HUMAN_HANDLE lives in litigpt.prompts. An interface redefining
        it locally is how the handle and the format drift apart.
        """
        for module in self.MODULES:
            source = self._source(module)
            assert 'DEFAULT_HUMAN_HANDLE = "' not in source, (
                f"{module} redefines DEFAULT_HUMAN_HANDLE; import it instead"
            )


class TestSystemPrompt:
    def test_names_the_persona_twice(self):
        # Both mentions are load-bearing: the model is told who it is and then
        # told whose style to write in.
        assert build_system_prompt("alice").count("alice") == 2

    def test_is_italian(self):
        assert build_system_prompt("alice").startswith("Sei alice")
