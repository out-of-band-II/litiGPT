"""
Which comments become training examples, and what context they get.

The post index was empty for the life of the project: the thread builder took
the comments file alone and filed every row with a 'parent_id' key as a
comment. Every reply to a post -- 10% of the eligible comments -- was dropped
for having no context, and no training thread ever opened with its post,
while the Reddit bot opened every thread with one. Loss curves looked normal;
it was found by checking what fraction of trained replies answered a post
(zero of 59,872).

The filters are pinned here for a quieter reason: each one is a config value,
and a value that stops reaching the code changes the dataset without a word.
"""

import logging
from pathlib import Path
from typing import ClassVar

import polars as pl
import pytest

from litigpt.data.extraction import RedditDataExtractor
from litigpt.data.preprocessing import RedditDataPreprocessor
from litigpt.prompts import POST_BODY_MAX_CHARS, render_post

ROOT = Path(__file__).resolve().parent.parent


def _threads(posts_have_parent_id_column=True):
    comments = pl.DataFrame({
        "id": ["c1", "c2"],
        "author": ["alice", "bob"],
        "parent_id": ["t3_p1", "t1_c1"],
        "body": ["risposta diretta al post, abbastanza lunga", "e questa risponde ad alice"],
    })
    posts = {
        "id": ["p1"],
        "author": ["op"],
        "title": ["Titolo del post"],
        "selftext": ["Corpo del post"],
    }
    if posts_have_parent_id_column:
        # The old submissions parquet carried these columns, empty. Under the
        # old rule that alone made every post look like a comment.
        posts["parent_id"] = [None]
        posts["body"] = [None]
    return RedditDataExtractor("unused").build_conversation_threads(
        comments, pl.DataFrame(posts)
    )


def _user_rows(rows):
    return pl.DataFrame(
        [{"type": "comment", "score": 1, "created_utc": 0, **r} for r in rows]
    )


class TestPostsAreContext:
    def test_posts_are_indexed_even_with_a_parent_id_column(self):
        threads = _threads(posts_have_parent_id_column=True)
        assert set(threads["posts_by_id"]) == {"p1"}
        assert "p1" not in threads["comments_by_id"]

    def test_a_reply_to_a_post_is_trained_on_with_the_post_as_context(self):
        pre = RedditDataPreprocessor()
        pairs = pre.create_training_pairs(
            _user_rows([{"parent_id": "t3_p1", "body": "la mia risposta al post di op"}]),
            _threads(), username="carol",
        )
        assert len(pairs) == 1
        assert pairs[0]["context"] == "op: Titolo del post\n\nCorpo del post"

    def test_a_nested_reply_sees_the_whole_thread_up_to_the_post(self):
        pre = RedditDataPreprocessor()
        pairs = pre.create_training_pairs(
            _user_rows([{"parent_id": "t1_c2", "body": "rispondo a bob con calma"}]),
            _threads(), username="carol",
        )
        assert pairs[0]["context"].splitlines() == [
            "op: Titolo del post",
            "",
            "Corpo del post",
            "alice: risposta diretta al post, abbastanza lunga",
            "bob: e questa risponde ad alice",
        ]


class TestRenderPost:
    def test_title_then_body(self):
        assert render_post("Titolo", "Corpo") == "Titolo\n\nCorpo"

    def test_a_link_post_is_its_title(self):
        # 42% of posts have an empty or removed body; the title is all they say.
        assert render_post("Solo titolo", "") == "Solo titolo"
        assert render_post("Solo titolo", None) == "Solo titolo"

    def test_body_is_cut_where_the_bot_always_cut_it(self):
        out = render_post("T", "x" * (POST_BODY_MAX_CHARS + 50))
        assert out == "T\n\n" + "x" * POST_BODY_MAX_CHARS


class TestOnePostRenderer:
    """Training and the bot must render a post the same way; both go through
    render_post, and both clean the text first."""

    MODULES: ClassVar[list[str]] = [
        "litigpt/data/preprocessing.py",
        "litigpt/deployment/reddit_bot.py",
    ]

    @pytest.mark.parametrize("module", MODULES)
    def test_uses_render_post(self, module):
        source = (ROOT / module).read_text(encoding="utf-8")
        assert "render_post(" in source
        assert "clean_text(" in source
        assert "selftext[:" not in source, f"{module} cuts the post body itself"


def _filter(rows, **kwargs):
    df = pl.DataFrame(rows)
    return RedditDataPreprocessor(**kwargs).filter_quality(df)["body"].to_list()


class TestFilters:
    def test_word_floor_replaces_the_character_floor(self):
        kept = _filter([
            {"body": "esatto", "score": 5},
            {"body": "esattamente", "score": 5},
            {"body": "non ce ne frega niente", "score": 5},  # 5 words
        ], min_words=5)
        assert kept == ["non ce ne frega niente"]

    def test_long_comments_dropped_over_the_cap(self):
        long_text = " ".join(["parola"] * 400)  # ~2800 chars
        kept = _filter([{"body": long_text, "score": 1},
                        {"body": "cinque parole giuste per passare", "score": 1}],
                       max_length=2000)
        assert kept == ["cinque parole giuste per passare"]

    def test_no_score_floor_keeps_downvoted_comments(self):
        rows = [{"body": "un commento impopolare ma molto sincero", "score": -4}]
        assert _filter(rows, min_score=None) == [rows[0]["body"]]
        assert _filter(rows, min_score=1) == []

    def test_moderator_comments_dropped(self):
        kept = _filter([
            {"body": "il tuo commento è stato rimosso per la regola 2", "score": 1,
             "distinguished": "moderator"},
            {"body": "questo invece l'ho scritto da utente normale", "score": 1,
             "distinguished": None},
        ])
        assert kept == ["questo invece l'ho scritto da utente normale"]

    def test_missing_distinguished_column_is_said_not_ignored(self, caplog):
        with caplog.at_level(logging.WARNING):
            kept = _filter([{"body": "cinque parole giuste per passare", "score": 1}])
        assert kept == ["cinque parole giuste per passare"]
        assert "distinguished" in caplog.text

    def test_deleted_bodies_never_survive(self):
        assert _filter([{"body": "[deleted]", "score": 1}], min_words=0) == []


class TestAuthorAliases:
    """One person is one persona. SpiegoReloaded is SpiegoLeDiscussioni's 2025
    account and Generale_Zod is Amlach's: left apart, each pair is two
    personas the model cannot separate, and the blind eval scores them as
    confusions."""

    @staticmethod
    def _dump(tmp_path):
        pl.DataFrame({
            "id": ["c1", "c2", "c3", "c4", "c5"],
            "author": ["main", "alt", "alt", "other", "other"],
            "parent_id": ["t3_p1"] * 5,
            "body": ["uno", "due", "tre", "quattro", "cinque"],
        }).write_parquet(tmp_path / "comments.parquet")
        pl.DataFrame({"id": ["p1"], "author": ["alt"], "title": ["t"], "selftext": [""]}
                     ).write_parquet(tmp_path / "posts.parquet")

    def test_a_merged_person_is_ranked_on_all_their_accounts(self, tmp_path):
        self._dump(tmp_path)
        ext = RedditDataExtractor(str(tmp_path), author_aliases={"alt": "main"})
        users = ext.extract_multiple_users(
            comments_file="comments.parquet", posts_file="posts.parquet",
            top_n=1, min_comments_per_user=1,
        )
        # Apart, "other" (2) would outrank "main" (1); together main has 3.
        assert list(users) == ["main"]
        data = users["main"]
        assert data.filter(pl.col("type") == "comment").height == 3
        assert data.filter(pl.col("type") == "post").height == 1

    def test_context_names_the_main_account(self, tmp_path):
        self._dump(tmp_path)
        ext = RedditDataExtractor(str(tmp_path), author_aliases={"alt": "main"})
        threads = ext.build_conversation_threads(ext.load_data("comments.parquet"),
                                                 ext.load_data("posts.parquet"))
        assert threads["comments_by_id"]["c2"]["author"] == "main"
        assert threads["posts_by_id"]["p1"]["author"] == "main"

    def test_a_misspelt_alias_raises_instead_of_merging_nothing(self, tmp_path):
        self._dump(tmp_path)
        ext = RedditDataExtractor(str(tmp_path), author_aliases={"Alt": "main"})
        with pytest.raises(ValueError, match="Alt"):
            ext.extract_multiple_users(comments_file="comments.parquet",
                                       posts_file="posts.parquet", top_n=1,
                                       min_comments_per_user=1)

    def test_chains_are_refused(self):
        with pytest.raises(ValueError, match="themselves aliases"):
            RedditDataExtractor("x", author_aliases={"a": "b", "b": "c"})


class TestSplit:
    def test_seeded_split_is_reproducible_and_leaves_input_alone(self):
        pre = RedditDataPreprocessor()
        data = [{"i": i} for i in range(100)]
        first = pre.split_data(data, 0.9)
        second = pre.split_data(data, 0.9)
        assert first == second
        assert data == [{"i": i} for i in range(100)]
        assert len(first[0]) == 90
