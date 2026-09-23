"""
Who the public Reddit bot is allowed to post as.

With bot.available_users empty -- as it is in every shipped config -- the bot
used to answer as DEFAULT_USERNAME, "anonimo": a persona the adapter never
trained on, so each reply was a blend of all thirty users rather than any one
of them, posted without a word. The list is now required and opt-in, checked
against the adapter's manifest, and a selector with no preference falls back
to a random listed user.
"""

import pytest

from litigpt.deployment.reddit_bot import RedditBot
from litigpt.inference.classifier import KeywordUserSelector, check_bot_users
from litigpt.model_utils import DEFAULT_USERNAME

TRAINED = ["alice", "bob", "carol"]


class TestCheckBotUsers:
    def test_empty_list_refuses_and_names_the_options(self):
        with pytest.raises(ValueError, match="alice, bob, carol"):
            check_bot_users([], TRAINED)

    def test_user_the_adapter_never_saw_is_refused(self):
        with pytest.raises(ValueError, match="dave"):
            check_bot_users(["alice", "dave"], TRAINED)

    def test_subset_of_the_cohort_is_accepted(self):
        assert check_bot_users(["bob"], TRAINED) == ["bob"]

    def test_adapter_without_manifest_warns_but_allows(self, caplog):
        assert check_bot_users(["bob"], []) == ["bob"]
        assert "no training manifest" in caplog.text


def _bot(users, selector):
    """A RedditBot without logging in to Reddit or loading a model."""
    bot = object.__new__(RedditBot)
    bot.available_users = users
    bot.user_selector = selector
    return bot


class TestSelection:
    def test_no_keyword_match_falls_back_to_a_listed_user(self):
        bot = _bot(["alice", "bob"], KeywordUserSelector({"alice": ["calcio"]}))
        picks = {bot.select_user_for_context("parliamo di cucina") for _ in range(50)}
        assert picks <= {"alice", "bob"}
        assert DEFAULT_USERNAME not in picks

    def test_keyword_match_is_respected(self):
        bot = _bot(["alice", "bob"], KeywordUserSelector({"bob": ["calcio"]}))
        assert bot.select_user_for_context("il calcio di ieri") == "bob"

    def test_bot_refuses_to_start_without_users(self):
        # Before any Reddit login or model load.
        with pytest.raises(ValueError, match="available_users"):
            RedditBot(model_path="x", base_model="x", subreddit_name="test",
                      bot_username="x", available_users=[])
