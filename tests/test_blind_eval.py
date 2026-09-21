"""
The two pieces of the blind evaluation that have to be right.

Name masking is what keeps a round blind: asked "chi sei?", the model will
often answer with its own name, which is a giveaway carrying no stylistic
information. Leaking it does not crash anything -- it quietly turns a
measurement of style recognition into a reading test, and every round
collected that way is worthless.

The binomial test is what turns a score into a claim. If it is wrong, the
scoreboard reports significance that is not there.
"""

import math

import pytest

from litigpt.interface.blind_eval import (
    NAME_PLACEHOLDER,
    binomial_at_least,
    mask_self_reference,
)


class TestMaskSelfReference:
    def test_masks_a_bare_mention(self):
        masked, fired = mask_self_reference("sono Tom_Hadar", "Tom_Hadar")
        assert masked == f"sono {NAME_PLACEHOLDER}"
        assert fired is True

    @pytest.mark.parametrize("form", ["Tom_Hadar", "u/Tom_Hadar", "/u/Tom_Hadar"])
    def test_masks_every_reddit_mention_form(self, form):
        masked, fired = mask_self_reference(f"ciao {form} come va", "Tom_Hadar")
        assert "Tom_Hadar" not in masked
        assert fired is True

    def test_is_case_insensitive(self):
        # Someone typing their own name casually will not match the registered
        # casing, and a leak is a leak regardless.
        masked, fired = mask_self_reference("sono TOM_HADAR", "Tom_Hadar")
        assert "TOM_HADAR" not in masked
        assert fired is True

    def test_reports_when_it_did_not_fire(self):
        # The flag is recorded per round: a model that volunteers its name is
        # itself a finding, so "nothing was masked" has to be distinguishable.
        masked, fired = mask_self_reference("non te lo dico", "Tom_Hadar")
        assert masked == "non te lo dico"
        assert fired is False

    def test_does_not_mask_a_longer_name_containing_it(self):
        # Tom_Hadar must not be found inside Tom_Hadaresque.
        masked, fired = mask_self_reference("parlo di Tom_Hadaresque", "Tom_Hadar")
        assert masked == "parlo di Tom_Hadaresque"
        assert fired is False

    def test_leaves_other_peoples_names_alone(self):
        """
        Deliberate: who a persona argues with is part of how they write.
        Blanking that would remove signal, not close a leak.
        """
        masked, fired = mask_self_reference(
            "sono d'accordo con kurlash", "Tom_Hadar"
        )
        assert "kurlash" in masked
        assert fired is False

    def test_masks_repeated_mentions(self):
        masked, _ = mask_self_reference(
            "Tom_Hadar dice, e Tom_Hadar ripete", "Tom_Hadar"
        )
        assert "Tom_Hadar" not in masked
        assert masked.count(NAME_PLACEHOLDER) == 2


class TestBinomialAtLeast:
    """P(X >= k) for X ~ Binomial(n, p), which decides every verdict shown."""

    def test_zero_correct_is_certain(self):
        assert binomial_at_least(0, 10, 0.25) == pytest.approx(1.0)

    def test_all_correct_is_p_to_the_n(self):
        assert binomial_at_least(4, 4, 0.5) == pytest.approx(0.5 ** 4)

    def test_known_value(self):
        # P(X >= 1) with n=2, p=1/2 is 3/4.
        assert binomial_at_least(1, 2, 0.5) == pytest.approx(0.75)

    def test_chance_performance_is_not_significant(self):
        # 5 of 20 at four options is exactly chance; it must not look special.
        assert binomial_at_least(5, 20, 0.25) > 0.2

    def test_strong_performance_is_significant(self):
        # 15 of 20 at four options is far above chance.
        assert binomial_at_least(15, 20, 0.25) < 0.01

    def test_is_monotonic_in_k(self):
        # More correct answers can never be less surprising.
        probs = [binomial_at_least(k, 20, 0.25) for k in range(21)]
        assert probs == sorted(probs, reverse=True)

    def test_sums_to_one_over_the_partition(self):
        n, p = 8, 0.3
        total = sum(
            math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)
        )
        assert total == pytest.approx(1.0)
        assert binomial_at_least(0, n, p) == pytest.approx(total)
