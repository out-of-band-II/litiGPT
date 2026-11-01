"""
A share link without a login.

--share puts the app at a public gradio.live URL, and the app impersonates
real, named people. Gradio starts happily without auth and says nothing, so
the only place this can be stopped is before launch -- and before the model
loads, so a refused start costs no GPU time.
"""

import sys

import pytest

from litigpt.interface.access import resolve_auth


class TestResolveAuth:
    def test_share_without_a_login_is_refused(self):
        with pytest.raises(ValueError, match="--auth"):
            resolve_auth(None, share=True, host="127.0.0.1")

    def test_pairs_split_on_the_first_colon(self):
        assert resolve_auth("ann:pa:ss, bob:x", share=True, host="127.0.0.1") == [
            ("ann", "pa:ss"), ("bob", "x"),
        ]

    @pytest.mark.parametrize("entry", ["ann", "ann:", ":pw"])
    def test_an_empty_user_or_password_is_refused(self, entry):
        # Not a crash otherwise: "ann" would become a login with no password.
        with pytest.raises(ValueError, match="user:pass"):
            resolve_auth(entry, share=False, host="127.0.0.1")

    def test_a_tunnel_needs_no_login(self):
        assert resolve_auth(None, share=False, host="127.0.0.1") is None


class TestEntryPoints:
    """Both apps must apply the check, before they build anything."""

    def test_chat_refuses_share_without_auth_before_loading(self, monkeypatch):
        pytest.importorskip("gradio")
        from litigpt.interface import gradio_app

        def must_not_load(**kwargs):
            raise AssertionError("model loaded before the share check")

        monkeypatch.setattr(gradio_app, "GradioChatInterface", must_not_load)
        monkeypatch.setattr(sys, "argv", ["gradio_app", "--model", "x", "--share"])
        with pytest.raises(SystemExit):
            gradio_app.main()

    def test_chat_passes_the_login_to_launch(self, monkeypatch):
        pytest.importorskip("gradio")
        from litigpt.interface import gradio_app

        seen = {}

        class FakeInterface:
            def __init__(self, **kwargs):
                pass

            def launch(self, **kwargs):
                seen.update(kwargs)

        monkeypatch.setattr(gradio_app, "GradioChatInterface", FakeInterface)
        monkeypatch.setattr(sys, "argv", ["gradio_app", "--model", "x",
                                          "--share", "--auth", "ann:pw"])
        gradio_app.main()
        assert seen["share"] is True
        assert seen["auth"] == [("ann", "pw")]

    def test_blind_eval_refuses_share_without_auth(self, monkeypatch):
        pytest.importorskip("gradio")
        from litigpt.interface import blind_eval

        def must_not_load(*args, **kwargs):
            raise AssertionError("interface built before the share check")

        monkeypatch.setattr(blind_eval, "BlindEvalInterface", must_not_load)
        monkeypatch.setattr(sys, "argv", ["blind_eval", "--oracle", "--share"])
        with pytest.raises(SystemExit):
            blind_eval.main()
