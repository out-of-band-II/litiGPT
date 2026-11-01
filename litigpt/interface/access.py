"""
Who can reach an interface.

Every interface serves a model impersonating real, named people. Bound to
localhost behind an SSH tunnel, only you can reach it. A --share link is
public by construction: gradio.live hands the URL out, and anyone holding it
gets in. A wider bind (0.0.0.0) may or may not be public -- on RunPod it is
whenever the port is an exposed HTTP port, or sits under one, since nginx
proxies 7861 to 7860 -- so that is a warning rather than a refusal.
"""

import logging

logger = logging.getLogger(__name__)

LOCAL_HOSTS = ("127.0.0.1", "localhost")

AUTH_HELP = (
    "Require a login: 'user:pass', or several separated by commas. Use this "
    "whenever the app is reachable by anyone but you -- a RunPod HTTP port "
    "and a --share link are both public to whoever has the URL."
)


def resolve_auth(value: str | None, *, share: bool,
                 host: str) -> list[tuple[str, str]] | None:
    """Parse --auth into Gradio's list of (user, password) pairs.

    Raises ValueError on a malformed entry, or on a share link with no login.
    """
    auth = None
    if value:
        auth = []
        for pair in value.split(","):
            pair = pair.strip()
            user, sep, password = pair.partition(":")
            if not sep or not user or not password:
                raise ValueError(f"--auth entry {pair!r} is not 'user:pass'")
            auth.append((user, password))

    if auth is None and share:
        raise ValueError(
            "--share publishes this app at a public URL, and it impersonates "
            "real, named people. Add --auth user:pass."
        )
    if auth is None and host not in LOCAL_HOSTS:
        logger.warning(
            "Serving on %s with no authentication. Anyone who can reach this "
            "port can chat with a model impersonating real, named people. "
            "Pass --auth user:pass unless the port is reachable only through "
            "an SSH tunnel.", host,
        )
    return auth
