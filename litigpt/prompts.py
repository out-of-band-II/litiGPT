"""
Shared prompt templates for training and inference.

All prompts are in Italian since the bot is deployed on an Italian subreddit.
"""

from collections.abc import Iterable

# What to call the person on the other side of a chat interface. Training data
# carries real Reddit usernames here; at inference there is no real name to
# use, and a generic Italian handle is closer to the training distribution than
# an English role label would be. Deliberately not a cohort member's name: that
# would invite the persona's feelings about that specific person into the
# reply, which is a confound in a test about style.
DEFAULT_HUMAN_HANDLE = "utente"


def render_thread(turns: Iterable[tuple[str, str]]) -> str:
    """
    Render (speaker, text) pairs as the Reddit thread format used in training.

    This is the single definition of that format. It used to be reimplemented
    in every interface, and the copies drifted: the Gradio app labelled the
    model's own turns "assistant" and the Ollama server labelled both sides
    "user"/"assistant" -- English role names that appear nowhere in the
    training data, which is built from real usernames. Feeding the model a
    format it never saw degrades output quality and raises no error, so the
    drift was invisible until someone read the two files side by side.

    Empty text is skipped rather than emitting a bare "speaker:" line.
    """
    return "\n".join(f"{speaker}: {text}" for speaker, text in turns if text)


# A post's body is cut here before it becomes a thread turn. The bot had cut
# at 500 characters on its own since it was written; the cut now lives where
# training can apply it too, so a long post costs the same share of the token
# budget in both.
POST_BODY_MAX_CHARS = 500


def render_post(title: str | None, selftext: str | None) -> str:
    """
    The text of a submission as one thread turn: its title, then its body.

    Callers clean the two strings first and pair the result with the post's
    author, so it renders as "author: title\\n\\nbody" like any other turn.

    The title is the only text a link or image post has -- 42% of this
    subreddit's posts have an empty or removed body -- so it is never dropped.
    There is no "Titolo:" label: a fixed label is a shape that real comments
    never have, and the blank line already separates the two.

    This is the single definition, shared by preprocessing and the Reddit bot.
    Training used to index no posts at all, while the bot rendered the post's
    body as the opening turn of every thread -- a turn the model had never
    seen.
    """
    title = (title or "").strip()
    body = (selftext or "").strip()[:POST_BODY_MAX_CHARS].strip()
    return "\n\n".join(part for part in (title, body) if part)


def count_words(text: str | None) -> int:
    """
    Whitespace-separated words: the one measure of reply length.

    Preprocessing drops training replies under data.min_comment_words, and
    inference resamples replies under inference.min_reply_words. Both count
    with this, so "5 words" means the same thing on either side.
    """
    return len((text or "").split())


def build_system_prompt(username: str) -> str:
    """
    Build the system prompt for a given user.

    Always includes the username, even in single-user mode,
    so training and inference prompts stay consistent.

    Args:
        username: Reddit username to impersonate.
    """
    return (
        f"Sei {username}, un utente di Reddit. "
        f"Rispondi nello stile e nel tono di scrittura di {username}."
    )


def build_alpaca_instruction(username: str) -> str:
    """Build the Alpaca-format instruction for a given user."""
    return (
        f"Rispondi alla seguente conversazione su Reddit "
        f"come farebbe {username}:"
    )
