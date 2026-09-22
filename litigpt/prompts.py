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
