"""
Shared prompt templates for training and inference.

All prompts are in Italian since the bot is deployed on an Italian subreddit.
"""


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
