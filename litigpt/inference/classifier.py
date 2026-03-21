"""
User Selection Strategies

Encapsulated user selection for the Reddit bot. Implementations decide
which user persona to respond as, given a conversation context.
"""

import logging
import random
from typing import Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class UserSelector(Protocol):
    """Protocol for user selection strategies."""

    def select_user(self, context: str, available_users: List[str]) -> Optional[str]:
        """Select a user to respond as, given conversation context.

        Args:
            context: The conversation context text.
            available_users: List of usernames the model can impersonate.

        Returns:
            Selected username, or None to fall back to default.
        """
        ...


class RandomUserSelector:
    """Selects a random user from available users."""

    def select_user(self, context: str, available_users: List[str]) -> Optional[str]:
        if not available_users:
            return None
        return random.choice(available_users)


class KeywordUserSelector:
    """
    Keyword-based user selection.
    Routes to the user whose keywords best match the conversation context.
    """

    def __init__(self, user_keywords: Dict[str, List[str]]):
        """
        Args:
            user_keywords: Dict mapping username to list of keywords they discuss.

        Example:
            {
                'tech_user': ['python', 'javascript', 'coding', 'programming'],
                'gaming_user': ['valorant', 'league', 'gaming', 'fps'],
            }
        """
        self.user_keywords = {
            username: [kw.lower() for kw in keywords]
            for username, keywords in user_keywords.items()
        }

    def select_user(self, context: str, available_users: List[str]) -> Optional[str]:
        """Select user based on keyword matching against available users."""
        context_lower = context.lower()

        scores = {}
        for username in available_users:
            keywords = self.user_keywords.get(username, [])
            scores[username] = sum(1 for kw in keywords if kw in context_lower)

        if not scores or max(scores.values()) == 0:
            return None

        return max(scores.items(), key=lambda x: x[1])[0]

    def get_matching_keywords(self, context: str, username: str) -> List[str]:
        """Utility: show which keywords matched for a user."""
        context_lower = context.lower()
        keywords = self.user_keywords.get(username, [])
        return [kw for kw in keywords if kw in context_lower]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: Random selection
    print("=" * 60)
    print("Random Selection")
    print("=" * 60)

    selector = RandomUserSelector()
    users = ["alice", "bob", "charlie"]
    for _ in range(5):
        print(f"  Selected: {selector.select_user('any context', users)}")

    # Example: Keyword selection
    print("\n" + "=" * 60)
    print("Keyword Selection")
    print("=" * 60)

    keyword_selector = KeywordUserSelector({
        'tech_enthusiast': ['python', 'javascript', 'coding', 'api', 'github'],
        'gamer': ['valorant', 'league', 'fps', 'gaming', 'rank'],
        'fitness_guru': ['gym', 'workout', 'protein', 'cardio', 'gains']
    })

    test_contexts = [
        "Just hit a new PR at the gym! 225 bench press!",
        "What's the best Python framework for web development?",
        "Anyone else stuck in silver rank in Valorant?"
    ]

    available = ['tech_enthusiast', 'gamer', 'fitness_guru']
    for ctx in test_contexts:
        user = keyword_selector.select_user(ctx, available)
        keywords = keyword_selector.get_matching_keywords(ctx, user) if user else []
        print(f"\nContext: {ctx}")
        print(f"Selected: {user}")
        print(f"Matched keywords: {keywords}")
