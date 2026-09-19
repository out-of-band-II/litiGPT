"""
Module 5: Reddit Bot Deployment
Deploy bot to monitor and respond to Reddit comments
"""

import praw
from praw.models import Comment
import time
import logging
import random
from collections import deque
from datetime import datetime
from typing import Optional, List
import os
from dotenv import load_dotenv

# Import inference module
from litigpt.inference.generator import RedditBotInference
from litigpt.inference.classifier import UserSelector, RandomUserSelector
from litigpt.model_utils import DEFAULT_USERNAME, DEFAULT_BASE_MODEL

logger = logging.getLogger(__name__)

class RedditBot:
    def __init__(self,
                 model_path: str,
                 base_model: str,
                 subreddit_name: str,
                 bot_username: str,
                 trigger_keywords: Optional[list] = None,
                 reply_probability: float = 0.3,
                 min_score_threshold: int = 1,
                 cooldown_seconds: int = 60,
                 available_users: Optional[List[str]] = None,
                 user_selector: Optional[UserSelector] = None,
                 max_depth: int = 3,
                 inference_config: Optional[dict] = None):
        """
        Initialize Reddit bot.

        The bot always generates responses as a specific user. When multiple
        users are configured it auto-selects via the user_selector; when a single
        user (or none) is configured it uses that user for every reply.

        Args:
            model_path: Path to trained model
            base_model: Base model name
            subreddit_name: Subreddit to monitor
            bot_username: Bot's Reddit username
            trigger_keywords: Keywords that trigger responses (None = all)
            reply_probability: Probability of replying to eligible comments
            min_score_threshold: Minimum comment score to respond to
            cooldown_seconds: Seconds between responses
            available_users: List of users model can impersonate
            user_selector: Strategy for selecting which user to respond as
            max_depth: Max parent comments for context
        """

        # Load environment variables
        load_dotenv()

        # Initialize Reddit API
        self.reddit = self._init_reddit_api()
        self.subreddit = self.reddit.subreddit(subreddit_name)
        self.bot_username = bot_username

        # Initialize model
        self.bot_inference = RedditBotInference(
            model_path=model_path,
            base_model=base_model,
            use_lora=True,
            load_in_4bit=True,
        )

        # User selection
        self.available_users = available_users or []
        self.user_selector = user_selector or RandomUserSelector()

        # Inference settings
        self.inference_config = inference_config or {}

        # Bot settings
        self.trigger_keywords = trigger_keywords or []
        self.reply_probability = reply_probability
        self.min_score_threshold = min_score_threshold
        self.cooldown_seconds = cooldown_seconds

        # Track processed comments (bounded to prevent unbounded memory growth)
        self._processed_ids = deque(maxlen=10000)
        self._processed_set: set = set()
        self.last_reply_time = 0
        self.max_depth = max_depth

        logger.info(f"Bot initialized for r/{subreddit_name}")
        logger.info(f"Available users: {', '.join(self.available_users) or '(default)'}")

    def _mark_processed(self, comment_id: str):
        """Mark a comment as processed, evicting oldest if at capacity."""
        if comment_id in self._processed_set:
            return
        if len(self._processed_ids) == self._processed_ids.maxlen:
            evicted = self._processed_ids[0]
            self._processed_set.discard(evicted)
        self._processed_ids.append(comment_id)
        self._processed_set.add(comment_id)

    def _is_processed(self, comment_id: str) -> bool:
        return comment_id in self._processed_set

    def _init_reddit_api(self) -> praw.Reddit:
        """Initialize PRAW Reddit API client"""

        required_vars = [
            "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
            "REDDIT_USER_AGENT", "REDDIT_USERNAME", "REDDIT_PASSWORD",
        ]
        missing = [v for v in required_vars if not os.getenv(v)]
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Add them to your .env file."
            )

        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD"),
        )

        logger.info(f"Logged in as: {reddit.user.me()}")
        return reddit

    def should_respond(self, comment) -> bool:
        """Determine if bot should respond to a comment"""

        # Skip own comments
        if comment.author and comment.author.name == self.bot_username:
            return False

        # Skip deleted/removed
        if comment.author is None or comment.body in ['[deleted]', '[removed]']:
            return False

        # Skip already processed
        if self._is_processed(comment.id):
            return False

        # Check score threshold
        if comment.score < self.min_score_threshold:
            return False

        # Check keywords if specified
        if self.trigger_keywords:
            comment_lower = comment.body.lower()
            if not any(keyword.lower() in comment_lower for keyword in self.trigger_keywords):
                return False

        # Probabilistic response
        if random.random() > self.reply_probability:
            return False

        # Cooldown check
        current_time = time.time()
        if current_time - self.last_reply_time < self.cooldown_seconds:
            return False

        return True

    def get_comment_context(self, comment) -> str:
        """Build conversation context from comment thread"""

        context_parts = []
        current = comment

        # Get parent comments
        for _ in range(self.max_depth):
            if current.is_root:
                # Get post title/body
                submission = current.submission
                context_parts.insert(0, f"Post: {submission.title}\n{submission.selftext[:500]}")
                break

            try:
                parent = current.parent()
                if isinstance(parent, Comment):
                    author = parent.author.name if parent.author else "[deleted]"
                    context_parts.insert(0, f"{author}: {parent.body}")
                    current = parent
                else:
                    # Parent is submission
                    context_parts.insert(0, f"Post: {parent.title}\n{parent.selftext[:500]}")
                    break
            except Exception as e:
                logger.warning(f"Error getting parent: {e}")
                break

        # Add the comment we're responding to
        author = comment.author.name if comment.author else "[deleted]"
        context_parts.append(f"{author}: {comment.body}")

        return "\n".join(context_parts)

    def select_user_for_context(self, context: str) -> str:
        """
        Select which user to respond as based on context.

        Returns:
            Username to impersonate. Falls back to DEFAULT_USERNAME if
            no users are configured or the selector returns None.
        """
        if self.available_users:
            selected = self.user_selector.select_user(context, self.available_users)
            if selected:
                logger.info(f"Selected user: {selected}")
                return selected

        return DEFAULT_USERNAME

    def generate_and_post_reply(self, comment):
        """Generate response and post it"""

        try:
            # Get context
            context = self.get_comment_context(comment)
            logger.info(f"\nContext:\n{context}\n")

            # Select user
            username = self.select_user_for_context(context)
            logger.info(f"Responding as: {username}")

            # Generate response
            response = self.bot_inference.generate_response(
                context,
                username=username,
                max_new_tokens=self.inference_config.get("max_new_tokens", 256),
                temperature=self.inference_config.get("temperature", 0.8),
                top_p=self.inference_config.get("top_p", 0.9),
            )

            logger.info(f"Generated response: {response}")

            # Add disclaimer
            disclaimer = f"\n\n---\n^(I'm a bot mimicking {username}'s style. Beep boop! [bot])"
            full_response = response + disclaimer

            # Post reply
            comment.reply(full_response)

            # Update tracking
            self._mark_processed(comment.id)
            self.last_reply_time = time.time()

            logger.info(f"Posted reply to comment {comment.id}")

        except Exception as e:
            logger.error(f"Error posting reply: {e}")

    def monitor_comments(self):
        """Monitor subreddit for new comments"""

        logger.info("Starting comment monitoring...")

        try:
            for comment in self.subreddit.stream.comments(skip_existing=True):
                try:
                    if self.should_respond(comment):
                        logger.info(f"\nProcessing comment {comment.id} by {comment.author}")
                        self.generate_and_post_reply(comment)
                    else:
                        self._mark_processed(comment.id)

                except Exception as e:
                    logger.error(f"Error processing comment: {e}")
                    continue

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise

    def reply_to_mentions(self):
        """Monitor and reply to username mentions"""

        logger.info("Monitoring mentions...")

        for mention in self.reddit.inbox.mentions(limit=25):
            if not self._is_processed(mention.id):
                try:
                    logger.info(f"Processing mention from {mention.author}")
                    self.generate_and_post_reply(mention)
                except Exception as e:
                    logger.error(f"Error replying to mention: {e}")

    def run(self, monitor_mentions: bool = True):
        """Run the bot"""

        logger.info(f"Starting Reddit bot for r/{self.subreddit.display_name}")
        logger.info(f"Trigger keywords: {self.trigger_keywords or 'None (all comments)'}")
        logger.info(f"Reply probability: {self.reply_probability}")

        # Check mentions first
        if monitor_mentions:
            self.reply_to_mentions()

        # Start monitoring
        self.monitor_comments()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('reddit_bot.log'),
            logging.StreamHandler()
        ]
    )

    bot = RedditBot(
        model_path="models/reddit_bot_lora",
        base_model=DEFAULT_BASE_MODEL,
        subreddit_name="test",
        bot_username="litiGPT",
        trigger_keywords=None,
        reply_probability=0.2,
        min_score_threshold=1,
        cooldown_seconds=120,
        available_users=["alice", "bob", "charlie"],
    )

    # bot.run()
