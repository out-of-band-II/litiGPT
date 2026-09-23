"""
Module 5: Reddit Bot Deployment
Deploy bot to monitor and respond to Reddit comments
"""

import logging
import os
import random
import time
from collections import deque

import praw
from dotenv import load_dotenv
from praw.models import Comment

from litigpt.inference.classifier import RandomUserSelector, UserSelector

# Import inference module
from litigpt.inference.generator import RedditBotInference
from litigpt.prompts import render_thread

logger = logging.getLogger(__name__)

class RedditBot:
    def __init__(self,
                 model_path: str,
                 base_model: str,
                 subreddit_name: str,
                 bot_username: str,
                 trigger_keywords: list | None = None,
                 reply_probability: float = 0.3,
                 min_score_threshold: int = 1,
                 cooldown_seconds: int = 60,
                 available_users: list[str] | None = None,
                 user_selector: UserSelector | None = None,
                 max_depth: int = 3,
                 inference_config: dict | None = None,
                 mention_poll_seconds: int = 300,
                 idle_sleep_seconds: int = 5):
        """
        Initialize Reddit bot.

        The bot always generates responses as one of available_users, which
        must be non-empty: the user_selector picks, and when it has no
        preference a random one is used. There is no anonymous fallback --
        see classifier.check_bot_users.

        Args:
            model_path: Path to trained model
            base_model: Base model name
            subreddit_name: Subreddit to monitor
            bot_username: Bot's Reddit username
            trigger_keywords: Keywords that trigger responses (None = all)
            reply_probability: Probability of replying to eligible comments
            min_score_threshold: Minimum comment score to respond to
            cooldown_seconds: Seconds between responses
            available_users: Users the bot may post as; required
            user_selector: Strategy for selecting which user to respond as
            max_depth: Max parent comments for context
        """

        # Checked before logging in or loading a model, so a missing list
        # costs nothing. pipeline.run_deployment also checks it against the
        # adapter's manifest.
        if not available_users:
            raise ValueError("RedditBot needs a non-empty available_users list")

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
        self.available_users = list(available_users)
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

        # Mentions are polled from inside the comment loop, whenever the
        # stream has caught up. 0 makes the first idle moment check them.
        self.mention_poll_seconds = mention_poll_seconds
        self.idle_sleep_seconds = idle_sleep_seconds
        self._last_mention_check = 0

        logger.info("Bot initialized for r/%s", subreddit_name)
        logger.info("Available users: %s", ', '.join(self.available_users))

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
            raise OSError(
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

        logger.info("Logged in as: %s", reddit.user.me())
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
        return current_time - self.last_reply_time >= self.cooldown_seconds

    @staticmethod
    def _speaker(thing) -> str:
        """Author name of a comment or submission, as training spells it."""
        return thing.author.name if thing.author else "[deleted]"

    def _submission_turn(self, submission) -> tuple[str, str]:
        """
        Render a submission as one thread turn.

        Deliberately the post's author and selftext only. Training builds
        context items from the raw dump and renders a post exactly like a
        comment -- see RedditDataPreprocessor._format_context, which reads
        'author' and falls back from 'body' to 'selftext'. The title is not
        part of that, and neither is a "Post:" label. Adding either here
        would feed the model a shape it never saw in training.
        """
        return (self._speaker(submission), submission.selftext[:500])

    def get_comment_context(self, comment) -> str:
        """
        Build conversation context from a comment thread.

        Goes through litigpt.prompts.render_thread rather than formatting the
        lines here: that renderer is the single definition of the thread
        format, and this module posts to a live subreddit, where a format the
        model never trained on degrades every reply and raises nothing.
        """

        turns: list[tuple[str, str]] = []
        current = comment

        # Walk up to the parents, newest last.
        for _ in range(self.max_depth):
            if current.is_root:
                turns.insert(0, self._submission_turn(current.submission))
                break

            try:
                parent = current.parent()
                if isinstance(parent, Comment):
                    turns.insert(0, (self._speaker(parent), parent.body))
                    current = parent
                else:
                    # Parent is the submission.
                    turns.insert(0, self._submission_turn(parent))
                    break
            except Exception as e:
                logger.warning("Error getting parent: %s", e)
                break

        # The comment being replied to closes the thread.
        turns.append((self._speaker(comment), comment.body))

        return render_thread(turns)

    def select_user_for_context(self, context: str) -> str:
        """
        Select which user to respond as based on context.

        Returns:
            Username to impersonate: the selector's choice, or a random one
            of available_users when it has none (e.g. no keyword matched).
        """
        selected = self.user_selector.select_user(context, self.available_users)
        if selected in self.available_users:
            logger.info("Selected user: %s", selected)
            return selected

        fallback = random.choice(self.available_users)
        logger.info("No selector preference; random user: %s", fallback)
        return fallback

    def generate_and_post_reply(self, comment):
        """Generate response and post it"""

        try:
            # Get context
            context = self.get_comment_context(comment)
            logger.info("\nContext:\n%s\n", context)

            # Select user
            username = self.select_user_for_context(context)
            logger.info("Responding as: %s", username)

            # Generate response
            response = self.bot_inference.generate_response(
                context,
                username=username,
                max_new_tokens=self.inference_config.get("max_new_tokens", 256),
                temperature=self.inference_config.get("temperature", 0.8),
                top_p=self.inference_config.get("top_p", 0.9),
            )

            logger.info("Generated response: %s", response)

            # Add disclaimer
            disclaimer = f"\n\n---\n^(I'm a bot mimicking {username}'s style. Beep boop! [bot])"
            full_response = response + disclaimer

            # Post reply
            comment.reply(full_response)

            # Update tracking
            self._mark_processed(comment.id)
            self.last_reply_time = time.time()

            logger.info("Posted reply to comment %s", comment.id)

        except Exception as e:
            logger.error("Error posting reply: %s", e)

    def monitor_comments(self, monitor_mentions: bool = True):
        """
        Monitor the subreddit for new comments, and service mentions in the gaps.

        The stream is opened with pause_after=-1 so it yields None once it has
        caught up instead of blocking. That idle moment is the only place the
        loop is free to do anything else, so it is where mentions are checked.
        Opening the stream without it is why mentions used to be polled once at
        startup and then never again for the life of the process.
        """

        logger.info("Starting comment monitoring...")

        try:
            for comment in self.subreddit.stream.comments(skip_existing=True,
                                                          pause_after=-1):
                if comment is None:
                    # Caught up with the subreddit.
                    if monitor_mentions:
                        self._poll_mentions_if_due()
                    time.sleep(self.idle_sleep_seconds)
                    continue

                try:
                    if self.should_respond(comment):
                        logger.info("\nProcessing comment %s by %s", comment.id, comment.author)
                        self.generate_and_post_reply(comment)
                    else:
                        self._mark_processed(comment.id)

                except Exception as e:
                    logger.error("Error processing comment: %s", e)
                    continue

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error("Fatal error: %s", e)
            raise

    def _poll_mentions_if_due(self):
        """Check mentions if enough time has passed since the last check."""
        now = time.time()
        if now - self._last_mention_check < self.mention_poll_seconds:
            return
        self._last_mention_check = now
        try:
            self.reply_to_mentions()
        except Exception as e:
            logger.error("Error polling mentions: %s", e)

    def reply_to_mentions(self):
        """
        Reply to username mentions of the bot.

        Open question: a mention is the one place a human states intent, so it
        is the natural way to ask the bot to answer *as* a particular persona
        ("u/litiGPT come tommyrugby"). Nothing parses that yet -- the persona
        still comes from self.user_selector like any other reply. Implementing
        it means a UserSelector that reads the request out of the context, not
        a special case here.
        """

        for mention in self.reddit.inbox.mentions(limit=25):
            if self._is_processed(mention.id):
                continue
            try:
                logger.info("Processing mention from %s", mention.author)
                self.generate_and_post_reply(mention)
            except Exception as e:
                logger.error("Error replying to mention: %s", e)
            finally:
                # Mark either way: a mention that failed once will fail again
                # on every poll, and retrying it forever starves the stream.
                self._mark_processed(mention.id)

    def run(self, monitor_mentions: bool = True):
        """Run the bot"""

        logger.info("Starting Reddit bot for r/%s", self.subreddit.display_name)
        logger.info("Trigger keywords: %s", self.trigger_keywords or 'None (all comments)')
        logger.info("Reply probability: %s", self.reply_probability)
        if monitor_mentions:
            logger.info("Mention poll interval: %ss", self.mention_poll_seconds)

        self.monitor_comments(monitor_mentions=monitor_mentions)


# No __main__ here on purpose. There used to be one, and it constructed a bot
# with a hardcoded model path, subreddit and persona list, read no config, and
# then left `bot.run()` commented out -- so `python -m litigpt.deployment
# .reddit_bot` loaded a 4-bit model, authenticated to Reddit and exited. The
# single entry point is the pipeline, which builds all of this from config:
#
#     python -m litigpt.pipeline --step deploy --config config.top30.yaml
