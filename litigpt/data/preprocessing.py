"""
Module 2: Data Preprocessing
Clean and format data for training
"""

import html
import random
import re
import logging
from typing import List, Dict, Tuple
import jsonlines
from pathlib import Path
import polars as pl
from argparse import ArgumentParser

from litigpt.prompts import (
    build_system_prompt,
    build_alpaca_instruction,
    render_thread,
)

logger = logging.getLogger(__name__)

class RedditDataPreprocessor:
    def __init__(self, min_length: int = 10, max_length: int = 512,
                 min_score: int = 1, strip_quoted_text: bool = True):
        self.min_length = min_length
        self.max_length = max_length
        self.min_score = min_score
        self.strip_quoted_text = strip_quoted_text

    # A quoted line is Reddit's "> ..." markup. The parent comment is already
    # supplied as context, so leaving the quote in a user's own reply trains the
    # model to reproduce text it cannot actually see at inference time.
    _QUOTE_LINE = re.compile(r"^[ \t]*>.*$", re.MULTILINE)
    _MD_LINK = re.compile(r"\[([^\]]*)\]\(https?://[^)]*\)")
    _BARE_URL = re.compile(r"https?://\S+")
    # Zero-width and BOM characters carry no style signal, only noise tokens.
    _ZERO_WIDTH = re.compile(r"[​‌‍﻿]")

    @staticmethod
    def _unescape_fully(text: str, max_passes: int = 3) -> str:
        """
        Decode HTML entities repeatedly until stable.

        This dump is double-escaped in places ("&amp;#x200B;"), so a single
        html.unescape pass leaves a live "&#x200B;" behind. Bounded to keep a
        pathological input from looping.
        """
        for _ in range(max_passes):
            decoded = html.unescape(text)
            if decoded == text:
                break
            text = decoded
        return text

    def clean_text(self, text: str) -> str:
        """
        Clean Reddit markup out of a comment body.

        Order matters: HTML entities are decoded first so that "&gt;" becomes a
        recognisable quote marker, quote lines are dropped while newlines still
        delimit them, and only then is whitespace normalised.
        """
        if not isinstance(text, str):
            return ""

        if text.strip().lower() in ("[deleted]", "[removed]", "none", ""):
            return ""

        # 1. Decode entities (&gt; &amp; &#x200B; ...), repeatedly: parts of
        #    this dump are double-escaped.
        text = self._unescape_fully(text)
        text = self._ZERO_WIDTH.sub('', text)

        # 2. Links: keep the anchor text, drop the URL. This runs before the
        #    quote strip because an anchor can hold a quote marker itself --
        #    "[&gt; quoted](url)" only reveals its ">" once unwrapped.
        text = self._MD_LINK.sub(r"\1", text)
        text = self._BARE_URL.sub("", text)

        # 3. Drop quoted parent text, while line boundaries still exist.
        if self.strip_quoted_text:
            text = self._QUOTE_LINE.sub("", text)

        # 4. Normalise whitespace but keep paragraph breaks, which carry some of
        #    the rhythm of how someone writes.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def filter_quality(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter low-quality entries"""
        # Clean body/selftext - coalesce body and selftext, then clean
        df = df.with_columns(
            pl.when(pl.col('body').is_not_null() & (pl.col('body') != ''))
            .then(pl.col('body'))
            .otherwise(
                pl.when('selftext' in df.columns)
                .then(pl.col('selftext'))
                .otherwise(pl.lit(''))
            )
            .alias('body')
        ) if 'selftext' in df.columns else df

        # Apply clean_text to body column
        df = df.with_columns(
            pl.col('body').map_elements(self.clean_text, return_dtype=pl.Utf8).alias('body')
        )

        # Add text length column
        df = df.with_columns(
            pl.col('body').str.len_chars().alias('text_length')
        )

        # Filter by length
        df = df.filter(
            (pl.col('text_length') >= self.min_length) &
            (pl.col('text_length') <= self.max_length)
        )

        # Filter deleted/removed
        df = df.filter(pl.col('body') != '')

        # Filter by score (optional - only keep upvoted content)
        if 'score' in df.columns:
            df = df.with_columns(pl.col('score').cast(pl.Int64, strict=False))
            df = df.filter(pl.col('score') >= self.min_score)

        logger.info(f"After filtering: {len(df)} entries")
        return df

    def create_training_pairs(self, user_data: pl.DataFrame,
                             thread_data: Dict,
                             username: str = None) -> List[Dict]:
        """
        Create (context, response) pairs for training

        Args:
            user_data: User's comments/posts (Polars DataFrame)
            thread_data: Pre-built conversation threads from RedditDataExtractor.build_conversation_threads
            username: Username to tag in training data (for multi-user)
        """
        from litigpt.data.extraction import get_context_for_comment

        training_pairs = []

        # Iterate over Polars DataFrame using to_dicts()
        for row in user_data.to_dicts():
            if row.get('type') != 'comment':
                continue

            # Get context
            context_items = get_context_for_comment(
                row,
                thread_data,
                max_context=3
            )

            if not context_items:
                continue

            # Format context
            context_text = self._format_context(context_items)
            response_text = row.get('body', '')

            if context_text and response_text:
                pair = {
                    'context': context_text,
                    'response': response_text,
                    'score': row.get('score', 0),
                    'created_utc': row.get('created_utc', 0)
                }

                # Add username if doing multi-user training
                if username:
                    pair['username'] = username

                training_pairs.append(pair)

        logger.info(f"Created {len(training_pairs)} training pairs")
        return training_pairs

    def create_multi_user_training_pairs(self,
                                        users_data: Dict[str, pl.DataFrame],
                                        thread_data: Dict,
                                        max_pairs_per_user: int = 0,
                                        seed: int = 0) -> List[Dict]:
        """
        Create training pairs for a cohort of users.

        Args:
            users_data: Mapping of username to that user's DataFrame.
            thread_data: Threads from RedditDataExtractor.build_conversation_threads.
            max_pairs_per_user: Cap per user so one prolific author cannot
                dominate the mix. 0 disables the cap. Sampling is random with a
                fixed seed rather than head-truncation, which would bias the set
                toward whichever period the dump happens to start in.
            seed: Seed for that sampling, so runs stay reproducible.

        Returns:
            Training pairs, each tagged with its username.
        """
        rng = random.Random(seed)
        all_pairs = []
        counts = {}

        for username, user_data in users_data.items():
            logger.info("Processing %s...", username)
            pairs = self.create_training_pairs(user_data, thread_data, username)

            if max_pairs_per_user and len(pairs) > max_pairs_per_user:
                logger.info("  capping %s: %d -> %d pairs",
                            username, len(pairs), max_pairs_per_user)
                pairs = rng.sample(pairs, max_pairs_per_user)

            counts[username] = len(pairs)
            all_pairs.extend(pairs)

        logger.info("Total training pairs: %d across %d users",
                    len(all_pairs), len(users_data))
        if counts:
            lo = min(counts.values())
            hi = max(counts.values())
            logger.info("  per user: min %d, max %d, imbalance %.1fx",
                        lo, hi, hi / max(lo, 1))
        return all_pairs

    def _format_context(self, context_items: List[Dict]) -> str:
        """
        Format context items into a single string.

        This is the definition of the thread format every interface has to
        match at inference time, so it goes through the shared renderer rather
        than building the string here.
        """
        return render_thread(
            (
                item.get('author', 'unknown'),
                self.clean_text(item.get('body') or item.get('selftext', '')),
            )
            for item in context_items
        )

    def format_for_training(self, pairs: List[Dict],
                           format_type: str = "chatml") -> List[Dict]:
        """
        Format training pairs for specific model format

        Formats:
        - chatml: ChatML format (for Llama, Mistral)
        - alpaca: Alpaca instruction format
        - raw: Simple context -> response

        Args:
            pairs: Training pairs
            format_type: Format to use
        """

        formatted_data = []

        for pair in pairs:
            username = pair.get('username')
            if not username:
                # Without a username the prompt would read "Sei unknown",
                # training the model on a persona that does not exist.
                logger.warning("Skipping pair with no username attached")
                continue

            if format_type == "chatml":
                system_content = build_system_prompt(username)

                formatted = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_content
                        },
                        {
                            "role": "user",
                            "content": pair['context']
                        },
                        {
                            "role": "assistant",
                            "content": pair['response']
                        }
                    ],
                    "username": username,
                }

            elif format_type == "alpaca":
                formatted = {
                    "instruction": build_alpaca_instruction(username),
                    "input": pair['context'],
                    "output": pair['response'],
                    "username": username,
                }

            else:  # raw
                formatted = {
                    "text": f"### User: {username}\n### Context:\n{pair['context']}\n\n### Response:\n{pair['response']}"
                }

            formatted_data.append(formatted)

        return formatted_data

    def split_data(self, data: List[Dict],
                   train_ratio: float = 0.9) -> Tuple[List[Dict], List[Dict]]:
        """Split into train/validation sets"""
        import random
        random.shuffle(data)

        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        logger.info(f"Train: {len(train_data)}, Validation: {len(val_data)}")
        return train_data, val_data

    def save_training_data(self, train_data: List[Dict], val_data: List[Dict],
                          output_dir: str = "data/training"):
        """Save formatted training data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as JSONL
        with jsonlines.open(output_path / "train.jsonl", 'w') as writer:
            writer.write_all(train_data)

        with jsonlines.open(output_path / "val.jsonl", 'w') as writer:
            writer.write_all(val_data)

        logger.info(f"Saved training data to {output_dir}")

def preprocessing_parser():
    parser = ArgumentParser(description="Preprocessing module",
                            epilog=f"""
    Examples:
    python %(prog)s -c litigi_comments.parquet -u outofband
    """)
    parser.add_argument("--dir", "-d", required=False, default="data/raw", help="Data directory")
    parser.add_argument("--comments", "-c", required=False, default="comments.parquet",
                       help="Comment data")
    parser.add_argument("--user", "-u", required=True)

    return parser


if __name__ == "__main__":

    parser = preprocessing_parser()
    args = parser.parse_args()

    comments_data_file = args.comments
    user = args.user
    raw_data_dir = args.dir

    preprocessor = RedditDataPreprocessor()

    # Load user data (Polars)
    user_data = pl.read_parquet(f"data/processed/{user}_data.parquet")

    # Load all comments and build conversation threads for context
    from litigpt.data.extraction import RedditDataExtractor
    extractor = RedditDataExtractor(raw_data_dir)
    all_comments = extractor.load_data(comments_data_file)
    thread_data = extractor.build_conversation_threads(all_comments)

    # Filter quality
    user_data = preprocessor.filter_quality(user_data)

    # Create training pairs
    pairs = preprocessor.create_training_pairs(user_data, thread_data)

    # Format for training
    formatted = preprocessor.format_for_training(pairs, format_type="chatml")

    # Split and save
    train, val = preprocessor.split_data(formatted)
    preprocessor.save_training_data(train, val)
