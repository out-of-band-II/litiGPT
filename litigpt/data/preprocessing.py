"""
Module 2: Data Preprocessing
Clean and format data for training
"""

import re
import logging
from typing import List, Dict, Tuple
import jsonlines
from pathlib import Path
import polars as pl
from argparse import ArgumentParser

from litigpt.prompts import build_system_prompt, build_alpaca_instruction

logger = logging.getLogger(__name__)

class RedditDataPreprocessor:
    def __init__(self, min_length: int = 10, max_length: int = 512):
        self.min_length = min_length
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """Clean Reddit text"""
        if not isinstance(text, str):
            return ""

        # Remove deleted/removed content
        if text.lower() in ['[deleted]', '[removed]', 'none']:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove Reddit formatting artifacts
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')

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
            df = df.filter(pl.col('score') > 0)

        logger.info(f"After filtering: {len(df)} entries")
        return df

    def create_training_pairs(self, user_data: pl.DataFrame,
                             all_comments: pl.DataFrame,
                             username: str = None, raw_data_dir: str = "data/raw") -> List[Dict]:
        """
        Create (context, response) pairs for training

        Args:
            user_data: User's comments/posts (Polars DataFrame)
            all_comments: All subreddit comments for context (Polars DataFrame)
            username: Username to tag in training data (for multi-user)
        """
        from litigpt.data.extraction import RedditDataExtractor

        extractor = RedditDataExtractor(raw_data_dir)
        thread_data = extractor.build_conversation_threads(all_comments)

        training_pairs = []

        # Iterate over Polars DataFrame using to_dicts()
        for row in user_data.to_dicts():
            if row.get('type') != 'comment':
                continue

            # Get context
            context_items = extractor.get_context_for_comment(
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
                                        all_comments: pl.DataFrame) -> List[Dict]:
        """
        Create training pairs for multiple users

        Args:
            users_data: Dictionary mapping username to their data (Polars DataFrames)
            all_comments: All subreddit comments (Polars DataFrame)

        Returns:
            List of training pairs with username tags
        """
        all_pairs = []

        for username, user_data in users_data.items():
            logger.info(f"Processing {username}...")
            pairs = self.create_training_pairs(user_data, all_comments, username)
            all_pairs.extend(pairs)

        logger.info(f"Total training pairs: {len(all_pairs)}")
        logger.info(f"Users: {list(users_data.keys())}")

        return all_pairs

    def _format_context(self, context_items: List[Dict]) -> str:
        """Format context items into a single string"""
        formatted = []

        for item in context_items:
            author = item.get('author', 'unknown')
            body = self.clean_text(item.get('body') or item.get('selftext', ''))

            if body:
                formatted.append(f"{author}: {body}")

        return "\n".join(formatted)

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
            username = pair.get('username', 'unknown')

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

    # Load all comments for context building
    from litigpt.data.extraction import RedditDataExtractor
    extractor = RedditDataExtractor(raw_data_dir)
    all_comments = extractor.load_data(comments_data_file)

    # Filter quality
    user_data = preprocessor.filter_quality(user_data)

    # Create training pairs
    pairs = preprocessor.create_training_pairs(user_data, all_comments)

    # Format for training
    formatted = preprocessor.format_for_training(pairs, format_type="chatml")

    # Split and save
    train, val = preprocessor.split_data(formatted)
    preprocessor.save_training_data(train, val)
