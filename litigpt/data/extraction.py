"""
Module 1: Data Extraction (Polars Version)
"""

import json
import logging
import polars as pl
from pathlib import Path
from typing import Dict, List
from argparse import ArgumentParser

logger = logging.getLogger(__name__)

class RedditDataExtractor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_data(self, filename: str) -> pl.DataFrame:
        """Load JSONL or Parquet file automatically"""
        filepath = self.data_dir / filename

        if filepath.suffix == '.parquet':
            return pl.read_parquet(filepath)
        elif filepath.suffix == '.jsonl':
            return pl.read_ndjson(filepath, infer_schema_length=None)
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")

    def extract_user_data(self,
                         username: str,
                         comments_file: str = "comments.jsonl",
                         posts_file: str = "submissions.jsonl") -> pl.DataFrame:
        """Extract user data (comments + posts)"""
        logger.info(f"Loading comments from {comments_file}...")
        comments = self.load_data(comments_file)

        # Filter by user
        user_comments = comments.filter(pl.col('author') == username)
        user_comments = user_comments.with_columns(pl.lit('comment').alias('type'))

        logger.info(f"Loading posts from {posts_file}...")
        posts = self.load_data(posts_file)

        user_posts = posts.filter(pl.col('author') == username)
        user_posts = user_posts.with_columns(pl.lit('post').alias('type'))

        # Combine (diagonal to handle different columns)
        user_data = pl.concat([
            user_comments,
            user_posts
        ], how='diagonal')

        logger.info(f"Extracted {len(user_comments)} comments and {len(user_posts)} posts for {username}")
        return user_data

    def extract_multiple_users(self,
                               usernames: List[str],
                               comments_file: str = "comments.jsonl",
                               posts_file: str = "submissions.jsonl",
                               min_comments_per_user: int = 100) -> Dict[str, pl.DataFrame]:
        """Extract data for multiple users"""
        users_data = {}

        for username in usernames:
            logger.info(f"{'='*60}")
            logger.info(f"Processing user: {username}")
            logger.info('='*60)

            user_data = self.extract_user_data(username, comments_file, posts_file)

            users_data[username] = user_data
            logger.info(f"Added {username} with {len(user_data)} items")

        logger.info(f"{'='*60}")
        logger.info(f"Total users extracted: {len(users_data)}")
        logger.info('='*60)

        return users_data

    def save_processed_data(self, data: pl.DataFrame, output_path: str):
        """Save processed data as JSONL or Parquet"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.suffix == '.parquet':
            data.write_parquet(output_path, compression='snappy')
        elif output_file.suffix == '.jsonl':
            data.write_ndjson(output_path)
        else:
            data.write_ndjson(output_path)

        logger.info(f"Saved data to {output_path}")

    def save_multi_user_data(self,
                            users_data: Dict[str, pl.DataFrame],
                            output_dir: str = "data/processed"):
        """Save multi-user data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save each user
        for username, data in users_data.items():
            user_file = output_path / f"{username}_data.parquet"
            data.write_parquet(user_file, compression='snappy')
            logger.info(f"Saved {username}: {user_file}")

        # Metadata
        metadata = {
            'users': list(users_data.keys()),
            'user_stats': {
                username: {
                    'total_items': len(data),
                    'comments': len(data.filter(pl.col('type') == 'comment')),
                    'posts': len(data.filter(pl.col('type') == 'post'))
                }
                for username, data in users_data.items()
            }
        }

        metadata_file = output_path / "users_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved: {metadata_file}")

    def build_conversation_threads(self, all_data_pl: pl.DataFrame) -> Dict[str, List[Dict]]:
        """Build conversation threads from all subreddit data"""

        all_data = all_data_pl.to_dicts()
        # Create lookup dictionaries
        comments_by_id = {}
        comments_by_parent = {}
        posts_by_id = {}

        for item in all_data:
            item_id = item.get('id') or item.get('name', '').split('_')[-1]

            if item.get('type') == 'comment' or 'parent_id' in item:
                comments_by_id[item_id] = item
                pid = item.get('parent_id', '')
                try:
                    parent_id = pid.split('_')[-1]
                except Exception:
                    logger.error(f"Error processing parent_id for {item}")
                    continue
                if parent_id not in comments_by_parent:
                    comments_by_parent[parent_id] = []
                comments_by_parent[parent_id].append(item)
            else:
                posts_by_id[item_id] = item

        return {
            'comments_by_id': comments_by_id,
            'comments_by_parent': comments_by_parent,
            'posts_by_id': posts_by_id
        }

    def get_context_for_comment(self, comment: Dict, thread_data: Dict,
                                max_context: int = 5) -> List[Dict]:
        """Get parent comments/posts for context"""
        context = []
        current = comment

        for _ in range(max_context):
            parent_id = current.get('parent_id', '').split('_')[-1]
            if not parent_id:
                break

            # Check if parent is a comment
            parent = thread_data['comments_by_id'].get(parent_id)
            if parent:
                context.insert(0, parent)
                current = parent
            else:
                # Check if parent is a post
                parent = thread_data['posts_by_id'].get(parent_id)
                if parent:
                    context.insert(0, parent)
                break

        return context

def data_extraction_parser():
    parser = ArgumentParser(description="Data extraction module",
                            epilog="""
    Examples:
    python %(prog)s -c litigi_comments.parquet -s submissions.jsonl --users alice bob
    """)

    parser.add_argument("--dir", "-d", required=False, default="data/raw", help="Data directory")
    parser.add_argument("--comments", "-c", required=False, default="comments.parquet",
                       help="Comment data")
    parser.add_argument("--submissions", "-s", required=False, default="submissions.jsonl", help="Submission data")
    parser.add_argument("--users", nargs="+", type=str, required=True,
                       help="One or more usernames to extract")

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = data_extraction_parser()
    args = parser.parse_args()
    extractor = RedditDataExtractor(args.dir)

    users_data = extractor.extract_multiple_users(
        usernames=args.users,
        comments_file=args.comments,
        posts_file=args.submissions
    )
    extractor.save_multi_user_data(users_data, "data/processed")
