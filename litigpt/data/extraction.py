"""
Module 1: Data Extraction (Polars Version)
"""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

class RedditDataExtractor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_data(self, filename: str) -> pl.DataFrame:
        """Load JSONL or Parquet file automatically"""
        filepath = self.data_dir / filename

        if filepath.suffix == '.parquet':
            return pl.read_parquet(filepath)
        if filepath.suffix == '.jsonl':
            return pl.read_ndjson(filepath, infer_schema_length=None)
        raise ValueError(f"Unsupported format: {filepath.suffix}")

    def select_top_users(self,
                         comments: pl.DataFrame,
                         n: int,
                         exclude: list[str] | None = None,
                         min_comments: int = 100) -> list[str]:
        """
        Pick the N most prolific authors, skipping bots and deleted accounts.

        Args:
            comments: Full comments DataFrame.
            n: How many authors to return.
            exclude: Authors to skip entirely (bots, [deleted], ...).
            min_comments: Floor below which an author has too little data to
                learn a style from.
        """
        exclude = exclude or []
        counts = (
            comments
            .filter(~pl.col("author").is_in(exclude))
            .group_by("author")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") >= min_comments)
            .sort("n", descending=True)
            .head(n)
        )
        users = counts["author"].to_list()
        if len(users) < n:
            logger.warning(
                "Only %d authors clear the %d-comment floor (asked for %d)",
                len(users), min_comments, n,
            )
        for row in counts.iter_rows(named=True):
            logger.info("  %-28s %7d comments", row["author"], row["n"])
        return users

    def extract_user_data(self,
                         username: str,
                         comments_file: str = "comments.jsonl",
                         posts_file: str = "submissions.jsonl",
                         comments: pl.DataFrame | None = None,
                         posts: pl.DataFrame | None = None) -> pl.DataFrame:
        """
        Extract one user's comments and posts.

        Pass `comments`/`posts` to reuse an already-loaded corpus — extracting a
        cohort otherwise re-reads the whole dump once per user.
        """
        if comments is None:
            logger.info("Loading comments from %s...", comments_file)
            comments = self.load_data(comments_file)
        if posts is None:
            logger.info("Loading posts from %s...", posts_file)
            posts = self.load_data(posts_file)

        user_comments = (comments.filter(pl.col('author') == username)
                                 .with_columns(pl.lit('comment').alias('type')))
        user_posts = (posts.filter(pl.col('author') == username)
                           .with_columns(pl.lit('post').alias('type')))

        user_data = pl.concat([user_comments, user_posts], how='diagonal')
        logger.info("Extracted %d comments and %d posts for %s",
                    len(user_comments), len(user_posts), username)
        return user_data

    def extract_multiple_users(self,
                               usernames: list[str] | None = None,
                               comments_file: str = "comments.jsonl",
                               posts_file: str = "submissions.jsonl",
                               min_comments_per_user: int = 100,
                               top_n: int = 0,
                               exclude_authors: list[str] | None = None) -> dict[str, pl.DataFrame]:
        """
        Extract data for a cohort of users.

        The corpus is loaded once and filtered per user. Give `usernames` for an
        explicit cohort, or `top_n` to select the most prolific authors
        automatically.
        """
        logger.info("Loading comments from %s...", comments_file)
        comments = self.load_data(comments_file)
        logger.info("Loading posts from %s...", posts_file)
        posts = self.load_data(posts_file)

        if not usernames:
            if top_n <= 0:
                raise ValueError("Provide either `usernames` or a positive `top_n`.")
            logger.info("Selecting top %d authors by comment count:", top_n)
            usernames = self.select_top_users(
                comments, n=top_n, exclude=exclude_authors,
                min_comments=min_comments_per_user,
            )

        users_data = {}
        skipped = []
        for username in usernames:
            user_data = self.extract_user_data(
                username, comments=comments, posts=posts,
            )
            n_comments = len(user_data.filter(pl.col('type') == 'comment'))
            if n_comments < min_comments_per_user:
                logger.warning("Skipping %s: %d comments < %d minimum",
                               username, n_comments, min_comments_per_user)
                skipped.append(username)
                continue
            users_data[username] = user_data

        logger.info("=" * 60)
        logger.info("Extracted %d users (%d skipped for too little data)",
                    len(users_data), len(skipped))
        if skipped:
            logger.info("Skipped: %s", ", ".join(skipped))
        logger.info("=" * 60)
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

        logger.info("Saved data to %s", output_path)

    def save_multi_user_data(self,
                            users_data: dict[str, pl.DataFrame],
                            output_dir: str = "data/processed"):
        """Save multi-user data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save each user
        for username, data in users_data.items():
            user_file = output_path / f"{username}_data.parquet"
            data.write_parquet(user_file, compression='snappy')
            logger.info("Saved %s: %s", username, user_file)

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

        logger.info("Metadata saved: %s", metadata_file)

    def build_conversation_threads(self, comments: pl.DataFrame,
                                   posts: pl.DataFrame | None = None) -> dict[str, dict]:
        """
        Index comments and posts by id, so a reply can be walked up to its post.

        Posts come in separately. This used to take the comments file alone
        and sort its rows into comments or posts by whether a row had a
        'parent_id' key -- which every row of a DataFrame with that column
        has. The post index was therefore always empty: every reply to a post
        (10% of the eligible comments) was dropped for having no context, and
        no training thread ever opened with its post, while the Reddit bot
        opened every thread with one. Post rows are tagged type='post' here,
        which is how preprocessing knows to render them with render_post.
        """
        comments_by_id = {}
        comments_by_parent = {}
        for item in comments.to_dicts():
            item_id = item.get('id') or (item.get('name') or '').split('_')[-1]
            comments_by_id[item_id] = item
            parent_id = (item.get('parent_id') or '').split('_')[-1]
            comments_by_parent.setdefault(parent_id, []).append(item)

        posts_by_id = {}
        if posts is None:
            logger.warning(
                "No posts given to build_conversation_threads: every reply to a "
                "post will have no context and be dropped."
            )
        else:
            for item in posts.to_dicts():
                item_id = item.get('id') or (item.get('name') or '').split('_')[-1]
                item['type'] = 'post'
                posts_by_id[item_id] = item

        logger.info("Thread index: %d comments, %d posts",
                    len(comments_by_id), len(posts_by_id))
        return {
            'comments_by_id': comments_by_id,
            'comments_by_parent': comments_by_parent,
            'posts_by_id': posts_by_id
        }

    def get_context_for_comment(self, comment: dict, thread_data: dict,
                                max_context: int = 5) -> list[dict]:
        """Get parent comments/posts for context"""
        return get_context_for_comment(comment, thread_data, max_context)


def get_context_for_comment(comment: dict, thread_data: dict,
                            max_context: int = 5) -> list[dict]:
    """Get parent comments/posts for context (standalone version)"""
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
