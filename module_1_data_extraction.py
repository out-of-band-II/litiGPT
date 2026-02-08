"""
Module 1: Data Extraction
Extract target user's comments and posts from Reddit JSONL data
"""

import jsonlines
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
import zstandard as zstd
import io
import polars as pl
from argparse import ArgumentParser

class RedditDataExtractor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_jsonl(self, filename: str) -> List[Dict]:
        """Load a JSONL file"""
        data = []
        filepath = self.data_dir / filename
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                data.append(obj)
        return data
    
    def extract_zstd(self,filename,condition=None):
        """
        Processes a JSON stream from a compressed file and yields objects based on condition.

        Args:
            filepath (str): Path to the compressed JSON file
            condition (callable): Function to evaluate each object; if None, all objects are yielded

        Yields:
            dict: Each JSON object that meets the condition
        """
        i=0
        filepath = self.data_dir /filename
        with open(filepath, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
            with dctx.stream_reader(compressed_file) as stream_reader:
                # Read all content into a buffer
                text_content = io.TextIOWrapper(stream_reader, encoding='utf-8')
                for line in text_content:
                    obj = json.loads(line)
                    if condition is None or condition(obj):
                        i=i+1
                        if i%1000==0:
                            print (i, ' rows extracted.')
                        yield obj

    def _load_reddit_data_file(self,filename:str):
        admitted_ext = [".jsonl",".zst",".parquet"]
        print(f"Loading data from {filename}...")
        if Path(filename).suffix == ".jsonl":
            data = self.load_jsonl(filename)
        elif Path(filename).suffix == ".zst":
            data :List[Dict] = [entry for entry in self.extract_zstd(filename)]
            df =pl.DataFrame(data,infer_schema_length=None).drop(['media_embed','secure_media_embed'])
            print(df.describe())
            try:
                df.write_parquet((Path(self.data_dir)/filename).with_suffix('.parquet'))
            except Exception as e:
                print(f"Error {e} occured")
                df.to_pandas().to_parquet((Path(self.data_dir)/filename).with_suffix('.parquet'))

        elif Path(filename).suffix == ".parquet":
            data = pl.read_parquet(self.data_dir/filename).to_dicts()
        else:
            raise ValueError(f"{filename} suffix ({Path(filename).suffix}) not recognized (must be onf of {', '.join(admitted_ext)})")
        
        return data
        
    def extract_user_data(self, username: str, 
                         comments_file: str = "comments.jsonl",
                         posts_file: str = "submissions.jsonl") -> pd.DataFrame:
        """Extract all comments and posts from a specific user"""
        
        # Load comments
        print(f"Loading comments from {comments_file}...")
        comments = self._load_reddit_data_file(comments_file)
        user_comments = [c for c in comments if c.get('author') == username]
        
        # Load posts
        print(f"Loading posts from {posts_file}...")
        posts = self._load_reddit_data_file(posts_file)
        user_posts = [p for p in posts if p.get('author') == username]
        
        # Convert to DataFrame
        comments_df = pd.DataFrame(user_comments)
        posts_df = pd.DataFrame(user_posts)
        
        # Add type column
        if not comments_df.empty:
            comments_df['type'] = 'comment'
        if not posts_df.empty:
            posts_df['type'] = 'post'
        
        # Combine
        user_data = pd.concat([comments_df, posts_df], ignore_index=True)
        
        print(f"Extracted {len(user_comments)} comments and {len(user_posts)} posts")
        return user_data
    
    def extract_multiple_users(self, usernames: List[str],
                               comments_file: str = "comments.jsonl",
                               posts_file: str = "submissions.jsonl",
                               min_comments_per_user: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Extract data for multiple users
        
        Args:
            usernames: List of usernames to extract
            comments_file: Path to comments JSONL
            posts_file: Path to posts JSONL
            min_comments_per_user: Minimum comments required per user
            
        Returns:
            Dictionary mapping username to their data
        """
        users_data = {}
        
        for username in usernames:
            print(f"\n{'='*60}")
            print(f"Processing user: {username}")
            print('='*60)
            
            user_data = self.extract_user_data(username, comments_file, posts_file)
            
            if len(user_data) < min_comments_per_user:
                print(f"   Warning: User {username} has only {len(user_data)} items (min: {min_comments_per_user})")
                print(f"   Skipping {username}")
                continue
            
            users_data[username] = user_data
            print(f"✓ Added {username} with {len(user_data)} items")
        
        print(f"\n{'='*60}")
        print(f"Total users extracted: {len(users_data)}")
        print('='*60)
        
        return users_data
    
    def save_multi_user_data(self, users_data: Dict[str, pd.DataFrame], 
                            output_dir: str = "data/processed"):
        """Save multi-user data with metadata"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each user's data
        for username, data in users_data.items():
            user_file = output_path / f"{username}_data.jsonl"
            data.to_json(user_file, orient='records', lines=True)
            print(f"Saved {username}: {user_file}")
        
        # Save metadata
        metadata = {
            'users': list(users_data.keys()),
            'user_stats': {
                username: {
                    'total_items': len(data),
                    'comments': len(data[data['type'] == 'comment']),
                    'posts': len(data[data['type'] == 'post'])
                }
                for username, data in users_data.items()
            }
        }
        
        import json
        metadata_file = output_path / "users_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nMetadata saved: {metadata_file}")
    
    def build_conversation_threads(self, all_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Build conversation threads from all subreddit data"""
        
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
                except:
                    print(f"Error processign {item}")
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
    
    def save_processed_data(self, data: pd.DataFrame, output_path: str):
        """Save processed data"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_json(output_path, orient='records', lines=True)
        print(f"Saved to {output_path}")

def data_extraction_parser():
    parser = ArgumentParser(description="Data extraction module",
                            epilog=f"""
    Examples:
    python %(prog)s -c litigi_comments.parquet -s submissions.jsonl --multi-user
    """)

    parser.add_argument("--dir","-d", required=False, default="data/raw", help="Data directory")
    parser.add_argument("--comments","-c", required=False, default="comments.parquet",
                       help="Comment data")
    parser.add_argument("--submissions","-s", required=False, default="submissions.jsonl", help="Submissison data")
    parser.add_argument("--multi-user", action="store_true", 
                       help="Enable multi-user mode")
    

    return parser
    

if __name__ == "__main__":
    # Example usage - Single user
    parser = data_extraction_parser()
    args = parser.parse_args()
    submission_data_file = args.submissions
    comments_data_file = args.comments

    multi_user:bool = args.multi_user
    extractor = RedditDataExtractor("data/raw")
    if not multi_user:
        
        user_data = extractor.extract_user_data("target_username",comments_data_file,submission_data_file)
        extractor.save_processed_data(user_data, "data/processed/user_data.jsonl")
    else:
        # Example usage - Multiple users
        users = ["user1", "user2", "user3"]
        users_data = extractor.extract_multiple_users(
            usernames=users,
            min_comments_per_user=100,
            comments_file=comments_data_file,
            posts_file = submission_data_file
        )
        extractor.save_multi_user_data(users_data, "data/processed")